import torch
import warnings
from diffusers import UNet2DConditionModel, T2IAdapter
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        text_embeds,
        time_ids,
        adapter_feat_0,
        adapter_feat_1, 
        adapter_feat_2,
        adapter_feat_3,
    ):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        down_intrablock_additional_residuals = [
            adapter_feat_0, adapter_feat_1, adapter_feat_2, adapter_feat_3
        ]
        
        result = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            return_dict=False,
        )
        return result[0]


def export_unet():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    unet.eval()

    wrapper = UNetWrapper(unet).to("cuda", torch.float16)

    batch = 1
    sample = torch.randn(batch, 4, 128, 128, device="cuda").half()
    timestep = torch.full((batch,), 1.0, device="cuda").half()
    encoder_hidden_states = torch.randn(batch, 77, 2048, device="cuda").half()
    text_embeds = torch.randn(batch, 1280, device="cuda").half()
    time_ids = torch.randn(batch, 6, device="cuda").half()

    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    depth = torch.randn(batch, 3, 1024, 1024, device="cuda").half()
    with torch.inference_mode():
        adapter_feats = adapter(depth)
        if isinstance(adapter_feats, torch.Tensor):
            adapter_feats = (adapter_feats,)
        down_intra = [f.contiguous() for f in adapter_feats]
        
        while len(down_intra) < 4:
            down_intra.append(torch.zeros_like(down_intra[0]))

    input_tensors = (
        sample,
        timestep,
        encoder_hidden_states,
        text_embeds,
        time_ids,
        down_intra[0],
        down_intra[1],
        down_intra[2],
        down_intra[3],
    )

    input_names = [
        "sample",
        "timestep",
        "encoder_hidden_states",
        "text_embeds",
        "time_ids",
        "adapter_feat_0",
        "adapter_feat_1",
        "adapter_feat_2",
        "adapter_feat_3",
    ]

    dynamic_axes = {
        "sample": {0: "batch", 2: "height", 3: "width"},
        "timestep": {0: "batch"},
        "encoder_hidden_states": {0: "batch"},
        "text_embeds": {0: "batch"},
        "time_ids": {0: "batch"},
        "adapter_feat_0": {0: "batch"},
        "adapter_feat_1": {0: "batch"},
        "adapter_feat_2": {0: "batch"},
        "adapter_feat_3": {0: "batch"},
        "out_sample": {0: "batch", 2: "height", 3: "width"},
    }

    print("Exporting ONNX model...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                input_tensors,
                "sdxl_unet_temp.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=False,
                input_names=input_names,
                output_names=["out_sample"],
                dynamic_axes=dynamic_axes,
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL,
            )
    
    import os
    import glob
    
    try:
        import onnx
        from onnx import save_model
        
        print("Consolidating ONNX model into single file...")
        model = onnx.load("sdxl_unet_temp.onnx", load_external_data=True)
        
        save_model(
            model,
            "sdxl_unet.onnx",
            save_as_external_data=False,
        )
        
        os.remove("sdxl_unet_temp.onnx")
        
        removed = 0
        for pattern in ("*.onnx.data", "onnx__*", "unet.*"):
            for filepath in glob.glob(pattern):
                try:
                    os.remove(filepath)
                    removed += 1
                except OSError:
                    pass
        
        if removed > 0:
            print(f"Cleaned up {removed} external/shard files")
            
    except ImportError:
        print("Warning: onnx package not available, may have external files")
        if os.path.exists("sdxl_unet_temp.onnx"):
            os.rename("sdxl_unet_temp.onnx", "sdxl_unet.onnx")

    print("SDXL UNet exported to ONNX successfully")

if __name__ == "__main__":
    export_unet()
