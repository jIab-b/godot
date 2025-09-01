import torch
import warnings
from diffusers import AutoencoderKL
import os
import glob


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv

    def forward(self, latents):
        latents = self.post_quant_conv(latents)
        image = self.decoder(latents)
        return image


def export_vae_decoder():
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    ).to("cuda").eval()
    
    wrapper = VAEDecoderWrapper(vae).to("cuda", torch.float16)

    batch = 1
    latent_channels = 4
    latent_height = 128
    latent_width = 128
    dummy_latents = torch.randn(batch, latent_channels, latent_height, latent_width, device="cuda").half()

    input_names = ["latents"]
    output_names = ["images"]
    
    dynamic_axes = {
        "latents": {0: "batch", 2: "height", 3: "width"},
        "images": {0: "batch", 2: "image_height", 3: "image_width"},
    }

    print("Exporting VAE Decoder...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (dummy_latents,),
                "vae_decoder_temp.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL,
            )
    
    try:
        import onnx
        from onnx import save_model
        
        print("Consolidating VAE Decoder into single file...")
        model = onnx.load("vae_decoder_temp.onnx", load_external_data=True)
        
        save_model(
            model,
            "vae_decoder.onnx",
            save_as_external_data=False,
        )
        
        os.remove("vae_decoder_temp.onnx")
        
        removed = 0
        for pattern in ("*.onnx.data", "onnx__*", "vae_decoder.*"):
            for filepath in glob.glob(pattern):
                if filepath != "vae_decoder.onnx":
                    try:
                        os.remove(filepath)
                        removed += 1
                    except OSError:
                        pass
        
        if removed > 0:
            print(f"Cleaned up {removed} external/shard files")
            
    except ImportError:
        print("Warning: onnx package not available, may have external files")
        if os.path.exists("vae_decoder_temp.onnx"):
            os.rename("vae_decoder_temp.onnx", "vae_decoder.onnx")

    print("VAE Decoder exported to ONNX successfully")

if __name__ == "__main__":
    export_vae_decoder()
