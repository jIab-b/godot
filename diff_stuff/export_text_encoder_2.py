import torch
import warnings
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
import os
import glob


class TextEncoder2Wrapper(torch.nn.Module):
    def __init__(self, text_encoder_2):
        super().__init__()
        self.text_encoder_2 = text_encoder_2

    def forward(self, input_ids):
        outputs = self.text_encoder_2(input_ids, return_dict=False)
        return outputs[0], outputs[1]


def export_text_encoder_2():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda").eval()
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer_2"
    )
    
    wrapper = TextEncoder2Wrapper(text_encoder_2).to("cuda", torch.float16)

    batch = 1
    max_length = 77
    dummy_input_ids = torch.randint(0, tokenizer_2.vocab_size, (batch, max_length), device="cuda")

    input_names = ["input_ids"]
    output_names = ["text_embeddings", "pooled_output"]
    
    dynamic_axes = {
        "input_ids": {0: "batch"},
        "text_embeddings": {0: "batch"},
        "pooled_output": {0: "batch"},
    }

    print("Exporting Text Encoder 2 (OpenCLIP ViT-bigG/14)...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (dummy_input_ids,),
                "text_encoder_2_temp.onnx",
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
        
        print("Consolidating Text Encoder 2 into single file...")
        model = onnx.load("text_encoder_2_temp.onnx", load_external_data=True)
        
        save_model(
            model,
            "text_encoder_2.onnx",
            save_as_external_data=False,
        )
        
        os.remove("text_encoder_2_temp.onnx")
        
        removed = 0
        for pattern in ("*.onnx.data", "onnx__*", "text_encoder_2.*"):
            for filepath in glob.glob(pattern):
                if filepath != "text_encoder_2.onnx":
                    try:
                        os.remove(filepath)
                        removed += 1
                    except OSError:
                        pass
        
        if removed > 0:
            print(f"Cleaned up {removed} external/shard files")
            
    except ImportError:
        print("Warning: onnx package not available, may have external files")
        if os.path.exists("text_encoder_2_temp.onnx"):
            os.rename("text_encoder_2_temp.onnx", "text_encoder_2.onnx")

    print("Text Encoder 2 exported to ONNX successfully")

if __name__ == "__main__":
    export_text_encoder_2()
