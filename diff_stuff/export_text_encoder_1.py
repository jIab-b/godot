import torch
import warnings
from transformers import CLIPTextModel, CLIPTokenizer
import os
import glob


class TextEncoder1Wrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        outputs = self.text_encoder(input_ids, return_dict=False)
        return outputs[0]


def export_text_encoder_1():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda").eval()
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer"
    )
    
    wrapper = TextEncoder1Wrapper(text_encoder).to("cuda", torch.float16)

    batch = 1
    max_length = 77
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch, max_length), device="cuda")

    input_names = ["input_ids"]
    output_names = ["text_embeddings"]
    
    dynamic_axes = {
        "input_ids": {0: "batch"},
        "text_embeddings": {0: "batch"},
    }

    print("Exporting Text Encoder 1 (CLIP ViT-L/14)...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (dummy_input_ids,),
                "text_encoder_1_temp.onnx",
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
        
        print("Consolidating Text Encoder 1 into single file...")
        model = onnx.load("text_encoder_1_temp.onnx", load_external_data=True)
        
        save_model(
            model,
            "text_encoder_1.onnx",
            save_as_external_data=False,
        )
        
        os.remove("text_encoder_1_temp.onnx")
        
        removed = 0
        for pattern in ("*.onnx.data", "onnx__*", "text_encoder_1.*"):
            for filepath in glob.glob(pattern):
                if filepath != "text_encoder_1.onnx":
                    try:
                        os.remove(filepath)
                        removed += 1
                    except OSError:
                        pass
        
        if removed > 0:
            print(f"Cleaned up {removed} external/shard files")
            
    except ImportError:
        print("Warning: onnx package not available, may have external files")
        if os.path.exists("text_encoder_1_temp.onnx"):
            os.rename("text_encoder_1_temp.onnx", "text_encoder_1.onnx")

    print("Text Encoder 1 exported to ONNX successfully")

if __name__ == "__main__":
    export_text_encoder_1()
