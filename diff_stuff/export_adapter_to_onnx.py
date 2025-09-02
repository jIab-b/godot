import torch
from diffusers import T2IAdapter


class AdapterWrapper(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, depth_input):
        features = self.adapter(depth_input)
        if isinstance(features, (list, tuple)):
            return tuple(features[:4]) if len(features) >= 4 else tuple(features)
        return (features,)


def export_adapter():
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
        torch_dtype=torch.float16,
    ).to("cuda", torch.float16)
    adapter.eval()

    wrapper = AdapterWrapper(adapter).to("cuda", torch.float16)

    dummy_input = torch.randn(1, 3, 1024, 1024, device="cuda").half()

    with torch.inference_mode():
        trial_outputs = wrapper(dummy_input)
    num_outputs = min(4, len(trial_outputs))

    output_names = [f"res_{i}" for i in range(num_outputs)]
    dynamic_axes = {"depth_input": {0: "batch", 2: "height", 3: "width"}}
    for i in range(num_outputs):
        dynamic_axes[f"res_{i}"] = {0: "batch"}

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            "t2i_adapter.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["depth_input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print("T2I-Adapter exported to ONNX successfully")

if __name__ == "__main__":
    export_adapter()
