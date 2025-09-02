@echo off
REM convert_to_tensorrt.bat - Convert all SDXL Lightning + T2I Adapter components
cd /d "%~dp0"

echo Converting Text Encoder 1 (CLIP ViT-L/14) to TensorRT...
trtexec --onnx=text_encoder_1.onnx ^
        --saveEngine=text_encoder_1.trt ^
        --fp16 ^
        --minShapes=input_ids:1x77 ^
        --optShapes=input_ids:1x77 ^
        --maxShapes=input_ids:4x77 ^
        --builderOptimizationLevel=5

echo Converting Text Encoder 2 (OpenCLIP ViT-bigG/14) to TensorRT...
trtexec --onnx=text_encoder_2.onnx ^
        --saveEngine=text_encoder_2.trt ^
        --fp16 ^
        --minShapes=input_ids:1x77 ^
        --optShapes=input_ids:1x77 ^
        --maxShapes=input_ids:4x77 ^
        --builderOptimizationLevel=5

echo Converting T2I-Adapter to TensorRT...
trtexec --onnx=t2i_adapter.onnx ^
        --saveEngine=t2i_adapter.trt ^
        --fp16 ^
        --minShapes=depth_input:1x3x1024x1024 ^
        --optShapes=depth_input:1x3x1024x1024 ^
        --maxShapes=depth_input:1x3x1024x1024 ^
        --builderOptimizationLevel=5

echo Converting SDXL UNet to TensorRT...
trtexec --onnx=sdxl_unet.onnx ^
        --saveEngine=sdxl_unet.trt ^
        --fp16 ^
        --minShapes=sample:1x4x128x128,encoder_hidden_states:1x77x2048,timestep:1,text_embeds:1x1280,time_ids:1x6,adapter_feat_0:1x320x128x128,adapter_feat_1:1x640x64x64,adapter_feat_2:1x1280x32x32,adapter_feat_3:1x1280x16x16 ^
        --optShapes=sample:1x4x128x128,encoder_hidden_states:1x77x2048,timestep:1,text_embeds:1x1280,time_ids:1x6,adapter_feat_0:1x320x128x128,adapter_feat_1:1x640x64x64,adapter_feat_2:1x1280x32x32,adapter_feat_3:1x1280x16x16 ^
        --maxShapes=sample:4x4x128x128,encoder_hidden_states:4x77x2048,timestep:1,text_embeds:4x1280,time_ids:4x6,adapter_feat_0:4x320x128x128,adapter_feat_1:4x640x64x64,adapter_feat_2:4x1280x32x32,adapter_feat_3:4x1280x16x16 ^
        --builderOptimizationLevel=5

echo Converting VAE Decoder to TensorRT...
trtexec --onnx=vae_decoder.onnx ^
        --saveEngine=vae_decoder.trt ^
        --fp16 ^
        --minShapes=latents:1x4x128x128 ^
        --optShapes=latents:1x4x128x128 ^
        --maxShapes=latents:4x4x128x128 ^
        --builderOptimizationLevel=5

echo All conversions complete!
echo.
echo Generated TensorRT engines:
echo - text_encoder_1.trt
echo - text_encoder_2.trt
echo - t2i_adapter.trt
echo - sdxl_unet.trt
echo - vae_decoder.trt
pause
