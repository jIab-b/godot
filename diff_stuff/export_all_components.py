import subprocess
import sys
import os
import glob


def run_export_script(script_name):
    """Run an export script and return success status"""
    try:
        print(f"\n{'='*50}")
        print(f"Running {script_name}...")
        print(f"{'='*50}")
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            return True
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ {script_name} failed with exception: {e}")
        return False


def cleanup_temp_files():
    """Clean up any remaining temporary files"""
    print("\nCleaning up temporary files...")
    
    removed = 0
    patterns = ["*_temp.onnx", "*.onnx.data", "onnx__*"]
    
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                removed += 1
                print(f"Removed: {filepath}")
            except OSError as e:
                print(f"Failed to remove {filepath}: {e}")
    
    if removed == 0:
        print("No temporary files found")
    else:
        print(f"Cleaned up {removed} temporary files")


def verify_exports():
    """Verify all expected ONNX files exist and have reasonable file sizes"""
    print("\nVerifying exports...")
    
    expected_files = [
        "text_encoder_1.onnx",
        "text_encoder_2.onnx", 
        "vae_decoder.onnx",
        "sdxl_unet.onnx",
        "t2i_adapter.onnx"
    ]
    
    all_good = True
    total_size = 0
    
    for filename in expected_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_mb = size / (1024 * 1024)
            total_size += size_mb
            
            if size_mb > 1:  # Reasonable minimum size
                print(f"✓ {filename}: {size_mb:.1f} MB")
            else:
                print(f"⚠ {filename}: {size_mb:.1f} MB (suspiciously small)")
                all_good = False
        else:
            print(f"✗ {filename}: Missing")
            all_good = False
    
    print(f"\nTotal size: {total_size:.1f} MB")
    return all_good


def export_all_components():
    """Export all pipeline components to ONNX"""
    
    print("Starting export of all SDXL Lightning + T2I Adapter components...")
    
    export_scripts = [
        "export_text_encoder_1.py",
        "export_text_encoder_2.py", 
        "export_vae_decoder.py",
        "export_unet_to_onnx.py",
        "export_adapter_to_onnx.py"
    ]
    
    success_count = 0
    
    for script in export_scripts:
        if os.path.exists(script):
            if run_export_script(script):
                success_count += 1
        else:
            print(f"✗ Script not found: {script}")
    
    cleanup_temp_files()
    
    print(f"\n{'='*50}")
    print(f"Export Summary")
    print(f"{'='*50}")
    print(f"Scripts run: {len(export_scripts)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(export_scripts) - success_count}")
    
    if verify_exports():
        print("\n🎉 All components exported successfully!")
        print("\nExported files:")
        print("- text_encoder_1.onnx (CLIP ViT-L/14)")
        print("- text_encoder_2.onnx (OpenCLIP ViT-bigG/14)")
        print("- vae_decoder.onnx (VAE decoder)")
        print("- sdxl_unet.onnx (UNet + T2I adapter)")
        print("- t2i_adapter.onnx (T2I depth adapter)")
        
        print("\nReady for TensorRT conversion with convert_to_tensorrt.bat")
        
    else:
        print("\n⚠ Some exports may have issues. Check the output above.")
        
    return success_count == len(export_scripts)


if __name__ == "__main__":
    export_all_components()
