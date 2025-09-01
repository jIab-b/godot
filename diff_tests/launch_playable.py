#!/usr/bin/env python3

import subprocess
import os
import sys

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Godot executable (adjust as needed)
    godot_path = os.path.join(script_dir, "..", "bin", "godot.windows.editor.x86_64.exe")
    
    # Path to the main scene
    scene_path = os.path.join(script_dir, "main.tscn")
    
    # Check if Godot executable exists
    if not os.path.exists(godot_path):
        print(f"Error: Godot executable not found at {godot_path}")
        print("Please build Godot first or adjust the path in this script.")
        sys.exit(1)
    
    # Check if scene exists
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found at {scene_path}")
        sys.exit(1)
    
    # Launch Godot with the scene using diff_tests as the project
    try:
        print(f"Launching scene: {scene_path}")
        subprocess.run([godot_path, "--path", script_dir, scene_path])
    except Exception as e:
        print(f"Error launching Godot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
