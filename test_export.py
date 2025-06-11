#!/usr/bin/env python3
"""
Test script for YOLOv8-Face CoreML export

This script tests the model export functionality with a simple example.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages for CoreML export."""
    try:
        import coremltools
        import ultralytics
        print("✅ Required packages already installed")
        return True
    except ImportError:
        print("📦 Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "coremltools", "ultralytics"])
            print("✅ Packages installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            return False


def test_basic_functionality():
    """Test basic YOLO functionality."""
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
        
        # Test with a basic YOLO model (this will download automatically)
        print("🔄 Testing with YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        model_name = getattr(model.model, 'yaml_file', 'yolov8n')
        if hasattr(model.model, 'yaml') and isinstance(model.model.yaml, dict):
            model_name = model.model.yaml.get('yaml_file', model_name)
        print(f"✅ Model loaded: {model_name} ({model.task} task)")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def check_model_availability():
    """Check if YOLOv8-Face models are available."""
    models = [
        "yolov8n-face.pt",
        "yolov8-lite-t.pt", 
        "yolov8-lite-s.pt"
    ]
    
    available_models = []
    for model in models:
        if Path(model).exists():
            available_models.append(model)
            print(f"✅ Found model: {model}")
        else:
            print(f"❌ Model not found: {model}")
    
    if not available_models:
        print("\n📥 No YOLOv8-Face models found locally.")
        print("Please download one of the following models:")
        print("- yolov8-lite-t: https://drive.google.com/file/d/1vFMGW8xtRVo9bfC9yJVWWGY7vVxbLh94/view?usp=sharing")
        print("- yolov8-lite-s: https://drive.google.com/file/d/1ckpBT8KfwURTvTm5pa-cMC89A0V5jbaq/view?usp=sharing")
        print("- yolov8n: https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view?usp=sharing")
        print("\nAfter downloading, place the .pt file in this directory and run the export script.")
        return False
    
    return available_models


def main():
    print("🧪 Testing YOLOv8-Face CoreML Export Setup\n")
    
    # Test 1: Install requirements
    if not install_requirements():
        return
    
    # Test 2: Basic functionality
    if not test_basic_functionality():
        return
    
    # Test 3: Check model availability
    available_models = check_model_availability()
    
    if available_models:
        print(f"\n🎉 Setup complete! You can now export models using:")
        print(f"python export_coreml.py --model {available_models[0]}")
    else:
        print("\n⚠️  Setup partially complete. Download a model to proceed with export.")
    
    print("\n📚 Next steps:")
    print("1. Download a YOLOv8-Face model (links shown above)")
    print("2. Run: python export_coreml.py --model <model_name.pt>")
    print("3. Follow the generated iOS_Integration_Guide.md")


if __name__ == "__main__":
    main() 