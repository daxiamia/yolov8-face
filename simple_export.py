#!/usr/bin/env python3
"""
YOLOv8-Face CoreML简单导出脚本

这个脚本使用修复后的Ultralytics导出功能。
用法: python simple_export.py
"""

from ultralytics import YOLO
import sys

def main():
    print("🧪 YOLOv8-Face 简单CoreML导出")
    print("=" * 50)
    
    try:
        # 加载模型
        print("📂 加载模型：yolov8n-face.pt")
        model = YOLO('yolov8n-face.pt')
        
        # 直接使用Ultralytics的CoreML导出
        print("📤 开始CoreML导出...")
        print("⚠️ 注意：这可能需要几分钟时间...")
        
        # 执行导出
        exported_model = model.export(
            format='coreml',
            imgsz=640,
            # 不使用int8量化，避免额外的复杂性
        )
        
        print(f"✅ 导出成功！")
        print(f"📁 导出文件：{exported_model}")
        print("\n📖 使用说明：")
        print("1. 可以在Xcode中使用此.mlmodel文件")
        print("2. 或者使用Core ML框架在iOS/macOS应用中加载")
        
    except Exception as e:
        print(f"❌ 导出失败：{e}")
        print("\n💡 提示：")
        print("- 确保已经运行了PyTorch兼容性修复")
        print("- 确保coremltools版本为7.2")
        print("- 如果还有问题，可能需要降级PyTorch版本")

if __name__ == "__main__":
    main() 