#!/usr/bin/env python3
"""
YOLOv8-Face CoreML最终解决方案

这个脚本提供了多种导出选项，包括:
1. 仅导出ONNX (推荐用于iOS开发)
2. 尝试ONNX到CoreML转换 (如果可能)

用法: python final_solution.py
"""

from ultralytics import YOLO
import sys
import os
from pathlib import Path

def export_onnx_only():
    """仅导出ONNX格式，这是最稳定的选择"""
    print("📤 导出ONNX模型...")
    try:
        model = YOLO('yolov8n-face.pt')
        onnx_path = model.export(
            format='onnx',
            imgsz=640,
            opset=11  # 使用较旧的opset版本增加兼容性
        )
        print(f"✅ ONNX导出成功：{onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ ONNX导出失败：{e}")
        return None

def try_simple_coreml_conversion(onnx_path):
    """尝试简单的ONNX到CoreML转换"""
    try:
        import coremltools as ct
        print("🔄 尝试简单的ONNX到CoreML转换...")
        
        # 直接转换，不指定复杂参数
        mlmodel = ct.convert(onnx_path)
        coreml_path = str(onnx_path).replace('.onnx', '.mlmodel')
        mlmodel.save(coreml_path)
        
        print(f"✅ CoreML转换成功：{coreml_path}")
        return coreml_path
    except Exception as e:
        print(f"❌ CoreML转换失败：{e}")
        return None

def main():
    print("🎯 YOLOv8-Face CoreML最终解决方案")
    print("=" * 60)
    
    # 步骤1：导出ONNX
    onnx_path = export_onnx_only()
    if not onnx_path:
        print("❌ 无法继续，ONNX导出失败")
        return
    
    # 步骤2：尝试CoreML转换
    print("\n" + "=" * 60)
    coreml_path = try_simple_coreml_conversion(onnx_path)
    
    # 总结结果
    print("\n" + "=" * 60)
    print("📊 导出结果总结：")
    print("=" * 60)
    
    if os.path.exists(onnx_path):
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✅ ONNX模型：{onnx_path} ({file_size:.1f} MB)")
        print("   👉 可用于：")
        print("      - iOS开发 (使用ONNX Runtime)")
        print("      - Android开发")
        print("      - 跨平台推理")
    
    if coreml_path and os.path.exists(coreml_path):
        file_size = os.path.getsize(coreml_path) / (1024 * 1024)
        print(f"✅ CoreML模型：{coreml_path} ({file_size:.1f} MB)")
        print("   👉 可用于：")
        print("      - iOS/macOS原生应用")
        print("      - Xcode项目集成")
    else:
        print("⚠️ CoreML模型：导出失败")
        print("   💡 解决方案：")
        print("      - 使用ONNX模型配合ONNX Runtime")
        print("      - 或者使用在线转换工具")
        print("      - 或者降级到PyTorch 2.2.0 + coremltools 7.0")
    
    print("\n📖 使用建议：")
    if os.path.exists(onnx_path):
        print("🥇 推荐：使用ONNX模型 + ONNX Runtime")
        print("   - 兼容性最好")
        print("   - 性能优秀")
        print("   - 支持所有平台")
        
        print("\n🍎 iOS使用示例:")
        print("   1. 将.onnx文件添加到Xcode项目")
        print("   2. 安装ONNX Runtime Swift: https://github.com/microsoft/onnxruntime")
        print("   3. 使用以下代码加载模型:")
        print("      ```swift")
        print("      import onnxruntime_objc")
        print("      let model = try ORTSession(modelPath: \"yolov8n-face.onnx\")")
        print("      ```")

if __name__ == "__main__":
    main() 