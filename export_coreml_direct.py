#!/usr/bin/env python3
"""
YOLOv8-Face CoreML直接导出脚本

这个脚本直接从PyTorch模型转换为CoreML，避免ONNX中间转换的问题。
用法: python export_coreml_direct.py --model yolov8n-face.pt --imgsz 640
"""

import argparse
import sys
import os
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
    import coremltools as ct
    import numpy as np
except ImportError as e:
    print(f"错误：缺少必要的软件包。请安装：{e}")
    print("运行：pip install ultralytics coremltools")
    sys.exit(1)


def direct_coreml_export(model_path, imgsz=640, int8=False):
    """
    直接从PyTorch模型导出CoreML，避免ONNX中间转换
    """
    try:
        print(f"🔧 正在进行CoreML直接导出...")
        print(f"📂 加载模型：{model_path}")
        
        # 加载YOLO模型
        model = YOLO(model_path)
        
        # 获取PyTorch模型
        pytorch_model = model.model
        pytorch_model.eval()
        
        # 创建示例输入
        example_input = torch.randn(1, 3, imgsz, imgsz)
        
        print(f"📤 直接导出为CoreML（图像大小：{imgsz}）...")
        
        # 使用torch.jit.trace创建traced模型
        print("🔄 创建traced模型...")
        traced_model = torch.jit.trace(pytorch_model, example_input)
        
        # 转换为CoreML，明确指定PyTorch源
        print("📱 转换为CoreML（使用PyTorch源）...")
        coreml_model_path = str(model_path).replace('.pt', '.mlmodel')
        
        # 定义输入
        input_spec = ct.TensorType(shape=(1, 3, imgsz, imgsz), name="image")
        
        # 转换模型
        coreml_model = ct.convert(
            traced_model,
            inputs=[input_spec],
            source="pytorch",
            minimum_deployment_target=ct.target.iOS14,
            convert_to="neuralnetwork"  # 强制使用neural network格式避免mlprogram问题
        )
        
        # 保存模型
        print(f"💾 保存CoreML模型...")
        coreml_model.save(coreml_model_path)
        print(f"✅ CoreML模型已保存：{coreml_model_path}")
        
        # 验证模型
        print("🔍 验证CoreML模型...")
        try:
            # 重新加载模型进行验证
            loaded_model = ct.models.MLModel(coreml_model_path)
            print("✅ CoreML模型验证成功！")
            
            # 显示模型信息
            spec = loaded_model.get_spec()
            print(f"📊 模型信息：")
            print(f"   - 输入：{spec.description.input}")
            print(f"   - 输出：{spec.description.output}")
            print(f"   - iOS最低版本：iOS14")
            print(f"   - 格式：Neural Network")
            
            return coreml_model_path
            
        except Exception as e:
            print(f"❌ CoreML模型验证失败：{e}")
            return None
            
    except Exception as e:
        print(f"❌ 导出失败：{e}")
        print("\n🔧 故障排除建议：")
        print("1. 确保PyTorch版本兼容")
        print("2. 尝试更小的图像尺寸：--imgsz 320")
        print("3. 检查模型文件是否有效")
        return None


def main():
    parser = argparse.ArgumentParser(description='YOLOv8-Face CoreML直接导出工具')
    parser.add_argument('--model', type=str, default='yolov8n-face.pt', 
                       help='模型文件路径 (默认: yolov8n-face.pt)')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--int8', action='store_true', 
                       help='使用INT8量化')
    
    args = parser.parse_args()
    
    # 打印环境信息
    print("🧪 YOLOv8-Face CoreML直接导出工具")
    print()
    print(f"🐍 Python版本: {sys.version}")
    print(f"⚡ PyTorch版本: {torch.__version__}")
    print(f"🍎 CoreMLTools版本: {ct.__version__}")
    print()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 错误：模型文件不存在：{args.model}")
        return
    
    # 执行导出
    result = direct_coreml_export(args.model, args.imgsz, args.int8)
    
    if result:
        print(f"\n🎉 导出成功！CoreML模型已保存到：{result}")
        print("\n📖 使用说明：")
        print("1. 可以在Xcode中使用此.mlmodel文件")
        print("2. 或者使用Core ML框架在iOS/macOS应用中加载")
        print("3. 模型输入：640x640 RGB图像")
        print("4. 模型输出：YOLOv8检测结果")
    else:
        print("\n❌ 导出失败，请检查上述错误信息")


if __name__ == "__main__":
    main() 