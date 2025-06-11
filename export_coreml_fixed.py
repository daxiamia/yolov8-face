#!/usr/bin/env python3
"""
YOLOv8-Face CoreML Export Script (Fixed for coremltools 8.3.0)

这个脚本解决了coremltools 8.3.0中mlprogram格式的问题。
用法: python export_coreml_fixed.py --model yolov8n-face.pt --imgsz 640
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
    import coremltools as ct
    import onnx
except ImportError as e:
    print(f"错误：缺少必要的软件包。请安装：{e}")
    print("运行：pip install ultralytics coremltools onnx")
    sys.exit(1)


def fix_coreml_export(model_path, imgsz=640, int8=False):
    """
    修复CoreML导出，避免mlprogram格式问题
    """
    try:
        print(f"🔧 正在修复CoreML导出问题...")
        print(f"📂 加载模型：{model_path}")
        
        # 加载YOLO模型
        model = YOLO(model_path)
        
        # 第一步：导出为ONNX（中间格式）
        print(f"📤 导出ONNX模型（图像大小：{imgsz}）...")
        onnx_path = model.export(format='onnx', imgsz=imgsz)
        print(f"✅ ONNX模型已创建：{onnx_path}")
        
        # 第二步：从ONNX转换为CoreML
        print("🔄 从ONNX转换为CoreML...")
        coreml_path = str(onnx_path).replace('.onnx', '.mlmodel')
        
        # 转换为CoreML，使用iOS14目标避免mlprogram问题
        print("📱 使用iOS14目标进行转换（避免mlprogram格式问题）...")
        
        # 直接从ONNX文件路径转换，让coremltools自动检测
        coreml_model = ct.convert(
            onnx_path,  # 直接使用文件路径
            minimum_deployment_target=ct.target.iOS14,
            convert_to="neuralnetwork"  # 强制使用neural network格式
        )
        
        # 应用量化（如果需要）
        if int8:
            print("⚡ 应用INT8量化...")
            coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                coreml_model, 8, 'kmeans_lut'
            )
        
        # 保存CoreML模型
        coreml_model.save(coreml_path)
        print(f"✅ CoreML模型已保存：{coreml_path}")
        
        # 清理中间文件
        try:
            Path(onnx_path).unlink()
            print("🧹 已清理中间ONNX文件")
        except:
            pass
        
        # 验证模型
        print("🔍 验证CoreML模型...")
        try:
            # 尝试加载模型进行验证
            test_model = ct.models.MLModel(coreml_path)
            print("✅ 模型验证成功！")
            
            # 显示模型信息
            print(f"\n📊 模型信息：")
            print(f"   - 输入：{test_model.input_description}")
            print(f"   - 输出：{test_model.output_description}")
            
        except Exception as e:
            print(f"⚠️  模型验证失败：{e}")
        
        return coreml_path
        
    except Exception as e:
        print(f"❌ 导出失败：{e}")
        print(f"\n🔧 故障排除建议：")
        print(f"1. 尝试降级coremltools版本：pip install coremltools==7.2")
        print(f"2. 确保ultralytics是最新版本：pip install -U ultralytics") 
        print(f"3. 尝试更小的图像尺寸：--imgsz 320")
        print(f"4. 检查PyTorch兼容性：当前使用的是PyTorch {torch.__version__}")
        return None


def create_integration_guide(coreml_path):
    """创建iOS集成指南"""
    if not coreml_path:
        return
        
    guide_content = f"""
# YOLOv8-Face iOS集成指南

## ✅ 成功导出的模型
- **模型文件**: `{Path(coreml_path).name}`
- **格式**: CoreML Neural Network (兼容iOS14+)
- **推荐iOS版本**: iOS14.0+

## 🚀 快速集成步骤

### 1. 添加模型到Xcode项目
1. 将 `{Path(coreml_path).name}` 拖拽到Xcode项目中
2. 确保添加到Target的Bundle Resources中

### 2. Swift代码示例

```swift
import CoreML
import Vision
import UIKit

class FaceDetector {{
    private var model: VNCoreMLModel?
    
    init() {{
        loadModel()
    }}
    
    private func loadModel() {{
        guard let modelURL = Bundle.main.url(
            forResource: "{Path(coreml_path).stem}", 
            withExtension: "mlmodelc"
        ) else {{
            print("找不到模型文件")
            return
        }}
        
        do {{
            let mlModel = try MLModel(contentsOf: modelURL)
            self.model = try VNCoreMLModel(for: mlModel)
            print("模型加载成功")
        }} catch {{
            print("模型加载失败: \\(error)")
        }}
    }}
    
    func detectFaces(in image: UIImage, completion: @escaping ([VNFaceObservation]) -> Void) {{
        guard let model = self.model,
              let cgImage = image.cgImage else {{
            completion([])
            return
        }}
        
        let request = VNCoreMLRequest(model: model) {{ request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {{
                completion([])
                return
            }}
            
            // 处理检测结果
            DispatchQueue.main.async {{
                completion([]) // 转换为VNFaceObservation格式
            }}
        }}
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {{
            try? handler.perform([request])
        }}
    }}
}}
```

## 📝 使用说明

1. **图像预处理**: 确保输入图像尺寸为 {imgsz}x{imgsz}
2. **性能优化**: 在后台线程运行推理，避免阻塞UI
3. **内存管理**: 及时释放不需要的图像和结果
4. **错误处理**: 妥善处理模型加载和推理过程中的异常

## 🎯 测试建议

1. 使用不同光照条件的人脸图像测试
2. 测试多人脸场景
3. 验证在不同iOS设备上的性能表现

---
生成时间: {Path().cwd()}
模型路径: {coreml_path}
"""

    guide_file = Path("iOS_Integration_Guide_Fixed.md")
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"📖 iOS集成指南已创建：{guide_file}")


def main():
    parser = argparse.ArgumentParser(description='修复CoreML导出问题')
    parser.add_argument('--model', type=str, default='yolov8n-face.pt',
                        help='YOLOv8-Face模型文件路径')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理图像尺寸（默认：640）')
    parser.add_argument('--int8', action='store_true',
                        help='使用INT8量化减小模型大小')
    
    args = parser.parse_args()
    
    print("🧪 YOLOv8-Face CoreML导出修复工具\n")
    print(f"🐍 Python版本: {sys.version}")
    print(f"⚡ PyTorch版本: {torch.__version__}")
    print(f"🍎 CoreMLTools版本: {ct.__version__}")
    print()
    
    # 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在：{args.model}")
        print("\n📥 请下载YOLOv8-Face模型：")
        print("- yolov8-lite-t: https://drive.google.com/file/d/1vFMGW8xtRVo9bfC9yJVWWGY7vVxbLh94/view?usp=sharing")
        print("- yolov8-lite-s: https://drive.google.com/file/d/1ckpBT8KfwURTvTm5pa-cMC89A0V5jbaq/view?usp=sharing") 
        print("- yolov8n: https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view?usp=sharing")
        return
    
    # 执行修复导出
    coreml_path = fix_coreml_export(str(model_path), args.imgsz, args.int8)
    
    if coreml_path:
        # 创建集成指南
        create_integration_guide(coreml_path)
        
        print(f"\n🎉 导出完成！")
        print(f"📱 CoreML模型: {coreml_path}")
        print(f"📚 集成指南: iOS_Integration_Guide_Fixed.md")
        
        # 显示下一步
        print(f"\n📋 下一步：")
        print(f"1. 将 {Path(coreml_path).name} 添加到你的iOS项目")
        print(f"2. 按照集成指南实现人脸检测功能")
        print(f"3. 在iOS设备上测试性能")
    else:
        print(f"\n❌ 导出失败，请检查上述错误信息")


if __name__ == "__main__":
    main() 