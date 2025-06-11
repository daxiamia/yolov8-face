# YOLOv8-Face iOS 集成指南

本文档详细说明如何将 YOLOv8-Face 模型集成到 iOS 应用中。

## 概述

YOLOv8-Face 是一个高性能的人脸检测模型，可以检测人脸并提供 5 个关键点（左眼、右眼、鼻子、左嘴角、右嘴角）。通过 CoreML 格式，我们可以将其集成到 iOS 应用中实现实时人脸检测。

## 前置条件

- Python 3.8+
- iOS 开发环境 (Xcode 12+)
- iOS 11.0+ 设备
- 预训练的 YOLOv8-Face 模型

## 第一步：环境设置

### 1. 安装依赖

```bash
# 安装 CoreML 支持
pip install coremltools>=6.0

# 安装 Ultralytics YOLO
pip install ultralytics

# 或者直接安装项目依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型

从以下链接下载预训练模型：

- **yolov8-lite-t** (轻量级): [下载链接](https://drive.google.com/file/d/1vFMGW8xtRVo9bfC9yJVWWGY7vVxbLh94/view?usp=sharing)
- **yolov8-lite-s** (小型): [下载链接](https://drive.google.com/file/d/1ckpBT8KfwURTvTm5pa-cMC89A0V5jbaq/view?usp=sharing)
- **yolov8n** (标准): [下载链接](https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view?usp=sharing)

下载后将 `.pt` 文件放置在项目根目录。

## 第二步：模型导出

### 1. 测试环境

```bash
python test_export.py
```

这将验证您的环境是否正确设置。

### 2. 导出 CoreML 模型

```bash
# 基本导出
python export_coreml.py --model yolov8n-face.pt

# 使用 INT8 量化（更小的模型大小）
python export_coreml.py --model yolov8n-face.pt --int8

# 指定输入图像大小
python export_coreml.py --model yolov8n-face.pt --imgsz 416
```

导出成功后，您将得到：
- `.mlmodel` 文件：CoreML 模型
- `iOS_Integration_Guide.md`：详细的 iOS 集成指南

## 第三步：iOS 项目集成

### 1. 添加模型到 Xcode 项目

1. 将生成的 `.mlmodel` 文件拖拽到 Xcode 项目中
2. 确保添加到 target 的 bundle resources

### 2. 创建人脸检测器类

```swift
import CoreML
import Vision
import UIKit

class YOLOv8FaceDetector {
    private var model: VNCoreMLModel?
    
    init() {
        setupModel()
    }
    
    private func setupModel() {
        guard let modelURL = Bundle.main.url(forResource: "yolov8n-face", withExtension: "mlmodelc"),
              let coreMLModel = try? MLModel(contentsOf: modelURL),
              let vnModel = try? VNCoreMLModel(for: coreMLModel) else {
            print("Failed to load CoreML model")
            return
        }
        
        self.model = vnModel
    }
    
    func detectFaces(in image: UIImage, completion: @escaping ([DetectedFace]) -> Void) {
        guard let cgImage = image.cgImage,
              let model = self.model else {
            completion([])
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                completion([])
                return
            }
            
            let faces = results.compactMap { observation -> DetectedFace? in
                guard observation.confidence > 0.5 else { return nil }
                
                return DetectedFace(
                    boundingBox: observation.boundingBox,
                    confidence: observation.confidence,
                    landmarks: self.extractLandmarks(from: observation)
                )
            }
            
            DispatchQueue.main.async {
                completion(faces)
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        DispatchQueue.global(qos: .userInitiated).async {
            try? handler.perform([request])
        }
    }
    
    private func extractLandmarks(from observation: VNRecognizedObjectObservation) -> [CGPoint] {
        // YOLOv8-Face 返回 5 个关键点
        // 实际实现需要根据模型输出格式调整
        return []
    }
}

struct DetectedFace {
    let boundingBox: CGRect
    let confidence: Float
    let landmarks: [CGPoint]
}
```

### 3. 实时相机检测

```swift
import AVFoundation

class CameraViewController: UIViewController {
    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var overlayView: UIView!
    
    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var faceDetector = YOLOv8FaceDetector()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .medium
        
        guard let backCamera = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: backCamera) else {
            return
        }
        
        captureSession.addInput(input)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = previewView.bounds
        previewLayer.videoGravity = .resizeAspectFill
        previewView.layer.addSublayer(previewLayer)
        
        captureSession.startRunning()
    }
    
    private func updateOverlay(with faces: [DetectedFace]) {
        overlayView.layer.sublayers?.removeAll()
        
        for face in faces {
            let faceBox = convertToScreenCoordinates(face.boundingBox)
            drawFaceBox(faceBox, confidence: face.confidence)
        }
    }
    
    private func convertToScreenCoordinates(_ boundingBox: CGRect) -> CGRect {
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -overlayView.bounds.height)
        return boundingBox.applying(transform)
    }
    
    private func drawFaceBox(_ rect: CGRect, confidence: Float) {
        let shapeLayer = CAShapeLayer()
        shapeLayer.path = UIBezierPath(rect: rect).cgPath
        shapeLayer.fillColor = UIColor.clear.cgColor
        shapeLayer.strokeColor = UIColor.red.cgColor
        shapeLayer.lineWidth = 2.0
        
        overlayView.layer.addSublayer(shapeLayer)
        
        // 添加置信度标签
        let label = CATextLayer()
        label.string = String(format: "%.2f", confidence)
        label.fontSize = 12
        label.foregroundColor = UIColor.white.cgColor
        label.backgroundColor = UIColor.red.cgColor
        label.frame = CGRect(x: rect.minX, y: rect.minY - 20, width: 50, height: 20)
        
        overlayView.layer.addSublayer(label)
    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(image, from: image.extent) else { return }
        
        let uiImage = UIImage(cgImage: cgImage)
        
        faceDetector.detectFaces(in: uiImage) { [weak self] faces in
            self?.updateOverlay(with: faces)
        }
    }
}
```

## 第四步：性能优化

### 1. 模型优化

- 使用 `--int8` 参数进行量化，减小模型大小
- 根据设备性能选择合适的输入尺寸（416, 640）
- 考虑使用 yolov8-lite 版本获得更好的移动设备性能

### 2. 运行时优化

```swift
// 限制检测频率
private var lastDetectionTime: Date = Date()
private let detectionInterval: TimeInterval = 0.1 // 100ms

func captureOutput(...) {
    let now = Date()
    guard now.timeIntervalSince(lastDetectionTime) >= detectionInterval else { return }
    lastDetectionTime = now
    
    // 执行检测...
}

// 图像预处理优化
private func preprocessImage(_ image: UIImage) -> UIImage? {
    // 调整图像大小到模型输入尺寸
    let targetSize = CGSize(width: 640, height: 640)
    
    UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
    image.draw(in: CGRect(origin: .zero, size: targetSize))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    
    return resizedImage
}
```

## 第五步：测试和部署

### 1. 测试建议

- 在不同光照条件下测试
- 测试多人脸场景
- 验证不同设备上的性能
- 检查内存使用和电池消耗

### 2. 权限设置

在 `Info.plist` 中添加相机权限：

```xml
<key>NSCameraUsageDescription</key>
<string>此应用需要访问相机来进行人脸检测</string>
```

### 3. 部署注意事项

- 确保目标 iOS 版本为 11.0+
- 考虑模型大小对应用下载大小的影响
- 遵守人脸检测相关的隐私规定

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否正确添加到 bundle
   - 验证文件名和扩展名是否正确

2. **检测结果不准确**
   - 调整置信度阈值
   - 确保输入图像预处理正确
   - 检查 NMS 设置

3. **性能问题**
   - 降低检测频率
   - 使用更小的输入尺寸
   - 考虑使用 lite 版本模型

### 调试技巧

```swift
// 启用详细日志
func setupModel() {
    // ... 模型加载代码 ...
    
    if let model = self.model {
        print("✅ Model loaded successfully")
        print("Model input description: \(model.model.modelDescription.inputDescriptionsByName)")
        print("Model output description: \(model.model.modelDescription.outputDescriptionsByName)")
    }
}
```

## 示例项目

完整的示例项目包含：

- [x] 模型导出脚本
- [x] iOS 集成代码
- [x] 性能优化示例
- [x] 错误处理和调试代码

## 技术支持

如遇到问题，请检查：

1. [Ultralytics Documentation](https://docs.ultralytics.com/)
2. [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)
3. 项目 Issues 和讨论区

## 更新日志

- **v1.0**: 初始版本，支持 CoreML 导出和基础 iOS 集成
- 计划支持更多导出格式（ONNX, TensorRT）

---

🎉 恭喜！您现在可以在 iOS 应用中使用 YOLOv8-Face 进行实时人脸检测了！ 