#!/usr/bin/env python3
"""
YOLOv8-Face CoreML Export Script

This script exports a YOLOv8-Face model to CoreML format for iOS integration.
Usage: python export_coreml.py --model yolov8n-face.pt --imgsz 640
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
    import coremltools as ct
except ImportError as e:
    print(f"Error: Missing required package. Please install: {e}")
    print("Run: pip install ultralytics coremltools")
    sys.exit(1)


def download_pretrained_model(model_name="yolov8n-face.pt"):
    """Download pre-trained YOLOv8-Face model if not exists."""
    model_path = Path(model_name)
    
    if not model_path.exists():
        print(f"Model {model_name} not found locally.")
        print("Available models from README:")
        print("- yolov8-lite-t: https://drive.google.com/file/d/1vFMGW8xtRVo9bfC9yJVWWGY7vVxbLh94/view?usp=sharing")
        print("- yolov8-lite-s: https://drive.google.com/file/d/1ckpBT8KfwURTvTm5pa-cMC89A0V5jbaq/view?usp=sharing") 
        print("- yolov8n: https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view?usp=sharing")
        print("\nPlease download one of these models and place it in the current directory.")
        return None
    
    return str(model_path)


def export_to_coreml(model_path, imgsz=640, int8=False):
    """Export YOLOv8-Face model to CoreML format."""
    try:
        # Load model
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        print(f"Exporting to CoreML (imgsz={imgsz})...")
        
        # Method 1: Try direct CoreML export with iOS14 target (simpler approach)
        try:
            # Patch the CoreML export to use iOS14 target and neural network format
            from ultralytics.yolo.engine.exporter import Exporter
            import coremltools as ct
            
            # Create custom export arguments
            export_args = {
                'format': 'coreml',
                'imgsz': imgsz,
                'int8': int8,
                'device': 'cpu'  # CoreML export should use CPU
            }
            
            # Monkey patch the CoreML export to fix the target issue
            original_export_coreml = Exporter.export_coreml
            
            def patched_export_coreml(self, prefix="CoreML:"):
                """Patched CoreML export with iOS14 target."""
                from ultralytics.yolo.utils import colorstr, LOGGER
                from ultralytics.yolo.utils.checks import check_requirements
                
                check_requirements('coremltools>=6.0')
                import coremltools as ct
                
                LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
                f = self.file.with_suffix('.mlmodel')
                
                bias = [0.0, 0.0, 0.0]
                scale = 1 / 255
                classifier_config = None
                
                if self.model.task == 'classify':
                    classifier_config = ct.ClassifierConfig(list(self.model.names.values())) if self.args.nms else None
                    model = self.model
                elif self.model.task == 'detect':
                    from ultralytics.yolo.engine.exporter import iOSDetectModel
                    model = iOSDetectModel(self.model, self.im) if self.args.nms else self.model
                else:
                    model = self.model
                
                import torch
                ts = torch.jit.trace(model.eval(), self.im, strict=False)
                
                # Convert with iOS14 target to avoid mlprogram format issues
                ct_model = ct.convert(
                    ts,
                    inputs=[ct.ImageType('image', shape=self.im.shape, scale=scale, bias=bias)],
                    classifier_config=classifier_config,
                    minimum_deployment_target=ct.target.iOS14,  # Use iOS14 target
                    convert_to="neuralnetwork"  # Force neural network format
                )
                
                # Apply quantization if requested
                bits, mode = (8, 'kmeans_lut') if self.args.int8 else (16, 'linear') if self.args.half else (32, None)
                if bits < 32:
                    if 'kmeans' in mode:
                        check_requirements('scikit-learn')
                    ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
                
                # Set metadata
                m = self.metadata.copy()
                ct_model.short_description = m.pop('description')
                ct_model.author = m.pop('author')
                ct_model.license = m.pop('license')
                ct_model.version = m.pop('version')
                ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
                
                ct_model.save(str(f))
                return f, ct_model
            
            # Apply the patch temporarily
            Exporter.export_coreml = patched_export_coreml
            
            # Export with the patched method
            coreml_model = model.export(**export_args)
            
            # Restore original method
            Exporter.export_coreml = original_export_coreml
            
            print(f"✅ CoreML model exported successfully: {coreml_model}")
            return coreml_model
            
        except Exception as e1:
            print(f"Direct export failed: {e1}")
            print("Trying alternative ONNX->CoreML conversion...")
            
            # Method 2: ONNX -> CoreML conversion (fallback)
            import coremltools as ct
            import onnx
            
            # Export to ONNX first
            onnx_model = model.export(format='onnx', imgsz=imgsz)
            print(f"Intermediate ONNX model created: {onnx_model}")
            
            # Convert ONNX to CoreML
            print("Converting ONNX to CoreML...")
            coreml_model_path = str(onnx_model).replace('.onnx', '.mlmodel')
            
            # Load and convert
            onnx_model_obj = onnx.load(onnx_model)
            coreml_model_obj = ct.convert(
                onnx_model_obj,
                minimum_deployment_target=ct.target.iOS14,
                convert_to="neuralnetwork"
            )
            
            coreml_model_obj.save(coreml_model_path)
            
            # Clean up ONNX file
            try:
                import os
                os.remove(onnx_model)
                print("Cleaned up intermediate ONNX file")
            except:
                pass
            
            print(f"✅ CoreML model exported successfully: {coreml_model_path}")
            return coreml_model_path
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Try downgrading coremltools: pip install coremltools==7.2")
        print("2. Ensure you have the latest ultralytics: pip install -U ultralytics")
        print("3. Try exporting with different image size: --imgsz 320")
        return None


def create_ios_integration_guide(coreml_model_path):
    """Create iOS integration guide."""
    guide_content = f"""
# iOS Integration Guide for YOLOv8-Face

## 1. Add CoreML Model to iOS Project

1. Drag and drop `{Path(coreml_model_path).name}` into your Xcode project
2. Make sure to add it to the target's bundle resources

## 2. iOS Swift Implementation

```swift
import CoreML
import Vision
import UIKit
import AVFoundation

class YOLOv8FaceDetector {{
    private var model: VNCoreMLModel?
    private var requests: [VNRequest] = []
    
    init() {{
        setupModel()
    }}
    
    private func setupModel() {{
        guard let modelURL = Bundle.main.url(forResource: "{Path(coreml_model_path).stem}", withExtension: "mlmodelc"),
              let coreMLModel = try? MLModel(contentsOf: modelURL),
              let vnModel = try? VNCoreMLModel(for: coreMLModel) else {{
            print("Failed to load CoreML model")
            return
        }}
        
        self.model = vnModel
        setupRequests()
    }}
    
    private func setupRequests() {{
        guard let model = self.model else {{ return }}
        
        let request = VNCoreMLRequest(model: model) {{ [weak self] request, error in
            self?.processDetections(request: request, error: error)
        }}
        
        // Configure request
        request.imageCropAndScaleOption = .scaleFill
        self.requests = [request]
    }}
    
    func detectFaces(in image: UIImage, completion: @escaping ([DetectedFace]) -> Void) {{
        guard let cgImage = image.cgImage else {{
            completion([])
            return
        }}
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        DispatchQueue.global(qos: .userInitiated).async {{
            do {{
                try handler.perform(self.requests)
            }} catch {{
                print("Failed to perform detection: \\(error)")
                DispatchQueue.main.async {{
                    completion([])
                }}
            }}
        }}
    }}
    
    private func processDetections(request: VNRequest, error: Error?) {{
        guard let results = request.results as? [VNRecognizedObjectObservation] else {{
            return
        }}
        
        let detectedFaces = results.compactMap {{ observation -> DetectedFace? in
            // Convert Vision coordinates to UIKit coordinates
            let boundingBox = observation.boundingBox
            let confidence = observation.confidence
            
            // Apply NMS (Non-Maximum Suppression) if needed
            guard confidence > 0.5 else {{ return nil }}
            
            return DetectedFace(
                boundingBox: boundingBox,
                confidence: confidence,
                landmarks: extractLandmarks(from: observation)
            )
        }}
        
        DispatchQueue.main.async {{
            // Call completion handler
        }}
    }}
    
    private func extractLandmarks(from observation: VNRecognizedObjectObservation) -> [CGPoint] {{
        // Extract facial landmarks if available
        // YOLOv8-Face typically returns 5 key points: left eye, right eye, nose, left mouth corner, right mouth corner
        return []
    }}
}}

// MARK: - Data Models

struct DetectedFace {{
    let boundingBox: CGRect
    let confidence: Float
    let landmarks: [CGPoint]
}}

// MARK: - Real-time Camera Integration

class CameraViewController: UIViewController {{
    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var overlayView: UIView!
    
    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var faceDetector = YOLOv8FaceDetector()
    
    override func viewDidLoad() {{
        super.viewDidLoad()
        setupCamera()
    }}
    
    private func setupCamera() {{
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .medium
        
        guard let backCamera = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: backCamera) else {{
            return
        }}
        
        captureSession.addInput(input)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = previewView.bounds
        previewLayer.videoGravity = .resizeAspectFill
        previewView.layer.addSublayer(previewLayer)
        
        captureSession.startRunning()
    }}
}}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {{
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {{
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {{ return }}
        
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(image, from: image.extent) else {{ return }}
        
        let uiImage = UIImage(cgImage: cgImage)
        
        faceDetector.detectFaces(in: uiImage) {{ [weak self] faces in
            self?.updateOverlay(with: faces)
        }}
    }}
    
    private func updateOverlay(with faces: [DetectedFace]) {{
        // Clear previous overlays
        overlayView.layer.sublayers?.removeAll()
        
        // Draw new face detection boxes
        for face in faces {{
            let faceBox = convertToScreenCoordinates(face.boundingBox)
            drawFaceBox(faceBox, confidence: face.confidence)
        }}
    }}
    
    private func convertToScreenCoordinates(_ boundingBox: CGRect) -> CGRect {{
        // Convert Vision coordinates (0,0 at bottom-left) to UIKit coordinates (0,0 at top-left)
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -overlayView.bounds.height)
        return boundingBox.applying(transform)
    }}
    
    private func drawFaceBox(_ rect: CGRect, confidence: Float) {{
        let shapeLayer = CAShapeLayer()
        shapeLayer.path = UIBezierPath(rect: rect).cgPath
        shapeLayer.fillColor = UIColor.clear.cgColor
        shapeLayer.strokeColor = UIColor.red.cgColor
        shapeLayer.lineWidth = 2.0
        
        overlayView.layer.addSublayer(shapeLayer)
        
        // Add confidence label
        let label = CATextLayer()
        label.string = String(format: "%.2f", confidence)
        label.fontSize = 12
        label.foregroundColor = UIColor.white.cgColor
        label.backgroundColor = UIColor.red.cgColor
        label.frame = CGRect(x: rect.minX, y: rect.minY - 20, width: 50, height: 20)
        
        overlayView.layer.addSublayer(label)
    }}
}}
```

## 3. Performance Optimization Tips

1. **Image Preprocessing**: Resize images to the model's expected input size (640x640) before detection
2. **Threading**: Always run detection on background threads to avoid blocking the UI
3. **Frame Rate Control**: Limit detection frequency (e.g., every 3rd frame) for better performance
4. **Model Quantization**: Use int8 quantization for smaller model size and faster inference

## 4. Model Input/Output

- **Input**: RGB image, size 640x640 (or as specified during export)
- **Output**: Bounding boxes, confidence scores, and facial landmarks
- **Classes**: Single class "face"
- **Landmarks**: 5 key points (left eye, right eye, nose, left mouth, right mouth)

## 5. Integration with SwiftUI

```swift
import SwiftUI

struct FaceDetectionView: View {{
    @StateObject private var cameraManager = CameraManager()
    
    var body: some View {{
        ZStack {{
            CameraPreview(session: cameraManager.session)
                .ignoresSafeArea()
            
            FaceOverlayView(faces: cameraManager.detectedFaces)
        }}
        .onAppear {{
            cameraManager.startSession()
        }}
        .onDisappear {{
            cameraManager.stopSession()
        }}
    }}
}}
```

## 6. Testing

1. Test with various lighting conditions
2. Test with multiple faces in the frame
3. Test performance on different iOS devices
4. Verify memory usage and battery consumption

## 7. Deployment Considerations

- **iOS Version**: Requires iOS 11.0+ for CoreML
- **Device Compatibility**: Works on all iOS devices, but performance varies
- **Model Size**: Consider the model size impact on app download size
- **Privacy**: Ensure compliance with face detection privacy requirements
"""
    
    guide_file = Path("iOS_Integration_Guide.md")
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"📖 iOS integration guide created: {guide_file}")


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8-Face to CoreML')
    parser.add_argument('--model', type=str, default='yolov8n-face.pt',
                        help='Path to YOLOv8-Face model file')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference (default: 640)')
    parser.add_argument('--int8', action='store_true',
                        help='Use INT8 quantization for smaller model size')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = download_pretrained_model(args.model)
    if not model_path:
        return
    
    # Export to CoreML
    coreml_model = export_to_coreml(model_path, args.imgsz, args.int8)
    
    if coreml_model:
        # Create iOS integration guide
        create_ios_integration_guide(coreml_model)
        print("\n🎉 Export completed successfully!")
        print(f"CoreML model: {coreml_model}")
        print("Next steps:")
        print("1. Download a pre-trained model from the links shown above")
        print("2. Run this script again with the downloaded model")
        print("3. Follow the iOS_Integration_Guide.md for iOS integration")
    else:
        print("\n❌ Export failed. Please check the error messages above.")


if __name__ == "__main__":
    main() 