from ultralytics import YOLO

def export_model():
    print("Loading PyTorch model...")
    model = YOLO('best.pt')

    print("Exporting to ONNX format...")
    success = model.export(format='onnx', dynamic=False, simplify=True)
    
    if success:
        print(f"Success! Model exported successfully. Check your directory for the .onnx file.")
    else:
        print("Export failed. Please check your dependencies.")

if __name__ == "__main__":
    export_model()