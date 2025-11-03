if __name__ == '__main__':
    from ultralytics import YOLO
    model_path = "best.pt"
    model = YOLO(model_path)
    model.val(data="meta.yaml",
            split="test",
            imgsz=640,
            batch=16,
            device=0,
            project="val",
            name="exp",
            save_json=True
            )