from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    # ── Device selection ───────────────────────────────────────────────────────
    # Automatically uses GPU if available, falls back to CPU safely.
    # Never hardcode device=0 — crashes on any machine without a GPU.
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[AstroVision] Training on: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

    # ── Model config path ──────────────────────────────────────────────────────
    # Path is relative to this train.py file — works on any machine.
    config_path = Path(__file__).parent / "yolov8_cbam.yaml"

    model = YOLO(str(config_path))

    model.train(
        data="cosmica.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        workers=4,
        device=device,
        name="astrovision_full"
    )

if __name__ == "__main__":
    main()
