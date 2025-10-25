
EPOCHS = 5 # Reduced epochs for quick demonstration
MOSAIC = 1.0 # The comment in your code says to not use 1.0, but for some datasets, it can improve accuracy
OPTIMIZER = 'AdamW'
MOMENTUM = 0.9
LR0 = 0.001 # Higher initial learning rate
LRF = 0.01 # Final learning rate factor
SINGLE_CLS = False
IMGSZ = 640 # Default image size

import argparse
from ultralytics import YOLO
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    # Add image size as a tunable parameter
    parser.add_argument('--imgsz', type=int, default=IMGSZ, help='Input image size')
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Use a larger model for potentially better accuracy
    # Change 'yolov8s.pt' to 'yolov8m.pt' or 'yolov8l.pt' if you have the resources.
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))

    results = model.train(
        data='coco8.yaml',  # Using built-in COCO8 dataset for demonstration
        epochs=args.epochs,
        device='cpu',
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        imgsz=args.imgsz, # Pass the image size argument
        patience=100, # Added patience for early stopping to prevent overfitting
    )
