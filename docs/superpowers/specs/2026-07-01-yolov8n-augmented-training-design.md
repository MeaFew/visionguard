# YOLOv8n Augmented Training for NEU-DET

## Goal
Improve real NEU-DET test mAP50 from the current **0.750** to **0.80+**, with special focus on low-AP classes `crazing` (0.41) and `rolled-in_scale` (0.60).

## Approach
Instead of jumping to a larger model (YOLOv8s) which would roughly triple training time on the RTX 4060 Laptop, keep the fast **YOLOv8n** backbone and invest the saved compute budget into:

1. Longer training: 100 → **150 epochs**
2. RAM image caching to remove disk I/O bottleneck
3. Stronger augmentation tuned for NEU-DET surface defects
4. A larger batch (still within 8 GB VRAM) to improve GPU utilization

## Hyperparameters

| Parameter | Baseline | New run |
|-----------|----------|---------|
| model | yolov8n.pt | yolov8n.pt |
| imgsz | 640 | 640 |
| batch | 8 | **16** (with cache=ram VRAM usage stays low) |
| epochs | 100 | **150** |
| cache | none | **ram** |
| workers | 8 | **16** |
| mosaic | 1.0 | 1.0 |
| mixup | 0.0 | **0.1** |
| copy_paste | 0.0 | **0.1** |
| degrees | 0.0 | **5.0** |
| translate | 0.1 | 0.1 |
| scale | 0.5 | 0.5 |
| hsv_h | 0.015 | 0.015 |
| hsv_s | 0.7 | 0.7 |
| hsv_v | 0.4 | 0.4 |

> Note: `batch=16` worked for the baseline run at 1.38 GB VRAM. With `cache=ram` the memory overhead is slightly higher but should still fit comfortably in 8 GB.

## Expected Outputs
- `runs/detect/models/real_train_aug/weights/best.pt`
- `runs/detect/models/real_train_aug/weights/best.onnx`
- `reports/real_aug_evaluation_test.json`
- `assets/real_aug_demo_detection.jpg`
- Updated README real-data results section

## Validation Plan
1. Train and record per-epoch val mAP.
2. Evaluate final `best.pt` on the held-out test split.
3. Export ONNX and benchmark CPU inference latency.
4. Run demo on one real test image and save visualization.
5. Compare metrics with the baseline yolov8n run:
   - Overall test mAP50, mAP50-95
   - Per-class AP for crazing and rolled-in_scale
   - Training wall-clock time

## Rollback / Abort Criteria
- If training OOMs with batch=16 + cache=ram, reduce batch to 12 or 8.
- If val mAP does not improve over baseline after 75 epochs, stop early and keep baseline.
