# Multi-Paradigm VLA for Robotic Grasping — Offline Real-Data Report

## 1. Overview
This run compares three VLA action decoders on held-out real DROID robot frames:
- Autoregressive: fine-tuned OpenVLA from Notebook 2
- Diffusion: lightweight ViT+BERT+DiffusionHead baseline
- Flow-Matching: lightweight ViT+BERT+FlowMatchingHead baseline

## 2. Dataset
- Source: cadene/droid_1.0.1_v30
- Total streamed frames: 500
- Train frames: 443
- Eval frames: 57
- Robot platform: real Franka Panda data from DROID
- Evaluation mode: offline one-step action prediction on held-out frames
- Eval open-majority baseline: 56.1%
- Eval zero-action baseline: 0.97 cm / 0.63 deg
- Control-relevant eval fraction: 84.2%

## 3. Metrics
- Translation MAE (cm): average absolute XYZ delta error after converting to this repo's control interface
- Rotation MAE (deg): average absolute roll/pitch/yaw delta error
- Gripper Accuracy: binary open/close agreement
- Gripper Balanced Accuracy: average of close recall and open recall
- P50 Latency: median inference latency per frame

## 4. Results
### Autoregressive
- Translation MAE: 0.97 cm
- Rotation MAE: 0.63 deg
- Gripper Accuracy: 84.2%
- Gripper Balanced Accuracy: 84.6%
- Parse Fail Rate: 1.8%
- P50 Inference Latency: 3500.2 ms
- Control-Relevant Subset: 1.15 cm / 0.74 deg / 85.4%

### Diffusion
- Translation MAE: 2.73 cm
- Rotation MAE: 2.36 deg
- Gripper Accuracy: 52.6%
- Gripper Balanced Accuracy: 49.1%
- Parse Fail Rate: 0.0%
- P50 Inference Latency: 23.2 ms
- Control-Relevant Subset: 2.74 cm / 2.40 deg / 45.8%

### Flow-Matching
- Translation MAE: 1.56 cm
- Rotation MAE: 1.00 deg
- Gripper Accuracy: 50.9%
- Gripper Balanced Accuracy: 47.1%
- Parse Fail Rate: 0.0%
- P50 Inference Latency: 22.6 ms
- Control-Relevant Subset: 1.66 cm / 1.02 deg / 43.8%

## 5. Summary
- Best translation error: Autoregressive (0.97 cm)
- Best gripper accuracy: Autoregressive (84.2%)
- Fastest inference: Flow-Matching (22.6 ms p50)
- Interpret gripper results relative to the eval open-majority baseline above; a model that mostly emits open can look deceptively good on idle-heavy splits.

## 6. Output Files
- flow_matching_vla.pt
- diffusion_vla.pt
- training_curve.png
- diffusion_training_curve.png
- real_offline_summary.json
- real_offline_table.md
- real_offline_metrics.png
- real_offline_examples.png
- real_offline_predictions.jsonl
- technical_report.md
