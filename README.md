# VLA Action Decoder Benchmark: Autoregressive vs Diffusion vs Flow-Matching 🦾

> **Which action decoder is best for VLA robotic manipulation?**
> This project systematically compares 3 paradigms using a Franka Panda in MuJoCo.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.5-green.svg)](https://mujoco.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Core Research Question

VLA models (Vision-Language-Action) take in camera images + language instructions and output robot actions. **But how should the model generate those actions?** Nobody knows which method is best. This project benchmarks all three:

```text
Core question: after the VLA "brain" decides what to do, how should it turn that intent into concrete robot arm actions?

Method A: Autoregressive   — used by OpenVLA
Method B: Diffusion        — used by Diffusion Policy
Method C: Flow-Matching    — used by Physical Intelligence π0
```

## ✅ Recommended Run Order

### A. Local Quick Sanity Check

```bash
# 1) Install Python dependencies
pip install -r requirements.txt

# 2) OpenGL dependencies needed for headless Linux / Kaggle
apt-get update -qq && apt-get install -y -qq \
  libgl1-mesa-glx libgl1-mesa-dev libegl1-mesa-dev \
  libosmesa6-dev libglew-dev patchelf

# 3) Verify the Franka Panda environment
python -m envs.franka_grasp_env

# 4) Run the key tests
python -m pytest -q \
  tests/test_env.py \
  tests/test_rendering.py \
  tests/test_notebooks_franka.py \
  tests/test_openvla_notebook.py \
  tests/test_flow_matching_head.py

# 5) Run the local quick pipeline
python scripts/run_demo.py --quick
```

### B. Full Kaggle Pipeline

Use this fixed order:

1. `notebooks/01_env_setup_and_demo.py`
2. `notebooks/02_openvla_qlora_finetune.py`
3. `notebooks/03_flow_matching_eval.py`

If `Notebook 1` and `Notebook 2` are not run in the same Kaggle session, upload `/kaggle/working/demos/` as a Dataset first, then mount it in `Notebook 2`.

---

## 🧠 Three Action Decoders Explained

### Method A: Autoregressive (like ChatGPT, one token at a time)

```text
Just like ChatGPT generates text:
  "I" → "feel" → "very" → "happy"

OpenVLA generates actions in the same way:
  first decide the x delta → then y → then z → ... → finally the gripper

How it works: continuous actions are discretized into 256 bins
  the x value lies in [-1, +1]
  0.03 → maps to bin 131 → output token "131"

So the action becomes: "131, 122, 89, 128, 128, 128, 1"
                      x    y    z   rx   ry   rz  gripper

At its core, this turns action generation into a classification problem:
pick one bin out of 256.
```

**Advantages**: simple, mature, already used by OpenVLA  
**Drawbacks**: limited precision (only 256 bins), slower (7 numbers = 7 forward passes), dimensions are predicted independently

---

### Method B: Diffusion (sculpting an action out of noise)

```text
Like image generation in Stable Diffusion:
  noisy image → progressively denoise → sharp image

Action generation works the same way:
  random noise        [0.83, -0.45, 0.12, ...]  (messy numbers)
  ↓ denoise step 1    [0.52, -0.23, 0.08, ...]
  ↓ denoise step 2    [0.31, -0.10, 0.01, ...]
  ↓ ... repeat 50-100 steps ...
  ↓ final denoise     [0.03, -0.01, -0.05, 0, 0, 0, 1]  ← final action
```

**Advantages**: continuous action space, effectively unlimited precision, can represent multiple valid ways to solve the task  
**Drawbacks**: many denoising steps, slow inference (50-100 steps)

---

### Method C: Flow-Matching (draw a straight line from noise to action)

```text
Diffusion models take a curved path:
  noise  ~~~curved path~~~>  action   (many steps)

Flow-matching takes a straight path:
  noise  ——straight line——>  action   (only 5-10 steps! 🚀)

How it works: train a network to predict a velocity field
  t=0.0:  pure noise    [0.83, -0.45, ...]
  t=0.5:  halfway there [0.43, -0.23, ...]  (velocity field points the direction)
  t=1.0:  target action [0.03, -0.01, ...]  (final action)

Physical Intelligence's π0 uses this paradigm.
```

**Advantages**: continuous space + 5-10× faster than diffusion (straight path vs curved path)  
**Drawbacks**: requires tuning and can be sensitive to the noise schedule

---

### Visual Comparison of the Three Methods

```text
Method A Autoregressive:  [image + text] → bin131 → bin122 → bin89 → ... (choose one by one)
Method B Diffusion:       [image + text] → noise ~~~> ~~> ~~> ~~> action  (gradual denoising)
Method C Flow-Matching:   [image + text] → noise ——————————> action        (straight path 🚀)
```

---

## 📦 Outputs After Running

The README home page no longer shows pre-baked demo GIFs from the repo. Instead, it shows what you will actually get after running the project yourself.

### Local Run: `python scripts/run_demo.py --quick`

This generates:

```text
assets/
  initial_frame.png
  view_frontview.png
  view_topdown.png
  view_sideview.png
  expert_demo.gif
  autoregressive/episode_000_*.gif
  autoregressive/eval_results.txt
  diffusion/episode_000_*.gif
  diffusion/eval_results.txt
  flow_matching/episode_000_*.gif
  flow_matching/eval_results.txt
  trajectories_3d.png
  trajectories_2d.png
  success_heatmap.png
data/demos/
  demo_0000.npz
  demo_0001.npz
  ...
assets_quick.zip
```

The terminal will also print an offline real-data summary for each decoder:

```text
autoregressive: translation_mae=..., gripper_acc=..., latency=...ms
diffusion:      translation_mae=..., gripper_acc=..., latency=...ms
flow_matching:  translation_mae=..., gripper_acc=..., latency=...ms
```

### After Finishing All 3 Kaggle Notebooks

You will have these outputs under `/kaggle/working/`:

1. `demos/demo_*.npz`
2. `expert_demo.gif`
3. `openvla-finetuned/final/`
4. `openvla-finetuned/final/franka_action_config.json`
5. `results/flow_matching_vla.pt`
6. `results/diffusion_vla.pt`
7. `results/training_curve.png`
8. `results/diffusion_training_curve.png`
9. `results/autoregressive/offline_eval.json`
10. `results/diffusion/offline_eval.json`
11. `results/flow_matching/offline_eval.json`
12. `results/real_offline_summary.json`
13. `results/real_offline_table.md`
14. `results/real_offline_metrics.png`
15. `results/real_offline_examples.png`
16. `results/technical_report.md`

---

## 🏗️ System Architecture

```text
┌─────────── Training Pipeline ───────────┐    ┌── Offline Real-Data Eval ─┐
│                                          │    │                          │
│ Notebook 1: MuJoCo Franka demos          │    │ Held-out DROID frame      │
│       │                                  │    │ (image + instruction)     │
│ Notebook 2: OpenVLA QLoRA                │    │       ↓                  │
│   + DROID real robot data                │    │ 3 decoders predict        │
│       │                                  │    │ normalized Franka action  │
│ Notebook 3: Flow/Diffusion               │    │       ↓                  │
│   trained on DROID train split           │    │ Compare against held-out  │
│       │                                  │    │ real action target        │
│  ┌────┴────────────────┐                 │    │       ↓                  │
│  │ Decoder A: AR        │ ← OpenVLA      │    │ translation / rotation    │
│  │ Decoder B: Diffusion │ ← Diffusion    │    │ gripper / latency metrics │
│  │ Decoder C: Flow      │ ← π0           │    │       ↓                  │
│  └─────────────────────┘                 │    │ report + plots + tables   │
└──────────────────────────────────────────┘    └──────────────────────────┘
```

---

## 🚀 From-Zero Setup (step by step)

### Step 0: Clone and Install

```bash
git clone https://github.com/langchengg/Multi-Paradigm-VLA-for-Robotic-Grasping.git
cd Multi-Paradigm-VLA-for-Robotic-Grasping
pip install -r requirements.txt
```

Headless Linux / Kaggle also needs native off-screen rendering libraries:

```bash
apt-get update -qq && apt-get install -y -qq \
  libgl1-mesa-glx libgl1-mesa-dev libegl1-mesa-dev \
  libosmesa6-dev libglew-dev patchelf
```

### Step 1: Verify Environment (10 seconds)

```bash
# Test that the Franka Panda environment loads correctly
python -m envs.franka_grasp_env
```

You should see:

```text
[FrankaGraspEnv] image=256x256, camera=frontview
  7-DOF Franka Panda + parallel gripper
  Objects: ['red_cube', 'blue_cube', 'green_cube']
✅ Franka Panda env test passed!
```

### Step 2: Run Unit Tests

```bash
python -m pytest tests/test_env.py -v
```

Expected: `20 passed` ✅

### Step 3: Run Full Local Pipeline (~15 seconds)

```bash
# Quick mode: collect demos + evaluate 3 decoders + generate GIFs
python scripts/run_demo.py --quick
```

This does everything:

1. Creates a MuJoCo env with a Franka Panda + 3 colored cubes
2. Collects expert demos with a scripted grasping policy
3. Evaluates **all 3 decoders** (autoregressive, diffusion, flow-matching)
4. Generates GIFs, trajectory plots, and comparison charts into `assets/`

### Step 4: Test Flow-Matching Decoder

```bash
# Standalone flow-matching head test
python models/flow_matching_head.py
```

Expected:

```text
✅ Flow loss: ~1.83
✅ Predicted actions: torch.Size([4, 4, 7])   # (batch, horizon, action_dim)
   FlowMatchingHead params: 0.40M
```

```bash
# Standalone diffusion head test (new)
python models/diffusion_head.py
```

Expected:

```text
✅ Diffusion loss: ~1.00
✅ Predicted actions: torch.Size([4, 4, 7])   # (batch, horizon, action_dim)
   DiffusionHead params: ~0.40M
```

### Step 5: Train on Kaggle T4 GPU

Upload the files to Kaggle and run the 3 notebooks in order:

| Step | Notebook | Time | GPU |
|------|----------|------|-----|
| 5a | `notebooks/01_env_setup_and_demo.py` | ~10 min | CPU ok |
| 5b | `notebooks/02_openvla_qlora_finetune.py` | ~1-3 hrs | T4 required |
| 5c | `notebooks/03_flow_matching_eval.py` | ~45-90 min | T4 required |

**Notebook 1** → Collects 100 expert demos in MuJoCo → upload as a Kaggle Dataset  
**Notebook 2** → Fine-tunes OpenVLA-7B with QLoRA on MuJoCo demos + DROID real-robot data  
**Notebook 3** → Trains lightweight FlowMatching/Diffusion VLAs + held-out DROID offline eval

If Kaggle throws `RuntimeError: Numpy is not available` while loading OpenVLA, pin `numpy==1.26.4`. The install cells in Notebook 2 and 3 do this intentionally because `torch==2.2.0` can break against NumPy 2.x when remote OpenVLA processor code calls `tensor.numpy()`.
Notebook 2 now auto-discovers `demo_*.npz` under common Kaggle mount points and can stream up to 1000 real Franka robot samples from [`cadene/droid_1.0.1_v30`](https://huggingface.co/datasets/cadene/droid_1.0.1_v30). Both Notebook 2 and Notebook 3 import the same DROID-to-Franka conversion helpers from `data/droid_utils.py`, so training and offline evaluation use one shared action interface. The default Kaggle-fast preset also runs for 1 epoch. If neither source yields data, it fails immediately instead of pretending external data was loaded.

### Step 6: View Results

```bash
ls assets/
ls data/demos/ | head
ls -lh assets_quick.zip
# → initial_frame.png, view_*.png, expert_demo.gif,
#   autoregressive/, diffusion/, flow_matching/,
#   trajectories_3d.png, trajectories_2d.png, success_heatmap.png
```

---

## 📁 Project Structure

```text
├── envs/
│   ├── franka_grasp_env.py         # 🦾 Franka Panda 7-DOF + parallel gripper (primary)
│   ├── simple_grasp_env.py         # Simplified 3-DOF gripper (for quick tests)
│   └── robosuite_wrapper.py        # Robosuite Lift/PickPlaceCan wrapper
├── models/
│   ├── flow_matching_head.py       # 🔥 Flow-matching action decoder (π0-inspired)
│   ├── diffusion_head.py           # 🔥 Diffusion action decoder (DDPM/DDIM)
│   └── dummy_vla.py                # Dummy VLA implementing all 3 decoders for testing
├── data/
│   ├── collect_demos.py            # Scripted expert demo collection
│   └── droid_utils.py              # Shared DROID parsing + Franka action adapters
├── evaluation/
│   ├── closed_loop_eval.py         # Legacy MuJoCo closed-loop evaluation helpers
│   └── generate_videos.py          # Legacy GIF/video export helpers
├── visualization/
│   ├── plot_trajectories.py        # 3D/2D end-effector trajectory plots
│   └── success_heatmap.py          # Grasp success rate by object position
├── notebooks/
│   ├── 01_env_setup_and_demo.py    # Kaggle: MuJoCo setup + 100 expert demos
│   ├── 02_openvla_qlora_finetune.py # Kaggle: OpenVLA-7B QLoRA fine-tuning on T4
│   └── 03_flow_matching_eval.py    # Kaggle: DROID offline train/eval comparison
├── scripts/
│   └── run_demo.py                 # One-click: full pipeline in ~15 seconds
├── tests/
│   ├── test_env.py                 # Unit tests (20/20 passing)
│   └── test_rendering.py           # Headless rendering tests
└── requirements.txt
```

---

## 🔬 Technical Details

### Franka Panda Environment

- **7-DOF** Franka Emika Panda arm (based on MuJoCo Menagerie geometry)
- **Parallel gripper** with 2 fingers and finger pads
- **7-DOF action space**: `[dx, dy, dz, dax, day, daz, gripper]` — Cartesian + orientation control
- **Backward compatible**: also accepts 4-DOF `[dx, dy, dz, gripper]` (rotation auto-padded to 0)
- **Jacobian-based IK**: resolved-rate inverse kinematics (damped least-squares) converts Cartesian + angular velocity to 7-DOF joint commands using both position and rotation Jacobians
- **3 colored objects** (red, blue, green cubes) with randomized positions
- **12 language instructions** across 3 objects
- **3 camera views** (front, top-down, side) at 256×256

### Flow-Matching Decoder (π0-inspired)

```python
# Training
t = sample_beta(α=1.5, β=1.0)                    # Shifted beta time schedule
x_t = (1-t) * noise + t * action_gt              # Linear interpolation
loss = MSE(velocity_net(x_t, t, features),       # Predict velocity field
           action_gt - noise)                    # Target: optimal transport direction

# Inference (only 10 ODE steps — 5× faster than diffusion)
x = randn(batch, horizon * action_dim)           # Start from noise
for i in range(10):
    x += velocity_net(x, t=i/10, features) * dt  # Euler integration
```

Key choices:

- **Shifted beta distribution** for time sampling (π0 recipe)
- **Action chunking** H=4 (predict 4 future actions at once)
- **Sinusoidal time embeddings** (same as diffusion models)
- **~0.4M parameters** for the flow head alone

### Diffusion Decoder (Diffusion-Policy-inspired)

```python
# Training (DDPM)
t = randint(0, T)                                 # Random timestep
noise = randn_like(action)                        # Sample noise
x_t = sqrt(ā_t) * action_gt + sqrt(1-ā_t) * noise # Forward diffusion
loss = MSE(noise_net(x_t, t, features), noise)    # Predict noise

# Inference (DDIM, 10 steps — deterministic, no stochastic noise)
x = randn(batch, horizon * action_dim)            # Start from noise
for t in reversed(ddim_timesteps):                # 10 evenly-spaced steps
    pred_noise = noise_net(x, t, features)
    pred_x0 = (x - sqrt(1-ā_t)*pred_noise) / sqrt(ā_t)
    x = sqrt(ā_{t-1}) * pred_x0 + sqrt(1-ā_{t-1}) * pred_noise
```

Key choices:

- **Cosine noise schedule** (improved DDPM, Nichol & Dhariwal 2021)
- **DDIM deterministic sampling** for faster inference (10 steps vs 50-100)
- **Action chunking** H=4 (same as flow-matching)
- **~0.4M parameters** for the diffusion head

### OpenVLA QLoRA Setup (Notebook 2)

```text
OpenVLA-7B (7 billion params)
  ↓ 4-bit NF4 quantization (bitsandbytes)
  ↓ LoRA adapters (rank=32, α=64)
  ↓ Only 0.4% params trainable (~28M)
  ↓ Fits on Kaggle T4 GPU (16GB VRAM)
  ↓ Effective batch size: 16 (BS=2 × grad_accum=8)
```

### Offline Real-Data Evaluation Pipeline

```text
For each held-out DROID frame:
  1. Load a real Franka RGB observation + natural language instruction
  2. Convert the DROID action into this repo's normalized Franka delta-pose interface
  3. Run each decoder offline:
     a. Autoregressive OpenVLA → text action → parsed 7-DOF action
     b. Diffusion VLA → predicted 7-DOF action
     c. Flow-Matching VLA → predicted 7-DOF action
  4. Compare predictions against the held-out real action:
     - translation MAE (cm)
     - rotation MAE (deg)
     - gripper open/close accuracy
     - inference latency
  5. Save summary tables, plots, and qualitative example panels
```

---

## 🔮 Future Directions

- [x] ~~Implement flow-matching decoder (ODE-based)~~
- [x] ~~Implement diffusion decoder (DDPM/DDIM)~~
- [x] ~~Add action chunking (H=4 future actions)~~
- [x] ~~OpenVLA QLoRA fine-tuning on T4~~
- [x] ~~Franka Panda with parallel gripper~~
- [x] ~~7-DOF action space (Cartesian + orientation + gripper)~~
- [ ] Domain randomization (lighting, textures, camera poses)
- [x] ~~Offline evaluation on held-out real DROID data~~
- [ ] Multi-object sequential manipulation

## 📄 References

- [OpenVLA: An Open-Source Vision-Language-Action Model](https://openvla.github.io/) — autoregressive baseline
- [π0: A Vision-Language-Action Flow Model](https://www.physicalintelligence.company/blog/pi0) — flow-matching inspiration
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — diffusion baseline
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — mathematical foundation
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Franka Panda model reference

---

*Built as a portfolio project demonstrating a systematic comparison of VLA action decoders for robotic manipulation. The full pipeline — custom Franka Panda environment, OpenVLA fine-tuning, lightweight decoder training, and offline real-data evaluation on DROID — runs from zero with the commands above.*
