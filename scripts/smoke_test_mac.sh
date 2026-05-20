#!/usr/bin/env bash
set -euo pipefail

run_step() {
  local name="$1"
  shift
  echo
  echo "== ${name} =="
  if ! "$@"; then
    echo
    echo "Smoke test failed at: ${name}"
    echo "Command: $*"
    echo "Likely causes: missing dependencies, unavailable MuJoCo renderer, or missing macOS GUI context."
    echo "Fix: install requirements, then retry from a GUI terminal; for viewer workflows use mjpython on macOS."
    exit 1
  fi
}

run_step "Python version" python --version
run_step "PyTorch device" python -c "from src.utils.device import get_device; print(get_device('auto'))"
run_step "Import MuJoCo" python -c "import mujoco; print(mujoco.__version__)"

if python -c "from src.envs.synthetic_franka_grasp_env import SyntheticFrankaGraspEnv; env=SyntheticFrankaGraspEnv(image_size=64); obs=env.reset(target_object='red_cube', randomize=False); print(obs['image'].shape, obs['robot_state']['eef_pos'].shape); env.close()"; then
  echo "Real MuJoCo renderer is available."
else
  echo
  echo "Real MuJoCo renderer is unavailable in this terminal; retrying with VLA_MUJOCO_FAKE_RENDERER=1."
  echo "Physics/control still run, but smoke RGB frames are black placeholders."
  export VLA_MUJOCO_FAKE_RENDERER=1
  run_step "Reset synthetic Franka env with explicit headless renderer fallback" python -c "from src.envs.synthetic_franka_grasp_env import SyntheticFrankaGraspEnv; env=SyntheticFrankaGraspEnv(image_size=64); obs=env.reset(target_object='red_cube', randomize=False); print(obs['image'].shape, obs['robot_state']['eef_pos'].shape); env.close()"
fi

run_step "Generate 3 synthetic HDF5 demos" python -m src.data.generate_synthetic_demos --output data/smoke_synthetic.hdf5 --num_episodes 3 --image_size 64 --max_steps 40 --noise_std 0.0
run_step "Train tiny diffusion model for 2 batches" python -m src.training.train --dataset synthetic --data_path data/smoke_synthetic.hdf5 --decoder diffusion --num_epochs 1 --batch_size 2 --horizon 4 --max_batches 2 --no_pretrained_clip --tiny_random_clip --hidden_dim 64 --num_layers 2 --diffusion_train_steps 10 --inference_steps 2 --checkpoint_dir checkpoints/smoke --results_dir results/smoke --disable_tensorboard
run_step "Run one evaluation rollout" python -m src.training.evaluate --dataset synthetic --decoder diffusion --num_episodes 1 --max_steps 10 --horizon 4 --no_pretrained_clip --tiny_random_clip --hidden_dim 64 --num_layers 2 --diffusion_train_steps 10 --inference_steps 2 --checkpoint_dir checkpoints/smoke --results_dir results/smoke --no_save_video

echo
echo "Smoke test completed successfully."
