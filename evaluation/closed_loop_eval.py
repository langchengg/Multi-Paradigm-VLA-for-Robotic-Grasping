"""
Closed-Loop VLA Evaluator
=========================
Deploys a VLA model in closed-loop with a MuJoCo environment:
  image → VLA predict → env.step → new image → repeat

Supports evaluation across multiple decoder paradigms and
generates comparison metrics + video assets.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image


class VLAMuJoCoEvaluator:
    """
    Closed-loop evaluator for VLA models in MuJoCo environments.

    Pipeline per episode:
    1. env.reset() → initial observation (image + instruction)
    2. VLA model predicts action from (image, instruction)
    3. env.step(action) → new observation + reward + done
    4. Repeat 2-3 until done or max steps
    5. Record success, trajectory, frames

    Usage:
        evaluator = VLAMuJoCoEvaluator(model, env)
        results = evaluator.evaluate(num_episodes=50)
    """

    def __init__(self, model, env, use_oracle_info=True):
        """
        Args:
            model: VLA model with predict_action(image, instruction, ...) method
            env: Environment with reset() and step() methods
            use_oracle_info: if True, pass gripper_pos and target_pos to model
                           (only for DummyVLA testing)
        """
        self.model = model
        self.env = env
        self.use_oracle_info = use_oracle_info

    def evaluate(self, num_episodes=50, max_steps=150, record_video=True,
                 save_dir="assets", target_objects=None, verbose=True,
                 visualize=False):
        """
        Run closed-loop evaluation.

        Args:
            num_episodes: number of evaluation episodes
            max_steps: maximum steps per episode
            record_video: whether to save episode videos as GIFs
            save_dir: directory to save videos and results
            target_objects: list of objects to evaluate on (None = random)
            verbose: print progress

        Returns:
            results: dict with metrics and per-episode data
        """
        os.makedirs(save_dir, exist_ok=True)

        # Launch interactive viewer if requested
        if visualize and hasattr(self.env, "launch_viewer"):
            self.env.launch_viewer()

        if target_objects is None:
            target_objects = getattr(self.env, "OBJECTS", ["red_cube", "blue_cube"])

        results = {
            "successes": [],
            "steps": [],
            "rewards": [],
            "inference_times": [],
            "trajectories": [],
        }

        if verbose:
            decoder = getattr(self.model, "decoder_type", "unknown")
            print(f"\n{'='*60}")
            print(f"Closed-Loop Evaluation: {decoder}")
            print(f"  Episodes: {num_episodes}, Max steps: {max_steps}")
            print(f"  Targets: {target_objects}")
            print(f"{'='*60}")

        for ep in range(num_episodes):
            target = np.random.choice(target_objects)
            obs = self.env.reset(target_object=target)

            frames = [obs["image"].copy()] if record_video else None
            trajectory = {
                "gripper_positions": [obs["gripper_pos"].copy()],
                "target_object": target,
                "instruction": obs["instruction"],
            }

            episode_reward = 0
            episode_inf_times = []

            for step in range(max_steps):
                # ── VLA Inference ──
                kwargs = {}
                if self.use_oracle_info:
                    kwargs["gripper_pos"] = obs.get("gripper_pos")
                    kwargs["target_pos"] = obs.get("target_pos")

                action, inf_info = self.model.predict_action(
                    obs["image"],
                    obs["instruction"],
                    **kwargs,
                )

                episode_inf_times.append(inf_info.get("inference_time_ms", 0))

                # ── Environment Step ──
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward

                if record_video:
                    frames.append(obs["image"].copy())

                trajectory["gripper_positions"].append(obs["gripper_pos"].copy())

                if done:
                    break

            # Record results
            success = info.get("success", False)
            results["successes"].append(success)
            results["steps"].append(step + 1)
            results["rewards"].append(episode_reward)
            results["inference_times"].extend(episode_inf_times)
            results["trajectories"].append(trajectory)

            # Save video for first few episodes
            if record_video and ep < 5:
                self._save_gif(
                    frames,
                    os.path.join(save_dir, f"episode_{ep:03d}_{target}.gif"),
                    fps=10,
                )

            if verbose and (ep + 1) % max(1, num_episodes // 5) == 0:
                rate = np.mean(results["successes"])
                avg_steps = np.mean(results["steps"])
                print(f"  Episode {ep+1}/{num_episodes}: "
                      f"success_rate={rate:.1%}, avg_steps={avg_steps:.1f}")

        # Compute summary statistics
        results["summary"] = self._compute_summary(results)

        if verbose:
            self._print_summary(results["summary"])

        # Save summary
        summary_path = os.path.join(save_dir, "eval_results.txt")
        self._save_summary(results["summary"], summary_path)

        return results

    def evaluate_comparison(self, models_dict, num_episodes=30, visualize=False,
                             **kwargs):
        """
        Compare multiple VLA models/decoders.

        Args:
            models_dict: {"name": model_instance, ...}
            num_episodes: episodes per model
            visualize: if True, open interactive MuJoCo viewer

        Returns:
            comparison: dict of results per model
        """
        comparison = {}

        for name, model in models_dict.items():
            print(f"\n{'─'*40}")
            print(f"Evaluating: {name}")
            print(f"{'─'*40}")

            self.model = model
            save_dir = os.path.join(
                kwargs.get("save_dir", "assets"), name
            )
            results = self.evaluate(
                num_episodes=num_episodes,
                save_dir=save_dir,
                visualize=visualize,
                **{k: v for k, v in kwargs.items() if k != "save_dir"},
            )
            comparison[name] = results

        # Print comparison table
        self._print_comparison(comparison)
        return comparison

    def _compute_summary(self, results):
        """Compute summary statistics."""
        return {
            "success_rate": np.mean(results["successes"]),
            "avg_steps": np.mean(results["steps"]),
            "std_steps": np.std(results["steps"]),
            "avg_reward": np.mean(results["rewards"]),
            "avg_inference_ms": np.mean(results["inference_times"]) if results["inference_times"] else 0,
            "p50_inference_ms": np.percentile(results["inference_times"], 50) if results["inference_times"] else 0,
            "p95_inference_ms": np.percentile(results["inference_times"], 95) if results["inference_times"] else 0,
            "num_episodes": len(results["successes"]),
        }

    def _print_summary(self, summary):
        """Print evaluation summary."""
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({summary['num_episodes']} episodes)")
        print(f"{'='*50}")
        print(f"  Success Rate:    {summary['success_rate']:.1%}")
        print(f"  Avg Steps:       {summary['avg_steps']:.1f} ± {summary['std_steps']:.1f}")
        print(f"  Avg Reward:      {summary['avg_reward']:.2f}")
        print(f"  Inference (p50): {summary['p50_inference_ms']:.1f}ms")
        print(f"  Inference (p95): {summary['p95_inference_ms']:.1f}ms")
        print(f"{'='*50}")

    def _print_comparison(self, comparison):
        """Print comparison table."""
        print(f"\n{'='*70}")
        print(f"{'Model':<20} {'Success%':>10} {'Avg Steps':>10} {'Latency(ms)':>12}")
        print(f"{'─'*70}")
        for name, result in comparison.items():
            s = result["summary"]
            print(f"{name:<20} {s['success_rate']:>9.1%} "
                  f"{s['avg_steps']:>10.1f} "
                  f"{s['p50_inference_ms']:>11.1f}")
        print(f"{'='*70}")

    def _save_gif(self, frames, path, fps=10):
        """Save frames as GIF."""
        if not frames:
            return
        images = [Image.fromarray(f) for f in frames[::2]]  # subsample
        duration = int(1000 / fps)
        images[0].save(
            path, save_all=True, append_images=images[1:],
            duration=duration, loop=0,
        )

    def _save_summary(self, summary, path):
        """Save summary to text file."""
        with open(path, "w") as f:
            f.write("VLA MuJoCo Closed-Loop Evaluation Results\n")
            f.write("=" * 50 + "\n")
            for key, val in summary.items():
                if isinstance(val, float):
                    f.write(f"{key}: {val:.4f}\n")
                else:
                    f.write(f"{key}: {val}\n")


# ─── CLI entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from envs.simple_grasp_env import SimpleGraspEnv
    from models.dummy_vla import DummyVLA

    print("=" * 60)
    print("Testing Closed-Loop VLA Evaluation")
    print("=" * 60)

    # Create environment
    env = SimpleGraspEnv(image_size=128, camera_name="frontview")

    # Compare 3 decoders
    models = {
        "autoregressive": DummyVLA("autoregressive"),
        "diffusion": DummyVLA("diffusion"),
        "flow_matching": DummyVLA("flow_matching"),
    }

    evaluator = VLAMuJoCoEvaluator(
        model=models["flow_matching"],
        env=env,
        use_oracle_info=True,
    )

    # Run comparison
    comparison = evaluator.evaluate_comparison(
        models,
        num_episodes=10,
        max_steps=100,
        record_video=True,
        save_dir=str(project_root / "assets"),
    )

    env.close()
    print(f"\n✅ Closed-loop evaluation test complete!")
