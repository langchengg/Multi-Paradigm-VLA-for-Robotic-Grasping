"""
Dummy VLA Model
===============
Lightweight placeholder for testing the closed-loop evaluation pipeline
without requiring GPU or loading a real 7B parameter model.

Provides three "decoder" modes to simulate different VLA architectures:
- autoregressive: sequential action token generation
- diffusion: denoising-based action prediction
- flow_matching: ODE-based action generation (fastest)
"""

import numpy as np
import time


class DummyVLA:
    """
    Dummy Vision-Language-Action model for pipeline testing.

    Simulates VLA inference by generating heuristic actions based on
    simple image statistics. This allows testing the full closed-loop
    pipeline without GPU resources.

    In production, this would be replaced by:
    - OpenVLA (autoregressive, 7B params)
    - Diffusion Policy head
    - Flow-matching head (π0-inspired)
    """

    DECODER_LATENCIES = {
        "autoregressive": 0.15,  # ~150ms (slowest, token-by-token)
        "diffusion": 0.08,      # ~80ms (10 denoising steps)
        "flow_matching": 0.03,  # ~30ms (fastest, few ODE steps)
    }

    def __init__(self, decoder_type="flow_matching", action_dim=4, simulate_latency=False):
        """
        Args:
            decoder_type: "autoregressive", "diffusion", or "flow_matching"
            action_dim: output action dimension
            simulate_latency: if True, adds artificial delay to mimic real inference
        """
        assert decoder_type in self.DECODER_LATENCIES, \
            f"Unknown decoder: {decoder_type}. Choose from {list(self.DECODER_LATENCIES.keys())}"

        self.decoder_type = decoder_type
        self.action_dim = action_dim
        self.simulate_latency = simulate_latency
        self._call_count = 0

        # Phase-based oracle state (reset each episode via reset_episode)
        self._phase = 0
        self._phase_step = 0
        self._prev_target_pos = None

        print(f"[DummyVLA] Decoder: {decoder_type}, "
              f"Action dim: {action_dim}, "
              f"Simulated latency: {self.DECODER_LATENCIES[decoder_type]*1000:.0f}ms")

    def reset_episode(self):
        """Reset internal state for a new episode."""
        self._phase = 0
        self._phase_step = 0
        self._prev_target_pos = None

    def predict_action(self, image, instruction, gripper_pos=None, target_pos=None):
        """
        Predict action given observation.

        For testing, uses a phase-based heuristic matching the expert:
        Phase 0: approach (move above target)
        Phase 1: descend (lower to grasp height)
        Phase 2: grasp (close fingers)
        Phase 3: lift (raise object)

        Args:
            image: (H, W, 3) uint8 array
            instruction: str, language instruction
            gripper_pos: (3,) optional, current gripper position
            target_pos: (3,) optional, target object position

        Returns:
            action: (action_dim,) float array
            info: dict with inference metadata
        """
        self._call_count += 1
        start_time = time.time()

        # Detect new episode (target position changed significantly)
        if target_pos is not None:
            if self._prev_target_pos is not None:
                if np.linalg.norm(target_pos - self._prev_target_pos) > 0.1:
                    self.reset_episode()
            else:
                self.reset_episode()
            self._prev_target_pos = target_pos.copy()

        if self.simulate_latency:
            time.sleep(self.DECODER_LATENCIES[self.decoder_type])

        if gripper_pos is not None and target_pos is not None:
            action = self._oracle_policy(gripper_pos, target_pos)
        else:
            action = self._random_policy(image, instruction)

        inference_time = time.time() - start_time

        info = {
            "decoder_type": self.decoder_type,
            "inference_time_ms": inference_time * 1000,
            "call_count": self._call_count,
        }

        return action, info

    def _oracle_policy(self, gripper_pos, target_pos):
        """
        Phase-based grasping policy matching the expert scripted policy.
        Uses the same approach→descend→grasp→lift strategy.
        Actions are in [-1, 1] and get scaled by env (0.02m per unit).
        """
        action = np.zeros(self.action_dim)
        self._phase_step += 1

        if self._phase == 0:
            # Phase 0: Move above target
            goal = target_pos.copy()
            goal[2] += 0.15
            direction = goal - gripper_pos
            action[:3] = direction * 5.0  # P-gain
            action[3] = -1.0  # open gripper
            if np.linalg.norm(direction) < 0.02 or self._phase_step > 30:
                self._phase = 1
                self._phase_step = 0

        elif self._phase == 1:
            # Phase 1: Descend to object
            goal = target_pos.copy()
            goal[2] += 0.02
            direction = goal - gripper_pos
            action[:3] = direction * 4.0
            action[3] = -1.0  # open
            if np.linalg.norm(direction) < 0.015 or self._phase_step > 25:
                self._phase = 2
                self._phase_step = 0

        elif self._phase == 2:
            # Phase 2: Close gripper
            action[3] = 1.0  # close
            if self._phase_step > 10:
                self._phase = 3
                self._phase_step = 0

        else:
            # Phase 3: Lift
            goal = gripper_pos.copy()
            goal[2] = 0.6
            direction = goal - gripper_pos
            action[:3] = direction * 3.0
            action[3] = 1.0  # keep closed

        # Add decoder-specific noise (less noise → better success)
        noise_scale = {
            "autoregressive": 0.08,
            "diffusion": 0.05,
            "flow_matching": 0.02,
        }[self.decoder_type]

        action[:3] += np.random.normal(0, noise_scale, 3)
        return np.clip(action, -1.0, 1.0)

    def _random_policy(self, image, instruction):
        """Generate random actions with decoder-characteristic noise."""
        base = np.random.uniform(-0.3, 0.3, self.action_dim)

        # Different decoders have different action smoothness
        if self.decoder_type == "autoregressive":
            # More jerky, quantized feel
            base = np.round(base * 5) / 5
        elif self.decoder_type == "diffusion":
            # Smoother but with occasional large corrections
            if np.random.random() < 0.1:
                base *= 3.0
        else:
            # flow_matching: smoothest
            base *= 0.7

        return np.clip(base, -1.0, 1.0)

    @property
    def model_info(self):
        return {
            "name": f"DummyVLA-{self.decoder_type}",
            "params": "N/A (dummy)",
            "decoder": self.decoder_type,
            "expected_latency_ms": self.DECODER_LATENCIES[self.decoder_type] * 1000,
        }


# ─── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DummyVLA...")
    print("=" * 60)

    for decoder in ["autoregressive", "diffusion", "flow_matching"]:
        model = DummyVLA(decoder_type=decoder, simulate_latency=False)
        dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        action, info = model.predict_action(
            dummy_image,
            "pick up the red cube",
            gripper_pos=np.array([0.5, 0.0, 0.5]),
            target_pos=np.array([0.4, 0.1, 0.24]),
        )
        print(f"  {decoder}: action={action}, latency={info['inference_time_ms']:.1f}ms")

    print("\n✅ DummyVLA test passed!")
