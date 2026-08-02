"""Difficulty-adaptive trajectory shaping for ClawGUI's standard GRPO loop."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _cfg(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class TaskDifficulty:
    tier: str = "hard"
    reference_steps: int = 12


class DifficultyTracker:
    """Tracks task success EMA and persists it across training restarts."""

    def __init__(
        self,
        *,
        stats_path: str | None,
        target_success: float = 0.5,
        momentum: float = 0.9,
        min_samples: int = 4,
        min_weight: float = 0.75,
        max_weight: float = 1.5,
    ) -> None:
        self.stats_path = os.path.expanduser(stats_path) if stats_path else None
        self.target_success = target_success
        self.momentum = momentum
        self.min_samples = min_samples
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.success_ema: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.stats_path or not os.path.exists(self.stats_path):
            return
        try:
            with open(self.stats_path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.success_ema = {str(k): float(v) for k, v in data.get("success_ema", {}).items()}
            self.counts = {str(k): int(v) for k, v in data.get("counts", {}).items()}
        except (OSError, ValueError, TypeError) as exc:
            print(f"[LiteGUI] ignoring invalid difficulty stats {self.stats_path}: {exc}")

    def save(self) -> None:
        if not self.stats_path:
            return
        path = Path(self.stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {"success_ema": self.success_ema, "counts": self.counts},
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        temporary.replace(path)

    def weight(self, task_name: str) -> float:
        if self.counts.get(task_name, 0) < self.min_samples:
            return 1.0
        rate = max(0.05, self.success_ema.get(task_name, self.target_success))
        weight = math.sqrt(self.target_success / rate)
        return max(self.min_weight, min(self.max_weight, weight))

    def update_grouped(self, outcomes: dict[str, list[float]]) -> None:
        """Update once per task group to avoid order dependence within GRPO groups."""
        for task_name, values in outcomes.items():
            group_rate = float(np.mean(values)) if values else 0.0
            previous = self.success_ema.get(task_name, group_rate)
            self.success_ema[task_name] = (
                self.momentum * previous + (1.0 - self.momentum) * group_rate
            )
            self.counts[task_name] = self.counts.get(task_name, 0) + len(values)
        self.save()


class LiteGUIRewardShaper:
    """Shapes ClawGUI episode returns without replacing verl's GRPO trainer.

    Final outcome remains dominant. Difficulty only scales dense shaping terms,
    because multiplying the complete return by a task-constant would be cancelled
    by GRPO's same-prompt normalization.
    """

    def __init__(self, config: Any = None) -> None:
        self.enabled = bool(_cfg(config, "enable", False))
        self.outcome_weight = float(_cfg(config, "outcome_weight", 1.0))
        self.state_change_weight = float(_cfg(config, "state_change_weight", 0.20))
        self.efficiency_weight = float(_cfg(config, "efficiency_weight", 0.20))
        self.invalid_action_penalty = float(_cfg(config, "invalid_action_penalty", 0.05))
        self.repeat_action_penalty = float(_cfg(config, "repeat_action_penalty", 0.03))
        self.state_change_threshold = float(_cfg(config, "state_change_threshold", 0.015))
        self.max_steps = int(_cfg(config, "max_steps", 15))
        self.difficulty: dict[str, TaskDifficulty] = {}
        difficulty_file = _cfg(config, "difficulty_file", None)
        if difficulty_file:
            self._load_difficulty(os.path.expanduser(str(difficulty_file)))
        self.tracker = DifficultyTracker(
            stats_path=_cfg(config, "stats_path", None),
            target_success=float(_cfg(config, "target_success", 0.5)),
            momentum=float(_cfg(config, "ema_momentum", 0.9)),
            min_samples=int(_cfg(config, "min_samples", 4)),
            min_weight=float(_cfg(config, "min_difficulty_weight", 0.75)),
            max_weight=float(_cfg(config, "max_difficulty_weight", 1.5)),
        )

    def _load_difficulty(self, path: str) -> None:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        self.difficulty = {
            name: TaskDifficulty(
                tier=str(value.get("tier", "hard")),
                reference_steps=max(1, int(value.get("reference_steps", 12))),
            )
            for name, value in raw.items()
        }

    @staticmethod
    def _tier_prior(tier: str) -> float:
        """Cold-start prior before enough online success statistics exist."""
        return {"easy": 0.85, "medium": 1.0, "hard": 1.15}.get(tier.lower(), 1.0)

    @staticmethod
    def _last_info(infos: list[dict[str, Any]]) -> dict[str, Any]:
        for info in reversed(infos):
            if info:
                return info
        return {}

    @staticmethod
    def _active_infos(
        infos: list[dict[str, Any]], steps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [info for info, step in zip(infos, steps) if bool(step.get("active_masks", False))]

    def _task_name(self, infos: list[dict[str, Any]], index: int) -> str:
        value = self._last_info(infos).get("task_name")
        return str(value or f"unknown_task_{index}")

    def shape_batch(
        self,
        *,
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        total_infos: list[list[dict[str, Any]]],
        total_batch_list: list[list[dict[str, Any]]],
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if not self.enabled:
            return episode_rewards, [{} for _ in range(len(episode_rewards))]

        shaped = np.zeros_like(episode_rewards, dtype=np.float32)
        details: list[dict[str, float]] = []
        grouped_outcomes: dict[str, list[float]] = defaultdict(list)

        for index, (base_reward, length, infos, steps) in enumerate(
            zip(episode_rewards, episode_lengths, total_infos, total_batch_list)
        ):
            active_infos = self._active_infos(infos, steps)
            task_name = self._task_name(infos, index)
            last_info = self._last_info(active_infos)
            if "won" in last_info:
                outcome = float(bool(last_info["won"]))
            elif "eval_score" in last_info:
                outcome = float(float(last_info["eval_score"]) >= 1.0)
            else:
                outcome = float(float(base_reward) >= 1.0)
            grouped_outcomes[task_name].append(outcome)

            valid_changes = [
                float(info.get("litegui_state_change", 0.0))
                for info in active_infos
                if bool(info.get("is_action_valid", True))
            ]
            if valid_changes:
                state_progress = float(
                    np.mean(
                        [min(1.0, score / max(self.state_change_threshold, 1e-6)) for score in valid_changes]
                    )
                )
            else:
                state_progress = 0.0

            invalid_count = sum(not bool(info.get("is_action_valid", True)) for info in active_infos)
            signatures = [
                str(info.get("litegui_action_signature", ""))
                for info in active_infos
                if info.get("litegui_action_signature")
            ]
            repeat_count = len(signatures) - len(set(signatures))

            spec = self.difficulty.get(task_name, TaskDifficulty())
            reference_steps = min(spec.reference_steps, self.max_steps)
            efficiency = outcome * min(1.0, reference_steps / max(1.0, float(length)))
            difficulty_weight = self._tier_prior(spec.tier) * self.tracker.weight(task_name)
            difficulty_weight = max(
                self.tracker.min_weight,
                min(self.tracker.max_weight, difficulty_weight),
            )

            state_term = difficulty_weight * self.state_change_weight * state_progress
            efficiency_term = difficulty_weight * self.efficiency_weight * efficiency
            invalid_term = -self.invalid_action_penalty * invalid_count
            repeat_term = -self.repeat_action_penalty * repeat_count
            score = (
                self.outcome_weight * outcome
                + state_term
                + efficiency_term
                + invalid_term
                + repeat_term
            )
            shaped[index] = score
            components = {
                "outcome": outcome,
                "state_progress": state_progress,
                "efficiency": efficiency,
                "difficulty_weight": difficulty_weight,
                "invalid_count": float(invalid_count),
                "repeat_count": float(repeat_count),
                "litegui_reward": float(score),
            }
            details.append(components)
            if active_infos:
                active_infos[-1]["litegui_reward_components"] = components

        self.tracker.update_grouped(grouped_outcomes)
        print(
            "[LiteGUI] shaped rewards: "
            f"mean={float(shaped.mean()):.4f}, "
            f"success={float(np.mean([item['outcome'] for item in details])):.4f}"
        )
        return shaped, details
