from pathlib import Path
import sys

import numpy as np

CLAWGUI_RL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CLAWGUI_RL))

from agent_system.litegui.reward import DifficultyTracker, LiteGUIRewardShaper


def step(active=True):
    return {"active_masks": active}


def info(*, won=False, change=0.02, valid=True, action="click(1,1)"):
    return {
        "task_name": "SetAlarmTask",
        "won": won,
        "litegui_state_change": change,
        "is_action_valid": valid,
        "litegui_action_signature": action,
    }


def test_success_outweighs_failed_screen_changes():
    shaper = LiteGUIRewardShaper(
        {
            "enable": True,
            "state_change_weight": 0.2,
            "efficiency_weight": 0.2,
            "state_change_threshold": 0.015,
        }
    )
    rewards, details = shaper.shape_batch(
        episode_rewards=np.array([1.0, 0.0], dtype=np.float32),
        episode_lengths=np.array([3.0, 3.0], dtype=np.float32),
        total_infos=[
            [info(), info(), info(won=True)],
            [info(action="click(2,2)"), info(action="click(3,3)"), info(action="click(4,4)")],
        ],
        total_batch_list=[[step(), step(), step()], [step(), step(), step()]],
    )
    assert rewards[0] >= 1.0
    assert rewards[0] > rewards[1]
    assert details[0]["outcome"] == 1.0


def test_invalid_and_repeated_actions_are_penalized():
    shaper = LiteGUIRewardShaper({"enable": True})
    clean, _ = shaper.shape_batch(
        episode_rewards=np.array([0.0], dtype=np.float32),
        episode_lengths=np.array([2.0], dtype=np.float32),
        total_infos=[[info(action="a"), info(action="b")]],
        total_batch_list=[[step(), step()]],
    )
    penalized, _ = shaper.shape_batch(
        episode_rewards=np.array([0.0], dtype=np.float32),
        episode_lengths=np.array([2.0], dtype=np.float32),
        total_infos=[[info(valid=False, action="a"), info(valid=False, action="a")]],
        total_batch_list=[[step(), step()]],
    )
    assert penalized[0] < clean[0]


def test_difficulty_tracker_persists(tmp_path):
    stats = tmp_path / "difficulty.json"
    tracker = DifficultyTracker(stats_path=str(stats), min_samples=2)
    tracker.update_grouped({"task": [0.0, 0.0]})
    assert tracker.weight("task") == tracker.max_weight
    restored = DifficultyTracker(stats_path=str(stats), min_samples=2)
    assert restored.counts["task"] == 2
    assert restored.weight("task") == restored.max_weight

