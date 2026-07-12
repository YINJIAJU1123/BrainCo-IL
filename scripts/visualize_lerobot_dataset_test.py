from __future__ import annotations

import numpy as np

from scripts.visualize_lerobot_dataset import EpisodeAudit
from scripts.visualize_lerobot_dataset import QualityConfig
from scripts.visualize_lerobot_dataset import _apply_home_check
from scripts.visualize_lerobot_dataset import _check_boundary_idle
from scripts.visualize_lerobot_dataset import _check_kinematics
from scripts.visualize_lerobot_dataset import _check_timestamps
from scripts.visualize_lerobot_dataset import _parse_dimension_selection


def test_boundary_idle_detects_both_ends_and_suggests_trim() -> None:
    fps = 10.0
    timestamps = np.arange(12) / fps
    values = np.zeros((12, 2))
    values[4:8, 0] = np.arange(4)
    values[8:, 0] = 3
    config = QualityConfig(max_start_idle_s=0.2, max_end_idle_s=0.2, trim_padding_frames=1)

    issues, metrics, trim = _check_boundary_idle(values, timestamps, fps, 7, "observation.state", config)

    assert {issue.code for issue in issues} == {"start_idle", "end_idle"}
    assert np.isclose(metrics["start_idle_s"], 0.4)
    assert np.isclose(metrics["end_idle_s"], 0.4)
    assert trim == {"start_frame_inclusive": 3, "end_frame_inclusive": 8}


def test_timestamp_gap_is_a_dropped_sample() -> None:
    timestamps = np.array([0.0, 0.1, 0.2, 0.4, 0.5])

    issues, metrics = _check_timestamps(timestamps, 10.0, 0, QualityConfig(timestamp_tolerance_s=0.01))

    assert "dropped_sample" in {issue.code for issue in issues}
    assert metrics["dropped_intervals"] == 1


def test_kinematics_detects_angle_velocity_and_acceleration_jump() -> None:
    timestamps = np.arange(5) / 10.0
    values = np.array([[0.0], [0.0], [2.0], [2.0], [2.0]])
    config = QualityConfig(max_joint_step=0.5, max_joint_velocity=5.0, max_joint_acceleration=20.0)

    issues, metrics, _ = _check_kinematics(values, timestamps, 10.0, 0, "state", config)

    assert {issue.code for issue in issues} == {"joint_step_jump", "velocity_limit", "velocity_jump"}
    assert metrics["max_step"] == 2.0
    assert metrics["max_velocity"] == 20.0


def test_dimension_selection_preserves_requested_revo3_order() -> None:
    dimensions = _parse_dimension_selection("14:21,28:49,21:28,49:70", 70)

    assert dimensions == list(range(14, 21)) + list(range(28, 49)) + list(range(21, 28)) + list(range(49, 70))
    assert len(dimensions) == 56


def _audit(episode_index: int, initial: list[float]) -> EpisodeAudit:
    state = np.array([initial, initial], dtype=np.float64)
    return EpisodeAudit(
        episode_index=episode_index,
        parquet_path=None,  # type: ignore[arg-type]
        timestamps=np.array([0.0, 0.1]),
        state=state,
        action=None,
        state_key="observation.state",
        action_key=None,
        vector_dimensions=[0, 1],
        frame_indices=np.array([0, 1]),
        dataset_indices=np.array([0, 1]),
        table=None,
        issues=[],
        metrics={},
        suggested_trim=None,
    )


def test_home_check_flags_outlier_episode() -> None:
    audits = [_audit(0, [0.0, 0.0]), _audit(1, [0.01, 0.0]), _audit(2, [1.0, 0.0])]

    reference, source = _apply_home_check(audits, None, QualityConfig(home_tolerance=0.1))

    np.testing.assert_allclose(reference, [0.01, 0.0])
    assert source == "coordinate_median_of_selected_episodes"
    assert not audits[0].issues
    assert not audits[1].issues
    assert [issue.code for issue in audits[2].issues] == ["home_position_mismatch"]
