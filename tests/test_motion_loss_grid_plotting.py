from __future__ import annotations

import json

from training.jobs.plot_motion_loss_grid_search_results import (
    collect_motion_loss_grid_trials,
    varying_loss_parameters,
    write_ranked_trial_outputs,
)


def test_collect_motion_loss_grid_trials_from_trial_dirs(tmp_path):
    grid_dir = tmp_path / "motion_loss_grid_run"
    first_trial = grid_dir / "trials" / "20260918_motion_transformer_trial_001_direction_weight_0_0"
    second_trial = grid_dir / "trials" / "20260918_motion_transformer_trial_002_direction_weight_0_1"
    write_trial(first_trial, "trial_001", direction_weight=0.0, macro_f1=0.51)
    write_trial(second_trial, "trial_002", direction_weight=0.1, macro_f1=0.62)

    trials = collect_motion_loss_grid_trials(grid_dir)

    assert [trial.trial_id for trial in trials] == ["trial_001", "trial_002"]
    assert trials[1].loss_config["direction_weight"] == 0.1
    assert trials[1].metrics["macro_f1"] == 0.62
    assert varying_loss_parameters(trials) == ["direction_weight"]


def test_write_ranked_trial_outputs(tmp_path):
    grid_dir = tmp_path / "motion_loss_grid_run"
    trial_dir = grid_dir / "trials" / "20260918_motion_transformer_trial_001_direction_weight_0_0"
    write_trial(trial_dir, "trial_001", direction_weight=0.0, macro_f1=0.51)
    trials = collect_motion_loss_grid_trials(grid_dir)
    csv_path = tmp_path / "ranked.csv"
    json_path = tmp_path / "ranked.json"

    write_ranked_trial_outputs(
        csv_path,
        json_path,
        trials=trials,
        parameter_names=["direction_weight"],
        metric_names=["macro_recall", "macro_precision", "macro_f1", "exact_match"],
    )

    csv_text = csv_path.read_text(encoding="utf-8")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "loss.direction_weight" in csv_text
    assert "macro_f1" in csv_text
    assert json_payload[0]["trial_id"] == "trial_001"


def write_trial(trial_dir, trial_id: str, *, direction_weight: float, macro_f1: float):
    metrics_dir = trial_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "macro_recall": 0.8,
                "macro_precision": 0.7,
                "macro_f1": macro_f1,
                "exact_match": 0.4,
                "per_class": {"IGNORED": {"f1": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "summary.json").write_text(json.dumps({"best_epoch": 3, "best_metric": macro_f1}), encoding="utf-8")
    (trial_dir / "config.resolved.yaml").write_text(
        f"""
grid_search:
  trial_id: {trial_id}
  loss_config:
    primary_loss: bce
    direction_weight: {direction_weight}
    phase_weight: 0.5
""".lstrip(),
        encoding="utf-8",
    )
