"""Admission checks for experiment-scoped, benchmarked input profiles."""

import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from openpi.training import config_io, video_loading


class VerifiedProfileTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        temporary = Path(self.temporary.name)
        root = Path(__file__).resolve().parents[3]
        self.config = config_io.load_train_config(
            root / "src/openpi/training/training_config_template/ego/posttrain_e5_r2.yaml")
        self.config = dataclasses.replace(self.config, checkpoint_base_dir=str(temporary / "checkpoints"))
        options = dataclasses.asdict(video_loading.VideoLoadingOptions(workers=4))
        self.evidence = temporary / "evidence.json"
        self.evidence.write_text(json.dumps(dict(passed=True, selected_options=options)))
        source = "src/openpi/training/video_loading.py"
        self.profile = dict(
            experiment=self.config.exp_name, original_workers=self.config.num_workers,
            config_sha256=video_loading.config_sha256(self.config), options=options,
            evidence_path=str(self.evidence),
            evidence_sha256=hashlib.sha256(self.evidence.read_bytes()).hexdigest(),
            source_sha256={source: hashlib.sha256((root / source).read_bytes()).hexdigest()})
        self.manifest = (Path(self.config.checkpoint_dir).parent / ".input_pipeline"
                         / f"{self.config.exp_name}.json")
        self.manifest.parent.mkdir(parents=True)
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.profile))

    def test_only_admitted_experiment_changes_workers(self):
        prepared, actual = video_loading.prepare_verified_profile(self.config)
        self.assertEqual(actual, self.profile)
        self.assertEqual(prepared.num_workers, 4)
        self.assertEqual(dataclasses.replace(prepared, num_workers=self.config.num_workers), self.config)
        other = dataclasses.replace(self.config, exp_name="posttrain_e3_r1")
        untouched, absent = video_loading.prepare_verified_profile(other)
        self.assertIs(untouched, other)
        self.assertIsNone(absent)

    def test_changed_config_rejects_profile(self):
        changed = dataclasses.replace(self.config, seed=self.config.seed + 1)
        with self.assertRaises(ValueError):
            video_loading.prepare_verified_profile(changed)

    def test_changed_evidence_rejects_profile(self):
        self.evidence.write_text(self.evidence.read_text() + " ")
        with self.assertRaises(ValueError):
            video_loading.prepare_verified_profile(self.config)

    def test_changed_source_rejects_profile(self):
        self.profile["source_sha256"]["src/openpi/training/video_loading.py"] = "0" * 64
        self.save_manifest()
        with self.assertRaises(ValueError):
            video_loading.prepare_verified_profile(self.config)

    def test_wrong_experiment_rejects_profile(self):
        self.profile["experiment"] = "posttrain_e3_r1"
        self.save_manifest()
        with self.assertRaises(ValueError):
            video_loading.prepare_verified_profile(self.config)

    def test_saved_profile_supports_resume(self):
        prepared, _ = video_loading.prepare_verified_profile(self.config)
        Path(prepared.checkpoint_dir).mkdir(parents=True)
        video_loading.save_profile(self.profile, prepared.checkpoint_dir)
        self.manifest.unlink()
        resumed, actual = video_loading.prepare_verified_profile(dataclasses.replace(prepared, resume=True))
        self.assertEqual(resumed.num_workers, 4)
        self.assertTrue(resumed.resume)
        self.assertEqual(actual, self.profile)


if __name__ == "__main__":
    unittest.main()
