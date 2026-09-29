import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ego.validate_review_manifest import validate


class ValidateReviewManifestTest(unittest.TestCase):
    def test_complete_partition_and_pending_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            meta = source / "meta"
            meta.mkdir(parents=True)
            info = b'{"fps":30}'
            (meta / "info.json").write_bytes(info)
            (meta / "episodes.jsonl").write_text('{"episode_index":0,"length":67}\n')
            path = Path(temporary) / "review.json"
            manifest = {"schema_version": 1, "source_info_sha256": hashlib.sha256(info).hexdigest(),
                        "intervals": [
                            {"episode_index": 0, "start_frame": 0, "end_frame_exclusive": 20,
                             "decision": "exclude", "reason": "human reset"},
                            {"episode_index": 0, "start_frame": 20, "end_frame_exclusive": 67,
                             "decision": "keep", "reason": ""}]}
            path.write_text(json.dumps(manifest))
            result = validate(source, path)
            self.assertEqual(result["reviewed_real_starts"], 47)
            self.assertEqual(result["rounded_real_starts"], 64)
            self.assertEqual(result["real_padding_repeats"], 17)
            self.assertEqual(result["steps"]["posttrain_e5_r2"], 14)
            manifest["intervals"][1]["decision"] = "pending"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "human keep/exclude"):
                validate(source, path)


if __name__ == "__main__":
    unittest.main()
