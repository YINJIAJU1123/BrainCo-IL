import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_policy_contract import package_checkpoint


class ExportPolicyContractTest(unittest.TestCase):
    def test_package_from_serialized_policy_io(self):
        template = (Path(__file__).resolve().parents[2] /
                    "src/openpi/training/training_config_template/ego/posttrain_e1_r1.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "123"
            (root / "params").mkdir(parents=True)
            (root / "train_config.yaml").write_bytes(template.read_bytes())
            result = package_checkpoint(root, robot_interface="revotron_mit_revo3_mit_3cam",
                                        contract_only=False)
            contract = json.loads((root / "policy_contract.json").read_text())
            self.assertEqual(contract["chunk_steps"], 16)
            self.assertEqual(contract["required_cameras"], [
                "observation.images.cam_head", "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist"])
            self.assertEqual(contract["policy_rate_hz"], 30.0)
            self.assertEqual(result["contract_hash"], contract["contract_hash"])
            self.assertEqual(sum(map(len, contract["output_joint_groups"].values())), 56)


if __name__ == "__main__":
    unittest.main()
