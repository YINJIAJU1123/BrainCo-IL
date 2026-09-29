"""Write disposable 50-step and one-step-resume recipes for the 100 h run."""

import dataclasses
from pathlib import Path

from openpi.training import config_io


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / 'src/openpi/training/training_config_template/ego/egodex_pi05_100h_offset0_20k.yaml'


def main() -> None:
    cfg = config_io.load_train_config(TEMPLATE)
    if cfg.resume or cfg.num_train_steps != 20000 or cfg.batch_size != 16:
        raise ValueError('Expected fresh 100 h / batch16 / 20k template')
    profile = dataclasses.replace(
        cfg, exp_name='offset0_bs16_profile50_v1', num_train_steps=50,
        save_interval=49, keep_period=49, log_interval=1, wandb_enabled=False,
    )
    resumed = dataclasses.replace(profile, num_train_steps=51, resume=True,
                                  save_interval=50, keep_period=50)
    for name, recipe in (('profile50', profile), ('profile_resume', resumed)):
        path = config_io.save_train_config(recipe, ROOT / 'runs/ego/100h_20k' / name)
        print(path)


if __name__ == '__main__':
    main()
