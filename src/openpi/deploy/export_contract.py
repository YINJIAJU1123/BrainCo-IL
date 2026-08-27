"""Generate the immutable deployment contract for an existing checkpoint."""

from __future__ import annotations

import argparse

from openpi.deploy.contract import save_policy_contract
from openpi.training import config_io


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    args = parser.parse_args()
    config = config_io.load_train_config(args.checkpoint)
    path = save_policy_contract(config, args.checkpoint, dataset_roots=args.dataset)
    print(path)


if __name__ == "__main__":
    main()
