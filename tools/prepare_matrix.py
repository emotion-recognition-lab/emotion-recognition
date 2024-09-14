from __future__ import annotations

import json
import sys

import toml


def prepare_matrix(config_path: str, commit_message: str) -> dict[str, list[dict[str, str]]]:
    config = toml.load(config_path)
    experiments = config["experiments"]

    matrix = {"include": []}
    for exp_name, exp_config in experiments.items():
        if commit_message.lower().startswith(f"run {exp_name}") or commit_message.lower() == "run all":
            matrix["include"].append(
                {
                    "name": exp_name,
                    "train_command": exp_config["train_command"],
                }
            )

    return matrix


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_matrix.py <config_path> <commit_message>")
        sys.exit(1)

    config_path = sys.argv[1]
    commit_message = sys.argv[2]

    matrix = prepare_matrix(config_path, commit_message)
    print(f"matrix={json.dumps(matrix)}")
