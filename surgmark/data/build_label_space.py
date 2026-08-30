import json
import argparse
from pathlib import Path

from surgmark.data.jsonl_dataset import build_label_space, iter_records, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--output", default="configs/label_space.json")
    parser.add_argument("--procedural-priors", default="", help="Optional expert transition-weight JSON.")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--duration-bin-size", type=float, default=30.0)
    args = parser.parse_args()

    paths = [Path(p) for p in args.jsonl]
    procedural_priors = None
    if args.procedural_priors:
        procedural_priors = json.loads(Path(args.procedural_priors).read_text(encoding="utf-8"))
    label_space = build_label_space(
        iter_records(paths),
        procedural_priors=procedural_priors,
        epsilon=args.epsilon,
        duration_bin_size=args.duration_bin_size,
    )
    write_json(args.output, label_space)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
