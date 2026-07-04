import argparse
from pathlib import Path

from surgmark.data.jsonl_dataset import build_label_space, iter_records, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--output", default="configs/label_space.json")
    args = parser.parse_args()

    paths = [Path(p) for p in args.jsonl]
    label_space = build_label_space(iter_records(paths))
    write_json(args.output, label_space)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
