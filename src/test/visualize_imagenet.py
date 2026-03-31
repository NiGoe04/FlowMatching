import argparse

from src.flow_matching.controller.utils import load_imagenet_databatch, resolve_imagenet_data_folder
from src.flow_matching.view.utils import visualize_rgb_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ImageNet batch samples in a given index range.")
    parser.add_argument("--part", type=int, required=True, help="Dataset part index (1..10).")
    parser.add_argument("--dim", type=int, required=True, choices=[8, 16, 32, 64], help="Image resolution.")
    parser.add_argument("--start", type=int, required=True, help="Start sample index (inclusive).")
    parser.add_argument("--end", type=int, required=True, help="End sample index (inclusive).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.part < 1 or args.part > 10:
        raise ValueError("--part must be in [1, 10]")

    dataset_folder = resolve_imagenet_data_folder(args.dim)
    x = load_imagenet_databatch(dataset_folder, args.part, img_size=args.dim)

    visualize_rgb_samples(
        x,
        start_idx=args.start,
        end_idx=args.end,
        title=f"ImageNet-{args.dim} part {args.part} samples [{args.start}, {args.end}]",
    )


if __name__ == "__main__":
    main()
