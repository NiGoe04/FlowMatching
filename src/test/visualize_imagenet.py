from src.flow_matching.controller.utils import (
    load_imagenet_databatch,
    resolve_imagenet_data_folder,
)
from src.flow_matching.view.utils import visualize_rgb_samples


# ============================================================
# CONFIG (just edit these)
# ============================================================

PART = 1        # 1..10
DIM = 32      # 8, 16, 32, 64
START = 0       # inclusive
END = 15        # inclusive


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if PART < 1 or PART > 10:
        raise ValueError("PART must be in [1, 10]")

    dataset_folder = resolve_imagenet_data_folder(DIM)
    x = load_imagenet_databatch(dataset_folder, PART, img_size=DIM)

    visualize_rgb_samples(
        x,
        start_idx=START,
        end_idx=END,
        title=f"ImageNet-{DIM} part {PART} samples [{START}, {END}]",
    )


if __name__ == "__main__":
    main()