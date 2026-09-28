"""Run a single-input ExecuTorch .pte program on an image or random tensor."""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import torch
from executorch.runtime import Runtime, Verification


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", required=True, type=Path, help="Path to .pte")
    parser.add_argument(
        "-is", "--input-shape", required=True, type=int, nargs=4,
        metavar=("N", "C", "H", "W"), help="Must exactly match the export shape.",
    )
    parser.add_argument("-i", "--input", type=Path, help="Optional image path; otherwise random input is used.")
    parser.add_argument("--matcher", choices=("lightglue", "superglue"),
                        help="Generate fixed-K random feature tensors for a matcher .pte.")
    parser.add_argument("--num-keypoints", type=int, default=256,
                        help="Static K used when --matcher is set.")
    parser.add_argument("--matcher-features", choices=("superpoint", "disk", "aliked", "sift", "doghardnet"),
                        default="superpoint", help="LightGlue descriptor family.")
    return parser.parse_args()


def make_input(image_path: Path | None, shape: list[int]) -> torch.Tensor:
    batch, channels, height, width = shape
    if batch != 1:
        raise ValueError("The image runner currently supports batch size 1 only.")
    if image_path is None:
        return torch.randn(shape, dtype=torch.float32)

    flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if channels == 1:
        tensor = torch.from_numpy(image).unsqueeze(0)
    elif channels == 3:
        tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    else:
        raise ValueError(f"Only 1- or 3-channel image inputs are supported, got C={channels}")
    return tensor.unsqueeze(0).to(torch.float32) / 255.0


def make_matcher_inputs(kind: str, keypoints: int, features: str) -> tuple[torch.Tensor, ...]:
    if keypoints <= 0:
        raise ValueError("--num-keypoints must be positive")
    if kind == "lightglue":
        descriptor_dim = 256 if features == "superpoint" else 128
        return (
            torch.rand(1, keypoints, 2), torch.rand(1, keypoints, 2),
            torch.rand(1, keypoints, descriptor_dim),
            torch.rand(1, keypoints, descriptor_dim),
        )
    return (
        torch.rand(1, keypoints, 2), torch.rand(1, keypoints, 2),
        torch.rand(1, 256, keypoints), torch.rand(1, 256, keypoints),
        torch.rand(1, keypoints), torch.rand(1, keypoints),
    )


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"No .pte file at {args.model}")
    inputs = (make_input(args.input, args.input_shape),) if args.matcher is None else make_matcher_inputs(
        args.matcher, args.num_keypoints, args.matcher_features
    )
    method = Runtime.get().load_program(
        str(args.model), verification=Verification.Minimal
    ).load_method("forward")
    started = perf_counter()
    outputs = method.execute(inputs)
    elapsed_ms = (perf_counter() - started) * 1000
    print(f"Executed in {elapsed_ms:.1f} ms")
    for index, output in enumerate(outputs):
        if isinstance(output, torch.Tensor):
            print(
                f"output{index}: shape={tuple(output.shape)}, dtype={output.dtype}, "
                f"min={output.min().item():.5g}, max={output.max().item():.5g}"
            )
        else:
            print(f"output{index}: {type(output).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
