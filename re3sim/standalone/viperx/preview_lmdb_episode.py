#!/usr/bin/env python3
"""Create sampled wrist/top previews from one ViperX LMDB episode."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


def sample_frame_indices(frame_count: int, stride: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    indices = list(range(0, frame_count, stride))
    last_index = frame_count - 1
    if indices[-1] != last_index:
        indices.append(last_index)
    return indices


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image.copy()
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def compose_preview_frame(
    wrist: np.ndarray,
    top: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    wrist = np.asarray(wrist)
    top = np.asarray(top)
    for name, image in (("wrist", wrist), ("top", top)):
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0:
            raise ValueError(f"{name} image must have shape (H, W, 3)")

    image_height = max(wrist.shape[0], top.shape[0])
    wrist_panel = _resize_to_height(wrist, image_height)
    top_panel = _resize_to_height(top, image_height)
    panels = np.concatenate([wrist_panel, top_panel], axis=1)

    header_height = 36
    preview = np.zeros(
        (header_height + image_height, panels.shape[1], 3), dtype=np.uint8
    )
    preview[header_height:] = panels
    cv2.putText(
        preview,
        f"frame {frame_index:06d} | wrist_camera",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "top_camera",
        (wrist_panel.shape[1] + 8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def _resolve_lmdb_path(input_path: Path) -> tuple[Path, Path]:
    input_path = Path(input_path).expanduser().resolve()
    if (input_path / "lmdb").is_dir():
        return input_path, input_path / "lmdb"
    if input_path.is_dir() and input_path.name == "lmdb":
        return input_path.parent, input_path
    raise FileNotFoundError(
        f"Expected an episode directory containing lmdb/ or an lmdb directory: "
        f"{input_path}"
    )


def _decode_jpeg(transaction: object, key: str) -> np.ndarray:
    payload = transaction.get(key.encode("utf-8"))
    if payload is None:
        raise KeyError(f"LMDB is missing '{key}'")
    encoded = np.asarray(pickle.loads(payload), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"LMDB image could not be decoded: '{key}'")
    return image


def generate_preview(
    input_path: Path,
    stride: int = 10,
    output_dir: Path | None = None,
) -> list[Path]:
    import lmdb

    episode_path, lmdb_path = _resolve_lmdb_path(input_path)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else episode_path / f"preview_stride{stride}"
    )
    previews: list[tuple[int, np.ndarray]] = []
    environment = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False
    )
    try:
        with environment.begin(write=False) as transaction:
            action_payload = transaction.get(b"action")
            if action_payload is None:
                raise KeyError(f"LMDB has no 'action' key: {lmdb_path}")
            frame_count = len(pickle.loads(action_payload))
            frame_indices = sample_frame_indices(frame_count, stride)

            for frame_index in frame_indices:
                wrist_key = (
                    f"observations/images/wrist_camera/{frame_index}"
                )
                top_key = f"observations/images/top_camera/{frame_index}"
                wrist = _decode_jpeg(transaction, wrist_key)
                top = _decode_jpeg(transaction, top_key)
                preview = compose_preview_frame(wrist, top, frame_index)
                previews.append((frame_index, preview))
    finally:
        environment.close()

    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for frame_index, preview in previews:
        output_path = destination / f"frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), preview):
            raise OSError(f"Failed to write preview image: {output_path}")
        outputs.append(output_path)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one ViperX LMDB episode and write side-by-side "
            "wrist/top GS-composited previews."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Episode directory containing lmdb/ or the lmdb directory itself.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sample every N trajectory frames (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <episode>/preview_strideN).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outputs = generate_preview(args.path, args.stride, args.output)
    print(f"preview_frames={len(outputs)}")
    print(f"preview_output={outputs[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
