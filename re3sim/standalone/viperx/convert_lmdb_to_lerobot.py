#!/usr/bin/env python3
"""Convert ViperX Re3Sim LMDB episodes to LeRobotDataset v3."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import cv2
import lmdb
import numpy as np


JOINT_NAMES = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
)
CAMERA_NAMES = ("wrist_camera", "top_camera")
IMAGE_SHAPE = (480, 640, 3)
CONTRACT_VERSION = "viperx_joint_gripper_v1"
DEFAULT_TASK = "pick the black block into the basket"
DEFAULT_REPO_ID = "local/viperx_pick_into_basket"
FPS = 10
ROBOT_TYPE = "viperx"


@dataclass(frozen=True)
class EpisodeSource:
    episode_path: Path
    lmdb_path: Path
    qpos: tuple[np.ndarray, ...]
    actions: tuple[np.ndarray, ...]

    @property
    def frame_count(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class ConversionSummary:
    episode_count: int
    frame_count: int
    output_path: Path


def lerobot_features() -> dict[str, dict]:
    vector_feature = {
        "dtype": "float32",
        "shape": (7,),
        "names": list(JOINT_NAMES),
    }
    camera_feature = {
        "dtype": "video",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.state": vector_feature.copy(),
        "action": vector_feature.copy(),
        "observation.images.wrist_camera": camera_feature.copy(),
        "observation.images.top_camera": camera_feature.copy(),
    }


def discover_episodes(
    input_path: Path, max_episodes: int | None
) -> list[Path]:
    """Return one episode or sorted direct episodes from a collection batch."""
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    resolved = Path(input_path).expanduser().resolve()
    if (resolved / "lmdb").is_dir():
        episodes = [resolved]
    else:
        episodes = sorted(
            path.resolve()
            for path in resolved.glob("log-*")
            if path.is_dir() and (path / "lmdb").is_dir()
        )

    if not episodes:
        raise FileNotFoundError(
            f"No episode directory containing lmdb/ found under: {resolved}"
        )
    return episodes[:max_episodes]


def _load_pickled(transaction: object, key: str, episode_path: Path) -> object:
    payload = transaction.get(key.encode("utf-8"))
    if payload is None:
        raise KeyError(f"{episode_path}: LMDB is missing '{key}'")
    try:
        return pickle.loads(payload)
    except Exception as error:
        raise ValueError(
            f"{episode_path}: LMDB value '{key}' is not valid pickle"
        ) from error


def _validate_metadata(metadata: object, episode_path: Path) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError(f"{episode_path}: metadata must be a dictionary")
    if metadata.get("success") is not True:
        raise ValueError(f"{episode_path}: metadata success must be true")
    if metadata.get("observation_before_action") is not True:
        raise ValueError(
            f"{episode_path}: observation_before_action must be true"
        )
    if set(metadata.get("camera_names", [])) != set(CAMERA_NAMES):
        raise ValueError(
            f"{episode_path}: camera_names must be {list(CAMERA_NAMES)}"
        )
    contract = metadata.get("data_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{episode_path}: data_contract must be a dictionary")
    if contract.get("version") != CONTRACT_VERSION:
        raise ValueError(
            f"{episode_path}: data_contract version must be {CONTRACT_VERSION}"
        )
    if contract.get("order") != list(JOINT_NAMES):
        raise ValueError(
            f"{episode_path}: data_contract order must be {list(JOINT_NAMES)}"
        )
    if contract.get("shape") != [7]:
        raise ValueError(f"{episode_path}: data_contract shape must be [7]")
    return metadata


def _as_vectors(values: object, field: str, episode_path: Path) -> tuple[np.ndarray, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{episode_path}: {field} must be a sequence")
    vectors = []
    for index, value in enumerate(values):
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{episode_path}: {field}[{index}] is not a numeric vector"
            ) from error
        if vector.shape != (7,):
            raise ValueError(
                f"{episode_path}: {field}[{index}] has shape {vector.shape}, expected (7,)"
            )
        if not np.isfinite(vector).all():
            raise ValueError(
                f"{episode_path}: {field}[{index}] contains non-finite values"
            )
        vectors.append(np.ascontiguousarray(vector.copy()))
    return tuple(vectors)


def inspect_episode(episode_path: Path) -> EpisodeSource:
    episode_path = Path(episode_path).expanduser().resolve()
    lmdb_path = episode_path / "lmdb"
    info_path = episode_path / "info.json"
    if not lmdb_path.is_dir():
        raise FileNotFoundError(f"{episode_path}: missing lmdb directory")
    if not info_path.is_file():
        raise FileNotFoundError(f"{episode_path}: missing info.json")

    try:
        info_metadata = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{episode_path}: info.json is invalid") from error
    _validate_metadata(info_metadata, episode_path)

    environment = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False
    )
    try:
        with environment.begin(write=False) as transaction:
            lmdb_metadata = _validate_metadata(
                _load_pickled(transaction, "json_data", episode_path),
                episode_path,
            )
            for field in (
                "success",
                "observation_before_action",
                "camera_names",
                "data_contract",
            ):
                if info_metadata.get(field) != lmdb_metadata.get(field):
                    raise ValueError(
                        f"{episode_path}: info.json and LMDB json_data disagree on '{field}'"
                    )
            qpos = _as_vectors(
                _load_pickled(transaction, "observations/qpos", episode_path),
                "observations/qpos",
                episode_path,
            )
            actions = _as_vectors(
                _load_pickled(transaction, "action", episode_path),
                "action",
                episode_path,
            )
            if not actions:
                raise ValueError(f"{episode_path}: episode has zero frames")
            if len(qpos) != len(actions):
                raise ValueError(
                    f"{episode_path}: observations/qpos has {len(qpos)} frames but "
                    f"action has {len(actions)}"
                )
            for frame_index in range(len(actions)):
                for camera_name in CAMERA_NAMES:
                    key = f"observations/images/{camera_name}/{frame_index}"
                    if transaction.get(key.encode("utf-8")) is None:
                        raise KeyError(f"{episode_path}: LMDB is missing '{key}'")
    finally:
        environment.close()

    return EpisodeSource(episode_path, lmdb_path, qpos, actions)


def _decode_rgb(transaction: object, key: str, episode_path: Path) -> np.ndarray:
    encoded = np.asarray(
        _load_pickled(transaction, key, episode_path), dtype=np.uint8
    )
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{episode_path}: JPEG '{key}' could not be decoded")
    if bgr.shape != IMAGE_SHAPE or bgr.dtype != np.uint8:
        raise ValueError(
            f"{episode_path}: JPEG '{key}' has shape {bgr.shape} and dtype "
            f"{bgr.dtype}, expected uint8 {IMAGE_SHAPE}"
        )
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def iter_lerobot_frames(
    source: EpisodeSource, task: str
) -> Iterator[dict[str, object]]:
    environment = lmdb.open(
        str(source.lmdb_path), readonly=True, lock=False, readahead=False
    )
    try:
        with environment.begin(write=False) as transaction:
            for frame_index, (qpos, action) in enumerate(
                zip(source.qpos, source.actions, strict=True)
            ):
                frame: dict[str, object] = {
                    "observation.state": qpos.copy(),
                    "action": action.copy(),
                    "task": task,
                }
                for camera_name in CAMERA_NAMES:
                    key = f"observations/images/{camera_name}/{frame_index}"
                    frame[f"observation.images.{camera_name}"] = _decode_rgb(
                        transaction, key, source.episode_path
                    )
                yield frame
    finally:
        environment.close()


def _validate_converted_dataset(
    root: Path,
    repo_id: str,
    expected_episode_frames: tuple[int, ...],
    task: str,
) -> None:
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    expected_total_frames = sum(expected_episode_frames)
    if metadata.info.get("codebase_version") != "v3.0":
        raise RuntimeError(
            f"Converted dataset has codebase_version "
            f"{metadata.info.get('codebase_version')!r}, expected 'v3.0'"
        )
    if metadata.fps != FPS:
        raise RuntimeError(
            f"Converted dataset has fps={metadata.fps}, expected {FPS}"
        )
    if metadata.total_episodes != len(expected_episode_frames):
        raise RuntimeError(
            f"Converted dataset has {metadata.total_episodes} episodes, expected "
            f"{len(expected_episode_frames)}"
        )
    if metadata.total_frames != expected_total_frames:
        raise RuntimeError(
            f"Converted dataset has {metadata.total_frames} frames, expected "
            f"{expected_total_frames}"
        )

    expected_features = lerobot_features()
    for key, expected in expected_features.items():
        actual = metadata.features.get(key)
        if actual is None:
            raise RuntimeError(f"Converted dataset is missing feature {key!r}")
        if actual.get("dtype") != expected["dtype"]:
            raise RuntimeError(
                f"Converted feature {key!r} has dtype {actual.get('dtype')!r}, "
                f"expected {expected['dtype']!r}"
            )
        if tuple(actual.get("shape", ())) != tuple(expected["shape"]):
            raise RuntimeError(
                f"Converted feature {key!r} has shape {actual.get('shape')!r}, "
                f"expected {expected['shape']!r}"
            )
        if actual.get("names") != expected["names"]:
            raise RuntimeError(
                f"Converted feature {key!r} has names {actual.get('names')!r}, "
                f"expected {expected['names']!r}"
            )

    for episode_index in range(len(expected_episode_frames)):
        data_path = root / metadata.get_data_file_path(episode_index)
        if not data_path.is_file():
            raise RuntimeError(f"Converted dataset is missing data file: {data_path}")
        for video_key in metadata.video_keys:
            video_path = root / metadata.get_video_file_path(
                episode_index, video_key
            )
            if not video_path.is_file():
                raise RuntimeError(
                    f"Converted dataset is missing video file: {video_path}"
                )

    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if len(dataset) != expected_total_frames:
        raise RuntimeError(
            f"LeRobotDataset loads {len(dataset)} frames, expected "
            f"{expected_total_frames}"
        )
    for frame_index in {0, expected_total_frames - 1}:
        frame = dataset[frame_index]
        if tuple(frame["observation.state"].shape) != (7,):
            raise RuntimeError("Loaded observation.state does not have shape (7,)")
        if tuple(frame["action"].shape) != (7,):
            raise RuntimeError("Loaded action does not have shape (7,)")
        for camera_name in CAMERA_NAMES:
            image = frame[f"observation.images.{camera_name}"]
            if tuple(image.shape) != (3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]):
                raise RuntimeError(
                    f"Loaded {camera_name} has shape {tuple(image.shape)}, expected "
                    f"(3, {IMAGE_SHAPE[0]}, {IMAGE_SHAPE[1]})"
                )
        if frame["task"] != task:
            raise RuntimeError(
                f"Loaded task is {frame['task']!r}, expected {task!r}"
            )


def convert_dataset(
    input_path: Path,
    output_path: Path,
    repo_id: str = DEFAULT_REPO_ID,
    max_episodes: int | None = None,
    task: str = DEFAULT_TASK,
) -> ConversionSummary:
    """Convert validated LMDB episodes into one local LeRobotDataset v3."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_path = Path(output_path).expanduser().resolve()
    incomplete_path = output_path.with_name(output_path.name + ".incomplete")
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")
    if incomplete_path.exists():
        raise FileExistsError(
            f"Incomplete output already exists: {incomplete_path}. "
            "Inspect or remove it before retrying."
        )
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must use the form 'namespace/dataset_name'")
    if not task.strip():
        raise ValueError("task must not be empty")

    episode_paths = discover_episodes(input_path, max_episodes)
    sources = tuple(inspect_episode(path) for path in episode_paths)
    episode_frames = tuple(source.frame_count for source in sources)

    dataset = None
    finalized = False
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=incomplete_path,
            fps=FPS,
            robot_type=ROBOT_TYPE,
            features=lerobot_features(),
            use_videos=True,
            batch_encoding_size=1,
        )
        for source in sources:
            for frame in iter_lerobot_frames(source, task):
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
        finalized = True

        _validate_converted_dataset(
            incomplete_path,
            repo_id,
            episode_frames,
            task,
        )
        incomplete_path.rename(output_path)
    except Exception:
        if dataset is not None and not finalized:
            with suppress(Exception):
                dataset.finalize()
        raise

    return ConversionSummary(
        episode_count=len(sources),
        frame_count=sum(episode_frames),
        output_path=output_path,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ViperX LMDB episodes to LeRobotDataset v3."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One episode directory or a batch directory containing log-* episodes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New local LeRobot dataset directory.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Local dataset identifier (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--max-episodes",
        type=_positive_int,
        default=None,
        help="Convert only the first N sorted episodes (default: all).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = convert_dataset(
        input_path=args.input,
        output_path=args.output,
        repo_id=args.repo_id,
        max_episodes=args.max_episodes,
        task=DEFAULT_TASK,
    )
    print(f"lerobot_conversion_episodes={summary.episode_count}")
    print(f"lerobot_conversion_frames={summary.frame_count}")
    print(f"lerobot_conversion_output={summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
