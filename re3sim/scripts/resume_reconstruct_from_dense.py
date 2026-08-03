#!/usr/bin/env python3

"""Resume the Re3Sim OpenMVS pipeline from an existing dense point cloud."""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def build_reconstruct_mesh_command(
    min_point_distance: float,
    max_threads: int,
    target_face_num: int,
) -> list[str]:
    return [
        "ReconstructMesh",
        "scene_dense.mvs",
        "-p",
        "scene_dense.ply",
        "-o",
        "scene_dense_mesh.mvs",
        "--min-point-distance",
        str(min_point_distance),
        "--max-threads",
        str(max_threads),
        "--target-face-num",
        str(target_face_num),
    ]


def build_refine_mesh_command(max_threads: int) -> list[str]:
    return [
        "RefineMesh",
        "scene_dense.mvs",
        "-m",
        "scene_dense_mesh.ply",
        "-o",
        "scene_dense_mesh_refine.mvs",
        "--scales",
        "1",
        "--resolution-level",
        "2",
        "--max-face-area",
        "16",
        "--max-threads",
        str(max_threads),
    ]


def build_texture_mesh_command(max_threads: int) -> list[str]:
    return [
        "TextureMesh",
        "scene_dense.mvs",
        "-m",
        "scene_dense_mesh_refine.ply",
        "-o",
        "scene_dense_mesh_refine_texture.mvs",
        "--max-threads",
        str(max_threads),
    ]


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required non-empty file is missing: {path}")


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory is missing: {path}")


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise FileNotFoundError(f"Required command is not available: {command}")


def run_stage(
    name: str,
    command: list[str],
    cwd: Path,
    expected_outputs: tuple[Path, ...],
) -> None:
    output_exists = [path.is_file() and path.stat().st_size > 0 for path in expected_outputs]
    if all(output_exists):
        print(f"[{name}] outputs already exist; skipping")
        return
    if any(path.exists() for path in expected_outputs):
        partial = [str(path) for path in expected_outputs if path.exists()]
        raise RuntimeError(
            f"[{name}] partial outputs exist; inspect or remove them before retrying: {partial}"
        )

    print(f"[{name}] {shlex.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)
    for path in expected_outputs:
        require_file(path)


def ensure_directory_link(link_path: Path, target_path: Path) -> None:
    target_path = target_path.resolve()
    if link_path.is_symlink():
        if link_path.resolve() != target_path:
            raise RuntimeError(
                f"Existing symlink points to the wrong target: {link_path} -> {link_path.resolve()}"
            )
        return
    if link_path.exists():
        require_directory(link_path)
        return
    link_path.symlink_to(target_path, target_is_directory=True)


def align_mesh_to_marker(
    input_dir: Path,
    mvs_dir: Path,
    colmap_dir: Path,
    re3sim_root: Path,
) -> None:
    output_path = mvs_dir / "mesh_to_marker.npy"
    if output_path.is_file() and output_path.stat().st_size > 0:
        print("[marker alignment] mesh_to_marker.npy already exists; skipping")
        return
    if output_path.exists():
        raise RuntimeError(f"Invalid marker-alignment output already exists: {output_path}")

    ensure_directory_link(mvs_dir / "sparse", colmap_dir / "sparse")
    ensure_directory_link(mvs_dir / "images", input_dir / "images")

    subprocess.run(
        [
            sys.executable,
            "../real-deployment/utils/compute_transform_to_marker_aruco.py",
            "--data_type",
            "openmvs",
            "--data_folder",
            str(mvs_dir),
            "--headless",
        ],
        cwd=re3sim_root,
        check=True,
    )

    colmap_transform = colmap_dir / "sparse" / "0" / "colmap_to_marker.npy"
    require_file(colmap_transform)
    shutil.copy2(colmap_transform, output_path)
    require_file(output_path)


def convert_mesh_to_usd(mesh_path: Path, mvs_dir: Path, re3sim_root: Path) -> None:
    usd_path = mesh_path.with_suffix(".usd")
    if usd_path.is_file() and usd_path.stat().st_size > 0:
        print(f"[USD conversion] output already exists; skipping: {usd_path}")
        return
    if usd_path.exists():
        raise RuntimeError(f"Invalid USD output already exists: {usd_path}")

    subprocess.run(
        [
            sys.executable,
            "utils/usd/obj_to_usd.py",
            "--obj_path",
            str(mesh_path),
            "--usd_dir",
            str(mvs_dir),
            "--collision-approximation",
            "meshSimplification",
            "--headless",
        ],
        cwd=re3sim_root,
        check=True,
    )
    require_file(usd_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume Re3Sim from existing scene_dense.mvs/scene_dense.ply outputs."
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-t", "--texture", action="store_true")
    parser.add_argument("--min-point-distance", type=float, default=4.0)
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument("--target-face-num", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_point_distance <= 0:
        raise ValueError("--min-point-distance must be positive")
    if args.max_threads <= 0:
        raise ValueError("--max-threads must be positive")
    if args.target_face_num <= 0:
        raise ValueError("--target-face-num must be positive")

    input_dir = args.input_dir.expanduser().resolve()
    mvs_dir = input_dir / "mvs"
    colmap_dir = input_dir / "colmap"
    re3sim_root = Path(__file__).resolve().parents[1]

    require_directory(input_dir / "images")
    require_directory(colmap_dir / "sparse" / "0")
    require_directory(mvs_dir)
    require_file(mvs_dir / "scene_dense.mvs")
    require_file(mvs_dir / "scene_dense.ply")
    require_file(
        re3sim_root.parent
        / "real-deployment"
        / "utils"
        / "compute_transform_to_marker_aruco.py"
    )
    require_file(re3sim_root / "utils" / "usd" / "obj_to_usd.py")

    for command in ("ReconstructMesh", "RefineMesh"):
        require_command(command)
    if args.texture:
        require_command("TextureMesh")

    run_stage(
        "ReconstructMesh",
        build_reconstruct_mesh_command(
            args.min_point_distance,
            args.max_threads,
            args.target_face_num,
        ),
        mvs_dir,
        (mvs_dir / "scene_dense_mesh.ply",),
    )
    run_stage(
        "RefineMesh",
        build_refine_mesh_command(args.max_threads),
        mvs_dir,
        (mvs_dir / "scene_dense_mesh_refine.ply",),
    )

    mesh_path = mvs_dir / "scene_dense_mesh_refine.ply"
    if args.texture:
        run_stage(
            "TextureMesh",
            build_texture_mesh_command(args.max_threads),
            mvs_dir,
            (mvs_dir / "scene_dense_mesh_refine_texture.ply",),
        )
        mesh_path = mvs_dir / "scene_dense_mesh_refine_texture.ply"

    align_mesh_to_marker(input_dir, mvs_dir, colmap_dir, re3sim_root)
    convert_mesh_to_usd(mesh_path, mvs_dir, re3sim_root)

    print("Resume reconstruction completed")
    print(f"mesh={mesh_path}")
    print(f"mesh_to_marker={mvs_dir / 'mesh_to_marker.npy'}")
    print(f"usd={mesh_path.with_suffix('.usd')}")


if __name__ == "__main__":
    main()
