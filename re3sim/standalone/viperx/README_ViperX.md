# ViperX Re3Sim Scene, Collection, Replay, and Preview

This document describes the routine procedures for Stage 9 scene-asset integration, Stage 10 ViperX operation, and the current single-episode LMDB collection, replay, and preview workflow.

Unless a command is explicitly marked as running on the host, all Isaac Sim, collection, replay, and preview commands must be run inside the `re3sim` container.

## 0. Before Every Session: Host Permissions

Run both commands in this section in an Ubuntu host terminal, not inside the container.

### 0.1 Allow the Container to Display the Isaac Sim X11 Window

This is required only when using `--interactive`. Headless collection and preview do not require X11 access.

```bash
xhost +local:docker
```

You can revoke the permission after finishing the session:

```bash
xhost -local:docker
```

### 0.2 Allow the Host User to Modify and Delete Container-Generated Data

The container writes to bind mounts as root by default. After collection or preview, run the following command on the host:

```bash
sudo chown -R "$(id -u):$(id -g)" \
  ~/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/calibration/data
```

This host directory is mounted as `/root/work/data` inside the container.

## 1. Create and Enter the Container

### 1.1 Create the Container (Host)

The following is the complete current command. It configures the GPU, X11, memory limits, and code/data/robot-asset mounts:

```bash
docker run --name re3sim --entrypoint bash -itd \
  --runtime=nvidia \
  --gpus='"device=0"' \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --rm \
  --network=bridge \
  --shm-size=28g \
  --memory=24g \
  --memory-swap=56g \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Projects/Re3Sim_ViperX/Re3Sim/real-deployment:/root/dev/real-deployment:rw \
  -v ~/Projects/Re3Sim_ViperX/Re3Sim/re3sim:/root/dev/real2sim2real:rw \
  -v ~/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/calibration/data:/root/work/data:rw \
  -v ~/Projects/Re3Sim_ViperX/viperx_asset:/root/dev/viperx_asset:ro \
  re3sim_yuzheng:openmvs
```

This command uses `--rm`: the container is deleted after `docker stop re3sim`, but the code, data, and assets in the bind-mounted directories are preserved.

If Docker reports that a container named `re3sim` already exists, check its status first:

```bash
docker ps -a --filter name=re3sim
```

If it is still running, do not create it again; enter the existing container directly.

### 1.2 Enter the Container (Host)

```bash
docker exec -it re3sim bash
```

### 1.3 Enter the Project Directory (Container)

```bash
cd /root/dev/real2sim2real
```

The remaining commands use the absolute path to the `py10` Python interpreter, so `conda activate py10` is not required.

## 2. Check Input Assets Before Running (Container)

The following paths must exist:

```bash
ls /root/work/data/reconstruction_source/mvs/scene_dense_mesh_refine_texture.usd
ls /root/work/data/reconstruction_source/gs/0
ls /root/work/data/viperx_hand_eye/marker_2_base.npy
ls /root/work/data/viperx_hand_eye/cam_to_hand_pose.npy
ls /root/work/data/top_camera_to_base.npy
ls /root/work/data/viperx_stage10/vx300s_full_mplib.srdf
ls /root/dev/viperx_asset/urdf/vx300s_full.urdf
```

Check the runtime dependencies in the current Python environment:

```bash
/root/miniconda/envs/py10/bin/python -c \
  "import lmdb, mplib; print('runtime_dependencies=PASS')"
```

`/root/work/data/viperx_stage10/vx300s.usd` does not need to exist in advance. On the first run, the entry point generates it from the URDF and saves it at that path. Later runs reuse the saved file. The SRDF is not regenerated during routine operation, so `vx300s_full_mplib.srdf` listed above must already exist.

The unified configuration file is:

```text
/root/dev/real2sim2real/configs/viperx/pick_into_basket/collect_data_viperx.yaml
```

It defines:

- The relationship between the Isaac world and the ViperX base.
- The reconstructed USD, 3DGS, and marker-to-base paths.
- The fixed-base ViperX, home pose, PD parameters, and `0.8 rad/s` joint-target speed limit.
- The seven-dimensional `[6 arm rad, 1 gripper scalar]` data contract.
- The wrist-camera and top-camera extrinsics.
- The black task block, randomized support face, and randomized yaw.

## 3. Startup Checks

### 3.1 Headless Startup Check (Container)

This command loads the scene, ViperX, task object, and cameras, moves the robot to home, and then exits. It does not execute a grasp or save an LMDB:

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml
```

A normal run should print:

```text
viperx_home=READY
runtime_cameras=('wrist_camera', 'top_camera')
viperx_runtime=READY
```

The order of the cameras in the tuple does not affect the data, but both `wrist_camera` and `top_camera` must be present.

### 3.2 Open the Interactive Isaac Sim Window (Container)

First confirm that `xhost +local:docker` was run on the host, then run:

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
  --interactive
```

Use this mode to check:

- The relative placement of the reconstructed scene and ViperX base.
- The ViperX home pose and black appearance.
- The task block's initial position, support face, and orientation.
- The presence of the wrist-camera and top-camera prims.

Close the Isaac Sim window or press `Ctrl+C` in the terminal to stop the program.

If `Authorization required` or `GLFW initialization failed` appears, exit the program and rerun the following command on the host:

```bash
xhost +local:docker
```

Then verify that the container-creation command includes:

```text
-e DISPLAY="${DISPLAY}"
-v /tmp/.X11-unix:/tmp/.X11-unix:rw
```

## 4. Collect One Diagnostic Trajectory

The current implementation provides `--collect-one`; the production `--collect N` interface for collecting multiple successful trajectories has not yet been implemented.

### 4.1 Run a Single Collection (Container)

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
  --collect-one
```

The command above performs headless collection. To observe the same collection in the Isaac Sim window, first grant X11 access on the host as described in Section 0.1, then run inside the container:

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
  --collect-one \
  --interactive
```

After the trajectory finishes, the window remains on the final scene. Close it or press `Ctrl+C` in the terminal to exit.

The sequence is:

```text
Load scene
  -> Move ViperX to home
  -> Randomly select a stable support face and yaw for the task block, then let it settle
  -> pregrasp
  -> approach
  -> close
  -> lift
  -> place
  -> open
  -> retreat
```

Only successful trajectories are written to LMDB. A successful run prints:

```text
viperx_collection_success=True
viperx_collection_lmdb_root=/root/work/data/sim_logs/viperx_pick_into_basket/.../log-...
```

A failed run prints:

```text
viperx_collection_success=False
viperx_collection_lmdb=NOT_SAVED
```

A single diagnostic LMDB currently contains:

- Seven-dimensional `observations/qpos`, `observations/qvel`, and `action`.
- `observations/ee_pose`.
- Final GS-composited RGB from `wrist_camera` and `top_camera`.
- Diagnostic fields including the raw Isaac image, GS background image, semantic mask, and label mapping.

The final composite images already contain the static GS background and correct occlusion by the dynamic Isaac robot and task object. Isaac does not need to be restarted to render previews after collection.

You can also override the calibration with a newer result, for example:

```bash
  /root/miniconda/envs/py10/bin/python \
    standalone/viperx/pick_into_basket_lmdb_viperx.py \
    --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
    --collect-one \
    --interactive \
    --wrist-camera-extrinsic-override \
    /root/work/data/viperx_sim_overrides/wrist_camera_r03.npy
```

## 5. Locate an LMDB

### 5.1 Path Inside the Container

All single episodes are stored under:

```text
/root/work/data/sim_logs/viperx_pick_into_basket/<collection-batch>/log-*/
```

Show the most recent episode:

```bash
ls -dt /root/work/data/sim_logs/viperx_pick_into_basket/*/log-* | head -n 1
```

The basic structure of a successful episode is:

```text
log-xxxxxx-xxxx/
├── info.json
└── lmdb/
    ├── data.mdb
    └── lock.mdb
```

### 5.2 Path on the Host

The same data is available on the host at:

```text
~/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/calibration/data/
  sim_logs/viperx_pick_into_basket/<collection-batch>/log-*/
```

If the files are owned by root, appear locked in the file manager, or cannot be deleted, run the `sudo chown -R ...` command from Section 0.2 on the host.

## 6. Replay One LMDB

You can pass either an episode directory or its internal `lmdb` directory. The complete command below automatically selects the most recently saved episode.

### 6.1 Replay (Container)

```bash
/bin/bash -c '
VIPERX_EPISODE="$(ls -dt /root/work/data/sim_logs/viperx_pick_into_basket/*/log-* | head -n 1)"
echo "replay_episode=${VIPERX_EPISODE}"
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
  --replay "${VIPERX_EPISODE}" \
  --interactive
'
```

If you already know the episode path, pass it directly to `--replay`; the automatic lookup step is unnecessary. Remove the final `--interactive` option for headless replay.

After replaying the action sequence, the program prints:

```text
viperx_replay_steps=<trajectory-frame-count>
viperx_replay_success=<True or False>
```

Replay reads only the seven-dimensional `action` and drives the simulation again. It does not create a new LMDB or restore the task block's randomized initial pose from the recorded episode. It is therefore intended for inspecting the robot action trajectory. `viperx_replay_success` represents a deterministic reproduction of the original episode only when the current task block's initial state happens to match the recorded state.

The `--interactive` window displays the dynamic Isaac scene, not the dual-camera GS-composited images stored in the LMDB. Use the preview command in Section 7 to inspect the complete GS observation trajectory.

## 7. Generate a Dual-Camera GS Trajectory Preview

The preview script reads the final composited RGB images already stored in the LMDB. It does not start Isaac Sim or rerun the GS renderer.

### 7.1 Generate One Image Every 10 Frames by Default (Container)

```bash
/bin/bash -c '
VIPERX_EPISODE="$(ls -dt /root/work/data/sim_logs/viperx_pick_into_basket/*/log-* | head -n 1)"
echo "preview_episode=${VIPERX_EPISODE}"
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/preview_lmdb_episode.py \
  "${VIPERX_EPISODE}"
'
```

Each preview image contains:

- `wrist_camera` on the left.
- `top_camera` on the right.
- The original trajectory frame number at the top.
- Frames `0, 10, 20, ...` by default, with the final frame always included.

The default output directory is:

```text
<single-log-directory>/preview_stride10/
```

A successful run prints:

```text
preview_frames=<number-of-generated-images>
preview_output=<preview-directory>
```

### 7.2 Set the Frame Stride and Output Directory (Container)

```bash
/bin/bash -c '
VIPERX_EPISODE="$(ls -dt /root/work/data/sim_logs/viperx_pick_into_basket/*/log-* | head -n 1)"
echo "preview_episode=${VIPERX_EPISODE}"
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/preview_lmdb_episode.py \
  "${VIPERX_EPISODE}" \
  --stride 10 \
  --output /root/work/data/viperx_episode_preview
'
```

An older LMDB containing only the wrist camera produces an explicit error:

```text
LMDB is missing 'observations/images/top_camera/<frame-number>'
```

The script does not substitute a raw Isaac image or raw GS image for a missing official dual-camera composite.

## 8. Current Implementation Boundaries

- Stage 9 reconstruction assets and coordinate alignment are integrated into the current YAML; use `--interactive` for routine inspection.
- Stage 10 fixed-base ViperX, TCP, gripper, dual cameras, GS composite occlusion, and the single-trajectory expert loop are integrated into the current entry point.
- `--collect-one`, `--replay`, and dual-camera frame-sampled preview are implemented.
- Production multi-trajectory collection through `--collect N` has not yet been implemented. Repeatedly running `--collect-one` must not be treated as an existing batch interface.
- `side_camera` is skipped automatically because its extrinsics are empty. The current official cameras are `wrist_camera` and `top_camera`.
- `planner.use_background_point_cloud` is currently `false`. MPLib does not use the noisy reconstructed point cloud for path planning; robot, task-object, and scene-physics collisions in Isaac are handled separately.

### 8.1 Known Non-Blocking Messages

The following messages come from legacy modules or optional dependencies that are not used by the current ViperX entry point. If `viperx_home=READY` and `viperx_runtime=READY` appear later, these messages do not indicate that this entry point failed:

```text
ModuleNotFoundError: No module named 'real2sim2real.controllers.stacking_curobo'
Field "model_path" has conflict with protected namespace "model_"
Pinocchio is not installed, some functions will not be available.
```

Issues that do require attention are exceptions subsequently raised by the entry point itself, `viperx_collection_success=False`, or the absence of `viperx_runtime=READY`.

## 9. Stop the Container and Restore Host Permissions

First exit the container shell:

```bash
exit
```

Then stop the container on the host:

```bash
docker stop re3sim
```

Because the container was created with `--rm`, it is deleted when stopped, while bind-mounted data is preserved.

Restore ownership of the data directory:

```bash
sudo chown -R "$(id -u):$(id -g)" \
  ~/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/calibration/data
```

Revoke X11 access:

```bash
xhost -local:docker
```
