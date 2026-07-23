# Re3Sim Real Deployment

This directory contains Re3Sim-side real robot and calibration entry points.
ViperX-specific runtime logic belongs here, while `viperx_asset` remains a pure
asset directory.

## ViperX Kinematics Model

Robotics Toolbox FK/IK wrapper:

```text
Re3Sim/real-deployment/viperx_model.py
```

validation script:

```text
Re3Sim/real-deployment/utils/validate_viperx_model.py
```

The wrapper directly loads the validated complete URDF:

```text
viperx_asset/urdf/vx300s_full.urdf
```

The full model contains 9 active joints,
but the Re3Sim-facing interface exposes only the 6 ViperX arm joints:

```text
waist
shoulder
elbow
forearm_roll
wrist_angle
wrist_rotate
```

The gripper and finger joints are kept internal at fixed default values during
FK/IK validation:

```text
gripper: 0.0
left_finger: 0.021
right_finger: -0.021
```

The current configured kinematic chain is:

```text
base_link: vx300s/base_link
end_link: vx300s/ee_gripper_link
```

Runtime interface:

```python
from viperx_model import load_viperx_model

model = load_viperx_model()
T_base_ee = model.fk(q_arm)

q_current = q_arm
T_target = T_base_ee
ik_result = model.ik(T_target, q0_arm=q_current)

if not ik_result.success:
    raise RuntimeError(ik_result.reason)

q_solution = ik_result.q_arm
```

Interface contract:

- `q_arm` is six values in radians.
- Joint order is `waist`, `shoulder`, `elbow`, `forearm_roll`, `wrist_angle`, `wrist_rotate`.
- `fk(q_arm)` returns a 4x4 `T_base_ee` transform.
- `ik(target_transform, q0_arm=...)` returns a `ViperXIKResult`, not a hardware command.
- In real execution, `q0_arm` should normally be the current measured arm joint state.
- IK output must still be checked by adapter/safety logic before any real robot motion.

## Environment

FK/IK validation uses the offline `re3sim-ros-xacro` environment.
It does not connect to hardware.

Use it directly for one command:

```bash
cd /home/kienzhu/Projects/Re3Sim_ViperX
mamba run -n re3sim-ros-xacro python Re3Sim/real-deployment/utils/validate_viperx_model.py
```

Or activate it first if running multiple commands:

```bash
cd /home/kienzhu/Projects/Re3Sim_ViperX
mamba activate re3sim-ros-xacro
```

If the environment was deleted, rebuild it from:

```bash
mamba env create -f viperx_asset/env/re3sim-ros-xacro.yml
mamba activate re3sim-ros-xacro
```

## Validate FK And IK

Run:

```bash
cd /home/kienzhu/Projects/Re3Sim_ViperX
mamba run -n re3sim-ros-xacro python Re3Sim/real-deployment/utils/validate_viperx_model.py
```

Equivalent after `mamba activate re3sim-ros-xacro`:

```bash
python Re3Sim/real-deployment/utils/validate_viperx_model.py
```

The validation script owns runtime-only details such as `MPLCONFIGDIR`; the
model module does not set process environment variables.

Validation method:

- Load `viperx_asset/urdf/vx300s_full.urdf` in Robotics Toolbox.
- Load the same URDF in PyBullet.
- Expand each 6-DOF arm sample to the full 9-joint model.
- Keep gripper and finger joints at fixed internal defaults.
- Assert every validation sample is finite, six-dimensional, and inside URDF/RTB joint limits.
- Compare `vx300s/ee_gripper_link` FK from Robotics Toolbox against PyBullet.
- Use PyBullet FK targets as IK goals.
- Solve IK in Robotics Toolbox through `viperx_model.py`, using a nearby but non-identical seed.
- Put each IK solution back into PyBullet and verify `FK(IK(target))` reaches the original target.
- Check an obviously unreachable target does not report success.

Current IK validation seed perturbation:

```text
[0.16, -0.18, 0.20, -0.15, 0.16, -0.18] rad
```

This is intentionally farther than a numerical epsilon but still local enough
to represent a nearby task target. It avoids using the original target
configuration as the IK initial guess.

Expected pass markers:

```text
rtb_full_n=9
wrapper_arm_n=6
base_link=vx300s/base_link
end_link=vx300s/ee_gripper_link
max_fk_position_error=...
max_fk_rotation_error=...
max_ik_position_error=...
max_ik_rotation_error=...
unreachable_target_check=PASS
validate_viperx_model=PASS
```

## URDF Mapping

This is the Stage 4 lab runbook for aligning real ViperX raw encoder ticks with
the validated URDF joint radians used by `viperx_model.py`.

Mapping formula:

```text
q_urdf = q_home_urdf + sign * (raw - raw_home) * scale_rad_per_tick
```

The result is stored in the adapter/config layer:

```text
/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

This procedure does not change URDF `q=0`, Dynamixel `Homing_Offset` or
`Drive_Mode`, or LeRobot calibration to fit the URDF. It reads raw
`Present_Position` values; LeRobot normalized `.pos` values are not radians and
are not used as URDF joint values.

The leader/follower types, IDs, and lab device names below follow
`/home/yuzzhu/Projects/Re3Sim_ViperX/lerobot/README-SL.md`.

### 1. Enter the Lab Project and Environment

The repository stays in the backed-up NFS home:

```bash
cd /home/yuzzhu/Projects/Re3Sim_ViperX
```

The Python environment and package caches stay on the local `/data` disk.
Stage 4 uses the user-level Miniforge installation at
`/data/yuzzhu/Re3Sim_ViperX/tools/miniforge3`. Do not run `conda init` or modify
`~/.bashrc`. In every new terminal or SSH session, source Conda once before its
first use; the same terminal does not need to source it again. This does not
change the accepted Stage 1/2 Mamba workflow.

Create the Stage 4 environment only once:

```bash
mkdir -p /data/yuzzhu/Re3Sim_ViperX/envs
mkdir -p /data/yuzzhu/Re3Sim_ViperX/cache/conda
mkdir -p /data/yuzzhu/Re3Sim_ViperX/cache/pip

source /data/yuzzhu/Re3Sim_ViperX/tools/miniforge3/etc/profile.d/conda.sh

CONDA_PKGS_DIRS=/data/yuzzhu/Re3Sim_ViperX/cache/conda \
conda create -y \
  -p /data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib \
  python=3.11

conda activate /data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib

CONDA_PKGS_DIRS=/data/yuzzhu/Re3Sim_ViperX/cache/conda \
conda install -y \
  -p /data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib \
  -c conda-forge \
  ffmpeg

python -m pip install --cache-dir /data/yuzzhu/Re3Sim_ViperX/cache/pip -e /home/yuzzhu/Projects/Re3Sim_ViperX/lerobot
python -m pip install --cache-dir /data/yuzzhu/Re3Sim_ViperX/cache/pip draccus pybullet roboticstoolbox-python spatialmath-python

which python
python --version
python -m pip -V
python -c "import lerobot, draccus, pybullet, roboticstoolbox, spatialmath; print('stage4 env ok')"
```

This uses Conda for environment management and FFmpeg, following the preparation
style in `lerobot/README-SL.md`. The explicit prefix and cache paths keep the
environment and downloaded packages out of the 15 GB NFS home. Python remains
at version 3.11 for this project. FFmpeg is installed as a dependency, but it
does not have a separate Stage 4 acceptance check.

On every later session, reuse the initialized environment without reinstalling
it:

```bash
cd /home/yuzzhu/Projects/Re3Sim_ViperX
source /data/yuzzhu/Re3Sim_ViperX/tools/miniforge3/etc/profile.d/conda.sh
conda activate /data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib

which python
python -m pip -V
```

`which python` should print:

```text
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python
```

Do not use `sudo`. After activation, use normal `python`, `python -m pip`, and
`lerobot-*` commands; there is no need to repeat the environment's full Python
path.

### 2. Confirm the Leader and Follower Ports

First check the lab aliases:

```bash
ls -l /dev/ttyDXL_follower_left /dev/ttyDXL_leader_left
```

The commands below assume:

```text
follower ViperX: /dev/ttyDXL_follower_left
leader WidowX:   /dev/ttyDXL_leader_left
```

If an alias is absent, run `lerobot-find-port` and replace that alias in the
later commands with the detected absolute device path, such as
`/dev/ttyUSB0`.

### 3. Calibrate Leader and Follower in One Run

This command uses the lab pairing from `lerobot/README-SL.md` and calibrates
both sides during the same `lerobot-teleoperate` startup:

```bash
lerobot-teleoperate \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_follower \
  --robot.calibration_dir=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/lerobot_calibration/robots/viperx \
  --teleop.type=widowx \
  --teleop.port=/dev/ttyDXL_leader_left \
  --teleop.id=left_leader \
  --teleop.calibration_dir=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/lerobot_calibration/teleoperators/widowx \
  --display_data=false
```

Follow the terminal prompts for the leader and follower calibration. LeRobot
creates the calibration directories automatically. Calibration on each side is
controlled by the existence of its exact JSON file:

```text
/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/lerobot_calibration/robots/viperx/left_follower.json
/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/lerobot_calibration/teleoperators/widowx/left_leader.json
```

- If a file is absent, that side enters calibration and saves the file.
- If a file already exists, that side loads it and skips calibration.
- With both files absent, this single command calibrates leader and follower.
- After both sides connect, the command enters teleoperation. Check the pairing,
  then press `Ctrl+C` when finished.

The following URDF mapping steps use only `left_follower.json`. The leader
calibration remains available for later teleoperation.

### 4. Create the URDF Mapping

Keep the follower supported while moving it by hand. Run:

```bash
python /home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/utils/create_viperx_urdf_mapping.py \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_follower \
  --robot.calibration_dir=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/lerobot_calibration/robots/viperx \
  --output=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

Interactive sequence:

1. Confirm the hardware prompt in the terminal.
2. In the PyBullet GUI, use the six sliders to set a safe and reproducible URDF
   home pose.
3. Support the real follower and move it to the same pose.
4. Return to the terminal and press `Enter`.
5. Keep supporting the arm until the terminal prints `raw_home=...`; this
   snapshot contains the six main arm actuators and the two shadow actuators.
6. Move the real arm to any other safe pose where both `shoulder` and `elbow`
   change. This is only a temporary shadow-sign sample; do not change the
   PyBullet home pose.
7. Press `Enter` again. If either paired joint moved too little, follow the
   terminal prompt and retry with a larger safe change.
8. Keep supporting the arm until both inferred shadow signs are printed, then
   return it to a stable supported position.
9. Wait for the script to save `viperx_urdf_mapping.json`.

If the mapping file already exists and you intentionally want to replace it,
rerun the same command with:

```text
--overwrite=true
```

For the six kinematic joints, the script applies the already validated URDF
sign flips for `shoulder`, `elbow`, and `wrist_angle` to the initial LeRobot
`drive_mode` signs. It infers the two shadow signs from the temporary second
pose. The original six-joint mapping fields remain unchanged; the two shadow
anchors are stored separately under `shadow_mapping`. The script does not send
`Goal_Position`.

### 5. Validate the Mapping Offline

This step does not connect to hardware:

```bash
python /home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/utils/validate_viperx_urdf_mapping.py \
  --mapping=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

Required result:

```text
validate_viperx_urdf_mapping=PASS
```

### 6. Validate Several Live Poses in PyBullet

Run:

```bash
python /home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/utils/validate_viperx_urdf_mapping.py \
  --live=true \
  --gui=true \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_follower \
  --robot.calibration_dir=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/lerobot_calibration/robots/viperx \
  --mapping=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

For each run:

1. The script connects to the follower and disables torque by default.
2. Support the follower and move it to a safe validation pose.
3. Return to the terminal and press `Enter`.
4. Keep supporting it until `live_raw=...` is printed or the PyBullet GUI
   opens; the raw snapshot has then been recorded.
5. Return the real arm to a stable supported position.
6. Compare the PyBullet pose with the real pose that was just recorded.
7. Return to the terminal and press `Enter` to close PyBullet.

Repeat the command for several distinct safe poses. Each run must end with:

```text
validate_viperx_urdf_mapping=PASS
```

### 7. Acceptance

Stage 4 has been accepted for `left_follower` using the procedure above:

- The environment import command printed `stage4 env ok`.
- Both `left_leader.json` and `left_follower.json` were created at the paths
  above.
- The home-anchor mapping was saved to
  `/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json`.
- The initial mapping showed reversed PyBullet motion for `shoulder`, `elbow`,
  and `wrist_angle` during live GUI validation.
- Those three mapping signs were flipped manually. The accepted mapping has
  sign `+1` for all six arm joints.
- Offline validation printed `validate_viperx_urdf_mapping=PASS` after the
  correction.
- Several distinct live GUI poses matched the corresponding real follower
  poses, and each run printed `validate_viperx_urdf_mapping=PASS`.

The repeated live raw-to-URDF PyBullet pose comparison is the final Stage 4
direction and alignment check.

When finished:

```bash
conda deactivate
```

## Torque-Off ViperX Pose Inspection

Use `inspect_viperx_pose.py` in the lab to inspect manually selected ViperX
poses before defining shooting regions or automatic targets. It reuses the
accepted Stage 4 mapping, the Stage 2 FK model, and one explicitly selected
RealSense camera. Each snapshot reports:

- raw encoder ticks for the six arm joints and two shadow motors;
- the six mapped URDF joint angles in radians;
- the end-effector XYZ position in metres, from `vx300s/base_link` to
  `vx300s/ee_gripper_link`;
- warnings when the measured raw values or mapped joint angles are outside the
  recorded ranges;
- one aligned `640 x 480` RGB PNG and one `640 x 480` raw depth NPY from the
  same camera frame pair.

### Extend the Existing Calibration Environment Once

Do not create another environment for pose inspection or the shooting
notebook. Reuse the Stage 3/4/5 environment at:

```text
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib
```

The earlier setup already installs the editable LeRobot fork, NumPy, SciPy,
OpenCV, Robotics Toolbox, and SpatialMath. Add only the camera-wrapper and
Jupyter-kernel dependencies that were not included in that setup:

```bash
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python \
  -m pip install \
  --cache-dir=/data/yuzzhu/Re3Sim_ViperX/cache/pip \
  "pyrealsense2>=2.55.1.6486,<2.57.0" \
  open3d \
  ipykernel
```

`open3d` is required at import time because the existing
`calibration/realsense/realsense.py` module imports it, even though
`inspect_viperx_pose.py` and the shooting notebook only use the module's RGB
and depth capture methods. These two capture entry points only use basic OpenCV
image conversion and writing; do not add a second OpenCV wheel solely for
ArUco/ChArUco here.

Verify the complete import chain without connecting the motor bus or camera:

```bash
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python - <<'PY'
import cv2
import ipykernel
import lerobot
import numpy
import open3d
import pyrealsense2
import roboticstoolbox
import scipy
import spatialmath

print("inspect_shooting_environment=PASS")
PY
```

Then verify that the inspection entry point itself can load. `--help` exits
before any hardware connection:

```bash
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python \
  /home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/utils/inspect_viperx_pose.py \
  --help
```

### Use the Same Environment as the Shooting Notebook Kernel

The Jupyter server and the notebook kernel are separate processes. The lab
launcher may start a system Jupyter server in a terminal and open Firefox; that
is acceptable. The selected kernel, rather than the server process, determines
which Python environment runs notebook cells.

Register the existing calibration environment as a Jupyter kernel once:

```bash
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python \
  -m ipykernel install \
  --user \
  --name=re3sim-viperx-calib \
  --display-name="re3sim-viperx-calib"
```

This writes only a small kernel specification under the user's Jupyter data
directory. It does not copy the environment into the NFS home and does not
require `sudo`.

After the automatic terminal and Firefox window open:

1. Open `hand_in_eye_shooting_ViperX.ipynb`.
2. Select `Kernel -> Change kernel -> re3sim-viperx-calib`.
3. Run this environment-only cell before running the notebook imports:

   ```python
   import sys

   expected = "/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python"
   print("kernel_python=", sys.executable)
   assert sys.executable == expected, "Wrong Jupyter kernel selected."
   print("shooting_kernel_environment=PASS")
   ```

The notebook currently carries a generic `python3` kernel name in its metadata,
so the displayed notebook title or the fact that Firefox opened successfully
does not prove that the calibration environment is active. The
`sys.executable` check above is the acceptance criterion. If
`re3sim-viperx-calib` does not appear in the kernel menu immediately after
registration, stop and restart the Jupyter server, then reopen the notebook.
Keep the server terminal open while using the notebook. The `Not Trusted`
banner and Jupyter widget warnings are separate from kernel selection and do
not replace the executable-path check. After the notebook has completed its
normal robot/camera cleanup, use `Kernel -> Shut Down`, then stop the Jupyter
server from its terminal with `Ctrl-C` twice.

Choose the wrist-camera serial number explicitly. Replace
`WRIST_CAMERA_SERIAL` below with the actual serial number, then run:

```bash
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python \
  /home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/utils/inspect_viperx_pose.py \
  --mapping=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json \
  --camera-serial=218622276584
```

Operation:

1. Support the arm, verify the selected serial belongs to the wrist camera, and
   manually place the arm at the pose to inspect.
2. Press `Enter`. The script connects to the motor bus, disables and verifies
   torque, starts the selected camera at RGB/depth `640 x 480` and `30 fps`,
   waits five seconds for warmup, then prints and captures the first snapshot.
3. Move the torque-off arm to another supported pose and press `Enter` again.
   Each press saves a new independent RGB/depth pair; no image stitching is
   performed.
4. Enter `E` and press `Enter` to stop the camera, disable torque again,
   disconnect, and exit.

Captured files are created under:

```text
/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/utils/assets/
  rgb_YYYYMMDD_HHMMSS_mmm.png
  depth_YYYYMMDD_HHMMSS_mmm.npy
```

The RGB file is an ordinary PNG. The depth file contains the aligned raw
RealSense depth array; use the camera depth scale when converting its values to
metres.

A normal exit ends with:

```text
viperx_pose_inspection=EXITED_CLEANLY
```

The script never enables torque and never writes `Goal_Position`. `Ctrl-C`,
terminal EOF, and runtime exceptions also stop the camera and enter the
torque-off disconnect path. An out-of-range snapshot is still printed for
diagnosis, but its `WARNING` status means that pose must not be treated as a
validated motion target.

## ViperX Adapter Live Validation

This is the Stage 5 runbook for validating the accepted Stage 4 mapping through
the `ViperXAdapter` command path.

**Current status: the startup-recovery adapter code and the single validation
plan are implemented, but the complete live plan, intentional stop/release,
and reconnect have not been accepted on hardware. Stage 5 therefore remains
unaccepted.**

The Re3Sim-facing state and action remain six arm joints. Every hardware write
contains eight absolute actuator targets: the six main joints plus
`shoulder_shadow` and `elbow_shadow`, with each shadow generated from its own
Stage 4 mapping.

### 1. Enter the Lab Project and Environment

Open a new terminal and run:

```bash
cd /home/yuzzhu/Projects/Re3Sim_ViperX
source /data/yuzzhu/Re3Sim_ViperX/tools/miniforge3/etc/profile.d/conda.sh
conda activate /data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib
```

The `source` command is required once in each new terminal or SSH session
before the first Conda command. It does not modify `~/.bashrc` and does not
require `conda init`.

### 2. Startup Recovery Boundary

The fixed startup staging joint vector is:

```python
STARTUP_STAGING_Q_RAD = (
    -0.050621,
    -1.000000,
    0.971010,
    0.115049,
    0.535359,
    0.000000,
)

```

`connect()` reads all eight actuator positions and writes the exact same raw
values back as the initial hold. It does not move toward home.

`prepare_for_motion()` is the only path that may read an initial mapped q
outside the URDF limits. It uses the existing closed-loop incremental command
logic to reach the fixed, strictly validated staging pose, while every raw
write remains inside all eight calibration ranges. After staging, normal
`read_joints()`, command targets, IK results, and waypoints all use strict
URDF/RTB limit validation.

This startup exception does not change URDF limits or `q=0`, Dynamixel
configuration, or LeRobot calibration.

### 3. Single Validation Plan

The validation script has one plan:

```text
offline:
  startup staging -> mapping home -> waist absolute +40 degrees
  -> base-XZ heart -> mapping home
  -> validate every six-joint q and every eight-actuator raw target

after MOVE:
  connect and hold current eight raw values
  -> prepare_for_motion()
  -> go_home()
  -> waist absolute +40 degrees
  -> heart
  -> go_home()
  -> stop_motion()
  -> await_release()
```

The waist target is an absolute `+40 degrees` value copied into the mapping
home vector, not a relative addition to the current waist value. The heart is
generated from that pose by ViperX IK. The script no longer contains a
shadow-only mode, a separate full-plan flag, or current/PWM/hardware-status
diagnostics.

`max_relative_target=5` still means maximum change per command rather than a
physical velocity limit. The current accepted design choice is to retain the
adapter's measured-state closed loop, incremental raw/radian step, command
period, arrival tolerance, and stable-sample check without adding a Dynamixel
profile or a separate software rad/s layer at this stage.

### 4. Run the Preflight and Live Gate

Run:

```bash
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python \
  /home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/utils/validate_viperx_adapter.py \
  --mapping=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

Before any hardware object is created, it must print:

```text
raw_plan_preflight=PASS
live_plan=startup_staging -> home -> waist_plus_40deg -> base_XZ_heart -> home
heart_waypoints=...
Type MOVE to execute this live test:
```

For an offline-only review, enter anything other than `MOVE`; the process exits
before connecting hardware. For the first live acceptance, inspect the plan
and physical worfrom scipy.optimize import least_squareskspace, then enter `MOVE`. Do not use `--yes` for the first
run.

A normal live completion prints:

```text
live_staging_home_waist_heart_home=PASS
validate_viperx_adapter=PASS
```

### 5. Stage 5 Acceptance Sequence

Run the following gates in order; a failed gate blocks all later gates:

1. Offline mapping and eight-actuator conversion checks.
2. Review the staging, home, waist, heart, and return-home preflight at the
   `MOVE` prompt.
3. One normal complete run from the usual startup pose, including the initial
   eight-actuator hold, startup recovery, home, waist, heart, return home, and
   supported release.
4. A separate intentional interruption run: the first `Enter` or `Ctrl+C` must
   stop and hold; after physical support, a second `Enter` must release torque
   and disconnect.
5. One final complete run to prove clean reconnect after the interruption.

Stage 5 is accepted only after all five gates pass on hardware.

## Boundary

The offline ViperX kinematics and the six-joint Stage 4 raw-to-URDF mapping are
accepted. The adapter startup recovery and the fixed validation plan are
implemented. All hardware commands contain the two mapped shadow targets.

The fixed full live plan, deliberate stop/hold/release, and reconnect remain
unaccepted. Stage 5 also does not establish general collision-free planning,
camera extrinsics, hand-eye calibration quality, or policy deployment safety.
