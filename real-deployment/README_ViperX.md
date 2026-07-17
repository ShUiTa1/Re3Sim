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

## ViperX Adapter Live Validation

This Stage 5 procedure validates the accepted Stage 4 mapping through the
`ViperXAdapter` command path. It performs a live
`home -> shoulder left 90 degrees -> base XZ heart -> home` motion and then
checks the operator stop, hold, torque-release, and reconnect behavior.

The default heart is approximately 8 cm wide. This procedure sends real
`Goal_Position` commands. Keep the robot within reach, clear the complete
motion volume, and do not run it unattended.

Do not consider Stage 5 accepted until both the normal live run and the
intentional interruption run below have been completed, followed by a
successful reconnect.

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

### 2. Prepare the Robot and Workspace

Before running the script:

1. Check that the accepted Stage 4 mapping exists at
   `/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json`.
2. Confirm that `/dev/ttyDXL_follower_left` still points to the follower
   ViperX.
3. Clear people, tools, cables, cameras, and other obstacles from the complete
   arm and gripper motion volume.
4. Put the follower in a stable, safely supported physical state.
5. Stay beside the robot with one hand ready to press `Enter` or `Ctrl+C` and
   enough room to support the arm before torque is released.

The script derives the robot ID, port, calibration directory, URDF path, and
mapping parameters from the accepted mapping file. Only provide explicit
overrides when intentionally testing a different configuration.

### 3. Run the Offline Preflight

Run:

```bash
/data/yuzzhu/Re3Sim_ViperX/envs/re3sim-viperx-calib/bin/python \
  /home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/utils/validate_viperx_adapter.py \
  --mapping=/home/yuzzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

Before connecting to the ViperX, the script plans the complete trajectory and
checks:

- IK for every pose;
- the maximum joint step between neighboring waypoints;
- every URDF joint limit;
- the complete `q -> raw -> safe_raw_range` conversion.

The terminal must print:

```text
raw_plan_preflight=PASS
live_plan=home -> shoulder_left_-90deg -> base_XZ_heart -> home
heart_waypoints=48
Type MOVE to execute this live test:
```

No hardware connection or motion command occurs before this prompt. Review
the plan and the physical workspace again. If any preflight check fails, or if
the workspace is not safe, do not type `MOVE`; press `Enter` or type anything
else to exit before hardware connection.

Do not add `--yes` during the first Stage 5 acceptance because it bypasses this
final confirmation.

### 4. Complete the Normal Live Run

At the confirmation prompt, type exactly:

```text
MOVE
```

The script then connects to the follower and executes:

1. Move from the current position to the accepted Stage 4 home pose.
2. Move the shoulder left by 90 degrees.
3. Trace the approximately 8 cm-wide, 48-waypoint heart in the robot base XZ
   plane.
4. Return to the home pose.
5. Stop active motion and hold the home pose with torque enabled.

After connection, the terminal also explains the live safety interaction:

```text
Safety: Enter or Ctrl-C halts at the current pose. After you support the arm, Enter releases torque and disconnects.
```

During this normal run, watch the real arm continuously. The motion must remain
smooth, visually correct, and clear of collisions. If anything is unsafe,
press `Enter` or `Ctrl+C` once and follow the interruption procedure in the
next section.

At the end of a normal trajectory, the robot remains at home and holds its
pose. Support the physical arm first, then press `Enter` when prompted. This
releases torque and disconnects the robot. Return it manually to a stable
supported position after torque is off.

A successful normal run ends with:

```text
live_home_shoulder_heart_home=PASS
validate_viperx_adapter=PASS
```

### 5. Validate Enter/Ctrl-C Stop and Release

The emergency interaction must be tested separately from the successful
normal run:

1. Run the same command from Section 3 again.
2. Confirm that the offline preflight prints `raw_plan_preflight=PASS`.
3. Type exactly `MOVE`.
4. During a safe part of the motion, press `Enter` or `Ctrl+C` once.
5. Confirm that the robot stops and holds its current pose rather than
   continuing the planned trajectory or immediately losing torque.
6. Physically support the arm.
7. Press `Enter` once more to release torque and disconnect.
8. Put the arm in a stable supported position after torque is off.

An intentional interruption raises `ViperXMotionInterrupted`, so a traceback
and a nonzero process exit are expected for this specific run. The normal PASS
markers are not expected because the planned trajectory was deliberately not
completed. Acceptance depends on the physical behavior: stop, hold, supported
release, and disconnect.

After the interruption test, rerun the full procedure and complete one normal
run. This final run verifies that the adapter can reconnect and still finish
the planned trajectory after an interrupted session.

### 6. Stage 5 Acceptance

Stage 5 is accepted only when all of the following have been observed on the
lab follower:

- The complete plan passes offline checks before any hardware connection and
  prints `raw_plan_preflight=PASS`.
- The normal live trajectory is smooth, visually correct, and collision-free
  in the prepared workspace.
- The approximately 8 cm heart is traced in the expected base XZ plane.
- The normal run returns home and prints
  `live_home_shoulder_heart_home=PASS` and
  `validate_viperx_adapter=PASS`.
- At normal completion, the arm holds at home until it is physically supported
  and the operator presses `Enter` to release torque.
- During the separate interruption run, one `Enter` or `Ctrl+C` stops and holds
  the current pose; a second `Enter`, after the arm is supported, releases
  torque and disconnects.
- A subsequent normal run reconnects and completes successfully.

This guide documents the Stage 5 acceptance procedure. It does not claim that
the hardware checks have already been performed.

## Boundary

The offline kinematics entry point and the Stage 4 real raw-to-URDF mapping
have been validated. The accepted mapping records the real encoder home, URDF
home, direction signs, and scale, and several live poses matched in PyBullet.

Until the Stage 5 acceptance procedure above is completed on the lab follower,
commanded hardware motion remains unvalidated. Stage 5 still does not establish
general collision-free planning, camera extrinsics, hand-eye calibration
quality, or policy deployment safety.
