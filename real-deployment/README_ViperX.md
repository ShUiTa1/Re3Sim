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

This section aligns real ViperX raw encoder ticks with the validated URDF joint
radians used by `viperx_model.py`. This is an adapter/config mapping step, not
a URDF edit and not a Dynamixel calibration rewrite.

Mapping formula:

```text
q_urdf = q_home_urdf + sign * (raw - raw_home) * scale_rad_per_tick
```

The generated mapping is saved by default to:

```text
Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

### Environment

Do not use `re3sim-ros-xacro` . That environment is for offline URDF
and FK/IK validation only. Mapping needs the LeRobot environment that can
connect to the real ViperX.

Use your existing working LeRobot/ViperX environment if it can already run
LeRobot calibration and connect to the arm. Otherwise create a dedicated
environment for Stage 4:

```bash
cd /home/kienzhu/Projects/Re3Sim_ViperX

mamba create -n re3sim-viperx-calib python=3.11
mamba activate re3sim-viperx-calib

which python
python -m pip -V

python -m pip install --upgrade pip
python -m pip install -e ./lerobot
python -m pip install pybullet roboticstoolbox-python spatialmath-python

python -c "import lerobot, draccus, pybullet, roboticstoolbox, spatialmath; print('stage4 env ok')"
```

`python -m pip install -e ./lerobot` is environment-local. It installs into the
currently active Python environment shown by `which python` and
`python -m pip -V`. Do not run it from `base` unless you intentionally want to
install LeRobot into `base`.

That command reads:

```text
lerobot/pyproject.toml
```

It installs:

- The local LeRobot package in editable mode.
- LeRobot command line entries such as `lerobot-calibrate`, `lerobot-find-port`, and `lerobot-teleoperate`.
- LeRobot base dependencies declared in `[project].dependencies`.
- ViperX hardware-side dependencies already declared by LeRobot, including `draccus`, `dynamixel-sdk`, `pyserial`, and `trossen_arm`.

It does not install:

- LeRobot optional extras such as `intelrealsense`, `kinematics`, `dev`, `test`, or `all`.
- Stage 4 model/GUI dependencies `pybullet`, `roboticstoolbox-python`, and `spatialmath-python`.
- OS-level USB permissions, udev rules, RealSense firmware/drivers, or serial-port access.
- Re3Sim/Isaac Sim/Docker dependencies.

If wrist camera or RealSense scripts are needed later, install the optional
camera extra in the same active environment:

```bash
python -m pip install -e "./lerobot[intelrealsense]"
```

A fresh Stage 4 environment must provide at least:

```text
LeRobot local fork and its hardware dependencies
draccus
pybullet
numpy
roboticstoolbox-python
spatialmath-python
```

The Stage 4 scripts do not manually inject `lerobot/src` into `sys.path`.
LeRobot must be importable from the active Python environment through the
editable install above. The scripts only add this directory internally so they
can import the local `viperx_model.py`:

```text
/home/kienzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment
```

When leaving the lab environment:

```bash
mamba deactivate
```

### Lab Command Checklist

Use this as the command-level order at the robot.

1. Enter the Stage 4 environment:

```bash
cd /home/kienzhu/Projects/Re3Sim_ViperX
mamba activate re3sim-viperx-calib
which python
python -m pip -V
```

2. If the environment is new, install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ./lerobot
python -m pip install pybullet roboticstoolbox-python spatialmath-python
python -c "import lerobot, draccus, pybullet, roboticstoolbox, spatialmath; print('stage4 env ok')"
```

3. Find or confirm the robot port:

```bash
lerobot-find-port
```

Use the detected port in later commands, for example `/dev/ttyUSB0`.

4. If you intentionally want to rerun LeRobot calibration:

```bash
lerobot-calibrate \
  --robot.type=viperx \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.calibration_dir=/path/to/lerobot/calibration
```

This is LeRobot's own motor calibration step. It is separate from the URDF
mapping below. The URDF mapping does not edit LeRobot calibration files.

5. Create the URDF mapping:

```bash
python Re3Sim/real-deployment/utils/create_viperx_urdf_mapping.py \
  --robot.type=viperx \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.calibration_dir=/path/to/lerobot/calibration \
  --output=Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

6. Validate the saved mapping offline:

```bash
python Re3Sim/real-deployment/utils/validate_viperx_urdf_mapping.py \
  --mapping=Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

7. Validate live raw encoder reading without sending motor goals:

```bash
python Re3Sim/real-deployment/utils/validate_viperx_urdf_mapping.py \
  --live=true \
  --robot.type=viperx \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.calibration_dir=/path/to/lerobot/calibration \
  --mapping=Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

8. Leave the environment:

```bash
mamba deactivate
```

The Stage 4 acceptance marker is:

```text
validate_viperx_urdf_mapping=PASS
```

### Create Mapping

Script:

```text
Re3Sim/real-deployment/utils/create_viperx_urdf_mapping.py
```

Responsibility:

- Parse a LeRobot-style `ViperXConfig` with `draccus`.
- Load the validated full URDF through `viperx_model.py`.
- Open PyBullet GUI and let the user choose the URDF home pose with sliders.
- Open the LeRobot Dynamixel bus directly.
- Disable torque by default so the real arm can be moved by hand.
- Read `raw_home` from `Present_Position` with `normalize=False`.
- Infer each joint `sign` by manual URDF-positive joint movement.
- Save `q_home_urdf`, `raw_home`, `sign`, encoder scale, raw safe range, and URDF limits.
- Never send `Goal_Position`.
- Never edit Dynamixel `Homing_Offset`, `Drive_Mode`, or LeRobot calibration files.

Typical command:

```bash
cd /home/kienzhu/Projects/Re3Sim_ViperX
mamba activate <your-lerobot-viperx-env>

python Re3Sim/real-deployment/utils/create_viperx_urdf_mapping.py \
  --robot.type=viperx \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.calibration_dir=/path/to/lerobot/calibration \
  --output=Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

If the output file already exists and you want to replace it:

```bash
python Re3Sim/real-deployment/utils/create_viperx_urdf_mapping.py \
  --robot.type=viperx \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.calibration_dir=/path/to/lerobot/calibration \
  --output=Re3Sim/real-deployment/configs/viperx_urdf_mapping.json \
  --overwrite=true
```

Interactive flow:

- The script asks for confirmation before touching the hardware bus.
- PyBullet GUI opens with six sliders for the arm joints.
- Move the PyBullet sliders to the chosen URDF home pose.
- Move the real arm by hand to the matching physical pose.
- Return to the terminal and press `ENTER`; the script records `q_home_urdf` and `raw_home`.
- For each joint, move only the prompted joint in the URDF-positive direction.
- Accept each inferred sign after checking the printed raw encoder delta.
- The mapping JSON is written after all checks pass.

Optional non-interactive sign input after signs are known:

```bash
python Re3Sim/real-deployment/utils/create_viperx_urdf_mapping.py \
  --robot.type=viperx \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.calibration_dir=/path/to/lerobot/calibration \
  --signs=1,-1,1,1,-1,1 \
  --overwrite=true
```

The `--signs` order is always:

```text
waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
```

### Validate Mapping

Script:

```text
Re3Sim/real-deployment/utils/validate_viperx_urdf_mapping.py
```

Responsibility:

- Load `viperx_urdf_mapping.json`.
- Check the mapping schema and six-joint order.
- Verify `raw_home` maps back exactly to `q_home_urdf`.
- Check `raw_home` against saved LeRobot calibration safe ranges.
- Check `q_home_urdf` against saved URDF limits.
- Run FK through `viperx_model.py` and print `T_base_ee`.
- In `live=true` mode, connect to the LeRobot bus and read current raw encoders only.
- Convert live raw encoders to `q_urdf`, check limits, and print live FK.
- Never send `Goal_Position`.
- Never edit Dynamixel or LeRobot calibration state.

Offline validation, no hardware:

```bash
cd /home/kienzhu/Projects/Re3Sim_ViperX
mamba activate <your-lerobot-viperx-env>

python Re3Sim/real-deployment/utils/validate_viperx_urdf_mapping.py \
  --mapping=Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

Live read-only validation:

```bash
cd /home/kienzhu/Projects/Re3Sim_ViperX
mamba activate <your-lerobot-viperx-env>

python Re3Sim/real-deployment/utils/validate_viperx_urdf_mapping.py \
  --live=true \
  --robot.type=viperx \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.calibration_dir=/path/to/lerobot/calibration \
  --mapping=Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

Use `--yes=true` only when you intentionally want to skip the read-only live
confirmation prompt:

```bash
python Re3Sim/real-deployment/utils/validate_viperx_urdf_mapping.py \
  --live=true \
  --yes=true \
  --robot.type=viperx \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.calibration_dir=/path/to/lerobot/calibration
```

Expected pass marker:

```text
validate_viperx_urdf_mapping=PASS
```

Validation output includes:

```text
home_q_urdf=(...)
home_ee_xyz=(...)
home_T_base_ee=
live_raw={...}          # only in live=true mode
live_q_urdf=(...)       # only in live=true mode
live_ee_xyz=(...)       # only in live=true mode
```

### Acceptance

Accept Stage 4 only after:

- `create_viperx_urdf_mapping.py` writes a mapping JSON for the real robot.
- Offline `validate_viperx_urdf_mapping.py` passes on that JSON.
- Live `validate_viperx_urdf_mapping.py --live=true` reads real raw encoders and passes.
- The printed `q_urdf` values are plausible for the real arm pose.
- The printed FK `T_base_ee` is plausible for the real arm pose.
- The mapping JSON is treated as the adapter/config source of truth for raw encoder to URDF radians.

## Boundary

This validates the offline software kinematics entry point only. It does not
prove real ViperX encoder zero, raw encoder direction, LeRobot calibration
neutral pose, camera mounting transform, or safe hardware motion. IK is a
mathematical joint solution only; deployment still belongs in the adapter and
hardware safety layer.

Mapping reduces part of this boundary by explicitly recording real
encoder home, URDF home, direction signs, and scale. It still does not prove
camera extrinsics, hand-eye calibration quality, collision-free motion, or
policy deployment safety.
