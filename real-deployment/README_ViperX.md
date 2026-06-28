# Re3Sim Real Deployment

This directory contains Re3Sim-side real robot and calibration entry points.
ViperX-specific runtime logic belongs here, while `viperx_asset` remains a pure
asset directory.

## ViperX Kinematics Model

Robotics Toolbox FK/IK wrapper:

```text
Re3Sim/real-deployment/viperx_model.py
```

Stage 2 validation script:

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

Use the existing validation environment directly for one command:

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

## Boundary

This validates the offline software kinematics entry point only. It does not
prove real ViperX encoder zero, raw encoder direction, LeRobot calibration
neutral pose, camera mounting transform, or safe hardware motion. IK is a
mathematical joint solution only; deployment still belongs in the adapter and
hardware safety layer.
