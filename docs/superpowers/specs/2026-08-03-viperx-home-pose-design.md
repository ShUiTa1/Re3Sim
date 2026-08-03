# ViperX Home Pose Before Policy Control

## Goal

Give simulation collection, LMDB replay, and later real-robot deployment the
same deterministic policy handoff state. The robot first moves to a fixed
joint-space home pose with the gripper aimed about 70 degrees down toward the
table. Only after the pose is reached does the task controller or policy take
over.

## Home pose

The six arm joints use the existing public order:

```text
[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
```

The configured home pose is:

```text
[0.0, 0.0, 0.0, 0.0, 1.22173048, 0.0] rad
```

It is stored as `control.home_q_rad`. `control.home_timeout_s` is `10.0`.
Homing reuses the existing `expert.joint_target_tolerance_rad` convergence
tolerance.

`wrist_angle` rotates about the URDF positive Y axis. With the other five arm
joints at zero, positive 70 degrees rotates the gripper direction toward
negative Z. The gripper scalar remains `1.0` (open), giving the complete
seven-dimensional home action:

```text
[0.0, 0.0, 0.0, 0.0, 1.22173048, 0.0, 1.0]
```

## Runtime sequence

1. Load and reset the Isaac scene as today.
2. Command the home action through the existing PD controller and the existing
   `0.8 rad/s` arm-target rate limit.
3. Wait until all six measured arm joints are within the configured joint
   tolerance. A finite timeout aborts cleanly if the pose cannot be reached.
4. Do not create an episode buffer or sample camera observations during this
   homing period.
5. After homing succeeds, read the settled task-item pose, construct the MPLib
   pick-and-place phases, and create the episode buffer.
6. Record the first policy observation at home. Record the complete motion from
   home to pregrasp and through the remaining pick-and-place phases.

LMDB replay follows the same homing sequence before applying its first stored
action. Interactive runtime without collection also homes before printing
`viperx_runtime=READY`, so the displayed ready state matches collection.

## Data and deployment contract

The episode begins at the policy handoff state, not at the imported URDF zero
state. Episode metadata records the configured six-joint home pose and open
gripper scalar. Later real-robot execution must command the same home pose,
wait for the same joint-space tolerance, then hand control to the policy.

The homing motion itself is intentionally excluded from training data. The
home-to-target motion is intentionally included.

## Scope

This change adds the YAML home pose and timeout plus a small reusable homing
loop in the existing ViperX runtime entry. It does not add a new Python script,
a new planner phase, TCP inverse kinematics, or unrelated validation
machinery.

## Verification

- Unit-test home-vector parsing and joint-error convergence behavior without
  Isaac Sim.
- Run one collection and confirm the first saved `observations/qpos` is within
  tolerance of the configured home action.
- Confirm later saved actions move from home toward pregrasp.
- Run LMDB replay and confirm it homes before applying the first stored action.
- Confirm no LMDB is saved when homing times out.
