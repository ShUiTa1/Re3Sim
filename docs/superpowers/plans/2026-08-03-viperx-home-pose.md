# ViperX Home Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move ViperX to a deterministic 70-degree-down joint-space home pose before policy collection or replay, excluding homing from LMDB while recording all motion from home onward.

**Architecture:** Keep the feature in the existing ViperX runtime boundary. Pure helpers parse and validate the home action and compute convergence; one small Isaac-facing loop commands the existing task adapter until measured arm joints reach home. The collection and replay branches run only after this common homing gate.

**Tech Stack:** Python 3.10, NumPy, Isaac Sim 4.0, YAML, `unittest`.

## Global Constraints

- Arm order is `[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]`.
- Home arm state is `[0.0, 0.0, 0.0, 0.0, 1.22173048, 0.0]` radians.
- Home gripper scalar is `1.0` (open).
- Homing uses the existing `0.8 rad/s` rate limiter and runtime PD controller.
- Homing is not written to LMDB; home-to-pregrasp and all later task motion are written.
- Collection, replay, and interactive ready state all pass through the same homing gate.
- Do not add a new runtime script, TCP IK, or planner phase.

---

### Task 1: Home action contract

**Files:**
- Modify: `re3sim/configs/viperx/pick_into_basket/collect_data_viperx.yaml`
- Modify: `re3sim/standalone/viperx/pick_into_basket_lmdb_viperx.py`
- Test: `re3sim/tests/test_viperx_stage10_contract.py`

**Interfaces:**
- Consumes: `params["control"]` from the existing task YAML.
- Produces: `_load_home_settings(control: dict[str, Any]) -> tuple[np.ndarray, float]`, returning a seven-dimensional action and timeout seconds; `_home_arm_error(current_qpos: Any, home_action: Any) -> float`.

- [ ] **Step 1: Write failing contract tests**

Add tests that require the exact YAML values and pure helper behavior:

```python
def test_yaml_defines_seventy_degree_policy_home(self) -> None:
    import yaml
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    control = config["tasks"][0]["params"]["control"]
    np.testing.assert_allclose(
        control["home_q_rad"], [0, 0, 0, 0, 1.22173048, 0]
    )
    self.assertEqual(control["home_timeout_s"], 10.0)

def test_home_settings_form_open_seven_dimensional_action(self) -> None:
    entry = _load_module("viperx_home_contract", ENTRY_PATH)
    action, timeout_s = entry._load_home_settings(
        {"home_q_rad": [0, 0, 0, 0, 1.22173048, 0], "home_timeout_s": 10}
    )
    np.testing.assert_allclose(action, [0, 0, 0, 0, 1.22173048, 0, 1])
    self.assertEqual(timeout_s, 10.0)
    self.assertAlmostEqual(entry._home_arm_error(action, action), 0.0)
```

- [ ] **Step 2: Run the focused tests and observe failure**

Run:

```bash
/root/miniconda/envs/py10/bin/python -m unittest \
  tests.test_viperx_stage10_contract.ViperXCollectionSurfaceTest.test_yaml_defines_seventy_degree_policy_home \
  tests.test_viperx_stage10_contract.ViperXCollectionSurfaceTest.test_home_settings_form_open_seven_dimensional_action -v
```

Expected: failure because the YAML keys and helper functions do not exist.

- [ ] **Step 3: Implement the minimal contract**

Add to `control`:

```yaml
home_q_rad: [0.0, 0.0, 0.0, 0.0, 1.22173048, 0.0]
home_timeout_s: 10.0
```

Add pure helpers to the runtime entry:

```python
def _load_home_settings(control: dict[str, Any]) -> tuple[np.ndarray, float]:
    arm = np.asarray(control["home_q_rad"], dtype=np.float64)
    timeout_s = float(control["home_timeout_s"])
    if arm.shape != (6,) or not np.all(np.isfinite(arm)):
        raise ValueError("control.home_q_rad must be six finite radians")
    if timeout_s <= 0.0:
        raise ValueError("control.home_timeout_s must be positive")
    return np.concatenate([arm, [1.0]]), timeout_s


def _home_arm_error(current_qpos: Any, home_action: Any) -> float:
    current = np.asarray(current_qpos, dtype=np.float64)
    home = np.asarray(home_action, dtype=np.float64)
    if current.shape != (7,) or home.shape != (7,):
        raise ValueError("home convergence requires two seven-dimensional states")
    return float(np.max(np.abs(current[:6] - home[:6])))
```

- [ ] **Step 4: Run focused tests and confirm pass**

Run the Step 2 command again. Expected: both tests pass.

### Task 2: Common homing gate before recording and replay

**Files:**
- Modify: `re3sim/standalone/viperx/pick_into_basket_lmdb_viperx.py`
- Test: `re3sim/tests/test_viperx_stage10_contract.py`

**Interfaces:**
- Consumes: `_load_home_settings`, `_home_arm_error`, `ViperXPickAndPlaceTask.set_desired_action`, `advance_control`, `robot.get_joint_positions`, and `arm_indices`.
- Produces: `_move_to_home(...) -> None`; collection metadata key `policy_home` containing `q_rad`, `gripper_scalar`, and `joint_tolerance_rad`.

- [ ] **Step 1: Write a failing homing-loop test**

Use fake world/task objects to require rate-limited stepping until convergence without camera observation calls:

```python
def test_homing_steps_without_sampling_observations(self) -> None:
    entry = _load_module("viperx_home_loop", ENTRY_PATH)

    class Robot:
        q = np.zeros(9, dtype=np.float64)
        def get_joint_positions(self):
            return self.q.copy()

    class Task:
        robot = Robot()
        arm_indices = tuple(range(6))
        desired = None
        observation_calls = 0
        def set_desired_action(self, action):
            self.desired = np.asarray(action, dtype=np.float64)
        def advance_control(self, _physics_dt):
            delta = np.clip(self.desired[:6] - self.robot.q[:6], -0.2, 0.2)
            self.robot.q[:6] += delta
        def get_observation(self):
            self.observation_calls += 1
            raise AssertionError("homing must not sample cameras")

    class World:
        steps = 0
        def step(self, *, render):
            self.steps += 1

    task, world = Task(), World()
    home = np.asarray([0, 0, 0, 0, 1.22173048, 0, 1], dtype=np.float64)
    entry._move_to_home(
        world=world,
        runtime_task=task,
        home_action=home,
        physics_dt=0.1,
        tolerance_rad=0.05,
        timeout_s=2.0,
        render=False,
    )
    self.assertGreater(world.steps, 0)
    self.assertEqual(task.observation_calls, 0)
    self.assertLessEqual(entry._home_arm_error(task.robot.q[:7], home), 0.05)
```

- [ ] **Step 2: Run the test and observe failure**

Run:

```bash
/root/miniconda/envs/py10/bin/python -m unittest \
  tests.test_viperx_stage10_contract.ViperXCollectionSurfaceTest.test_homing_steps_without_sampling_observations -v
```

Expected: failure because `_move_to_home` does not exist.

- [ ] **Step 3: Implement the common homing gate**

Implement `_move_to_home` with these exact behaviors:

```python
def _move_to_home(*, world, runtime_task, home_action, physics_dt,
                  tolerance_rad, timeout_s, render) -> None:
    runtime_task.set_desired_action(home_action)
    max_steps = int(np.ceil(timeout_s / physics_dt))
    for step in range(max_steps + 1):
        positions = np.asarray(runtime_task.robot.get_joint_positions(), dtype=np.float64)
        measured = np.concatenate(
            [positions[list(runtime_task.arm_indices)], [home_action[6]]]
        )
        error = _home_arm_error(measured, home_action)
        if error <= tolerance_rad:
            print(f"viperx_home=READY steps={step} max_error_rad={error:.6f}")
            return
        if step < max_steps:
            runtime_task.advance_control(physics_dt)
            world.step(render=render)
    raise RuntimeError(
        f"ViperX home timed out after {timeout_s:g} s: max error={error:.6f} rad"
    )
```

In `run`, call `_load_home_settings` and `_move_to_home` after the existing startup loop and before `viperx_runtime=READY` or any branch-specific controller construction. Reuse `expert.joint_target_tolerance_rad`. Because `_run_collection` constructs `EpisodeBuffer` after this call, no homing samples enter LMDB. Replay automatically receives the same gate.

Add metadata:

```python
"policy_home": {
    "q_rad": list(params["control"]["home_q_rad"]),
    "gripper_scalar": 1.0,
    "joint_tolerance_rad": float(params["expert"]["joint_target_tolerance_rad"]),
},
```

- [ ] **Step 4: Run the focused test and full contract suite**

Run:

```bash
/root/miniconda/envs/py10/bin/python -m unittest tests.test_viperx_stage10_contract -v
```

Expected: all tests pass.

- [ ] **Step 5: Run Isaac collection smoke verification**

Run:

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
  --collect-one
```

Expected startup evidence includes `viperx_home=READY` before `viperx_runtime=READY`. If collection succeeds, inspect the first `observations/qpos` and confirm its first six values are within `0.05 rad` of `home_q_rad`.
