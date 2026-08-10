# ViperX Re3Sim 场景、采集、回放与预览

本文记录 Stage 9 场景资产接入、Stage 10 ViperX 运行以及当前单条 LMDB 采集/回放/预览的日常操作。

除特别标为“宿主机”的命令外，Isaac Sim、采集、回放和预览命令都在 `re3sim` 容器内执行。

## 0. 每次开始前：宿主机权限

以下两条命令都在 Ubuntu 宿主机终端执行，不是在容器内执行。

### 0.1 允许容器显示 Isaac Sim X11 窗口

只有使用 `--interactive` 时必须执行；纯 headless 采集和预览不需要 X11。

```bash
xhost +local:docker
```

工作结束后可以收回授权：

```bash
xhost -local:docker
```

### 0.2 让容器生成的数据可由宿主机用户修改和删除

容器默认以 root 写入 bind mount。采集或预览完成后，在宿主机执行：

```bash
sudo chown -R "$(id -u):$(id -g)" \
  ~/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/calibration/data
```

该宿主机目录在容器内对应 `/root/work/data`。

## 1. 创建并进入容器

### 1.1 创建容器（宿主机）

下面是当前完整命令，包含 GPU、X11、内存限制以及代码/数据/机器人资产挂载：

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

这里使用了 `--rm`：执行 `docker stop re3sim` 后容器会被删除，但这些 bind mount 中的代码、数据和资产不会被删除。

如果提示容器名 `re3sim` 已存在，先检查状态：

```bash
docker ps -a --filter name=re3sim
```

如果它仍在运行，不需要重新创建，直接进入即可。

### 1.2 进入容器（宿主机）

```bash
docker exec -it re3sim bash
```

### 1.3 进入项目目录（容器内）

```bash
cd /root/dev/real2sim2real
```

本文后续统一直接使用 py10 的绝对 Python 路径，因此不要求额外执行 `conda activate py10`。

## 2. 运行前检查输入资产（容器内）

下面这些路径必须存在：

```bash
ls /root/work/data/reconstruction_source/mvs/scene_dense_mesh_refine_texture.usd
ls /root/work/data/reconstruction_source/gs/0
ls /root/work/data/viperx_hand_eye/marker_2_base.npy
ls /root/work/data/viperx_hand_eye/cam_to_hand_pose.npy
ls /root/work/data/top_camera_to_base.npy
ls /root/work/data/viperx_stage10/vx300s_full_mplib.srdf
ls /root/dev/viperx_asset/urdf/vx300s_full.urdf
```

检查当前 Python 环境中的运行依赖：

```bash
/root/miniconda/envs/py10/bin/python -c \
  "import lmdb, mplib; print('runtime_dependencies=PASS')"
```

`/root/work/data/viperx_stage10/vx300s.usd` 不要求预先存在；第一次运行时，入口会从 URDF 生成它并保存到该路径。之后会直接复用。SRDF 不会在日常运行中重新生成，因此上面的 `vx300s_full_mplib.srdf` 必须存在。

当前统一配置文件是：

```text
/root/dev/real2sim2real/configs/viperx/pick_into_basket/collect_data_viperx.yaml
```

它定义：

- Isaac world 与 ViperX base 的关系。
- reconstructed USD、3DGS 和 marker-to-base 路径。
- fixed-base ViperX、home pose、PD 参数和 `0.8 rad/s` 关节目标限速。
- 7 维 `[6 arm rad, 1 gripper scalar]` 数据契约。
- 腕部相机与顶部相机外参。
- 黑色任务块、随机支撑面和随机 yaw。

## 3. 启动检查

### 3.1 Headless 启动检查（容器内）

该命令加载场景、ViperX、任务物体和相机，移动到 home，然后退出；不会执行抓取，也不会保存 LMDB：

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml
```

正常情况下应看到：

```text
viperx_home=READY
runtime_cameras=('wrist_camera', 'top_camera')
viperx_runtime=READY
```

相机 tuple 的打印顺序不影响数据，但必须同时包含 `wrist_camera` 和 `top_camera`。

### 3.2 打开 Isaac Sim 交互窗口（容器内）

先确认宿主机已执行 `xhost +local:docker`，然后运行：

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
  --interactive
```

该模式用于检查：

- reconstructed scene 与 ViperX base 的相对位置。
- ViperX home 姿态和黑色外观。
- 任务块初始位置、支撑面和朝向。
- 腕部、顶部相机 prim 是否存在。

关闭 Isaac Sim 窗口或在终端按 `Ctrl+C` 结束。

若出现 `Authorization required` 或 `GLFW initialization failed`，先退出程序，在宿主机重新执行：

```bash
xhost +local:docker
```

然后确认创建容器时没有遗漏：

```text
-e DISPLAY="${DISPLAY}"
-v /tmp/.X11-unix:/tmp/.X11-unix:rw
```

## 4. 采集一条诊断轨迹

当前已实现的是 `--collect-one`；正式的 `--collect N` 多条成功轨迹接口尚未实现。

### 4.1 执行单条采集（容器内）

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
  --collect-one
```

以上是 headless 采集。若要在 Isaac Sim 窗口中观察同一次采集，先在宿主机执行第 0.1 节的 X11 授权，再在容器内运行：

```bash
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/pick_into_basket_lmdb_viperx.py \
  --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
  --collect-one \
  --interactive
```

轨迹结束后窗口会停留在最终场景，关闭窗口或在终端按 `Ctrl+C` 退出。

流程为：

```text
加载场景
  -> ViperX 到 home
  -> 任务块随机选择稳定支撑面和 yaw 后落稳
  -> pregrasp
  -> approach
  -> close
  -> lift
  -> place
  -> open
  -> retreat
```

只有成功轨迹会写入 LMDB。成功时应看到：

```text
viperx_collection_success=True
viperx_collection_lmdb_root=/root/work/data/sim_logs/viperx_pick_into_basket/.../log-...
```

失败时会看到：

```text
viperx_collection_success=False
viperx_collection_lmdb=NOT_SAVED
```

单条诊断 LMDB 当前包含：

- 7 维 `observations/qpos`、`observations/qvel` 和 `action`。
- `observations/ee_pose`。
- `wrist_camera`、`top_camera` 的最终 GS 合成 RGB。
- 纯 Isaac 图、GS 背景图、semantic mask 和 label mapping 等诊断字段。

最终合成图已经包含 GS 静态背景，以及 Isaac 动态机器人/任务物体的正确遮挡。采集后不需要重新启动 Isaac 来渲染预览。

也可以用新的标定覆盖，比如

```bash
  /root/miniconda/envs/py10/bin/python \
    standalone/viperx/pick_into_basket_lmdb_viperx.py \
    --config configs/viperx/pick_into_basket/collect_data_viperx.yaml \
    --collect-one \
    --interactive \
    --wrist-camera-extrinsic-override \
    /root/work/data/viperx_sim_overrides/wrist_camera_r03.npy
```



## 5. 查找 LMDB

### 5.1 容器内路径

所有单条 episode 位于：

```text
/root/work/data/sim_logs/viperx_pick_into_basket/<采集批次>/log-*/
```

查看最近的 episode：

```bash
ls -dt /root/work/data/sim_logs/viperx_pick_into_basket/*/log-* | head -n 1
```

一个成功 episode 的基本结构为：

```text
log-xxxxxx-xxxx/
├── info.json
└── lmdb/
    ├── data.mdb
    └── lock.mdb
```

### 5.2 宿主机路径

同一数据在宿主机位于：

```text
~/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/calibration/data/
  sim_logs/viperx_pick_into_basket/<采集批次>/log-*/
```

如果文件显示为 root 所有、文件管理器带锁或无法删除，在宿主机执行本文 0.2 的 `sudo chown -R ...` 命令。

## 6. 回放一条 LMDB

可以传 episode 目录，也可以传它内部的 `lmdb` 目录。下面的完整命令自动选择最近保存的一条 episode。

### 6.1 回放（容器内）

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

如果已经知道具体 episode 路径，可以直接把它传给 `--replay`，不需要前面的自动查找步骤。去掉最后的 `--interactive` 即为 headless 回放。

动作序列重放完成后会打印：

```text
viperx_replay_steps=<轨迹帧数>
viperx_replay_success=<True 或 False>
```

回放只读取 7 维 `action` 并重新驱动仿真，不会创建新的 LMDB，也不会从 LMDB 恢复采集时随机生成的物块初始姿态。因此它用于检查机器人动作轨迹；只有当前物块初始状态恰好匹配时，`viperx_replay_success` 才代表原 episode 的确定性任务复现。

`--interactive` 窗口显示的是 Isaac 动态场景，不是 LMDB 中保存的双相机 GS 合成画面。检查完整 GS 观测轨迹应使用第 7 节的预览命令。

## 7. 生成双相机 GS 轨迹预览

预览脚本直接读取 LMDB 中已经保存的最终合成 RGB，不启动 Isaac Sim，也不重新运行 GS renderer。

### 7.1 默认每 10 帧生成一张（容器内）

```bash
/bin/bash -c '
VIPERX_EPISODE="$(ls -dt /root/work/data/sim_logs/viperx_pick_into_basket/*/log-* | head -n 1)"
echo "preview_episode=${VIPERX_EPISODE}"
/root/miniconda/envs/py10/bin/python \
  standalone/viperx/preview_lmdb_episode.py \
  "${VIPERX_EPISODE}"
'
```

每张预览图：

- 左侧为 `wrist_camera`。
- 右侧为 `top_camera`。
- 顶部标注原始轨迹帧号。
- 默认抽取 `0, 10, 20, ...`，并保证包含最后一帧。

默认输出到：

```text
<单条log目录>/preview_stride10/
```

正常结束时会打印：

```text
preview_frames=<生成图片数>
preview_output=<预览目录>
```

### 7.2 指定抽帧间隔和输出目录（容器内）

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

旧 LMDB 如果只包含腕部相机，会明确报错：

```text
LMDB is missing 'observations/images/top_camera/<帧号>'
```

脚本不会用纯 Isaac 图或纯 GS 图代替缺失的正式双相机合成图。

## 8. 当前实现边界

- Stage 9 重建资产和坐标对齐已接入当前 YAML；日常检查通过 `--interactive` 完成。
- Stage 10 的 fixed-base ViperX、TCP、gripper、双相机、GS 合成遮挡和单条专家闭环已经进入当前入口。
- `--collect-one`、`--replay` 和双相机抽帧预览已实现。
- `--collect N` 正式多条采集尚未实现，不能把 `--collect-one` 重复运行自动视作现有批量接口。
- `side_camera` 因外参为空会被自动跳过；当前正式相机是 `wrist_camera` 与 `top_camera`。
- `planner.use_background_point_cloud` 当前为 `false`。MPLib 不使用噪声较大的 reconstructed point cloud 做路径规划；Isaac 中的机器人、任务块和场景物理碰撞是独立层。

### 8.1 当前已知的非阻断提示

下列消息来自项目中未被当前 ViperX 入口使用的旧模块或可选依赖。如果后面仍出现 `viperx_home=READY` 和 `viperx_runtime=READY`，它们不是本入口失败的依据：

```text
ModuleNotFoundError: No module named 'real2sim2real.controllers.stacking_curobo'
Field "model_path" has conflict with protected namespace "model_"
Pinocchio is not installed, some functions will not be available.
```

真正需要处理的是入口自身随后抛出的异常、`viperx_collection_success=False`，或缺少 `viperx_runtime=READY`。

## 9. 结束容器与恢复宿主机权限

先退出容器 shell：

```bash
exit
```

然后在宿主机停止容器：

```bash
docker stop re3sim
```

由于创建命令使用 `--rm`，容器会被删除，bind mount 数据会保留。

恢复数据目录所有权：

```bash
sudo chown -R "$(id -u):$(id -g)" \
  ~/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/calibration/data
```

收回 X11 授权：

```bash
xhost -local:docker
```
