# ViperX 标定任务与运行环境边界

本文说明 ViperX hand-in-eye、marker-to-base、场景重建和 Re3Sim 对齐分别在哪里运行。核心原则是：实验室只完成依赖真实设备和固定物理场景的工作；数据完整采集后，标定求解、场景重建和仿真对齐都可以离线完成。

## 1. 总体边界

```text
实验室 host
  ├── ViperX calibration、mapping 和 adapter 实机验证
  ├── customized scene 多视角照片采集
  └── ViperX + 腕部 RealSense hand-eye 数据采集
            │
            │ 完整数据集和配置文件
            ▼
离线工作环境
  ├── hand-in-eye 求解 -> cam_to_hand_pose.npy
  └── marker-to-base 求解 -> marker_2_base.npy
            │
            ▼
Re3Sim Docker
  ├── reconstruct.py -> gs_to_marker.npy / mesh_to_marker.npy / scene assets
  └── Isaac Sim / Re3Sim 场景对齐与仿真验证
```

实验室不需要为了 hand-eye 数据采集启动 Re3Sim/Isaac Sim Docker。当前 Re3Sim Docker 主要服务场景重建、Isaac Sim 和最终仿真，不负责直接连接 ViperX 或 RealSense。

## 2. 环境分工总表

| 工作 | 推荐地点与环境 | 需要 ViperX | 需要 RealSense | 需要实验室场景 | 需要 Re3Sim Docker |
|---|---|---:|---:|---:|---:|
| LeRobot calibration | 实验室 host | 是 | 否 | 否 | 否 |
| URDF mapping 生成与 live GUI 验证 | 实验室 host | 是 | 否 | 否 | 否 |
| Adapter 实机验收 | 实验室 host | 是 | 否 | 否 | 否 |
| 场景重建照片采集 | 实验室 | 否 | 使用拍摄设备 | 是 | 否 |
| Hand-eye 数据采集 | 实验室 host | 是 | 是 | 是 | 否 |
| `cam_to_hand_pose.npy` 求解 | 任意离线主机 | 否 | 否 | 否 | 否 |
| `marker_2_base.npy` 求解 | 任意离线主机 | 否 | 否 | 否 | 否 |
| `reconstruct.py` | Re3Sim Docker | 否 | 否 | 否，使用已采照片 | 是 |
| Re3Sim/Isaac Sim 对齐 | Re3Sim Docker | 否 | 否 | 否 | 是 |

离线求解也可以放进 Docker，但算法本身不依赖 Isaac Sim。更简单的方式是在包含 NumPy、OpenCV ArUco/ChArUco、Robotics Toolbox 和 ViperX 模型的轻量 Python 环境中运行。

## 3. 实验室必须完成的内容

实验室部署目录：

```text
/home/yuzzhu/Projects/Re3Sim_ViperX
```

大型采集数据应写入本地数据盘，例如：

```text
/data/yuzzhu/Re3Sim_ViperX/data/viperx_hand_eye
```

不要把 RGB、depth 和场景重建照片长期写入容量有限的 NFS home。`/data/yuzzhu` 没有自动备份，离开实验室前必须把不可替代的数据复制到有备份的允许位置。

### 3.1 ViperX 硬件准备

实验室内完成：

- LeRobot follower calibration。
- `create_viperx_urdf_mapping.py` 的 mapping 生成。
- `validate_viperx_urdf_mapping.py` 的实机/PyBullet 多姿态检查。
- `validate_viperx_adapter.py` 的 adapter 实机运动与人工介入验收。

主要产物：

```text
Re3Sim/real-deployment/configs/lerobot_calibration/robots/viperx/left_follower.json
Re3Sim/real-deployment/configs/viperx_urdf_mapping.json
```

这些步骤依赖真实 ViperX 串口，不属于 Docker 工作。

### 3.2 Customized scene 照片采集

在实验室拍摄 `reconstruct.py` 所需的多视角照片。这里只负责采集，不要求在实验室运行重建。

采集期间必须保持以下物理关系不变：

- 场景中的 marker 不移动。
- 机器人 base 不移动。
- 桌面、固定背景和主要场景结构不移动。
- 重建照片中的 marker 与 hand-eye/marker-to-base 数据使用同一个坐标基准。

可以使用 iPhone 主摄拍摄，并转成按顺序命名的小写 JPEG/PNG。实验桌后方不可进入时不必强求 360°：从左前、正前、右前形成 U 形/半环绕，再补充较高俯视角，保证相邻照片约 70–80% 重叠并覆盖任务平台和真实相机常见视角。固定背景可以保留；待抓取或之后会移动的物体应先移走。hand-eye 数据与 reconstruction 照片必须分目录保存。

照片可以带回其他机器，再放入 Re3Sim Docker 执行 `reconstruct.py`。

### 3.3 Hand-eye 数据采集

入口：

```text
Re3Sim/real-deployment/calibration/hand_in_eye_shooting_ViperX.ipynb
```

该 notebook 只负责：

```text
Stage 4 mapping + ViperXModel
  -> 读取人工确认的 Q_CENTERS_RAD
  -> 当前配置每个中心独立生成 5 个样本
  -> 当前配置每个样本随机选择 5 个关节并在逐关节小范围内扰动
  -> 连接前预检 staging、anchor、全部六维 q_target 和八电机 raw target
  -> 使用 FK 记录每个 q_target 对应的 T_base_hand
  -> 输入 CAPTURE 后 connect 原地保持
  -> prepare_for_motion 到固定 staging
  -> 移动到 anchor
  -> ViperXAdapter 逐目标运动，当前配置到位后额外等待 3.0 s
  -> 腕部 RealSense 拍摄 RGB/depth
  -> 读取同一静止姿态的六主关节 raw、q_rad、时间戳和 FK
  -> 保存逐帧标定数据
  -> 正常完成时返回 mapping home，再 stop/hold/release
```

它不负责：

- 求解 `cam_to_hand_pose.npy`。
- 求解 `marker_2_base.npy`。
- 运行场景重建。
- 启动 Isaac Sim。
- 把 LeRobot 归一化 `.pos` 当作弧度。

正式采集时，相机必须刚性固定在腕部，且采集完成前不能重新安装或改变安装姿态。当前采集计划不使用 IK；q/raw/FK 数值预检不等于碰撞、线缆或相机视野检查。每个 q center 必须是 inspect 中人工确认的安全姿态，并为随机扰动保留足够余量。

当前状态：Stage 6 的 joint-space/FK、八电机运动、相机与数据保存链路已通过实机验收。ViperX shooting 使用 Stage 4 mapping、`ViperXModel` 和 `ViperXAdapter`，不包含 `.pos`/rad 占位转换；当前 notebook/data 为 11 个现场确认的 q center、每中心 5 张、每样本随机五关节小扰动，逐关节范围为 `[3°, 3°, 3°, 5°, 3°, 4°]`，到位后等待 `3.0 s`。

当前固定 ChArUco 参数为 5×5、`DICT_6X6_250`、整板 180×180 mm、方格 36 mm、marker 27 mm。该实体板已经用于本次正式采集。

下一轮实验室重采的目标不是在原中心附近简单增加重复照片，而是使用 `inspect_viperx_pose.py` 重新选择安全且方向覆盖更丰富的 q centers：在标定板完整清晰的前提下，覆盖正面、左右倾斜、上下倾斜和适量绕相机光轴旋转，并兼顾不同位置和距离。新数据必须保存到独立目录并完整备份，不能先覆盖当前数据再比较。

### 3.4 固定顶部 RealSense 外参标定

入口：

```text
Re3Sim/real-deployment/utils/calibrate_fixed_realsense_to_base.py
```

这一步直接在实验室宿主机上、使用前面运行 RealSense 脚本的 Python 环境执行；不需要 Docker，也不需要连接或移动 ViperX。标定前要求：

- 顶部相机、marker 和机器人 base 均已固定，三者关系不再变化。
- `calibration/data/viperx_hand_eye/marker_2_base.npy` 已存在，且对应当前 marker/base 关系。
- 顶部相机能清晰、完整地看到同一块 5×5 ChArUco 板。

从 `real-deployment` 目录运行。只接入一台 RealSense 时：

```bash
python utils/calibrate_fixed_realsense_to_base.py
```

同时接入多台 RealSense 时，必须指定顶部相机序列号：

```bash
python utils/calibrate_fixed_realsense_to_base.py --serial <TOP_CAMERA_SERIAL>
```

不指定 `--serial` 时，如果脚本检测到多台相机，会列出可用序列号并退出。脚本默认预热 30 帧，从 RealSense 直接读取当前 640×480@30 RGB 流的内参和畸变，检测一帧有效 ChArUco，然后计算：

```text
T_base_camera = T_base_marker @ inverse(T_camera_marker)
```

默认输出到 `calibration/data/viperx_hand_eye/`：

```text
top_camera_to_base.npy       # T_base_camera，将相机光学坐标系中的点变换到 ViperX base
top_camera_intrinsics.npz    # RGB 内参、畸变和分辨率
top_camera_preview.png       # ChArUco 检测与坐标轴叠加预览
```

先检查 `top_camera_preview.png`：画面应清晰，板子应完整，坐标轴应贴在板上而不是悬空或明显翻转。该预览用于排除检测错误，不是独立的外参精度真值验证。

脚本结束时会打印可直接写入采集配置的 `camera_params`。在 `collect_data_viperx.yaml` 中填入：

```yaml
- name: top_camera
  parent_frame: base
  extrinsic: /root/work/data/viperx_hand_eye/top_camera_to_base.npy
  camera_params: [fx, fy, cx, cy, width, height]  # 替换为脚本实际打印的 6 个值
```

宿主机上的 `real-deployment/calibration/data` 在 Re3Sim Docker 中挂载为 `/root/work/data`，所以 YAML 使用上面的容器路径。如果顶部相机、marker 或机器人 base 中任何一个移动，或采集分辨率改变，必须重新运行标定。

## 4. 回家后可以离线完成的内容

只要实验室数据完整，后续求解不再需要连接 ViperX、RealSense 或实验室网络。

### 4.1 Hand-in-eye 求解

入口：

```text
Re3Sim/real-deployment/calibration/hand_in_eye_calib_ViperX.py
```

通用算法：

```text
Re3Sim/real-deployment/calibration/calibration/hand_in_eye.py
```

输入：

- 每帧 RGB 图像。
- RGB 相机内参和畸变参数。
- 与图像对应的六关节 `q_rad`。
- 与实体标定板一致的 ChArUco 字典、方格尺寸和 marker 尺寸。
- 已验收的 ViperX URDF 和 `ViperXModel`。

逐帧机器人链路：

```text
q_rad
  -> ViperXModel.fk(q_rad)
  -> T_base_hand
```

逐帧视觉链路：

```text
RGB + intrinsics + ChArUco board
  -> T_camera_target
```

多姿态求解：

```text
{T_base_hand, T_camera_target}
  -> cv2.calibrateHandEye(...)
  -> T_hand_camera
  -> cam_to_hand_pose.npy
```

`hand` 必须始终对应已经固定的 ViperX 末端 frame：

```text
vx300s/ee_gripper_link
```

`hand_in_eye_calib_ViperX.py` 负责 ViperX 数据和运动学边界；`calibration/hand_in_eye.py` 只负责通用 ChArUco 检测和 OpenCV hand-eye 求解，不应连接硬件或导入 LeRobot。

当前状态：`hand_in_eye_calib_ViperX.py` 的机器人边界和当前必要门控均已实现。它从数据集中的 mapping snapshot 恢复 joint order、URDF、base/end frame，读取六维 `q_rad`，使用 `ViperXModel` FK 和统一的 `vx300s/ee_gripper_link` hand frame，并使用 5×5、36/27 mm ChArUco 参数。通用求解器对 `read_data()` 的 RGB 图像使用 `COLOR_RGB2GRAY`，检测失败时同步跳过对应 RGB/机器人 pose，并从 manifest 恢复 center id；当前 `MIN_VALID_SAMPLES_PER_CENTER=1`。

当前 55 帧数据检测成功 54 帧；仅 frame `27` 失败，无中心被删除，最终 54 帧进入 Stage 7 并生成 `cam_to_hand_pose.npy`。最终仍需用方向覆盖更丰富的新数据检查激励、不同样本子集稳定性、旋转合法性和物理安装尺度。

### 4.2 Marker-to-base 求解

入口：

```text
Re3Sim/real-deployment/utils/get_marker2base_aruco_ViperX.py
```

输入：

- 每帧 RGB、相机内参和对应 `q_rad`。
- `cam_to_hand_pose.npy`。
- 与 hand-eye 完全相同的 ViperX FK 和末端 frame。
- 固定在实验室场景中的 marker/ChArUco 坐标系。

每帧计算：

```text
T_base_marker
  = T_base_hand
  @ T_hand_camera
  @ T_camera_marker
```

多帧 `T_base_marker` 应在同一结果附近聚类。正式实现需要分别对平移和 SO(3) 旋转做稳健统计，报告每帧相对最终结果的位置和旋转残差，然后保存：

```text
marker_2_base.npy
```

该脚本不负责移动机器人、读取串口、拍照、修改 mapping 或运行 Isaac Sim。

当前状态：`get_marker2base_aruco_ViperX.py` 已完成单文件必要迁移并用当前 Stage 7 结果跑通现有 55 帧数据：正式机器人状态读取 shooting 保存的六维 joints，从数据集 mapping snapshot 加载完整 ViperX URDF 和统一 hand frame，使用实际 RGB 畸变及 5×5、36/27 mm ChArUco 参数；Stage 8 的 `cv2.imread()` BGR 输入使用 `COLOR_BGR2GRAY`；多帧合成采用平移均值和 SO(3) 旋转均值。脚本会跳过检测失败帧，当前中心门槛为 1，并把全局、逐中心和逐帧残差打印并保存到 `marker_2_base_analysis.json`。

当前运行检测并使用 54/55，仅 frame 27 失败，无中心被删除，并生成 `marker_2_base.npy`。当前计算结果为：

- 平移残差 mean `14.203 mm`、p95 `23.889 mm`、max `27.501 mm`。
- 旋转残差 mean `2.037°`、p95 `3.212°`、max `3.612°`。
- EPFL-compatible `total_error=0.0076057`；XYZ 平移绝对误差均值 `[5.240, 11.421, 5.274] mm`；ZYX Euler 旋转绝对误差均值 `[1.142°, 0.836°, 1.182°]`。

这些数值表示样本内拟合或多帧结果相对当前均值的离散程度，不是真值误差。新数据需独立保存并与当前结果比较；如果没有显著改善且当前结果物理方向正常，可以选用当前 `.npy` 继续后续工作，但不能据此声称获得了真值精度。

### 4.3 场景重建

入口：

```text
Re3Sim/re3sim/scripts/reconstruct.py
```

该步骤推荐在家里的 Re3Sim Docker 中运行。它只消费实验室已经拍摄好的场景图片，负责生成：

```text
gs_to_marker.npy
mesh_to_marker.npy
3DGS point cloud
OpenMVS mesh
USD scene asset
```

它不需要 ViperX、LeRobot calibration、adapter 或 RealSense 实时连接，也不求解 `cam_to_hand_pose.npy` 或 `marker_2_base.npy`。

### 4.4 Re3Sim/Isaac Sim 对齐

Docker 中最终消费：

```text
gs_to_marker.npy
mesh_to_marker.npy
marker_2_base.npy
cam_to_hand_pose.npy
ViperX URDF/meshes
```

核心组合关系为：

```text
T_base_gs = T_base_marker @ T_marker_gs
T_base_mesh = T_base_marker @ T_marker_mesh
```

这一阶段只验证仿真场景、marker 和机器人 base 的空间对齐，不再访问实验室硬件。

## 5. 离开实验室前的数据交接检查

至少保存以下内容：

```text
viperx_hand_eye/
├── rgb/
├── depth/
├── poses/
├── joints/
├── raw/
├── metadata/
├── configs/
│   ├── <本次 LeRobot follower calibration JSON>
│   └── <本次 Stage 4 mapping JSON>
├── rgb_intrinsics.npz
├── depth_intrinsics.npz
└── capture_manifest.json
```

这是 shooting notebook 当前实际写出的 hand-eye 数据根目录。`reconstruct.py` 使用的场景多视角照片属于独立支线，应另存到其输入目录的 `images/` 下，不嵌套进上述 hand-eye 数据集。

当前 55 帧数据在家中仓库中的位置为：

```text
/home/kienzhu/Projects/Re3Sim_ViperX/Re3Sim/real-deployment/calibration/data/viperx_hand_eye
```

这是离开实验室后的本地归档位置，不改变实验室 notebook 使用的 `/data/yuzzhu/Re3Sim_ViperX/data/viperx_hand_eye`。

逐帧最低信息：

- 唯一 frame id。
- RGB 和 depth 文件名。
- 相机时间戳。
- 机器人状态读取时间戳。
- 六个主关节的 measured raw snapshot；这是当前 `get_obs()` 和逐帧 `raw_*.npy` 的实际数据契约。
- 六关节 URDF `q_rad`。
- `T_base_hand`。
- 当前 mapping 文件标识和末端 frame。
- 标定板检测是否成功可以在采集时记录，也可以回家后离线计算。

运动命令边界与保存格式不同：每次 `Goal_Position` 写入都包含六个主关节和两个 shadow 的八电机目标，但 notebook 当前逐帧只保存六个主关节的实测 raw。

离开实验室前检查：

1. RGB、depth、joint、raw 和 metadata 的 frame id 一一对应。
2. 随机打开若干 RGB，确认标定板清晰且没有严重模糊或遮挡。
3. 相机内参文件存在，分辨率与实际图像一致。
4. 确认 `configs/` 已复制本次 calibration JSON 和 mapping JSON，manifest 已记录相机、mapping、anchor、q centers、扰动参数和 target plan；board 参数另行核对并记录。本次使用的已验收 URDF 应在仓库或备份中可单独追溯，notebook 当前不会把 URDF 文件复制到数据目录。
5. 确认腕部相机在整个采集过程中没有松动。
6. 确认 marker、机器人 base 和场景在两类图片采集期间没有改变相对位置。
7. 把 `/data/yuzzhu` 中不可替代的数据复制到有备份的允许位置。

## 6. 推荐执行顺序

```text
实验室：
  使用 inspect 重新选择安全、清晰且末端方向多样的 q centers
  -> 更新 shooting 的 Q_CENTERS_RAD 和现场确认的采样参数
  -> 重采 hand-eye 数据并完整备份
  -> 场景照片采集与数据备份
  -> adapter 主动安全介入/重连可独立补验

回家离线：
  hand_in_eye_calib_ViperX.py
  -> cam_to_hand_pose.npy
  -> get_marker2base_aruco_ViperX.py
  -> marker_2_base.npy
  -> marker_2_base_analysis.json
  -> 与当前计算结果比较并选择用于后续工作的标定结果

家里 Docker：
  reconstruct.py
  -> Re3Sim/Isaac Sim 场景对齐
```

只要实验室阶段保存了完整、同步且坐标关系没有变化的数据，hand-eye 求解、marker-to-base、场景重建和仿真对齐都不需要再次连接真实机械臂。
