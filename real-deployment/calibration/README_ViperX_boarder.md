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
  -> 每个中心独立生成 10 个样本
  -> 每个样本随机选择 3 个关节并在逐关节小范围内扰动
  -> 连接前预检 staging、anchor、全部六维 q_target 和八电机 raw target
  -> 使用 FK 记录每个 q_target 对应的 T_base_hand
  -> 输入 CAPTURE 后 connect 原地保持
  -> prepare_for_motion 到固定 staging
  -> 移动到 anchor
  -> ViperXAdapter 逐目标运动，到位后额外等待 1.5 s
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

当前状态：ViperX shooting 已使用 Stage 4 mapping、`ViperXModel` 和 `ViperXAdapter`，不再包含 `.pos`/rad 占位转换。腕部相机连接、同帧 RGB PNG/depth NPY 保存、相机参数选择、staging/anchor 候选姿态和旧计划的基本执行链已在实验室确认。旧 Cartesian/IK 计划因频繁出现大关节变化而改为 q center + 每样本随机三关节小扰动；当前默认每中心 10 张，逐关节范围为 `[2°, 2°, 2°, 4°, 3°, 4°]`，`STAGE5_LIVE_ACCEPTED=True`，但仓库 `Q_CENTERS_RAD` 仍只有 anchor。新计划的小样、完整中心列表、标定板检测和正式数据尚未验收。

当前固定 ChArUco 参数为 5×5、`DICT_6X6_250`、整板 180×180 mm、方格 36 mm、marker 27 mm。生成文件已经打印，但纸面尺寸测量与实际相机检测仍需现场验收。

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

当前状态：`hand_in_eye_calib_ViperX.py` 的机器人边界迁移已经写完。它从数据集中的 mapping snapshot 恢复 joint order、URDF、base/end frame，读取六维 `q_rad`，使用 `ViperXModel` FK 和统一的 `vx300s/ee_gripper_link` hand frame，并使用 5×5、36/27 mm ChArUco 参数。尚未用正式 ViperX shooting 数据生成并验收 `cam_to_hand_pose.npy`；通用求解器的失败样本筛选、有效样本下限和结果稳定性检查仍需补充。

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

当前状态：`get_marker2base_aruco_ViperX.py` 已完成单文件必要迁移：正式机器人状态读取 shooting 保存的六维 joints，从数据集 mapping snapshot 加载完整 ViperX URDF 和统一 hand frame，使用实际 RGB 畸变及 5×5、36/27 mm ChArUco 参数；多帧合成已改为平移均值和 SO(3) 旋转均值，不再使用 Panda FK、零畸变或 4×4 逐元素平均。代码尚未完成用户复核、离线测试和正式数据验收。

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
  adapter 默认 0.8 rad/s 实机体感与主动安全介入补验
  -> shooting q centers 与扰动余量复核
  -> shooting 小样
  -> 场景照片采集
  -> 正式 hand-eye 数据采集
  -> 数据完整性和备份检查

回家离线：
  hand_in_eye_calib_ViperX.py
  -> cam_to_hand_pose.npy
  -> get_marker2base_aruco_ViperX.py
  -> marker_2_base.npy

家里 Docker：
  reconstruct.py
  -> Re3Sim/Isaac Sim 场景对齐
```

只要实验室阶段保存了完整、同步且坐标关系没有变化的数据，hand-eye 求解、marker-to-base、场景重建和仿真对齐都不需要再次连接真实机械臂。
