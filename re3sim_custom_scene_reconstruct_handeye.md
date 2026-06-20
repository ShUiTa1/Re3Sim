# Re3Sim Customized Scene：`reconstruct.py` 与 `hand_in_eye_shooting.ipynb` 使用说明

这份说明只针对 README 里最简略的两步：

```bash
python reconstruct.py -i /path/to/input
```

以及：

```text
real-deployment/calibration/hand_in_eye_shooting.ipynb
```

核心理解：

```text
reconstruct.py
= 用真实场景照片重建“环境资产”
= 生成 3DGS 背景、OpenMVS mesh、USD、GS/mesh 到 marker 的变换

hand_in_eye_shooting.ipynb
= 用真实机器人 + RealSense 采集“手眼标定数据”
= 保存 rgb/depth/机器人末端 pose/关节角
= 后续再交给 hand_in_eye_calib.py 和 get_marker2base_aruco.py 计算相机外参、marker_2_base.npy
```

---

## 0. Customized scene 的总体链路

最终目标是得到类似 predefined scene 的资产结构：

```text
data/
├── align/
│   └── marker_2_base.npy
├── gs-data/
│   └── <scene_name>/
│       ├── gs_to_marker.npy
│       └── point_cloud/iteration_xxxxx/point_cloud.ply
├── usd/
│   └── <scene_name>/
│       ├── scene_dense_mesh_refine_texture.usd
│       └── mesh_to_marker.npy
├── items/
│   └── your_object.usd
└── urdfs/
    └── your_robot/
```

其中：

```text
reconstruct.py 负责：
images/ → colmap/ → gs/0/ → mvs/ → usd + gs_to_marker.npy + mesh_to_marker.npy

hand_in_eye_shooting.ipynb + 后续 calibration scripts 负责：
真实机器人多姿态采集 → camera/hand-eye calibration → marker_2_base.npy
```

所以这两步不是重复的：

```text
reconstruct.py：解决“场景在哪里”
hand-eye calibration：解决“真实机器人 base 在这个场景里在哪里”
```

---

# Part 1：`reconstruct.py`

## 1.1 你需要准备什么输入？

最小输入目录必须是：

```text
/path/to/input/
└── images/
    ├── 0001.png
    ├── 0002.png
    ├── ...
    └── N.png
```

代码里明确检查：

```python
image_dir = os.path.join(input_dir, "images")
assert os.path.exists(image_dir)
```

所以 `/path/to/input/images` 是硬要求。

README 里写的：

```text
/path/to/input/align_image.png
```

在你当前这个 `reconstruct.py` 里没有被直接使用。脚本虽然定义了 `--calibration_image` 和 `--depth_image` 参数，但主流程没有调用它们。也就是说：当前版本真正跑重建时主要依赖 `images/`。

不过，因为后面要计算 `gs_to_marker.npy` 和 `mesh_to_marker.npy`，你的 `images/` 里最好能看到 ArUco marker。否则 `compute_transform_to_marker_aruco.py` 很可能无法算出 marker 对齐关系。

---

## 1.2 照片应该怎么拍？

`images/` 里的照片用于 COLMAP、3DGS、OpenMVS，所以要求是多视角、有重叠、有纹理、能覆盖场景。

建议：

```text
1. 固定场景，不要移动桌子、marker、背景物体
2. 拍 50-200 张比较稳；小场景可先 50-80 张试跑
3. 相邻照片视角要有明显重叠
4. 多拍桌面边缘、背景、目标区域、marker
5. 光照尽量稳定，不要强反光、曝光跳变
6. 不要把机器人手臂频繁拍进去，除非它是固定背景
7. marker 最好在多张图里清晰可见
```

如果你只想先验证 pipeline，可以先拍一个很小的桌面场景：

```text
桌子 + ArUco marker + 一个固定篮子/盒子 + 背景纹理
```

不要一开始就搞复杂机械臂和多个物体。

---

## 1.3 推荐目录组织

host 上：

```text
~/Projects/Re3Sim_Host/custom_scene_001/
└── input/
    └── images/
        ├── 0001.png
        ├── 0002.png
        └── ...
```

container 里对应：

```text
/root/resources/custom_scene_001/input/images/
```

运行：

```bash
cd /isaac-sim/src
conda activate py10

python reconstruct.py \
  -i /root/resources/custom_scene_001/input \
  -t
```

`-t` 表示最后给 OpenMVS mesh 做 texture，生成 textured mesh。predefined scene 里用的也是 `scene_dense_mesh_refine_texture.usd`，所以建议加 `-t`。

---

## 1.4 `reconstruct.py` 的实际流程

这个脚本不是一个单一算法，而是串联了五个阶段：

```text
Stage 1: COLMAP sparse reconstruction
Stage 2: 3D Gaussian Splatting training
Stage 3: COLMAP dense undistortion
Stage 4: OpenMVS dense reconstruction + mesh refinement + optional texture
Stage 5: marker alignment + USD conversion
```

---

## 1.5 Stage 1：COLMAP 稀疏重建

对应函数：

```python
run_colmap(image_dir, colmap_dir)
```

内部流程：

```text
extract_features()
→ download_vocab_tree()
→ feature_matching()
→ pycolmap.incremental_mapping()
→ reconstruction.write_text()
```

### 输入

```text
input/images/
```

### 输出

```text
input/colmap/
├── database.db
├── vocab_tree.bin
└── sparse/
    ├── 0/
    └── text/
```

### 每一步含义

`extract_features()`：

```text
用 pycolmap 提取 SIFT 特征
camera_model 默认是 PINHOLE
```

`download_vocab_tree()`：

```text
根据图片数量下载 COLMAP vocabulary tree
<1000 张用 flickr100K_words32K
<10000 张用 flickr100K_words256K
更多用 words1M
```

`feature_matching()`：

```text
默认 sequential matching
适合视频抽帧或按顺序拍摄的照片
```

`incremental_mapping()`：

```text
估计每张图的相机位姿
生成 sparse reconstruction
```

### 这一阶段失败通常说明

```text
图片重叠不够
场景纹理太少
照片顺序太乱但用了 sequential matching
曝光/模糊严重
marker/桌面反光严重
```

如果图片不是连续视频抽帧，而是乱序拍摄，可能要把 matching 改成 exhaustive。

---

## 1.6 Stage 2：训练 3DGS

代码调用：

```bash
python train.py \
  -s <input>/colmap \
  -i <input>/images \
  -m <input>/gs/0 \
  --random_background
```

运行目录：

```text
gaussian_splatting/
```

### 输入

```text
input/images/
input/colmap/
```

### 输出

```text
input/gs/0/
├── point_cloud/
│   └── iteration_xxxxx/
│       └── point_cloud.ply
└── ...
```

### 作用

这一步生成 Re3Sim 的真实视觉背景。

后面 simulator 里相机看到的背景，不是纯 Isaac Sim 渲染，而是通过这个 3DGS 模型渲染出来，再和机器人/物体前景混合。

### 注意

如果 `diff-gaussian-rasterization` 或 `simple-knn` 没装好，这一步会挂。你之前 commit 的 `re3sim_yuzheng:openmvs` 主要就是为了把这些依赖固定下来。

---

## 1.7 Stage 3：COLMAP dense undistort

代码调用：

```bash
colmap image_undistorter \
  --image_path input/images \
  --input_path input/colmap/sparse/0 \
  --output_path input/colmap/dense \
  --output_type COLMAP
```

### 输入

```text
input/images/
input/colmap/sparse/0/
```

### 输出

```text
input/colmap/dense/
├── images/
├── sparse/
└── ...
```

### 作用

这一步把 COLMAP 稀疏重建结果整理成 OpenMVS 能读取的 dense 格式。

---

## 1.8 Stage 4：OpenMVS 重建 dense mesh

输出目录：

```text
input/mvs/
```

代码依次调用：

```bash
InterfaceCOLMAP
DensifyPointCloud
ReconstructMesh
RefineMesh
TextureMesh  # 只有加 -t 才跑
```

### 主要输出

不加 `-t`：

```text
input/mvs/
├── scene.mvs
├── scene_dense.mvs
├── scene_dense.ply
├── scene_dense_mesh.ply
└── scene_dense_mesh_refine.ply
```

加 `-t`：

```text
input/mvs/
├── scene_dense_mesh_refine_texture.mvs
└── scene_dense_mesh_refine_texture.ply
```

### 每一步含义

`InterfaceCOLMAP`：

```text
把 COLMAP dense 结果转成 OpenMVS 格式 scene.mvs
```

`DensifyPointCloud`：

```text
生成 dense point cloud
```

`ReconstructMesh`：

```text
从 dense point cloud 生成初始 mesh
```

`RefineMesh`：

```text
细化 mesh，输出 scene_dense_mesh_refine.ply
```

`TextureMesh`：

```text
给 refined mesh 贴图，输出 scene_dense_mesh_refine_texture.ply
```

### 这一阶段输出的 mesh 用来干嘛？

它主要给 Isaac Sim 做环境几何和碰撞。

也就是说：

```text
3DGS = 视觉背景
OpenMVS mesh / USD = 物理几何背景
```

---

## 1.9 Stage 5：marker 对齐

`reconstruct.py` 会自动调用：

```bash
python ../real-deployment/utils/compute_transform_to_marker_aruco.py \
  --data_type gaussian \
  --data_folder input/gs/0 \
  --headless
```

输出：

```text
input/gs/0/gs_to_marker.npy
```

然后再调用：

```bash
python ../real-deployment/utils/compute_transform_to_marker_aruco.py \
  --data_type openmvs \
  --data_folder input/mvs \
  --headless
```

输出/复制：

```text
input/mvs/mesh_to_marker.npy
```

### 这两个文件的作用

```text
gs_to_marker.npy
= 3DGS 坐标系 → marker 坐标系

mesh_to_marker.npy
= OpenMVS mesh 坐标系 → marker 坐标系
```

这两个文件保证：

```text
3DGS 背景
OpenMVS mesh
真实 marker
```

都能落在同一个坐标系下。

### 注意

这里还不是 robot base 对齐。

它只解决：

```text
GS / mesh → marker
```

不解决：

```text
marker → robot base
```

`marker_2_base.npy` 要靠 hand-eye / robot calibration 相关步骤生成。

---

## 1.10 Stage 6：PLY 转 USD

最后脚本调用：

```bash
python utils/usd/obj_to_usd.py \
  --obj_path <mvs_path> \
  --usd_dir input/mvs \
  --collision_approximation meshSimplification
```

如果你加了 `-t`，`mvs_path` 是：

```text
input/mvs/scene_dense_mesh_refine_texture.ply
```

否则是：

```text
input/mvs/scene_dense_mesh_refine.ply
```

输出大概率是：

```text
input/mvs/scene_dense_mesh_refine_texture.usd
```

或：

```text
input/mvs/scene_dense_mesh_refine.usd
```

这个 USD 是之后 YAML 里 `background.usd_path` 要指向的文件。

---

## 1.11 `progress.json` 是干嘛的？

脚本会在 `input/progress.json` 里记录哪些阶段已经跑完：

```json
{
  "colmap": true,
  "gaussian": true,
  "mvs_colmap_dense": true,
  "mvs_colmap_dense_mvs": true
}
```

如果你重复运行脚本，它会跳过已经完成的阶段。

如果你想强制重跑某一步，最直接的方法是删除：

```bash
rm /root/resources/custom_scene_001/input/progress.json
```

或者删除对应输出目录：

```bash
rm -rf /root/resources/custom_scene_001/input/colmap
rm -rf /root/resources/custom_scene_001/input/gs
rm -rf /root/resources/custom_scene_001/input/mvs
```

---

## 1.12 `reconstruct.py` 跑完后你应该检查什么？

```bash
ls /root/resources/custom_scene_001/input/colmap/sparse/0
ls /root/resources/custom_scene_001/input/gs/0
ls /root/resources/custom_scene_001/input/mvs
```

你应该至少看到：

```text
input/gs/0/
├── gs_to_marker.npy
└── point_cloud/...

input/mvs/
├── mesh_to_marker.npy
├── scene_dense_mesh_refine.ply
├── scene_dense_mesh_refine_texture.ply       # 如果加 -t
└── scene_dense_mesh_refine_texture.usd       # 如果加 -t 后 obj_to_usd 成功
```

之后可以整理成：

```text
data/
├── gs-data/custom_scene/
│   ├── gs_to_marker.npy
│   └── point_cloud/...
└── usd/custom_scene/
    ├── scene_dense_mesh_refine_texture.usd
    └── mesh_to_marker.npy
```

或者不整理，直接在 YAML 里写绝对路径也可以。

---

# Part 2：`hand_in_eye_shooting.ipynb`

## 2.1 这个 notebook 是干嘛的？

它不是直接算最终标定结果，而是采集手眼标定数据。

它会控制真实 Franka/Panda 机器人移动到多个随机扰动姿态，然后用 RealSense 拍 RGB/depth，同时保存机器人末端 pose 和关节角。

输出数据用于后续：

```bash
python real-deployment/calibration/hand_in_eye_calib.py --data_root <calibrate_folder>
python real-deployment/utils/get_marker2base_aruco.py --data_root <calibrate_folder>
```

所以完整关系是：

```text
hand_in_eye_shooting.ipynb
= 采集 calibration dataset

hand_in_eye_calib.py
= 根据 rgb/depth/poses 求 hand-eye calibration

get_marker2base_aruco.py
= 根据 ArUco marker 结果求 marker_2_base.npy
```

---

## 2.2 notebook 的输入条件

它默认假设你有：

```text
Franka FR3 / Panda 真实机械臂
RealSense 相机
panda_py / libfranka 可连接机器人
相机能看到 ArUco marker / calibration target
```

notebook 一开始写：

```python
hostname = '172.16.0.2'
username = 'TQSH'
password = 'tqshanghai'
base_dir = '../../data/hand_in_eye8'
```

你需要改：

```python
hostname
base_dir
```

其中 `base_dir` 是保存标定数据的目录。

---

## 2.3 Connect 部分

代码：

```python
import panda_py
from panda_py import libfranka

panda = panda_py.Panda(hostname)
panda.enable_logging(int(1e2))
gripper = libfranka.Gripper(hostname)
```

作用：

```text
连接真实 Franka/Panda
打开 robot state logging
连接 gripper
```

`panda.enable_logging(int(1e2))` 表示以一定频率记录机器人状态，后面会从 log 里取关节角：

```python
last_qpos = panda.get_log()["q"][-1]
```

---

## 2.4 `generate_and_move_to_pose()`

函数：

```python
def generate_and_move_to_pose(
    init_pose,
    roll, pitch, yaw,
    z_add, x_add, y_add,
    max_roll_deviation,
    max_pitch_deviation,
    max_yaw_deviation
):
```

作用：

```text
以 init_pose 为基准
叠加 roll/pitch/yaw 的随机扰动
叠加 x/y/z 平移偏移
让机器人移动到这个新 pose
返回实际目标 pose
```

具体过程：

```text
1. roll/pitch/yaw 加随机扰动
2. 用 scipy Rotation 变成旋转矩阵
3. 新旋转 = init_pose 原旋转 × 扰动旋转
4. 新位置 = init_pose 原位置 + x/y/z offset
5. panda.move_to_pose(pose)
6. 返回 pose
```

这个函数的目的不是执行任务，而是让相机从多个不同视角观察标定板/marker，从而让 hand-eye calibration 更稳定。

---

## 2.5 `save_pose()` 和 `save_joints()`

`save_pose()` 保存：

```text
base_dir/poses/pose_<frame_num>.npy
```

内容是机器人末端 pose，通常是 4x4 齐次变换矩阵。

`save_joints()` 保存：

```text
base_dir/joints/joints_<frame_num>.npy
```

内容是机器人当前关节角。

这两个数据是后续标定使用的 robot-side measurement。

---

## 2.6 RealSense 相机初始化

notebook 里：

```python
from realsense.realsense import Camera
from realsense.realsense import get_devices

device_serials = get_devices()
camera = Camera(device_serials[0], rgb_resolution, depth_resolution)
```

作用：

```text
枚举 RealSense 设备
选择第一个设备
设置 RGB/depth 分辨率
```

默认分辨率：

```python
rgb_resolution = (640, 480)
depth_resolution = (640, 480)
```

---

## 2.7 `capture_images()`

这是 notebook 的核心采集函数：

```python
def capture_images(
    camera,
    delay_before_shooting,
    start_frame,
    picture_nums,
    base_dir,
    init_pose,
    roll, pitch, yaw,
    z_add, x_add, y_add,
    max_roll_deviation,
    max_pitch_deviation,
    max_yaw_deviation
):
```

它做的事情：

```text
1. camera.start()
2. 读取 RGB/depth intrinsics
3. 保存 rgb_intrinsics.npz 和 depth_intrinsics.npz
4. 丢掉第一帧并等待相机稳定
5. 循环 picture_nums 次：
   5.1 随机生成并移动到一个新 pose
   5.2 拍 RGB 和 depth
   5.3 保存 RGB PNG
   5.4 保存 depth NPY
   5.5 保存 robot EE pose
   5.6 保存 robot joint qpos
6. panda.move_to_start()
7. camera.stop()
```

输出目录：

```text
base_dir/
├── rgb/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── depth/
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── poses/
│   ├── pose_0.npy
│   ├── pose_1.npy
│   └── ...
├── joints/
│   ├── joints_0.npy
│   ├── joints_1.npy
│   └── ...
├── rgb_intrinsics.npz
└── depth_intrinsics.npz
```

这就是后续 `hand_in_eye_calib.py --data_root <base_dir>` 需要读取的数据。

---

## 2.8 `image_configs` 是什么？

notebook 最后定义了多个 config：

```python
image_configs = [
    {...},
    {...},
    {...},
    {...}
]
```

每个 config 定义一组相机/末端姿态采集区域：

```text
init_pose：初始 pose
roll/pitch/yaw：基础姿态扰动方向
z_add/x_add/y_add：相对初始位姿的平移偏移
max_*_deviation：随机扰动幅度
```

然后：

```python
for i, config in enumerate(image_configs):
    capture_images(..., start_frame=10*i, picture_nums=10, ...)
```

默认是 4 组 config，每组 10 张，所以总共 40 组标定数据：

```text
rgb/0.png ... rgb/39.png
depth/0.npy ... depth/39.npy
poses/pose_0.npy ... pose_39.npy
joints/joints_0.npy ... joints_39.npy
```

---

## 2.9 notebook 里有一个明显需要注意的点

一开始定义：

```python
base_dir = '../../data/hand_in_eye8'
```

但是 `image_configs` 里面每个 config 又写了：

```python
'base_dir': '../hand_in_eye2'
```

实际调用 `capture_images()` 时传的是外层的 `base_dir`，不是 `config['base_dir']`：

```python
capture_images(..., base_dir, config['init_pose'], ...)
```

所以 `image_configs` 里的 `'base_dir': '../hand_in_eye2'` 实际没有用上。

你应该只改最上面的：

```python
base_dir = '/root/resources/custom_scene_001/calibrate'
```

并保证：

```text
/root/resources/custom_scene_001/calibrate/
├── rgb/
├── depth/
├── poses/
└── joints/
```

---

## 2.10 采集时你要保证什么？

最关键：

```text
每张 RGB 图里都应该能清楚看到 ArUco marker 或标定目标
机器人每次运动不能碰撞桌面/相机/线缆
相机不能松动
marker 不能移动
机器人 base 和 marker 之间的真实关系不能变
```

如果相机是 eye-in-hand：

```text
RealSense 固定在末端
机器人运动时相机跟着动
```

这就符合 hand-in-eye calibration 的名字。

如果相机是固定在外部，那这套 notebook 逻辑就不完全适合，要改成 eye-to-hand calibration。

---

## 2.11 和 `reconstruct.py` 的坐标关系

`reconstruct.py` 生成：

```text
gs_to_marker.npy
mesh_to_marker.npy
```

意思是：

```text
3DGS / mesh → marker
```

hand-eye / marker calibration 最终要生成：

```text
marker_2_base.npy
```

意思是：

```text
marker → robot base
```

最终 simulator 需要把这些连起来：

```text
3DGS coordinate
    ↓ gs_to_marker.npy
marker coordinate
    ↓ marker_2_base.npy
robot base coordinate

mesh coordinate
    ↓ mesh_to_marker.npy
marker coordinate
    ↓ marker_2_base.npy
robot base coordinate
```

这就是为什么两个步骤都要做。

---

# Part 3：你实际该怎么跑

## 3.1 启动 container

建议 run 时挂载源码和资源：

```bash
docker run --name re3sim --entrypoint bash -itd \
  --runtime=nvidia --gpus='"device=0"' \
  -e "ACCEPT_EULA=Y" \
  --rm --network=bridge --shm-size="28g" \
  -e "PRIVACY_CONSENT=Y" \
  -v ~/Projects/Re3Sim_Host:/root/resources:rw \
  -v ~/Projects/Re3Sim/re3sim:/root/dev/real2sim2real:rw \
  re3sim_yuzheng:openmvs
```

进入：

```bash
docker exec -it re3sim bash
conda activate py10
cd /isaac-sim/src
```

---

## 3.2 准备 reconstruction 输入

host：

```bash
mkdir -p ~/Projects/Re3Sim_Host/custom_scene_001/input/images
```

把图片放进去。

container 检查：

```bash
ls /root/resources/custom_scene_001/input/images
```

---

## 3.3 跑重建

```bash
cd /isaac-sim/src
conda activate py10

python reconstruct.py \
  -i /root/resources/custom_scene_001/input \
  -t
```

跑完检查：

```bash
ls /root/resources/custom_scene_001/input/gs/0
ls /root/resources/custom_scene_001/input/mvs
```

---

## 3.4 采集 hand-eye 数据

在 notebook 里改：

```python
hostname = '<your_franka_ip>'
base_dir = '/root/resources/custom_scene_001/calibrate'
```

然后运行 notebook。

采集完成后应有：

```text
/root/resources/custom_scene_001/calibrate/
├── rgb/
├── depth/
├── poses/
├── joints/
├── rgb_intrinsics.npz
└── depth_intrinsics.npz
```

---

## 3.5 运行后续标定脚本

README 后面写的命令是：

```bash
python real-deployment/calibration/hand_in_eye_calib.py \
  --data_root /root/resources/custom_scene_001/calibrate

python real-deployment/utils/get_marker2base_aruco.py \
  --data_root /root/resources/custom_scene_001/calibrate
```

期望得到：

```text
marker_2_base.npy
```

这个文件之后放到：

```text
data/align/marker_2_base.npy
```

或者 YAML 里直接指向它。

---

# Part 4：最重要的检查清单

## reconstruct.py 跑完后

必须有：

```text
input/gs/0/gs_to_marker.npy
input/gs/0/point_cloud/...
input/mvs/mesh_to_marker.npy
input/mvs/scene_dense_mesh_refine_texture.usd
```

如果缺 `gs_to_marker.npy` 或 `mesh_to_marker.npy`：

```text
大概率是 marker 没被识别出来
或者 images 里 marker 不够清楚
```

如果缺 USD：

```text
大概率是 obj_to_usd.py 转换失败
或者 mvs_path 没生成
```

## hand-eye 跑完后

必须有：

```text
calibrate/rgb/*.png
calibrate/depth/*.npy
calibrate/poses/pose_*.npy
calibrate/joints/joints_*.npy
calibrate/rgb_intrinsics.npz
calibrate/depth_intrinsics.npz
```

后续必须生成：

```text
marker_2_base.npy
```

如果 marker_2_base 算不出来：

```text
检查 RGB 图里 marker 是否清楚
检查相机内参是否保存
检查 poses 是否和图片编号一一对应
检查 marker 是否在采集过程中移动过
```

---

# Part 5：一句话总结

`reconstruct.py` 是“从多视角图片生成仿真场景资产”的脚本，输出 3DGS、mesh、USD、`gs_to_marker.npy`、`mesh_to_marker.npy`；`hand_in_eye_shooting.ipynb` 是“用真实机器人和 RealSense 采集手眼标定数据”的 notebook，输出 RGB、depth、末端 pose、关节角和相机内参，后续再由 `hand_in_eye_calib.py` 和 `get_marker2base_aruco.py` 生成 `marker_2_base.npy`。前者解决场景重建，后者解决机器人 base 和 marker 的坐标对齐。
