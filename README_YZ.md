# Local building Notes

This README file offers detailed buidling steps and file fixing. It's a supplement guide following the official README file.

## Build Image

When building Docker image by following command:

```bash
docker build -t re3sim:1.0.0 .
```

The following error may appear

```bash
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels. Please accept or remove them before proceeding:
0.937     - https://repo.anaconda.com/pkgs/main
0.937     - https://repo.anaconda.com/pkgs/r

```

That's because inside the Docker build, Conda wants to download Python 3.10 from  Anaconda’s default channels, but those channels now require accepting  their Terms of Service. Since Docker build is **non-interactive**, it cannot ask you “Do you accept?”

To fix that, open the **Dockerfile** and add the following content before `conda creat` (which located in *Re3Sim/re3sim*)

```bash
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    
```

Then rebuild it:

```bash
docker build -t re3sim:1.0.0 .
```

## Run container 

``` bash 
docker run --name re3sim --entrypoint bash -itd --runtime=nvidia --gpus='"device=0"' -e "ACCEPT_EULA=Y" --rm --network=bridge --shm-size="28g" -e "PRIVACY_CONSENT=Y" \
    -v ~/Projects/Re3Sim_Host:/root/resources:rw \
    re3sim:1.0.0
```

Here u could change the mount address on your host machine by replacing `~/Projects/Re3Sim_Host` 

Meanwhile, remember to check your **memory** size limit and change `--shm-size="32g"` to suitable value that you re able to apply.



Now enter the container by command 

```bash
docker exec -it re3sim bash
```

Then install isaac lab V1.1.0 **in env `py10`**

Check your `env` by

```
conda env list
```

you would see sth like 

```
# conda environments:
#
# * -> active
# + -> frozen
base                 *   /root/miniconda
py10                     /root/miniconda/envs/py10


```

Isaac Lab version requires Python 3.10 + torch 2.2.2.

switch to `py10` by 

```
conda activate py10
```

## Install isaaclab

run the installation

```bash
# in docker
cd /root/IsaacLab
./isaaclab.sh --install none
```

### Fail Notes 

This step might fail with error :`/root/miniconda/envs/py10/bin/python: No module named pip`

diagnosis command `python -m pip --version` may output

`pip 21.2.1+nv1 from /root/IsaacLab/_isaac_sim/kit/python/lib/python3.10/site-packages/pip (python 3.10)`

if we temporarily clean python env path by using

`PYTHONPATH= /root/miniconda/envs/py10/bin/python -m pip --version`

and it gives output: 

`pip 26.1.1 from /root/miniconda/envs/py10/lib/python3.10/site-packages/pip (python 3.10)`

That means `py10` actually has its own `pip` module  but the normal shell environment is polluted by Isaac Sim's bundled Python path.

**Fix** 

replace the upper command by:

```bash
# in docker
cd /root/IsaacLab
PYTHONPATH= ./isaaclab.sh --install none
```



You might find dependency lack and could install them based on log. Here is the dependency version alignment used in my setup:

```text
Python: 3.10
Isaac Sim: 4.0.0
IsaacLab: v1.1.0
torch: 2.2.2+cu121
torchvision: 0.17.2
numpy: 1.26.4
protobuf: 4.25.9
gymnasium: 0.29.0
pyglet: 1.5.31
open3d: 0.19.0
mplib: 0.2.1
```

`CUDA` and `torch` versions followed the instructions of auther.

you could verify your installation by`PYTHONPATH= python -m pip check`  and by test importing process.

## Install CUDA 11.8 and `diff-gaussian-rasterization` and `simple-knn`

The guide in README 

```bash
./cuda_11.8.0_520.61.05_linux.run --silent --toolkit
pip install src/gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install src/gaussian_splatting/submodules/simple-knn/
```

requires file `cuda_11.8.0_520.61.05_linux.run` and does not offer correct file address in two installation command. First download it in container by:

```
cd /root
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run
```

Then install it by 

```
./cuda_11.8.0_520.61.05_linux.run --silent --toolkit
```

Check your installation 

```
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
nvcc --version
```

Then go to Re3Sim project root and install Gaussian Splatting CUDA extensions into py10

```bash
cd /root/dev/real2sim2real

## accelerate the compile.
PYTHONPATH= python -m pip install ninja

PYTHONPATH= python -m pip install --no-build-isolation gaussian_splatting/submodules/diff-gaussian-rasterization/
PYTHONPATH= python -m pip install --no-build-isolation gaussian_splatting/submodules/simple-knn/
```

Similarly using `PYTHONPATH` to make sure you're using the correct `pip`.

`--no-build-isolation` means that pip will not create a temporary isolated build environment such as `/tmp/pip-build-env-xxxx`. Instead, it builds the package directly using the current `py10` environment. In this way, `setup.py` can import the `torch` package that is already installed in the current `py10` environment.

## Install OpenMVS

Follow the official building guide in https://github.com/cdcseacave/openMVS/wiki/Building

or commands below.



```bash
apt-get update

apt-get install -y \
  git cmake autoconf autoconf-archive automake libtool bison gfortran pkg-config \
  libxi-dev libx11-dev libxft-dev libxtst-dev libxext-dev libxrandr-dev \
  libxinerama-dev libxcursor-dev xorg-dev libgl-dev libglu1-mesa-dev \
  nasm libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
  curl zip unzip tar
  

PYTHONPATH= python -m pip install "cmake>=3.24"
export PATH=/root/miniconda/envs/py10/bin:$PATH
hash -r

cd /root

git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh

export VCPKG_ROOT=/root/vcpkg
echo 'export VCPKG_ROOT=/root/vcpkg' >> /root/.bashrc

cd /root

git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git

cd /root/openMVS
mkdir -p make
cd make

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=/root/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DOpenMVS_USE_CUDA=OFF \
  -DOpenMVS_USE_PYTHON=OFF
  
cmake --build . -j4

export PATH=/root/openMVS/make/bin:$PATH
echo 'export PATH=/root/openMVS/make/bin:$PATH' >> /root/.bashrc
```



Here I've commit and save the up-to-now container as a new image re3sim_yuzheng:openmvs for convenience.

`docker save -o re3sim_yuzheng_openmvs.tar re3sim_yuzheng:openmvs`

you could also simply load this image 

`docker load -i re3sim_yuzheng_openmvs.tar`

and create a container from the loaded image by

```bash
docker run --name re3sim --entrypoint bash -itd --runtime=nvidia --gpus='"device=0"' -e "ACCEPT_EULA=Y" --rm --network=bridge --shm-size="28g" -e "PRIVACY_CONSENT=Y" \
    -v ~/Projects/Re3Sim_Host:/root/resources:rw \
    -v ~/Projects/Re3Sim/re3sim:/root/dev/real2sim2real:rw \
    re3sim_yuzheng:openmvs
```



## Reconfigure customized scene

