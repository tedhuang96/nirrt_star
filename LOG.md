# LOG

## 250608
1. Added `environment1.yml` for creating the conda environment `pngenv1` on Ubuntu 22.04 with CUDA 12.8, conda 25.3.1, Python 3.12.0, and PyTorch 2.7.1. Fixed the bug of `weights_only` argument in torch.load.
```
conda create -n pngenv1 python==3.12
conda activate pngenv1
pip install open3d
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python
```
Tested on Quick Test section in `README.md`. Haven't tested on training or evaluation yet.

## 240817
1. Removed `pip install -e .` from setup instructions in `README.md`. Added the required version of numpy for setup instructions in `README.md`. Tested.
2. Added ICRA paper link and updated citation in `README.md`.

## 240421
1. Fixed typo of `circle_radius_range` in `generate_random_world_env_2d.py`. Adjusted `env_configs/random_2d.yml` to make the 2D configurations still the same as the previous setup, but with no typos.

## 240415
1. Fixed `img_folder` bug in `visualize_data_samples_2d.py`.
2. nirrt_star v1.0.0 is released.

## 240229
1. The public repo nirrt_star is created.
2. Add links to demo GitHub repo and Google project website. 