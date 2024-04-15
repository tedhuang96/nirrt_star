# NIRRT*

This is the implementation of Neural Informed RRT* (NIRRT*), which is the algorithm in our ICRA 2024 paper

### Neural Informed RRT*: Learning-based Path Planning with Point Cloud State Representations under Admissible Ellipsoidal Constraints

##### [Zhe Huang](https://tedhuang96.github.io/), [Hongyu Chen](https://www.linkedin.com/in/hongyu-chen-91996b22b), [John Pohovey](https://www.linkedin.com/in/johnp14/), [Katherine Driggs-Campbell](https://krdc.web.illinois.edu/)

[Paper] [[arXiv](https://arxiv.org/abs/2309.14595)] [[Main GitHub Repo](https://github.com/tedhuang96/nirrt_star)] [[Robot Demo GitHub Repo](https://github.com/tedhuang96/PNGNav)] [[Project Google Sites](https://sites.google.com/view/nirrt-star)] [[Presentation on YouTube](https://youtu.be/xys6XxMqFqQ)] [[Robot Demo on YouTube](https://youtu.be/XjZqUJ0ufGA)]

All code was developed and tested on Ubuntu 20.04 with CUDA 12.0, conda 23.11.0, Python 3.9.0, and PyTorch 2.0.1. We also offer implmentations on RRT*, Informed RRT*, and Neural RRT* as baselines. 


## Citation
If you find this repo useful, please cite
```
@article{huang2023neural,
  title={Neural Informed RRT*: Learning-based Path Planning with Point Cloud State Representations under Admissible Ellipsoidal Constraints},
  author={Huang, Zhe and Chen, Hongyu and Pohovey, John and Driggs-Campbell, Katherine},
  journal={arXiv preprint arXiv:2309.14595},
  year={2023}
}
```

## Setup
```
conda env create -f environment.yml
```
or
```
conda create -n pngenv python=3.9.0
conda activate pngenv
pip install -e .
pip install numpy
pip install pyyaml
pip install matplotlib
pip install opencv-python
pip install torch==2.0.1
pip install torchvision==0.15.2
pip install open3d==0.17.0
```

## Quick Test

### Data for ICRA 2024
Download [nirrt_star-icra24-data.zip](https://drive.google.com/file/d/1omIxfoASMzWBUcdXS7-9WiJbc2xTikf2/view?usp=drive_link) and move the zip file into the root folder of this repo. Run
```
cd nirrt_star/
unzip nirrt_star-icra24-data.zip
```

### Model Weights for ICRA 2024
Download [nirrt_star-icra24-model-weights.zip](https://drive.google.com/file/d/1jQ7yrxceSq1aHZjdDQYq2gB2HOHRPvCg/view?usp=drive_link) and move the zip file into the root folder of this repo. Run
```
cd nirrt_star/
unzip nirrt_star-icra24-model-weights.zip
```

### Evaluation for ICRA 2024
Download [nirrt_star-icra24-evaluation.zip](https://drive.google.com/file/d/1CZyPvj3K53vaXxOVTOpZCIU33zIg9dAS/view?usp=drive_link) and move the zip file into the root folder of this repo. Run
```
cd nirrt_star/
unzip nirrt_star-icra24-evaluation.zip
```

### Visualize 2D Random World samples
- For example, to visualize a 2D Random World test sample with token `100_2`, which means env_idx = 100, start_goal_idx = 2, run
```
conda activate pngenv
python visualize_data_samples_2d.py --visual_example_token 100_2
```
Check out images in `visualization/img_with_labels_2d/`.

- To visualize all 2D Random World test samples, run
```
conda activate pngenv
python visualize_data_samples_2d.py
```

### Planning Demo
- For 2D, run
```
conda activate pngenv
python demo_planning_2d.py -p nirrt_star -n pointnet2 -c bfs --problem {2D_problem_type} --iter_max 500
python demo_planning_2d.py -p nrrt_star -n unet --problem {2D_problem_type} --iter_max 500
python demo_planning_2d.py -p nrrt_star -n pointnet2 --problem {2D_problem_type} --iter_max 500
python demo_planning_2d.py -p irrt_star --problem {2D_problem_type} --iter_max 500
python demo_planning_2d.py -p rrt_star --problem {2D_problem_type} --iter_max 500
```
where `{2D_problem_type}` can be `random_2d`, `block`, or `gap`. Note `unet` cannot be used for `block`, as `unet` requires `img_height % 32 == 0 and img_width % 32 == 0`, while `block` may change the environment range randomly and does not meet this requirements. Visualizations can be found in `visualization/planning_demo/`.

- For 3D, run
```
conda activate pngenv
python demo_planning_3d.py -p nirrt_star -n pointnet2 -c bfs --problem random_3d --iter_max 500
python demo_planning_3d.py -p nrrt_star -n unet --problem random_3d --iter_max 500
python demo_planning_3d.py -p nrrt_star -n pointnet2 --problem random_3d --iter_max 500
python demo_planning_3d.py -p irrt_star --problem random_3d --iter_max 500
python demo_planning_3d.py -p rrt_star --problem random_3d --iter_max 500
```
Visualization will be in GUI.

### Visualizations in ICRA 2024 Paper
If you run [Result Analysis](#result-analysis) with the downloaded evaluation, check `visualization/evaluation/` and you will find the images used in Fig. 5 of NIRRT* ICRA 2024 paper.

## Data Collection
Instructions for collecting your own data.

### Collect 2D random world data
```
conda activate pngenv
python generate_random_world_env_2d.py
python generate_random_world_env_2d_point_cloud.py
```

### Collect 3D random world data
```
conda activate pngenv
python generate_random_world_env_3d_raw.py
python generate_random_world_env_3d_astar_labels.py
python generate_random_world_env_3d_point_cloud.py
```

### Generate block and gap environment configurations
```
conda activate pngenv
python generate_block_gap_env_2d.py
```

## Model Training
Instructions for collecting your own models.

### Training and Evaluation of PointNet++
To train and evaluate PointNet++ for guidance state inference, run
```
conda activate pngenv
python train_pointnet_pointnet2.py --random_seed 100 --model pointnet2 --dim 2
python eval_pointnet_pointnet2.py --random_seed 100 --model pointnet2 --dim 2
python train_pointnet_pointnet2.py --random_seed 100 --model pointnet2 --dim 3
python eval_pointnet_pointnet2.py --random_seed 100 --model pointnet2 --dim 3
```
If you want to train PointNet, you can replace `--model pointnet2` with `--model pointnet`. Note that `results/model_training/pointnet2_2d/checkpoints/best_pointnet2_2d.pth` is equivalent as the `pointnet2_sem_seg_msg_pathplan.pth` you will be putting in [PNGNav](https://github.com/tedhuang96/PNGNav) if you are going to deploy NIRRT* in ROS for your robot applications.

### Training and Evaluation of U-Net
```
conda activate pngenv
python train_unet.py
python eval_unet.py
```

## Evaluation of Planning Methods

### 2D
Run
```
conda activate pngenv
python eval_planning_2d.py -p nirrt_star -n pointnet2 -c bfs --problem {2D_problem_type} 
python eval_planning_2d.py -p nirrt_star -n pointnet2 --problem {2D_problem_type} 
python eval_planning_2d.py -p nrrt_star -n pointnet2 -c bfs --problem {2D_problem_type} 
python eval_planning_2d.py -p nrrt_star -n pointnet2 --problem {2D_problem_type} 
python eval_planning_2d.py -p nrrt_star -n unet --problem {2D_problem_type}
python eval_planning_2d.py -p irrt_star --problem {2D_problem_type} 
python eval_planning_2d.py -p rrt_star --problem {2D_problem_type} 
```
where `{2D_problem_type} ` can be `random_2d`, `block`, or `gap`. Note `unet` cannot be used for `block`, as `unet` requires `img_height % 32 == 0 and img_width % 32 == 0`, while `block` may change the environment range randomly and does not meet this requirements.

### 3D
Run
```
conda activate pngenv
python eval_planning_3d.py -p nirrt_star -n pointnet2 -c bfs
python eval_planning_3d.py -p nirrt_star -n pointnet2
python eval_planning_3d.py -p nrrt_star -n pointnet2 -c bfs
python eval_planning_3d.py -p nrrt_star -n pointnet2
python eval_planning_3d.py -p irrt_star
python eval_planning_3d.py -p rrt_star
```

### Result Analysis
Run
```
conda activate pngenv
python result_analysis_random_world_2d.py
python result_analysis_random_world_3d.py
python result_analysis_block.py
python result_analysis_gap.py
```
Visualizations are saved in `visualization/evaluation/`.


## References

[zhm-real/PathPlanning](https://github.com/zhm-real/PathPlanning)

[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

[rawmarshmellows/pytorch-unet-resnet-50-encoder](https://github.com/rawmarshmellows/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py)

[UCSD CSE 291 Collision Detection Material](https://cseweb.ucsd.edu/classes/sp19/cse291-d/Files/CSE291_13_CollisionDetection.pdf)

[Simple Intersection Tests for Games](https://www.gamedeveloper.com/game-platforms/simple-intersection-tests-for-games)

## Contact
Please feel free to open an issue or send an email to zheh4@illinois.edu.