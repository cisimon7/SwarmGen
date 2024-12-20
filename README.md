### Swarm-Gen: Fast Generation of Diverse Feasible Swarm Behaviors

This repository contains the source code to reproduce the experiments in our paper: "Swarm-Gen: Fast Generation of Diverse Feasible Swarm Behaviors" where we combine a generative model (CVAE/VQ-VAE) with a safty filter (SF) to learn to generate diverse swarm behavior trajectories given a fixed start and goal points.

<img src="./utils/imgs/teaser.png" width="800"/>


## Getting Started

1. Clone this repository:

```
git clone https://github.com/cisimon7/SwarmGen.git
cd SwarmGen
```
2. Create a conda environment and install the dependencies:

```
conda create -n venv python=3.8
conda activate venv
pip install -r requirements.txt
```

3. Create a directory `resources`

4. Download the [weights](https://owncloud.ut.ee/owncloud/s/K7MqFwSjdBk78a9) for our trained Models and save to the `resources` directory

5. Download the [train and test dataset](https://owncloud.ut.ee/owncloud/s/zHQgBDLndDJPbkc) and also save to the `resources` directory.

6. You should be able to run any of the train and/or test norebooks. Note that for `num_agent=4` or `num_agent=8`, the variables `lwidth` and `lheight` should both be set to 5, while for `num_agent=16`, they are set to 10. This applies for all the notebooks.


## Visualization
Visualization of diverse trajectories generated using the VQ-VAE model for a given fixed start and goal points.

<img src="./utils/imgs/scene_1.gif" width="300"/> <img src="./utils/imgs/scene_2.gif" width="300"/>
<br>
<img src="./utils/imgs/scene_3.gif" width="300"/> <img src="./utils/imgs/scene_4.gif" width="300"/>


###