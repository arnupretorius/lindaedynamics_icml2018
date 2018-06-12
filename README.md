# Code: Learning dynamics of linear denoising autoencoders
This repository provides the code to reproduce all the results in the paper: "Learning dynamics of linear denoising autoencoders." (ICML 2018)

## Basic requirements for Figures 1-4

To reproduce Figures 1-4, all that is required is `numpy` and `matplotlib`.

## Requirements for larger scale experiments for Figures 5-6: 

To reproduce Figures 5 and 6, a docker image was created to provide an identical research environment to the one used to run the initial experiments. Below are the instructions to reproduce these plots using this docker images and the notebooks provided.

## Quick Start (GPU required)

### Installation

#### Step 1. Install [Docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

#### Step 2. Obtain the research environment image from [Docker Hub](https://hub.docker.com/r/arnu/research_env/).

```bash
docker pull arnu/research_env
```
#### Step 3. Clone the research code repository. 
```bash
git clone https://github.com/arnupretorius/lindaedynamics_icml2018.git
```

### Usage

Change directory to the cloned repository on your local machine and run the bash script.
```bash
research_up.sh
```
This should create a volume bind mount with the current directory for persistent data storage as well as launch a Jupyter notebook accessible at http://0.0.0.0:8888/. Now, you can simply run the notebook corresponding to the figure in the paper you wish to reproduce. 

To stop the docker container from running simply shutdown the notebook by pressing ctrl+c (the container will automatically be removed once stopped). 
