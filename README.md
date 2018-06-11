# Learning dynamics of linear denoising autoencoders
Code to reproduce all the results in the paper: "Learning dynamics of linear denoising autoencoders." (ICML 2018)

**Note: this repo is still a work in progress. Additional notebooks coming soon...**

# Research Code #

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
This should create a volume bind mount with the current directory for persistent data storage as well as launch a Jupyter notebook accessible at http://0.0.0.0:8888/. To stop the docker container from running simply shutdown the notebook by pressing ctrl+c (the container will automatically be removed once stopped). 
