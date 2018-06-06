sudo nvidia-docker run -v "$(pwd)":/data --rm --name research -it -p 8888:8888 arnu/research_env jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/data'
