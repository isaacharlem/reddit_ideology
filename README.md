# Computational Methods for Social Science: Reddit Ideology

## Instructions
Follow the following instructions to run the model and analysis pipeline

It is highly recommended to use "cuda" if you have it available as the embedding and BERTopic processes require a descent amount of compute. It is possible to run on "mps" (metal) or "cpu", but expect the pipeline to take >60 minutes if so. We ran the pipeline using 4x NVIDIA A100 GPUs (80 GB) and it took about 5 minutes.

1. Clone the repository:

        git clone git@github.com:isaacharlem/reddit_ideology.git

    or 

        git clone https://github.com/isaacharlem/reddit_ideology.git

2. Create virtual environment (using conda):

    conda create -n red_id python=3.10

3. Activate virtual environment (using conda):

    conda activate red_id

4. Install package in editable mode

    pip install -e .

5. Set up OpenAI API key (here)[https://openai.com/index/openai-api/]

5. Set up config.yaml:

    cp config.yaml my_config.yaml

6. Set up OpenAI API key

5. (Optional) Request UChicago SLURM resources:

    srun -p general --gres=gpu:a100:4 --pty --cpus-per-task=32 --mem=200G -t 4:00:00 /bin/bash

6. 