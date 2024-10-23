# A Study on Pan-Cancer Generalization on Specific Tasks
This repo contains code for running the experiments that compare pan-cancer models trained on a gene expression task against models trained with a single cohort.


## Dataset
A link to download the processed dataset used will be added yet.

## Usage
To run the experiments, first create a new conda environment with the necessary packages. 
One way of doing it is by running

```bash
conda env create --file environment.yml
conda activate pc-generalization-310
pip install -e .
```

The experiments can be reproduced and the artifacts generated running the `run_all.sh` script. 
Note that there some randomness in the training process so the results may present some level of variation.
