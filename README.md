# Evaluating the Generalization of Neural Network-Based Pan-Cancer Classification Models for Cohort-Specific Predictions
This repo contains code for running the experiments that compare pan-cancer (PC) models trained on a gene expression task against models trained with a single cohort.


## Dataset
The dataset and the indices for each experiment can be downloaded [here](https://www.dropbox.com/scl/fi/981azce85uj9sy2dmixek/data_pc.tar.gz?rlkey=fahi3yakgylo86hv4s001x3bo&st=0ve0dnoy&dl=0).


The results, including fitted models for each run, can be downloaded [here](https://www.dropbox.com/scl/fi/nr0ktuqx1dnjzkmvgcbsx/results_pc_generalization.tar.gz?rlkey=lcxsn08jyd2529ve2o45omh68&st=1ywytppp&dl=0)

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

## Citation
If you find this code useful, consider citing the paper:

```bibtex
@inproceedings{fontanari2024pcgeneralization,
 author = {Thomas Fontanari and Mariana Recamonde-Mendoza},
    title = {Evaluating the Generalization of Neural Network-Based Pan-Cancer Classification Models for Cohort-Specific Predictions},
    booktitle = {Anais do XVII Simpósio Brasileiro de Bioinformática},
    location = {Vitória/ES},
    year = {2024},
    keywords = {},
    issn = {2316-1248},
    pages = {12--23},
    publisher = {SBC},
    address = {Porto Alegre, RS, Brasil},
    doi = {10.5753/bsb.2024.245165},
    url = {https://sol.sbc.org.br/index.php/bsb/article/view/32629}
}
```