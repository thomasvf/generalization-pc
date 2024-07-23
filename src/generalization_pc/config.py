from pathlib import Path
from typing import Union
import pandas as pd


# each raytune trial will block
raytune_gpu_per_trial = 0.1
raytune_cpu_per_trial = 2


n_runs = 5
max_epochs = 31
num_samples = 8
path_dataset = str(Path("data/tcga_cohorts_and_tumor_classification/").resolve())
path_indices = str(Path("data/indices").resolve())


class TCGADatasetParameters:
    def __init__(
        self,
        path_dataset: Union[str, Path] = path_dataset,
        path_indices: Union[str, Path] = path_indices,
        n_classes = 2,
        n_input_dims = 14133,
    ):
        self._path_dataset = Path(path_dataset)
        self._path_indices = Path(path_indices)
        self._n_classes = n_classes
        self._n_input_dims = n_input_dims

    @property
    def path_dataset(self):
        return self._path_dataset

    @property
    def path_indices(self):
        return self._path_indices
    
    @property
    def _path_metadata(self):
        return self._path_dataset / "sample_metadata.csv"
    
    @property
    def n_classes(self):
        return self._n_classes
    
    @property
    def n_input_dims(self):
        return self._n_input_dims
    
    def get_cohorts(self):
        df = pd.read_csv(self._path_dataset / "sample_metadata.csv")
        cohorts = df["cohort"].unique().tolist()
        return cohorts

