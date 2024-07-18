from pathlib import Path
from typing import Literal, Sequence, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset


def list_available_cohorts(path_tcga_dataset: Union[str, Path]):
    path_metadata = path_tcga_dataset / "sample_metadata.csv"
    metadata = pd.read_csv(path_metadata, index_col=0)
    cohorts = metadata["cohort"].unique()

    return cohorts


class TCGAExpressionDatasetCreator:
    """
    Creator for TCGAExperssionDataset used in an experiment.

    The mode (train, val or test) and run are only available during execution and can
    be set in the getter.
    """

    def __init__(
        self,
        path_tcga_dataset: Union[str, Path],
        path_indices: Union[str, Path],
        experiment: Literal[1, 2],
        mode: Literal["train", "val", "test"] = "train",
        run: int = 0,
        cohort: str = "pancancer",
    ):
        self._path_tcga_dataset = Path(path_tcga_dataset)
        self._path_indices = Path(path_indices)
        self._experiment = experiment
        self._mode = mode
        self._run = run
        self._cohort = cohort

    def set_mode_and_run(
        self,
        mode: Literal["train", "val", "test"] = None,
        run: int = None,
    ):
        if mode is not None:
            self._mode = mode

        if run is not None:
            self._run = run

        return self

    def get(self):
        indices = self._get_indices()
        dataset = self._build_dataset(indices)
        return dataset

    def _get_indices(self):
        indices_getter = IndicesLoader(
            path_indices_dir=self._path_indices,
            experiment=self._experiment,
            mode=self._mode,
            run=self._run,
            cohort=self._cohort,
        )

        indices = indices_getter.get()
        return indices

    def _build_dataset(self, indices: Sequence):
        tcga_dataset = TCGAExpressionDataset(
            path_tcga_dataset=self._path_tcga_dataset,
            tcga_indices=indices,
            metadata_column="sample_type",
            random_state=123,
        )

        return tcga_dataset

    def get_n_dims(self):
        return 14133

    def get_n_classes(self):
        return 2


class TCGAExpressionDSExp2PCCreator:
    """Creator for TCGA Expression Pan-cancer datasets for the second experiment."""

    def __init__(
        self,
        path_tcga_dataset: Union[str, Path],
        path_indices: Union[str, Path],
        target_cohort: str = None,
    ) -> None:
        self._path_tcga_dataset = Path(path_tcga_dataset)
        self._path_indices = Path(path_indices)
        self._target_cohort = target_cohort

        self._mode = None
        self._run = None

    def set_mode_and_run(
        self,
        mode: Literal["train", "val", "test"] = None,
        run: int = None,
    ):
        if mode is not None:
            self._mode = mode

        if run is not None:
            self._run = run

        return self
    
    def get(self):
        self._check_is_set()

        indices = self._get_indices()
        ds = self._build_dataset(indices)

        return ds
    
    def get_n_dims(self):
        return 14133

    def get_n_classes(self):
        return 2

    def _get_indices(self):
        if self._mode == "train" or self._mode == "val":
            filename = f"{self._target_cohort}_run{self._run}.csv"
            relative_path = (
                f"experiment_2/{self._mode}/pancancer/{filename}"
            )
        elif self._mode == "test":
            relative_path = f"experiment_2/test/indices_test_{self._target_cohort}_{self._run}.csv"
        else:
            raise ValueError("Mode must be 'train', 'val' or 'test'.")
        
        path_indices = self._path_indices / relative_path
        
        indices = self._read_indices_file(path_indices)
        
        return indices
    
    def _build_dataset(self, indices: Sequence):
        tcga_dataset = TCGAExpressionDataset(
            path_tcga_dataset=self._path_tcga_dataset,
            tcga_indices=indices,
            metadata_column="sample_type",
            random_state=123,
        )
        return tcga_dataset
    
    def _check_is_set(self):
        if self._mode is None or self._run is None:
            err_msg = (
                "Mode and run must be set before getting the dataset. "
                "Use the method _set_mode_and_run to set them."
            )
            raise ValueError(err_msg)
        
    def _read_indices_file(self, path_indices: Path):
        df = pd.read_csv(path_indices, index_col=0)
        indices = df["index"].to_numpy()
        return indices


class IndicesLoader:
    def __init__(
        self,
        path_indices_dir: Path,
        experiment: Literal[1, 2],
        mode: Literal["train", "val", "test"],
        run: Literal[0, 1, 2, 3, 4],
        cohort: str = "pancancer",
    ):
        self._path_indices_dir = path_indices_dir
        self._experiment = experiment
        self._mode = mode
        self._run = run
        self._cohort = cohort

    def get(self):
        path_experiment = self._path_indices_dir / f"experiment_{self._experiment}"
        if not self._is_test_set():
            indices = self._get_train_and_val_dataset_indices(path_experiment)
        else:
            indices = self._get_test_dataset_indices(path_experiment)

        return indices

    def _read_indices_file(self, path_indices):
        df = pd.read_csv(path_indices, index_col=0)
        indices = df["index"].to_numpy()
        return indices

    def _get_test_dataset_indices(self, path_experiment):
        if self._cohort == "pancancer":
            return self._get_pancancer_test_indices(path_experiment)

        filename = f"indices_{self._mode}_{self._cohort}_{self._run}.csv"
        path_indices = path_experiment / self._mode / filename
        indices = self._read_indices_file(path_indices)
        return indices

    def _get_train_and_val_dataset_indices(self, path_experiment):
        if self._cohort == "pancancer":
            filename = f"pancancer_run{self._run}.csv"
            path_indices = path_experiment / self._mode / "pancancer" / filename
        else:
            filename = f"indices_{self._mode}_{self._cohort}_{self._run}.csv"
            path_indices = path_experiment / self._mode / "cohort_specific" / filename

        indices = self._read_indices_file(path_indices)

        return indices

    def _is_test_set(self):
        return self._mode == "test"

    def _get_pancancer_test_indices(self, path_experiment):
        path_cohort_indices = path_experiment / "test"
        run_paths = []
        for p in path_cohort_indices.glob("*.csv"):
            mode, cohort, run = self._extract_mode_cohort_run_from_filename(p.stem)
            if run == self._run:
                run_paths.append(p)

        run_paths = sorted(run_paths)

        indice_dataframes = []
        for p in run_paths:
            indices = self._read_indices_file(p)
            indice_dataframes.append(indices)

        indices = np.concatenate(indice_dataframes, axis=0)
        indices = np.sort(indices)

        return indices

    def _extract_mode_cohort_run_from_filename(self, filename):
        parts = filename.split("_")
        mode = parts[1]
        cohort = parts[2]
        run = int(parts[3])

        return mode, cohort, run


class TCGAExpressionDataset(Dataset):
    def __init__(
        self,
        path_tcga_dataset: Union[str, Path],
        tcga_indices: Sequence,
        metadata_column: str = "sample_type",
        random_state: int = 123,
        scaler=None,
    ):
        self._path_tcga_dataset = path_tcga_dataset
        self._indices = tcga_indices
        self._scaler = scaler

        self._path_samples = self._path_tcga_dataset / "samples"

        self._metadata = pd.read_csv(
            self._path_tcga_dataset / "sample_metadata.csv", index_col=0
        )
        # if random_state is not None:
        #     self._metadata = self._metadata.sample(frac=1, random_state=random_state)

        self._metadata_column = metadata_column

        self.samples = self._metadata.index
        self.label_encoder, self.y = self._encode_labels()
        self.n_classes = len(np.unique(self.y))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        index_tcga = self._indices[index]
        path_sample = self.get_path_sample(index_tcga)

        sample = self._read_sample_expression(path_sample)
        if self._scaler is not None:
            sample = self.scaler.transform(sample.reshape(1, -1)).reshape(-1)
        sample = torch.from_numpy(sample).to(torch.float)

        y = self.y[index]
        return sample, y

    def _read_sample_expression(self, path_sample):
        df_sample = pd.to_numeric(pd.read_csv(path_sample, index_col=0).iloc[:, 0])
        sample = df_sample.to_numpy().transpose()
        return sample

    def get_examples_metadata(self):
        """Return metadata for the examples in the dataset.

        Return only the ones that are indexed by the given attribute `tcga_indices`.
        """
        return self._metadata.loc[self._indices, :]

    def get_path_sample(self, sample_index: str):
        path_sample = self.tcga_index_to_relative_path(
            sample_idx=sample_index, path_prefix=self._path_samples
        )
        return path_sample

    def get_genes(self):
        path_sample = self.get_path_sample(0)
        genes = pd.read_csv(path_sample, index_col=0).iloc[:, 0].index
        return genes

    def tcga_index_to_relative_path(self, sample_idx: str, path_prefix: str = None):
        """Convert a TCGA identifier to a path in the filesystem.

        Parameters
        ----------
        sample_idx : str
            TCGA sample identifier
        path_prefix : str, optional
            Path prefix to add to the sample path, by default None

        Returns
        -------
        Path
            Path to sample file
        """
        name_parts = sample_idx.split("-")

        index_path = Path("/".join(name_parts[:-1])) / f"{sample_idx}.csv"
        if path_prefix is None:
            return index_path

        return Path(path_prefix) / index_path

    def _encode_labels(self) -> np.ndarray:
        label_encoder = LabelEncoder()

        label_encoder.fit(self._metadata.loc[:, self._metadata_column].to_numpy())
        sample_labels = self._metadata.loc[
            self._indices, self._metadata_column
        ].to_numpy()
        y = label_encoder.transform(sample_labels)

        return label_encoder, y

    def _get_class_weights(self) -> torch.Tensor:
        """
        Get class weights for the dataset using scikit-learn's `compute_class_weight` function.
        """
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(self.y), y=self.y
        )
        class_weights = torch.from_numpy(class_weights).float()
        return class_weights
