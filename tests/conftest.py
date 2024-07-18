import json
import torch
from sklearn.datasets import make_blobs
import pytest
import importlib.resources

from generalization_pc.engines import EvaluateOnDataloaderLoop, TrainEpochLoop


class BlobsDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target


def generate_blobs_dataset():
    X, y = make_blobs(
        n_samples=50, 
        n_features=2, 
        centers=[[-10, -10], [10, 10]], 
        random_state=42
    )
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return BlobsDataset(X_tensor, y_tensor)


def get_linear_blobs_model():
    return torch.nn.Linear(in_features=2, out_features=2)


def get_trained_blobs_linear_model():
    resource_package = "generalization_pc"
    trav = importlib.resources.files(resource_package)
    resources_path = trav.joinpath("resources")
    model_path = resources_path.joinpath("blobs_linear_model.pt")

    model = get_linear_blobs_model() 
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def get_metrics_trained_blobs_linear_model():
    resource_package = "generalization_pc"
    trav = importlib.resources.files(resource_package)
    resources_path = trav.joinpath("resources")
    metrics_path = resources_path.joinpath("blobs_linear_model_metrics.json")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


@pytest.fixture
def blobs_dataset():
    return generate_blobs_dataset()


@pytest.fixture
def blobs_dataloader():
    ds = generate_blobs_dataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    return dl


@pytest.fixture
def linear_blobs_model():
    return torch.nn.Linear(in_features=2, out_features=2)


@pytest.fixture
def trained_linear_blobs_model():
    return get_trained_blobs_linear_model()


@pytest.fixture
def trained_linear_blobs_model_metrics():
    return get_metrics_trained_blobs_linear_model()
