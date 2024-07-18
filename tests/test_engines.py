import torch
import pytest


from generalization_pc.engines import EvaluateOnDataloaderLoop, TrainEpochLoop


# def test_EvaluateOnDataloaderLoop(model, dataset):
#     pass


def test_TrainEpochLoop_call(linear_blobs_model, blobs_dataloader):
    model = linear_blobs_model
    dl = blobs_dataloader

    for epoch in range(50):
        trainer = TrainEpochLoop(epoch=epoch)
        model, metrics = trainer.__call__(
            model=model,
            dataloader=dl,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            loss_fn=torch.nn.CrossEntropyLoss(),
        )

    assert isinstance(model, torch.nn.Module)
    assert "loss" in metrics
    assert "accuracy" in metrics

    assert metrics["loss"] < 0.01
    assert metrics["accuracy"] == 1.0


def test_EvaluateOnDataloaderLoop_call(
    trained_linear_blobs_model, 
    blobs_dataloader, 
    trained_linear_blobs_model_metrics
):
    model = trained_linear_blobs_model
    dl = blobs_dataloader

    evaluator = EvaluateOnDataloaderLoop()
    metrics = evaluator.__call__(
        model=model,
        dataloader=dl,
        loss_fn=torch.nn.CrossEntropyLoss(),
    )

    assert metrics == trained_linear_blobs_model_metrics