from pathlib import Path
from typing import Literal
import ray.tune
import ray.tune.schedulers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import ray
import ray.tune as tune
import ray.air as air

import generalization_pc as gpc
from generalization_pc.experiment import ExperimentExecutor
from generalization_pc.utils import save_output_artifacts
from generalization_pc.engines import EvaluateOnDataloaderLoop, TrainAndEvalLoop, TrainEpochLoop
from generalization_pc.models import FCModel, FCModelCreator
from generalization_pc.datasets import IndicesLoader, TCGAExpressionDataset, TCGAExpressionDatasetCreator, list_available_cohorts



def test_creator():
    path_dataset = Path("/home/thomas/work/mestrado/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification")
    path_indices = Path("/home/thomas/work/mestrado/generalization_pc/indices")
    cohorts = list_available_cohorts(path_dataset)
    
    cohort = cohorts[0]
    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=path_dataset,
        path_indices=path_indices,
        experiment=1,
        mode="train",
        cohort="pancancer",
        run=0
    )
    dataset = ds_creator.get()
    X, y = dataset[0]
    print(X.shape, y.shape)


def test_model():
    path_dataset = Path("/home/thomas/work/mestrado/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification")
    path_indices = Path("/home/thomas/work/mestrado/generalization_pc/indices")

    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=path_dataset,
        path_indices=path_indices,
        experiment=1,
        mode="train",
        cohort="pancancer",
        run=0
    )
    dataset = ds_creator.get()

    n_genes = dataset[0][0].shape[-1]
    n_classes = 2  # this implies using CELoss, put a BCELoss would be more usual

    model = FCModel(
        input_dim=n_genes, 
        output_dim=n_classes, 
        hidden_dim=(256)
    )

    print(model)
    example = torch.randn(size=(10, n_genes))
    output = model(example)
    print(output.shape)


def test_loop():
    path_dataset = Path("/home/thomas/work/mestrado/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification")
    path_indices = Path("/home/thomas/work/mestrado/generalization_pc/indices")

    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=path_dataset,
        path_indices=path_indices,
        experiment=1,
        mode="train",
        cohort="pancancer",
        run=0
    )
    dataset = ds_creator.get()

    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=path_dataset,
        path_indices=path_indices,
        experiment=1,
        mode="val",
        cohort="pancancer",
        run=0
    )
    val_dataset = ds_creator.get()

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    n_genes = dataset[0][0].shape[-1]
    n_classes = 2  # this implies using CELoss, put a BCELoss would be more usual

    model = FCModel(
        input_dim=n_genes, 
        output_dim=n_classes, 
        hidden_dim=(256)
    )

    lr = 1e-3
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger = logging.getLogger(__name__)
    
    train_func = TrainEpochLoop(
        device="cuda",
        logger=logger,
        max_batches=2
    )
    eval_func = EvaluateOnDataloaderLoop(
        device="cuda",
        max_batches=2,
        return_outputs=True
    )

    metrics, (outputs, labels) = eval_func(
        model=model,
        dataloader=val_dataloader,
        loss_fn=nn.CrossEntropyLoss()
    )

    # model, train_metrics = train_func(
    #     model=model,
    #     dataloader=dataloader,
    #     optimizer=optimizer,
    #     loss_fn=nn.CrossEntropyLoss(),
    # )


def test_train_eval_loop_raytune():
    path_dataset = Path("/home/thomas/work/mestrado/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification")
    path_indices = Path("/home/thomas/work/mestrado/generalization_pc/indices")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=path_dataset,
        path_indices=path_indices,
        experiment=1,
        mode="train",
        cohort="pancancer",
        run=0
    )
    train_dataset = ds_creator.get()

    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=path_dataset,
        path_indices=path_indices,
        experiment=1,
        mode="val",
        cohort="pancancer",
        run=0
    )
    val_dataset = ds_creator.get()


    max_epochs = 2
    device = "cpu"
    cpu_per_trial = 1
    gpu_per_trial = 0


    train_func = TrainEpochLoop(
        device=device,
        logger=logger,
        max_batches=2
    )
    eval_func = EvaluateOnDataloaderLoop(
        device=device,
        max_batches=2,
        return_outputs=False
    )

    train_eval_loop = TrainAndEvalLoop(
        train_epoch_function=train_func,
        validation_function=eval_func,
        datasets={"train": train_dataset, "val": val_dataset},
        n_input_dims=14133,
        n_classes=2,
        max_epochs=max_epochs,
        logger=logger,
        report_to_raytune=True
    )
    
    ray_scheduler = tune.schedulers.ASHAScheduler(
        max_t=max_epochs,
        grace_period=max_epochs,  # let warm restarts work
        reduction_factor=2,
    )

    if 'cuda' in device:
        trainable = tune.with_resources(
            train_eval_loop, {"cpu": cpu_per_trial, "gpu": gpu_per_trial}
        )
    else:
        trainable = tune.with_resources(train_eval_loop, {"cpu": cpu_per_trial})
    
    hp_config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1)
    }

    path_ray = Path("./tmp_outputs/raytune/").resolve()
    tuner = tune.Tuner(
        trainable=trainable,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=ray_scheduler,
            num_samples=4,
        ),
        run_config=air.RunConfig(
            # local_dir=str(path_ray),
            storage_path=str(path_ray),
            name="test_experiment",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="accuracy", num_to_keep=1
            ),
        ),
        param_space=hp_config,
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_configs = best_result.config
    print(best_configs)


def train_and_test_dev():
    path_dataset = Path("/home/thomas/work/mestrado/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification")
    path_indices = Path("/home/thomas/work/mestrado/generalization_pc/indices")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=path_dataset,
        path_indices=path_indices,
        experiment=1,
        mode="train",
        cohort="pancancer",
        run=0
    )
    train_dataset = ds_creator.get()

    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=path_dataset,
        path_indices=path_indices,
        experiment=1,
        mode="val",
        cohort="pancancer",
        run=0
    )
    val_dataset = ds_creator.get()

    max_epochs = 3
    device = "cpu"
    cpu_per_trial = 1
    gpu_per_trial = 0

    train_func = TrainEpochLoop(
        device=device,
        logger=logger,
        max_batches=2
    )
    eval_func = EvaluateOnDataloaderLoop(
        device=device,
        max_batches=2,
        return_outputs=False
    )

    ## training and testing the final model
    train_eval_loop = TrainAndEvalLoop(
        train_epoch_function=train_func,
        validation_function=eval_func,
        datasets={"train": train_dataset, "val": val_dataset},
        n_input_dims=14133,
        n_classes=2,
        max_epochs=max_epochs,
        logger=logger,
        report_to_raytune=False
    )

    train_eval_loop.run()
    print(
        train_eval_loop._records
    )

    print("Evaluating on: ")
    val_set_samples = val_dataset._metadata.loc[val_dataset._indices, :]

    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_func = EvaluateOnDataloaderLoop(
        device=device,
        max_batches=2,
        return_outputs=True
    )
    metrics, (pred, labels) = test_func(
        model=train_eval_loop.model, 
        dataloader=test_loader, 
        loss_fn=nn.CrossEntropyLoss()
    )

    save_output_artifacts(
        metric_records=train_eval_loop._records,
        outputs=pred,
        labels=labels,
        configs={},
        output_dir=Path("./tmp_outputs/test"),
        category_names=val_dataset.label_encoder.classes_,
        model=train_eval_loop.model
    )


def test_pancancer_test_indices():
    ds = TCGAExpressionDatasetCreator(
        path_tcga_dataset=Path("/home/thomas/work/mestrado/PoolingGenomicGNNs/data/tcga_cohorts_and_tumor_classification"),
        path_indices=Path("/home/thomas/work/mestrado/generalization_pc/indices"),
        experiment=2,
        mode="test",
        cohort="pancancer",
        run=0
    ).get()
    df_pc = ds._metadata.loc[ds._indices, :]
    df_pc_blca = df_pc[df_pc["cohort"] == "blca"]
    df_pc_blca.to_csv("blca_test_indices.csv")


def test_experiment_class():
    ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=gpc.config.path_dataset,
        path_indices=gpc.config.path_indices,
        experiment=1,
        cohort="pancancer",
    )
    model_creator = FCModelCreator(
        hidden_dim=(256,)
    )

    exp = ExperimentExecutor(
        dataset_creator=ds_creator,
        model_creator=model_creator
    )
    exp.run_experiment()


if __name__ == "__main__":
    # pc_exp1 = PanCancerAllCohorts()
    test_experiment_class()
