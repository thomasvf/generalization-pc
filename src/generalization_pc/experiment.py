import logging
from pathlib import Path
from typing import Literal, Union
import ray.tune as tune
import ray.air as air
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from generalization_pc.models import ModelCreator
from generalization_pc.utils import save_output_artifacts
from generalization_pc.datasets import TCGAExpressionDataset, TCGAExpressionDatasetCreator
from generalization_pc.engines import EvaluateOnDataloaderLoop, TrainAndEvalLoop, TrainEpochLoop
from generalization_pc import config


class ExperimentExecutor:
    def __init__(
        self,
        dataset_creator: TCGAExpressionDatasetCreator,
        model_creator: ModelCreator,
        output_dir: Union[str, Path] = "./output/experiment",
        n_runs: int = 5,
        max_epochs: int = 31,
        cpu_per_trial: int = config.raytune_cpu_per_trial,
        gpu_per_trial: float = config.raytune_gpu_per_trial,
        num_samples: int = 8
    ):
        self._output_dir = Path(output_dir)
        self._dataset_creator = dataset_creator
        self._model_creator = model_creator
        
        self._logger = self._get_logger()

        self._n_runs = n_runs
        self._max_epochs = max_epochs
        self._device = "cuda"
        self._cpu_per_trial = cpu_per_trial
        self._gpu_per_trial = gpu_per_trial
        self._max_batches = None
        self._n_classes = dataset_creator.get_n_classes()
        self._n_input_dims = dataset_creator.get_n_dims()
        self._num_samples = num_samples

        self._tuning_evaluation_metric = "val_loss"
        self._tuning_evaluation_mode = "min"

        # state variables
        self._current_run = 0
        self._tuning_results = None
        self._run_output_dir = None
        self._path_ray = None
        self._model = None
    
    def run_experiment(self):
        for run in range(self._n_runs):
            self._logger.info(f"Running experiment for run {run}")
            self._set_run_state(run)
            self._execute_run()

    def _set_run_state(self, run: int):
        self._current_run = run
        
        self._run_output_dir = self._output_dir / Path(f"run_{run}")
        self._path_ray = (self._run_output_dir / Path("raytune")).resolve()
        self._path_ray.mkdir(parents=True, exist_ok=True)

        self._set_current_datasets()
        self._model = self._get_model_for_dataset(self._train_set)

    def _execute_run(self):
        self._tune_hyperparameters()
        self._train_and_evaluate_best_model_config()

    def _tune_hyperparameters(self):
        tuning_function = self._get_raytune_tuning_function()
        tuner = self._get_tuner(tuning_function)
        
        self._tuning_results = tuner.fit()

    def _train_and_evaluate_best_model_config(self):
        best_result = self._get_best_result_from_tuning()
        best_configs = best_result.config

        self._logger.info(f"Best configs: {best_configs}")
        self._logger.info(
            f"Running final training (train + val set) and evaluation (on test set) with best configs"
        )
        train_func = TrainEpochLoop(
            device=self._device,
            logger=self._logger,
            max_batches=self._max_batches
        )
        eval_func = EvaluateOnDataloaderLoop(
            device=self._device,
            max_batches=self._max_batches,
            return_outputs=False
        )
        train_eval_loop = TrainAndEvalLoop(
            train_epoch_function=train_func,
            validation_function=eval_func,
            datasets={"train": self._train_val_set, "val": self._test_set},
            model=self._model,
            max_epochs=self._max_epochs,
            logger=self._logger,
            report_to_raytune=False,
            lr=best_configs["lr"],
            weight_decay=best_configs["weight_decay"],
            loss_fn=self._get_train_val_loss_fn()
        )
        train_eval_loop.run()

        test_loader = DataLoader(self._test_set, batch_size=32, shuffle=False)
        test_func = EvaluateOnDataloaderLoop(
            device=self._device,
            max_batches=self._max_batches,
            return_outputs=True
        )
        metrics, (pred, labels) = test_func(
            model=train_eval_loop.model, 
            dataloader=test_loader, 
            loss_fn=self._get_train_val_loss_fn()
        )
        save_output_artifacts(
            metric_records=train_eval_loop._records,
            outputs=pred,
            labels=labels,
            configs=best_configs,
            output_dir=self._run_output_dir,
            category_names=self._test_set.label_encoder.classes_,
            model=train_eval_loop.model
        )

    def _get_best_result_from_tuning(self):
        error_msg = "You must run the tuning before getting the best result"
        assert self._tuning_results is not None, error_msg
        
        best_result = self._tuning_results.get_best_result(
            metric=self._tuning_evaluation_metric, mode=self._tuning_evaluation_mode
        )

        return best_result

    def _get_raytune_tuning_function(self):
        train_func = TrainEpochLoop(
            device=self._device,
            logger=self._logger,
            max_batches=self._max_batches
        )
        eval_func = EvaluateOnDataloaderLoop(
            device=self._device,
            max_batches=self._max_batches,
            return_outputs=False
        )
        train_eval_loop = TrainAndEvalLoop(
            train_epoch_function=train_func,
            validation_function=eval_func,
            datasets={"train": self._train_set, "val": self._val_set},
            model=self._model,
            max_epochs=self._max_epochs,
            logger=self._logger,
            report_to_raytune=True,
            loss_fn=self._get_train_loss_fn()
        )
        
        return train_eval_loop

    def _get_tuner(self, tuning_function):
        ray_scheduler = self._get_scheduler()
        trainable = self._get_trainable(tuning_function)
        hp_config = self._get_hyperparameter_configs()
        
        tuner = self._build_tuner(ray_scheduler, trainable, hp_config)
        
        return tuner

    def _build_tuner(self, ray_scheduler, trainable, hp_config):
        tuner = tune.Tuner(
            trainable=trainable,
            tune_config=tune.TuneConfig(
                metric=self._tuning_evaluation_metric,
                mode=self._tuning_evaluation_mode,
                scheduler=ray_scheduler,
                num_samples=self._num_samples,
            ),
            run_config=air.RunConfig(
                storage_path=str(self._path_ray),
                name=self.__class__.__name__,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_score_attribute="accuracy", num_to_keep=1
                ),
            ),
            param_space=hp_config,
        )
        
        return tuner

    def _get_hyperparameter_configs(self):
        hp_config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-4, 1e-1)
        }
        
        return hp_config

    def _get_trainable(self, train_eval_loop):
        if 'cuda' in self._device:
            trainable = tune.with_resources(
                train_eval_loop, {"cpu": self._cpu_per_trial, "gpu": self._gpu_per_trial}
            )
        else:
            trainable = tune.with_resources(train_eval_loop, {"cpu": self._cpu_per_trial})
        return trainable

    def _get_scheduler(self):
        ray_scheduler = tune.schedulers.ASHAScheduler(
            max_t=self._max_epochs,
            grace_period=self._max_epochs,  # let warm restarts work
            reduction_factor=2,
        )
        
        return ray_scheduler

    def _set_current_datasets(self):
        self._train_set = self._get_mode_dataset(mode="train")
        self._val_set = self._get_mode_dataset(mode="val")
        self._test_set = self._get_mode_dataset(mode="test")

    @property
    def _train_val_set(self):
        return ConcatDataset([self._train_set, self._val_set])
    
    def _get_mode_dataset(self, mode: Literal['train', 'val', 'test']):
        dataset = self._dataset_creator.set_mode_and_run(
            mode=mode, run=self._current_run
        ).get()
        return dataset
    
    def _get_model_for_dataset(self, dataset: TCGAExpressionDataset):
        X, _ = dataset[0]
        n_input_dims = X.shape[0]
        n_classes = dataset.n_classes

        model = self._model_creator.get(
            input_dim=n_input_dims, output_dim=n_classes
        )

        return model
    
    def _get_logger(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
        return logger
    
    def _get_train_loss_fn(self):
        class_weights = self._train_set._get_class_weights()
        return self._get_loss_fn(class_weights)
    
    def _get_train_val_loss_fn(self):
        class_weights = self._train_set._get_class_weights()
        return self._get_loss_fn(class_weights)

    def _get_loss_fn(self, class_weights: Union[None, torch.tensor] = None):
        self._logger.info(f"Using class weights: {class_weights}")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        return loss_fn
