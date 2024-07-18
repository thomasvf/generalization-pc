import logging
from pathlib import Path
from typing import Dict, Callable, Tuple, Union

import numpy as np
import ray.train
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ray import tune
import ray

from generalization_pc.models import FCModel


TrainFuncType = Callable[
    [
        nn.Module, 
        DataLoader, 
        torch.optim.Optimizer, 
        nn.Module,
        torch.optim.lr_scheduler.LRScheduler,
        int
    ],
    Tuple[nn.Module, Dict[str, float]],
]


class TrainAndEvalLoop:
    """Fitting and validation loop for the models.

    It can be called as a function with a dictionary of hyperparameters, just like
    RayTune requires.
    """

    def __init__(
        self,
        train_epoch_function: TrainFuncType,
        validation_function: Callable,
        datasets: Dict[str, Dataset],
        model: nn.Module,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        report_to_raytune: bool = False,
        logger: logging.Logger = None,
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        T_0: int = 1,
        T_mult: int = 2,
        eta_min: float = 1e-5,
    ):
        self._train_epoch_function = train_epoch_function
        self._validation_function = validation_function

        self._model = model
        self._datasets = datasets

        self._loss_fn = loss_fn
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._lr = lr
        self._weight_decay = weight_decay
        self._T_0 = T_0
        self._T_mult = T_mult
        self._eta_min = eta_min

        self._report_to_raytune = report_to_raytune

        self._records = []

        if logger is None:
            self._logger = self._create_logger()
        else:
            self._logger = logger

    def __call__(self, hp_config: Union[Dict, None] = None):
        self.run(hp_config)

    def run(self, hp_config: Union[Dict, None] = None):
        self._set_tuning_configs(hp_config)

        train_loader, val_loader = self._get_train_and_val_loaders()

        self.model = self._get_model()

        self.optimizer = self._get_optimizer(self.model)

        scheduler = self._get_lr_scheduler(self.optimizer)

        loss_fn = self._get_loss_fn()

        for epoch in range(self._max_epochs):
            self.model, train_metrics = self._train_epoch_function(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                epoch=epoch,
            )

            validation_metrics = self._validation_function(
                model=self.model, 
                dataloader=val_loader, 
                loss_fn=loss_fn,
                epoch=epoch
            )

            self._report_metrics(epoch, train_metrics, validation_metrics)

    def _get_loss_fn(self):
        return self._loss_fn

    def _get_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self._T_0,
            T_mult=self._T_mult,
            eta_min=self._eta_min,
        )

    def _get_optimizer(self, model):
        return torch.optim.AdamW(
            model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )

    def _report_metrics(self, epoch, train_metrics, validation_metrics):
        record_metrics = {"epoch": epoch}
        for k, v in train_metrics.items():
            record_metrics[f"train_{k}"] = v
        
        for k, v in validation_metrics.items():
            record_metrics[f"val_{k}"] = v

        self._logger.info("Metrics: ", extra=record_metrics)
        self._records.append(record_metrics)
        
        if self._report_to_raytune:
            ray.train.report(record_metrics)

    def _set_tuning_configs(self, config: Union[Dict, None] = None):
        if config is None:
            return
        
        if "lr" in config:
            self._lr = config["lr"]
        if "weight_decay" in config:
            self._weight_decay = config["weight_decay"]
        if "T_0" in config:
            self._T_0 = config["T_0"]
        if "T_mult" in config:
            self._T_mult = config["T_mult"]
        if "eta_min" in config:
            self._eta_min = config["eta_min"]

    def _get_train_and_val_loaders(self):
        drop_last_train = len(self._datasets["train"]) % self._batch_size == 1
        train_loader = DataLoader(
            self._datasets["train"], batch_size=self._batch_size, shuffle=True, drop_last=drop_last_train
        )

        val_loader = DataLoader(
            self._datasets["val"], batch_size=self._batch_size, shuffle=True
        )

        return train_loader, val_loader

    def _get_model(self):
        return self._model

    def _create_logger(self):
        logger = logging.getLogger(self.__class__.__name__)
        return logger


class TrainEpochLoop:
    """A class for abstracting the training loop for this specific project."""

    def __init__(
        self,
        device: str = "cpu",
        epoch: int = 0,
        log_period: int = 10,
        logger: logging.Logger = None,
        max_batches: int = None,
    ):
        self._device = device
        self._epoch = epoch
        self._log_period = log_period
        self._logger = logger
        self._max_batches = max_batches

    def __call__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        epoch: int = 0,
    ) -> Tuple[nn.Module, Dict[str, float]]:
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._dataloader = dataloader
        self._epoch = epoch
        self._scheduler = scheduler

        return self.run()

    def run(self):
        self._send_modules_to_device()

        self._model.train()

        iters = len(self._dataloader)
        correct, total, train_loss, train_steps = 0, 0, 0, 0
        for i, batch in enumerate(self._dataloader, 0):
            if self._should_stop(train_steps):
                break

            x, target = batch
            x, target = x.to(self._device), target.to(self._device)

            self._optimizer.zero_grad()
            y_predicted = self._model(x)
            loss = self._loss_fn(y_predicted, target)

            loss.backward()
            self._optimizer.step()

            with torch.no_grad():
                class_predicted = torch.argmax(y_predicted, dim=1)
                correct += (class_predicted == target).type(torch.float).sum().item()
                total += target.shape[0]

                train_loss += loss.detach().cpu().numpy()
                train_steps += 1

            self._scheduler_step_if_exists(iters, i)

            with torch.no_grad():
                self._log_metrics(correct, total, i, loss)

        train_loss = train_loss / train_steps
        train_accuracy = correct / total
        metrics = {"loss": train_loss, "accuracy": train_accuracy}

        return self._model, metrics

    def _log_metrics(self, correct, total, i, loss):
        if i % self._log_period == 0:
            with torch.no_grad():
                print("Batch: {}".format(i))
                print("Mean batch loss: {}".format(loss.item()))
                print("Batch Accuracy: {}".format(correct / total))

    def _scheduler_step_if_exists(self, iters, i):
        if self._scheduler is not None:
            self._scheduler.step(self._epoch + i / iters)

    def _send_modules_to_device(self):
        self._model.to(device=self._device)
        self._loss_fn.to(device=self._device)

    def _should_stop(self, batch_iteration: int):
        if self._max_batches is not None:
            return batch_iteration >= self._max_batches
        return False


class EvaluateOnDataloaderLoop:
    def __init__(
        self,
        device: str = "cpu",
        epoch: int = 0,
        max_batches: int = None,
        return_outputs: bool = False,
    ):
        self._device = device
        self._epoch = epoch
        self._max_batches = max_batches
        self._return_outputs = return_outputs

    def __call__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        epoch: int = 0,
    ):
        self._model = model
        self._loss_fn = loss_fn
        self._dataloader = dataloader
        self._epoch = epoch

        return self.run()

    def run(self):
        self._send_modules_to_device()

        self._model.eval()

        class_predictions = []
        outputs = []
        labels = []
        iters = len(self._dataloader)
        with torch.no_grad():
            total_loss = 0
            for i, batch in enumerate(self._dataloader, 0):
                if self._should_stop(i):
                    break

                x, t = batch
                x, t = x.to(self._device), t.to(self._device)

                y_predicted = self._model(x)
                loss = self._loss_fn(y_predicted, t)

                outputs.append(y_predicted.detach().cpu().numpy())
                class_predicted = torch.argmax(y_predicted, dim=1)
                class_predictions.append(class_predicted.detach().cpu().numpy())
                labels.append(t.detach().cpu().numpy())
                total_loss += loss.detach().cpu().numpy()

        outputs = np.concatenate(outputs, axis=0)
        class_predictions = np.concatenate(class_predictions, axis=0)
        labels = np.concatenate(labels, axis=0)

        metrics = {
            "loss": total_loss / iters,
            "balanced_accuracy": balanced_accuracy_score(labels, class_predictions),
            "accuracy": accuracy_score(labels, class_predictions),
        }

        if self._return_outputs:
            return metrics, (outputs, labels)

        return metrics

    def _send_modules_to_device(self):
        self._model.to(device=self._device)
        self._loss_fn.to(device=self._device)

    def _should_stop(self, batch_iteration: int):
        if self._max_batches is not None:
            return batch_iteration >= self._max_batches
        return False
