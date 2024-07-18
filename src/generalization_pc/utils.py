from pathlib import Path
from typing import Dict, Sequence, Union
import json

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_output_artifacts(
    metric_records: Sequence[Dict],
    outputs: np.ndarray,
    labels: np.ndarray,
    configs: Dict,
    output_dir: Union[str, Path],
    category_names: Sequence[str],
    model=None,
):
    output_dir = Path(output_dir) / "final_model_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    df_metrics = pd.DataFrame.from_records(metric_records)

    # Save network outputs and labels
    df_outputs = pd.DataFrame(outputs, columns=category_names)
    df_outputs["labels"] = labels
    df_outputs["labels"] = df_outputs["labels"].map(lambda x: category_names[x])
    df_outputs.to_csv(output_dir / "outputs.csv")

    # Save predictions and labels
    predictions = np.argmax(outputs, axis=1)
    data = {"predictions": predictions, "labels": labels}
    df_test_predictions = pd.DataFrame.from_dict(data)
    df_test_predictions["predictions"] = df_test_predictions["predictions"].map(
        lambda x: category_names[x]
    )
    df_test_predictions["labels"] = df_test_predictions["labels"].map(
        lambda x: category_names[x]
    )

    df_test_predictions.to_csv(output_dir / "predictions.csv")

    # Save confusion matrix
    fig, ax = plot_confusion_matrix(
        df_test_predictions,
        true_label_column="labels",
        predicted_label_column="predictions",
    )
    savefig(fig, output_dir, "confusion_matrix")

    # Save a few metrics per epoch
    df_metrics.to_csv(output_dir / "metrics.csv")

    # Save configs
    write_json(obj=configs, file_path=(output_dir / "model_configs.json"))

    # Save model
    if model is not None:
        torch.save(model.state_dict(), output_dir / "final_model.pt")


def plot_confusion_matrix(
    df_classif: pd.DataFrame,
    true_label_column: str = "true_label",
    predicted_label_column: str = "predicted",
):
    labels = df_classif[true_label_column].unique().tolist()
    cm = confusion_matrix(
        y_true=df_classif[true_label_column],
        y_pred=df_classif[predicted_label_column],
        labels=labels,
    )
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots()
    sns.heatmap(df_cm, annot=True, ax=ax, fmt="g", cbar=False, cmap=sns.cm.rocket_r)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Label")

    return fig, ax


def savefig(fig, output_dir, filename_stem, extensions=("pdf", "jpg")):
    for ext in extensions:
        fig.savefig(output_dir / f"{filename_stem}.{ext}")


def write_json(obj, file_path: Union[str, Path]):
    file_path = Path(file_path)
    with open(str(file_path), "w") as f:
        return json.dump(obj, f)
