"""
Compares the pan-cancer model on a specific task against cohort-specific models on 
the same task.
"""

from pathlib import Path

import pandas as pd
from generalization_pc.config import n_runs
from generalization_pc.config import max_epochs
from generalization_pc.config import num_samples
from generalization_pc.datasets import TCGAExpressionDSExp2PCCreator, TCGAExpressionDatasetCreator
from generalization_pc.experiment import ExperimentExecutor
from generalization_pc.config import TCGADatasetParameters
from generalization_pc.models import FCModelCreator
import argparse


ds_defs = TCGADatasetParameters()


def run_pancancer_models(output_dir: str):
    output_dir = Path(output_dir)

    pancancer_ds_creator = TCGAExpressionDatasetCreator(
        path_tcga_dataset=ds_defs.path_dataset,
        path_indices=ds_defs.path_indices,
        experiment=1,
        cohort="pancancer",
    )
    model_creator = FCModelCreator(hidden_dim=(256,))

    exp_output_dir = output_dir / "outputs_pc/"
    exp = ExperimentExecutor(
        dataset_creator=pancancer_ds_creator,
        model_creator=model_creator,
        output_dir=exp_output_dir,
        max_epochs=max_epochs,
        n_runs=n_runs,
        num_samples=num_samples,
    )

    exp.run_experiment()

    _add_cohort_label_to_predictions(n_runs, pancancer_ds_creator, exp_output_dir)


def run_cohort_specific_models(output_dir: str):
    output_dir = Path(output_dir)
    
    cohorts = ds_defs.get_cohorts()
    for cohort in cohorts:
        cs_ds_creator = TCGAExpressionDatasetCreator(
            path_tcga_dataset=ds_defs.path_dataset,
            path_indices=ds_defs.path_indices,
            experiment=1,
            cohort=cohort,
        )
        model_creator = FCModelCreator(hidden_dim=(256,))

        exp = ExperimentExecutor(
            dataset_creator=cs_ds_creator,
            model_creator=model_creator,
            output_dir=(
                output_dir
                / "outputs_cs_with_pc_indices/{}_tumor_prediction".format(cohort)
            ),
            max_epochs=max_epochs,
            n_runs=n_runs,
            num_samples=num_samples,
        )
        exp.run_experiment()


def run_pancancer_with_unseen_target_cohorts(output_dir: str):
    output_dir = Path(output_dir)

    cohorts = ds_defs.get_cohorts()
    for cohort in cohorts:
        print("Running pan-cancer model without cohort: {}".format(cohort))

        pc_ds_creator = TCGAExpressionDSExp2PCCreator(
            path_tcga_dataset=ds_defs.path_dataset,
            path_indices=ds_defs.path_indices,
            target_cohort=cohort,
        )
        model_creator = FCModelCreator(hidden_dim=(256,))

        exp = ExperimentExecutor(
            dataset_creator=pc_ds_creator,
            model_creator=model_creator,
            output_dir=(
                output_dir
                / "outputs_without_target_cohorts/{}_tumor_prediction".format(cohort)
            ),
            max_epochs=max_epochs,
            n_runs=n_runs,
            num_samples=num_samples,
        )
        exp.run_experiment()


def _add_cohort_label_to_predictions(n_runs, pancancer_ds_creator, exp_output_dir):
    for run in range(n_runs):
        test_set = pancancer_ds_creator.set_mode_and_run(mode='test', run=run).get()
        path_to_pc_run_predictions = exp_output_dir / f"run_{run}/final_model_results/predictions.csv"
        path_to_pc_run_preds_with_cohorts = path_to_pc_run_predictions.parent / 'predictions_with_cohorts.csv'

        df_predictions = pd.read_csv(path_to_pc_run_predictions, index_col=0)
        
        cohorts = test_set.get_examples_metadata()['cohort']
        df_predictions['cohort'] = cohorts.tolist()
        df_predictions.to_csv(path_to_pc_run_preds_with_cohorts)


def main():
    parser = argparse.ArgumentParser("Run all experiments.")
    parser.add_argument(
        "--output-dir", type=str, help="Output directory where to save results", required=True
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    run_cohort_specific_models(output_dir)
    run_pancancer_models(output_dir)
    run_pancancer_with_unseen_target_cohorts(output_dir)


if __name__ == "__main__":
    main()
