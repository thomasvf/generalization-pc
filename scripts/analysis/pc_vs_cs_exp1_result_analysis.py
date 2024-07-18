from functools import partial
from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


from generalization_pc.config import TCGADatasetParameters


from analysis_utils import (
    CohortSpecificPredictions,
    PanCancerPredictions,
    ResultPathsIterable,
    SingleModelComparisonArtifactsBuilder,
)


PATH_CS_WITH_PC_INDICES_RESULTS = Path("results/outputs_cs_with_pc_indices")
PATH_PC_RESULTS = Path("results/outputs_pc")
PATH_ARTIFACTS = Path("artifacts")
DS_PARAMS = TCGADatasetParameters()
path_dataset_ratios_output = PATH_ARTIFACTS / "dataset_info.csv"


COHORTS = [
    "brca",
    "kirc",
    "luad",
    "lusc",
    "esca",
    "kich",
    "blca",
    "thca",
    "kirp",
    "coad",
    "lihc",
    "ucec",
    "stad",
    "hnsc",
    "read",
    "prad",
]


def build_default_predictions_handles(cohorts):
    """Construct default PanCancerPredictions and CohortSpecificPredictions objects.

    Returns
    -------
    Tuple[pd.DataFrame, List[pd.DataFrame]]
        Pan-cancer results and list of cohort-specific results
    """

    cs_results_list = []
    for cohort in cohorts:
        results_iterator = ResultPathsIterable(
            path_base=PATH_CS_WITH_PC_INDICES_RESULTS / f"{cohort}_tumor_prediction",
            fname_template="run_{}/final_model_results/predictions.csv",
        )
        cs_results = CohortSpecificPredictions(
            iterable_path_results=results_iterator, cohort=cohort
        )
        cs_results_list.append(cs_results)

    results_iterator = ResultPathsIterable(
        path_base=PATH_PC_RESULTS,
        fname_template="run_{}/final_model_results/predictions_with_cohorts.csv",
    )
    pc_results = PanCancerPredictions(
        path_dataset=DS_PARAMS.path_dataset,
        iterable_path_results=results_iterator,
    )
    return pc_results, cs_results_list


def dataset_size_influence():
    cohorts = COHORTS

    cohort_sizes = pd.read_csv(path_dataset_ratios_output, index_col=0).iloc[:-1, :][
        ["Total"]
    ]
    cohort_sizes = cohort_sizes.sort_values("Total", ascending=False)
    print(cohort_sizes)

    pc_results, cs_results_list = build_default_predictions_handles(cohorts=cohorts)
    art_builder = SingleModelComparisonArtifactsBuilder(
        pc_results=pc_results,
        cs_results_list=cs_results_list,
        output_dir=PATH_ARTIFACTS / "pc_vs_cs_sample_size",
    )

    ## Plot ordered by sample size
    art_builder.make_comparison_plot(
        score_func=partial(f1_score, average="macro"),
        cohorts_list=cohort_sizes.index.tolist(),
        score_name="Test F1 Macro Average",
        suffix="_sample_size",
    )

    # Plot scatter of f1score difference vs sample size
    scores = art_builder.build_scores_table(partial(f1_score, average="macro"))
    scores_pc = scores[scores["model"] == "pan-cancer"]
    scores_cs = scores[scores["model"] == "cohort-specific"]

    scores_diff = pd.merge(
        left=scores_pc, right=scores_cs, on=("cohort", "run"), suffixes=("_pc", "_cs")
    )
    scores_diff["F1 Difference"] = scores_diff["score_pc"] - scores_diff["score_cs"]
    scores_diff["cohort"] = scores_diff["cohort"].str.upper()
    scores_diff = pd.merge(
        left=scores_diff, right=cohort_sizes, left_on="cohort", right_index=True
    )
    print(scores_diff)

    x = "Total"
    y = "F1 Difference"
    fig, ax = plt.subplots()
    sns.scatterplot(data=scores_diff, x=x, y=y, ax=ax, s=70)
    ax.grid()
    fig.savefig(art_builder.output_dir / "scatter_scores_sample_size.jpg")
    fig.savefig(art_builder.output_dir / "scatter_scores_sample_size.pdf")

    corr_result = stats.pearsonr(scores_diff[x], scores_diff[y])
    print("Pearson R: {}".format(corr_result))


def imbalance_influence():
    cohorts = COHORTS

    cohort_imbalances = (
        1 / pd.read_csv(path_dataset_ratios_output, index_col=0).iloc[:-1, :][["Ratio"]]
    )
    cohort_imbalances = cohort_imbalances.sort_values("Ratio", ascending=True)
    print(cohort_imbalances)

    pc_results, cs_results_list = build_default_predictions_handles(cohorts=cohorts)
    art_builder = SingleModelComparisonArtifactsBuilder(
        pc_results=pc_results,
        cs_results_list=cs_results_list,
        output_dir=PATH_ARTIFACTS / "pc_vs_cs_imbalance",
    )
    ## Plot ordered by sample size
    art_builder.make_comparison_plot(
        score_func=partial(f1_score, average="macro"),
        cohorts_list=cohort_imbalances.index.tolist(),
        score_name="Test F1 Macro Average",
        suffix="_imbalance",
    )

    # Plot scatter of f1score difference vs sample size
    scores = art_builder.build_scores_table(partial(f1_score, average="macro"))
    scores_pc = scores[scores["model"] == "pan-cancer"]
    scores_cs = scores[scores["model"] == "cohort-specific"]

    scores_diff = pd.merge(
        left=scores_pc, right=scores_cs, on=("cohort", "run"), suffixes=("_pc", "_cs")
    )
    scores_diff["F1 Difference"] = scores_diff["score_pc"] - scores_diff["score_cs"]
    scores_diff["cohort"] = scores_diff["cohort"].str.upper()
    scores_diff = pd.merge(
        left=scores_diff, right=cohort_imbalances, left_on="cohort", right_index=True
    )
    print(scores_diff)

    x = "Ratio"
    y = "F1 Difference"
    fig, ax = plt.subplots()
    sns.scatterplot(data=scores_diff, x=x, y=y, ax=ax, s=85)
    ax.grid()
    ax.set_xlabel("Imbalance Ratio")
    fig.savefig(art_builder.output_dir / "scatter_scores_imbalance.jpg")
    fig.savefig(art_builder.output_dir / "scatter_scores_imbalance.pdf")

    corr_result = stats.pearsonr(scores_diff[x], scores_diff[y])
    print("Pearson R: {}".format(corr_result))


def compare_models():
    """Compare the performance of the pan-cancer and cohort-specific models."""
    cohorts = COHORTS
    cohorts = sorted(cohorts, key=lambda x: x)

    pc_results, cs_results_list = build_default_predictions_handles(cohorts=cohorts)

    art_builder = SingleModelComparisonArtifactsBuilder(
        pc_results=pc_results,
        cs_results_list=cs_results_list,
        output_dir=PATH_ARTIFACTS / "pc_vs_cs",
    )
    art_builder.make_comparison_plot(
        score_func=partial(f1_score, average="macro"),
        cohorts_list=cohorts,
        score_name="Test F1 Macro Average",
        y_lim=(0.6, 1.02),
    )
    art_builder.performance_vs_variance_plot(
        score_func=partial(f1_score, average="macro"),
        cohorts_list=cohorts,
        score_name="Test F1 Macro Average",
    )


def test_counts():
    """
    Assert that the pan-cancer and cohort-specific results have the same number of
    samples per cohort.
    """
    print("Testing counts")
    
    cohorts = COHORTS
    pc_results, cs_results_list = build_default_predictions_handles(cohorts=cohorts)

    art_builder = SingleModelComparisonArtifactsBuilder(
        pc_results=pc_results,
        cs_results_list=cs_results_list,
        output_dir="artifacts/pc_vs_cs_counts",
    )
    pc_counts, cs_counts = art_builder.get_count_tables()
    cs_counts = cs_counts.sort_index()
    pc_counts_csindex = pc_counts.loc[
        pc_counts.index.intersection(cs_counts.index), :
    ].sort_index()

    diff = pc_counts_csindex.compare(cs_counts)
    print(pc_counts_csindex)
    print(cs_counts)
    print("Differences:")
    print(diff)


def analyze_dataset_ratios():
    path_metadata = DS_PARAMS._path_metadata

    metadata = pd.read_csv(path_metadata, index_col=0)
    counts = metadata.value_counts().reset_index()
    counts = counts.pivot(index=["cohort"], columns=["sample_type"])
    counts.columns = counts.columns.droplevel(0)
    counts.index = counts.index.str.upper()
    counts.loc["Total"] = counts.sum(numeric_only=True, axis=0)
    counts.loc[:, "Total"] = counts.sum(numeric_only=True, axis=1)
    counts.loc[:, "Ratio"] = counts["Solid Tissue Normal"] / counts["Primary Tumor"]
    counts.to_csv(path_dataset_ratios_output)


def main():
    test_counts()
    analyze_dataset_ratios()
    compare_models()
    dataset_size_influence()
    imbalance_influence()


if __name__ == "__main__":
    main()
