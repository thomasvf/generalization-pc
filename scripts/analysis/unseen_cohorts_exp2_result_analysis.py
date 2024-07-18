from functools import partial
from pathlib import Path
from typing import Iterable, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


PATH_CS_WITH_PC_INDICES_RESULTS = Path("results/outputs_cs_with_pc_indices")
PATH_UNSEEN_COHORT_RESULTS = Path("results/outputs_without_target_cohorts")
PATH_ARTIFACTS = Path("artifacts")
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


class ResultPathsIterable:
    def __init__(self, path_base: str, fname_template: str, runs=(0, 1, 2, 3, 4)):
        self.path_base = path_base
        self.fname_template = fname_template
        self.runs = runs

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, i):
        path_file = Path(self.path_base) / self.fname_template.format(self.runs[i])
        return path_file


class PanCancerPredictions:
    def __init__(self, path_dataset, iterable_path_results, cohort: str) -> None:
        self.path_dataset = path_dataset
        self.paths = iterable_path_results
        self.cohort = cohort

        n_holdouts = 5
        self.random_states = []
        rng = np.random.default_rng(seed=123)
        for rep in range(n_holdouts):
            random_state = int(rng.integers(500))
            self.random_states.append(random_state)

    def build_samples_per_run_table(self):
        counts = pd.DataFrame()
        for i, filepath in enumerate(self.paths):
            predictions = pd.read_csv(filepath, index_col=0)
            run_counts = predictions["cohort"].value_counts().rename(i)
            counts = pd.concat((counts, run_counts), axis=1)

        return counts

    def build_scores_table(self, score_func, wide_form: bool = False):
        """Build a table with the f1 scores for each cohort

        Returns
        -------
        pd.DataFrame
            A dataframe with the results
        """
        results = pd.DataFrame()
        for i, filepath in enumerate(self.paths):
            predictions = pd.read_csv(filepath, index_col=0)
            run_results = predictions.groupby("cohort").apply(
                lambda x: score_func(x["labels"], x["predictions"])
            )
            run_results = run_results.rename(i)
            results = pd.concat((results, run_results), axis=1)

        if wide_form:
            return results

        m = pd.melt(results, value_name="score", var_name="run", ignore_index=False)
        m = m.rename_axis(index="cohort").reset_index()
        return m

    def build_f1_scores_table(self, **kwargs):
        """Build a table with the f1 scores for each cohort

        Returns
        -------
        pd.DataFrame
            A dataframe with the results
        """
        scores = self.build_scores_table(
            score_func=partial(f1_score, average="macro"), **kwargs
        )
        return scores


class CohortSpecificPredictions:
    def __init__(self, iterable_path_results, cohort) -> None:
        self.paths = iterable_path_results
        self.cohort = cohort

        n_holdouts = 5
        self.random_states = []
        rng = np.random.default_rng(seed=123)
        for rep in range(n_holdouts):
            random_state = int(rng.integers(500))
            self.random_states.append(random_state)

    def build_scores_table(self, score_func, **kwargs):
        """Build a table with the f1 scores on the runs

        Returns
        -------
        pd.DataFrame
            A dataframe with the results
        """
        records = []
        for i, filepath in enumerate(self.paths):
            predictions = pd.read_csv(filepath, index_col=0)
            records.append(
                {
                    "run": i,
                    "score": score_func(
                        predictions["labels"], predictions["predictions"]
                    ),
                }
            )
        results = pd.DataFrame.from_records(records)
        return results

    def build_f1_scores_table(self, **kwargs):
        """Build a table with the f1 scores

        Returns
        -------
        pd.DataFrame
            A dataframe with the results
        """
        scores = self.build_scores_table(
            score_func=partial(f1_score, average="macro"), **kwargs
        )
        return scores

    def get_n_samples(self, run=0):
        n_samples = len(pd.read_csv(self.paths[run]))
        return n_samples


class SingleModelComparisonArtifactsBuilder:
    def __init__(
        self,
        pc_results_list: Iterable[CohortSpecificPredictions],
        cs_results_list: Iterable[CohortSpecificPredictions],
        output_dir: str,
    ) -> None:
        self.pc_results_list = pc_results_list
        self.cs_results_list = cs_results_list
        self.output_dir = Path(output_dir)

    def build_scores_table(self, score_func, **kwargs):
        scores_all = pd.DataFrame()

        for pc_test_results in self.pc_results_list:
            pc_test_scores_cohort = pc_test_results.build_f1_scores_table()
            pc_test_scores_cohort["cohort"] = pc_test_results.cohort
            pc_test_scores_cohort["model"] = "pan-cancer-test"
            scores_all = pd.concat((scores_all, pc_test_scores_cohort), axis=0)

        for cs_results in self.cs_results_list:
            cs_scores_cohort = cs_results.build_f1_scores_table()
            cs_scores_cohort["cohort"] = cs_results.cohort
            cs_scores_cohort["model"] = "cohort-specific"
            scores_all = pd.concat((scores_all, cs_scores_cohort), axis=0)

        return scores_all.reset_index(drop=True)

    def get_count_tables(
        self,
    ):
        """
        Count samples in each cohort and run for the pan-cancer and cohort-specific datasets.
        The counts should match.
        """
        pc_counts = self.pc_results.build_samples_per_run_table()

        cs_counts = {}
        for cs_results in self.cs_results_list:
            cohort = cs_results.cohort
            n_samples = []
            for run in range(5):
                cs_n_samples_run = cs_results.get_n_samples(run=run)
                n_samples.append(cs_n_samples_run)
            cs_counts[cohort] = n_samples
        cs_counts = pd.DataFrame.from_dict(data=cs_counts, orient="index")

        return pc_counts, cs_counts

    def build_f1_scores_table(self, **kwargs):
        scores = self.build_scores_table(
            score_func=partial(f1_score, average="macro"), **kwargs
        )
        return scores

    def make_comparison_plot(
        self,
        score_func,
        score_name: str = None,
        ax=None,
        cohorts_list: List = None,
        save_fig: bool = True,
        y_lim=None,
        suffix="",
    ):
        self.output_dir.mkdir(exist_ok=True, parents=True)

        scores = self.build_scores_table(score_func=score_func)
        scores["cohort"] = scores["cohort"].str.upper()
        scores = scores.replace(
            {"pan-cancer-test": "Pan-cancer", "cohort-specific": "Cohort-specific"}
        )
        if score_name is not None:
            scores = scores.rename(columns={"score": score_name})
        else:
            score_name = "score"
        scores = scores.rename(columns={"model": "Model", "cohort": "Cohort"})

        if cohorts_list is not None:
            cohorts_list = [c.upper() for c in cohorts_list]
            scores = scores[scores["Cohort"].isin(cohorts_list)]

        if ax is None:
            fig, ax = plt.subplots()
        sns.barplot(
            data=scores,
            x="Cohort",
            y=score_name,
            hue="Model",
            ax=ax,
            order=cohorts_list,
        )
        ax.tick_params(axis="x", labelrotation=90)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        fig.tight_layout()
        if save_fig:
            fig.savefig(
                self.output_dir / f"{score_name}_comparison_pctest_vs_cs{suffix}.jpg"
            )
            fig.savefig(
                self.output_dir / f"{score_name}_comparison_pctest_vs_cs{suffix}.pdf"
            )

        scores_mean = (
            scores.groupby(["Model", "Cohort"]).agg({score_name: "mean"}).reset_index()
        )
        scores_mean = scores_mean.pivot(columns=["Model"], index="Cohort")
        scores_mean[("Test F1 Macro Average", "Difference")] = (
            scores_mean[(score_name, "Pan-cancer")]
            - scores_mean[(score_name, "Cohort-specific")]
        )
        scores_mean[("Test F1 Macro Average", "Ratio")] = (
            scores_mean[(score_name, "Pan-cancer")]
            / scores_mean[(score_name, "Cohort-specific")]
        )
        scores_mean.to_csv(self.output_dir / f"{score_name}_table.csv")

        return ax

    def performance_vs_variance_plot(
        self,
        score_func,
        score_name: str = None,
        ax=None,
        cohorts_list: List = None,
        save_fig: bool = True,
    ):
        self.output_dir.mkdir(exist_ok=True, parents=True)

        scores = self.build_scores_table(score_func=score_func)
        scores["cohort"] = scores["cohort"].str.upper()
        scores = scores.replace(
            {"pan-cancer-test": "Pan-cancer", "cohort-specific": "Cohort-specific"}
        )
        if score_name is not None:
            scores = scores.rename(columns={"score": score_name})
        else:
            score_name = "score"
        scores = scores.rename(columns={"model": "Model", "cohort": "Cohort"})

        if cohorts_list is not None:
            cohorts_list = [c.upper() for c in cohorts_list]
            scores = scores[scores["Cohort"].isin(cohorts_list)]
        print(scores)

        scores_stats = (
            scores.groupby(["Model", "Cohort"])
            .agg({score_name: ["mean", "var"]})
            .droplevel(level=0, axis=1)
        )
        stats_diff: pd.DataFrame = (
            scores_stats.loc[("Pan-cancer",)] - scores_stats.loc[("Cohort-specific",)]
        )
        stats_diff["var"] = -stats_diff["var"]

        x = "Variance Reduction"
        y = "Performance Increase"
        stats_diff = stats_diff.rename(columns={"var": x, "mean": y})

        print(stats_diff)
        fig, ax = plt.subplots()
        sns.scatterplot(data=stats_diff, x=x, y=y, ax=ax, s=70)
        ax.grid()
        if save_fig:
            fig.savefig(self.output_dir / f"{score_name}_perf_vs_var.jpg")
            fig.savefig(self.output_dir / f"{score_name}_perf_vs_var.pdf")

        corr_result = stats.pearsonr(stats_diff[x], stats_diff[y])
        print("Variance Influence Pearson R: {}".format(corr_result))

    def get_pc_sample_counts(self, run=0):
        counts = self.pc_results.build_samples_per_run_table()[run]
        return counts

    def make_f1_comparison_plot(self, **kwargs):
        self.make_comparison_plot(
            score_func=partial(f1_score, average="macro"), **kwargs
        )

    def make_accuracy_comparison_plot(self, **score_kwargs):
        self.make_comparison_plot(score_func=partial(accuracy_score), **score_kwargs)


def build_default_predictions_handles(cohorts):
    """Construct default PanCancerPredictions and CohortSpecificPredictions objects.

    Returns
    -------
    Tuple[pd.DataFrame, List[pd.DataFrame]]
        Pan-cancer results and list of cohort-specific results
    """

    cs_results_list = []
    for cohort in cohorts:
        results_it = ResultPathsIterable(
            path_base=PATH_CS_WITH_PC_INDICES_RESULTS / f"{cohort}_tumor_prediction",
            fname_template="run_{}/final_model_results/predictions.csv",
        )
        cs_results = CohortSpecificPredictions(
            iterable_path_results=results_it, cohort=cohort
        )
        cs_results_list.append(cs_results)

    pc_test_results_list = []
    for cohort in cohorts:
        results_it = ResultPathsIterable(
            path_base=PATH_UNSEEN_COHORT_RESULTS / f"{cohort}_tumor_prediction",
            fname_template="run_{}/final_model_results/predictions.csv",
        )
        pc_test_results = CohortSpecificPredictions(
            iterable_path_results=results_it, cohort=cohort
        )
        pc_test_results_list.append(pc_test_results)

    return pc_test_results_list, cs_results_list


def dataset_size_influence():
    cohorts = COHORTS

    cohort_sizes = pd.read_csv(path_dataset_ratios_output, index_col=0).iloc[:-1, :][
        ["Total"]
    ]
    cohort_sizes = cohort_sizes.sort_values("Total", ascending=False)
    print(cohort_sizes)

    pc_results_list, cs_results_list = build_default_predictions_handles(
        cohorts=cohorts
    )
    art_builder = SingleModelComparisonArtifactsBuilder(
        pc_results_list=pc_results_list,
        cs_results_list=cs_results_list,
        output_dir=PATH_ARTIFACTS / "unseen_sample_size",
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
    scores_pc = scores[scores["model"] == "pan-cancer-test"]
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
    sns.scatterplot(data=scores_diff, x=x, y=y, ax=ax)
    ax.grid()
    fig.savefig(art_builder.output_dir / "scatter_scores_sample_size.jpg")
    fig.savefig(art_builder.output_dir / "scatter_scores_sample_size.pdf")

    corr_result = stats.pearsonr(scores_diff[x], scores_diff[y])
    print("Dataset Influence Pearson R: {}".format(corr_result))


def imbalance_influence():
    cohorts = COHORTS

    cohort_imbalances = (
        1 / pd.read_csv(path_dataset_ratios_output, index_col=0).iloc[:-1, :][["Ratio"]]
    )
    cohort_imbalances = cohort_imbalances.sort_values("Ratio", ascending=True)
    print(cohort_imbalances)

    pc_results_list, cs_results_list = build_default_predictions_handles(
        cohorts=cohorts
    )
    art_builder = SingleModelComparisonArtifactsBuilder(
        pc_results_list=pc_results_list,
        cs_results_list=cs_results_list,
        output_dir=PATH_ARTIFACTS / "unseen_imbalance",
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
    scores_pc = scores[scores["model"] == "pan-cancer-test"]
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
    sns.scatterplot(data=scores_diff, x=x, y=y, ax=ax)
    ax.grid()
    ax.set_xlabel("Imbalance Ratio")
    fig.savefig(art_builder.output_dir / "scatter_scores_imbalance.jpg")
    fig.savefig(art_builder.output_dir / "scatter_scores_imbalance.pdf")

    corr_result = stats.pearsonr(scores_diff[x], scores_diff[y])
    print("Imbalance Influence Pearson R: {}".format(corr_result))


def compare_models(sort_by_pc: bool = True):
    cohorts = COHORTS
    cohorts = sorted(cohorts, key=lambda x: x)

    pc_results_list, cs_results_list = build_default_predictions_handles(
        cohorts=cohorts
    )
    art_builder = SingleModelComparisonArtifactsBuilder(
        pc_results_list=pc_results_list,
        cs_results_list=cs_results_list,
        output_dir=PATH_ARTIFACTS / "unseen_cohorts",
    )
    scores = art_builder.build_scores_table(
        score_func=partial(f1_score, average="macro")
    )
    if sort_by_pc:
        cohort_pc_means = (
            scores[scores["model"] == "pan-cancer-test"]
            .groupby("cohort")
            .agg({"score": np.mean})
        )
        cohorts = sorted(cohorts, key=lambda x: -cohort_pc_means.loc[x, "score"])

    art_builder.make_comparison_plot(
        score_func=partial(f1_score, average="macro"),
        cohorts_list=cohorts,
        score_name="Test F1 Macro Average",
        y_lim=(0.5, 1.02),
    )
    art_builder.performance_vs_variance_plot(
        score_func=partial(f1_score, average="macro"),
        cohorts_list=cohorts,
        score_name="Test F1 Macro Average",
    )


def main():
    # test_counts()
    compare_models()
    dataset_size_influence()
    imbalance_influence()


if __name__ == "__main__":
    main()
