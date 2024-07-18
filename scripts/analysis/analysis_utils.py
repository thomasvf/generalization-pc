from pathlib import Path
from typing import Iterable, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score


from functools import partial


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
    def __init__(self, path_dataset, iterable_path_results) -> None:
        self.path_dataset = path_dataset
        self.paths = iterable_path_results

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
            run_results = predictions.groupby("cohort")[["labels", "predictions"]].apply(
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


class SingleModelComparisonArtifactsBuilder:
    def __init__(
        self,
        pc_results: PanCancerPredictions,
        cs_results_list: Iterable[CohortSpecificPredictions],
        output_dir: str,
    ) -> None:
        self.pc_results = pc_results
        self.cs_results_list = cs_results_list
        self.output_dir = Path(output_dir)

    def build_scores_table(self, score_func, **kwargs):
        scores_all = pd.DataFrame()

        pc_scores = self.pc_results.build_scores_table(score_func=score_func, **kwargs)
        pc_scores["model"] = "pan-cancer"
        scores_all = pd.concat((scores_all, pc_scores), axis=0)

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
        y_lim=(0.5, 1.05),
        suffix="",
    ):
        self.output_dir.mkdir(exist_ok=True, parents=True)

        scores = self.build_scores_table(score_func=score_func)
        scores["cohort"] = scores["cohort"].str.upper()
        scores = scores.replace(
            {"pan-cancer": "Pan-cancer", "cohort-specific": "Cohort-specific"}
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
        ax.set_ylim(y_lim)
        fig.tight_layout()
        if save_fig:
            fig.savefig(
                self.output_dir / f"{score_name}_comparison_pc_vs_cs{suffix}.jpg"
            )
            fig.savefig(
                self.output_dir / f"{score_name}_comparison_pc_vs_cs{suffix}.pdf"
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
            {"pan-cancer": "Pan-cancer", "cohort-specific": "Cohort-specific"}
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
        sns.scatterplot(data=stats_diff, x=x, y=y, ax=ax, s=85)
        ax.grid()
        if save_fig:
            fig.savefig(self.output_dir / f"{score_name}_perf_vs_var.jpg")
            fig.savefig(self.output_dir / f"{score_name}_perf_vs_var.pdf")

        corr_result = stats.pearsonr(stats_diff[x], stats_diff[y])
        print("Pearson R: {}".format(corr_result))

    def get_pc_sample_counts(self, run=0):
        counts = self.pc_results.build_samples_per_run_table()[run]
        return counts

    def make_f1_comparison_plot(self, **kwargs):
        self.make_comparison_plot(
            score_func=partial(f1_score, average="macro"), **kwargs
        )

    def make_accuracy_comparison_plot(self, **score_kwargs):
        self.make_comparison_plot(score_func=partial(accuracy_score), **score_kwargs)