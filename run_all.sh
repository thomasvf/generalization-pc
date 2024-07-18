#!/bin/bash

python scripts/run_experiments.py --output-dir output
python scripts/analysis/pc_vs_cs_exp1_result_analysis.py
python scripts/analysis/unseen_cohorts_exp2_result_analysis.py
