# Feature-level attention on the UCI heart disease data: code, data and results

Code, data splits, seeds and result files for the study *Does Feature-Level Attention Improve Heart Disease Classification? A Controlled Benchmark with Matched Shapley Explanations and External Validation*. Everything here reproduces every number, table and figure of the manuscript from the four public UCI files.

## Contents

| Path | What it is |
|---|---|
| `code/pipeline.py` | The experiment. Nine models (logistic regression, decision tree, random forest, XGBoost, MLP, single-token attention and its affine control, feature-token attention and its affine control) on the same 50 folds of 10 x 5 repeated stratified cross-validation; matched Kernel SHAP for every model; attention maps; the integer-coding sensitivity run; the heads x dropout grid; extra initialisation seeds; external tests on the Hungarian, Swiss and VA Long Beach cohorts. All settings, seeds and hyperparameters are in the configuration block at the top of the file. |
| `code/revision2_analyses.py` | The two analyses added in revision: classical models refitted on 85% (the networks' own fit subset), 70% and 50% of each training fold (Section 7.4, Table 11), and the class-stratified bootstrap intervals and Swiss leave-one-out analysis of the external results (Section 7.1, Table 8). |
| `code/build_tables.py` | Derives the manuscript tables from the raw result files. Trains nothing. |
| `code/make_figures.py` | Draws every figure from `results/`. Trains nothing. |
| `data/` | The four processed UCI files and `heart-disease.names`, from https://archive.ics.uci.edu/dataset/45/heart+disease (DOI 10.24432/C52P4X), CC BY 4.0. |
| `results/split_indices_onehot.csv` | Patient indices (row numbers of the 297 complete Cleveland records) of the fit, inner-validation and test subset of every repeat and fold. `split_indices_integer.csv` and `split_indices_ext.csv` hold the same for the sensitivity and 8-feature runs. |
| `results/run_info.json` | Library versions, the full configuration (seeds included) and the code hash of the run that produced the results. |
| `results/dataset_info.json` | SHA-256 of each data file the results were computed from, and the analysed-set counts. |
| `results/cv_folds_*.csv`, `oof_predictions_*.csv` | Per-fold metrics and per-patient out-of-fold predictions. |
| `results/external_*.csv` | External-cohort predictions for every training seed, and their summaries. |
| `results/shap_*.csv`, `attention_*.csv` | SHAP values, rankings and agreement; attention maps and attention-SHAP comparisons. |
| `results/rev2_*.csv`, `training_size_table.csv`, `external_table_rev2.csv` | Outputs of `revision2_analyses.py`. |
| `figures/` | Every figure, as used in the manuscript. |

## Reproduce

```bash
pip install -r requirements.txt
cd code
python pipeline.py --stage cv --smoke          # quick check, about a minute
python pipeline.py --stage all --log full_run.log
python revision2_analyses.py
python build_tables.py
python make_figures.py
```

The full `pipeline.py` run took about 44 minutes on a 12-core CPU (stage times in `results/run_info.json`); `revision2_analyses.py` takes about 5 minutes. No GPU is needed.

## Reproducibility

- Seeds: global seed 42; fold seed = 42 + 1000 x repeat + fold; external training seeds = 42 + 7919 x (i + 1) for i = 0..9. Each network's initialisation, batch order and dropout use the fold seed; the five extra initialisation seeds of the initialisation analysis are fold seed + 100003 x (s + 1) for s = 0..4. `run_info.json` records the configuration and seed counts.
- Every split, seed and inner-validation subset is fixed and the networks train single-threaded, so the numbers are identical across runs on the same library versions.
- Finished folds are checkpointed under `.cache/<hash of pipeline.py>/`; an interrupted run resumes and gives the same numbers as an uninterrupted one.
- Fitting the classical models on 100% of each training fold in `revision2_analyses.py` reproduces the main analysis fold by fold; the check is recorded in `results/rev2_info.json`.

## Licence

Data: CC BY 4.0, from the UCI Machine Learning Repository (Janosi, Steinbrunn, Pfisterer and Detrano, 1989).
