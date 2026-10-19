# Code README

Setup: Colab T4, seed 0, data in /content/data, run from the folder that contains code/.

## Commands
- Task 1: python code/task1.py --data /content/data
- Task 2: python code/task2.py
- Task 3: python code/task3.py --data /content/data
- Task 4 sce, filter: python code/task4.py --data /content/data --methods sce,filter
- Task 4 coteach: python code/task4_coteach.py --data /content/data
- Task 5: python code/task5.py --data /content/data  (submitted best run is tagged f40, see outputs/t5_f40_summary.json)
Shared code: common.py, model.py, train.py. test_common.py is a unit test.

## Submitted vs exploratory
Submitted Task 4 methods: sce, filter, coteach (best: coteach).
Exploratory runs also executed and not submitted: fsce (task4_fsce.py), ls, relabel (task4_relabel.py), plus and cotplus (task4_cotplus.py), and extra forget rates f35, f40, f45. Their logs are in outputs/.

Note: the rules allow 1-3 methods; I submitted 3 (sce, filter, coteach) but also ran the exploratory variants above while tuning, and I am reporting them for transparency. Selection used noisy validation accuracy only.

## Label usage
Clean labels were used only for measurement. val_acc_clean was logged but never used for model selection, early stopping or hyperparameter choice. The forget rate was chosen on noisy validation accuracy; Task 5 selection used synthetic validation labels.
