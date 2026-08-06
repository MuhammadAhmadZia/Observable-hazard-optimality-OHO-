# Experiment scripts to add here

The figure generator (make_figures.py) is included and runnable. The experiment
suite that produced the CSVs in ../results should be added to this folder before
the repository is made public. Add:

1. vb_setup.py          shared data loading, hazard lexicons, metrics
2. verification_suite.ipynb   the FAST-flag notebook that runs sections A to H
3. observability_fixed.py     the fixed-policy observability analysis (lambda=0.6)

These exist in the working environment and should be exported from the final
Colab notebook. Do not publish until the CSVs in ../results can be regenerated
end to end from this folder.
