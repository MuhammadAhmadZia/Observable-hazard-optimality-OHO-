# Observable Hazard Optimality in Health-Aware Recommendation

Code and data for the paper *Observable hazard optimality and the role of machine
learning in health-aware recommender systems*.

## What this repository contains

The study formalizes safety-aware ranking under a deterministic, fully observable
hazard signal, proves that transparent analytical re-ranking is optimal in that
setting, and maps the boundary where learning starts to help. This repository holds
the experiment code, the result files behind every table and figure, and the figure
generation script.

## Layout

```
code/          experiment scripts and the figure generator
results/       one CSV per experiment, consumed by the figure script
figures/       generated figures (run the figure script to populate)
docs/          notes on reproducing each table and figure
```

## Result files

Each CSV in `results/` maps to a section of the paper.

| File                    | Paper element                          |
|-------------------------|----------------------------------------|
| A_proposition.csv       | Table 2, numerical verification        |
| B_decomposition.csv     | Table 11 (withdrawn decomposition)     |
| C_dfi_sensitivity.csv   | Table 3, depth index by segment        |
| D_battery.csv           | Table 4-6, mitigation battery          |
| E_steelman.csv          | Table 7-8, learned formulations        |
| E_theta.csv             | learned penalty weights                |
| F_observability.csv     | Table 10, observability study          |
| G_parity.csv            | representation parity (withdrawn)      |
| H_domains.csv           | Table 10 conditionality / second axis  |

## Reproducing the figures

```
pip install -r requirements.txt
cd code
python make_figures.py --csv ../results --out ../figures
```

The script also runs in Colab or Jupyter; see the header of `make_figures.py`.

## Environment

All experiments run on free-tier cloud notebooks. No GPU is required. The full
pipeline was executed on Google Colab.

## Data source

The primary catalogue is the public cosmetics ingredient dataset
`farahjbara/Comparing-Cosmetics-by-Ingredients`, downloaded at runtime. Hazard
reference lists are derived from public regulatory sources cited in the paper.

## License

Released under the MIT License. See LICENSE.

## Citation

If you use this code or data, please cite the paper. See CITATION.cff.
