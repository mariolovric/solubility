# Solubility prediction project

The paper for this repo is published on doi.org/10.26434/chemrxiv.12746948.v1

## Project structure

    .
    ├── data                 # Data for modelling
    ├── results              # Results 
    ├── experiments              # Automated tests and run as .py
    ├── src                  # Source, models, tools, utilities
    ├── LICENSE
    └── README.md            # Brief repo description and installation recommendation
______________________________________________
## Running it

There are 3 important data files, all sharing the same index.
The data is originally published at http://doi.org/10.5281/zenodo.4008331

* `data/descriptors.csv`          | Descriptors file
* `data/fingerprints.csv`         | Fingerprints file
* `data/solubility_data.csv`      | Predictive target and data splits

## Best regressor

Should be run as
>> python best_regressor.py

## Running the chosen models

>> python run_indi_model_lasso.py
>> python run_indi_model_rf.py

______________________________________________
The code is set up as follows:

> `src` has all the modules necessary for modelling

> `src/configs.py` 		   | Parameter space definitions for ML models
> `src/models.py`  		   | Optimization and modelling modules
> `src/model_support.py`   | Preprocessing routines
> `src/utils.py`   		   | Auxiliary modules for data handling

______________________________________________


Installation recommendation

`conda create -n env python=3.6 scikit-learn=0.22 numpy pandas jupyter`
`conda install -c conda-forge imbalanced-learn bayesian-optimization eli5 ` 

