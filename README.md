# Solubility prediction project

The results generated using the code in this repository have been published in Journal of Chemometrics: https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.3349

If you use the code/library in (other) projects, please consider citing our paper.

## Project structure

    .
    ├── data                 # Data for modelling
    ├── results              # Results 
    ├── experiments          # Automated tests
    ├── src                  # Source, models, tools, utilities
    ├── LICENSE
    └── README.md            # Brief repo description and installation recommendation
______________________________________________
## The repository

There are 3 important data files, all sharing the same index (canonical SMILES).
The data is also published at http://doi.org/10.5281/zenodo.4008331

* `data/descriptors.csv`          | Descriptors file
* `data/fingerprints.csv`         | Fingerprints file
* `data/solubility_data.csv`      | Predictive target and data splits (random, picking, pca split)


The /src directory has all relevant modules and functions for modelling and preprocessing.
The /results directory is the data drop from trained models. 


## Best regressor

This scripts creates a pickle file with model parameters and results.
Should be run as:
>> python best_regressor.py
 

## Running the winning models

The two winning models in our work (LASSO and Random Forest).
Model parameters are included in the files.
Should be run as 

>> python run_indi_model_lasso.py

or

>>> python run_indi_model_rf.py

______________________________________________
The code is set up as follows:

> `src` has all the modules necessary for modelling

> `src/configs.py` 		   | Parameter space definitions for ML models
> `src/models.py`  		   | Optimization and modelling modules
> `src/model_support.py`   | Preprocessing routines
> `src/utils.py`   		   | Aux
