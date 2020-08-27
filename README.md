# xxxx  

## Project structure


    .
    ├── data                 # Data for modelling
    ├── results              # Results 
    ├── scripts              # Automated tests and run as .py
    ├── src                  # Source, models, tools, utilities
    ├── LICENSE
    ├── README.md
    └── env.yml              # Conda env for linux
______________________________________________
## Running it

There are 3 important data files, all sharing the same index

* `data/xxx.csv`          | xxx
* `data/xxx.csv`          | xxx 
* `data/xxx.csv`          | xxx

______________________________________________
The code is set up as follows:

> `src` has all the modules necessary for modelling

> `src/configs.py` xxxx
> `src/models.py` xxxx
> `src/utils.py` xxxx

______________________________________________
Preprocessing

* TB

Installation

`conda create -n env python=3.6 scikit-learn=0.22 numpy pandas seaborn jupyter`
`conda install -c conda-forge imbalanced-learn bayesian-optimization eli5 umap-learn pandas-profiling molvs` 

