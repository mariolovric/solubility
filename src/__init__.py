# Basic imports
import pandas as pd
import numpy as np
import time
from math import sqrt
import pickle

# Model imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from lightgbm import LGBMRegressor

# sklearn support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

# visualization
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

# metrics import
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from eli5.sklearn import PermutationImportance
from eli5 import explain_weights_df

# Filtering out warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

# Data paths
data_folder = "../data/"
target_path = data_folder + "solubility_data.csv"
fingerprint_path = data_folder + "fingerprints.csv"
descriptor_path = data_folder + "descriptors.csv"
