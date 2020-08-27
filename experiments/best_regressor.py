import sys
import warnings
sys.path.append('..')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

from src.model_support import *
from src.models import regression_test
from src.utils import *


if __name__ == '__main__':

    # load data and set indices
    df_log, df_fp, df_desc, df_all, predictive_dataset = load_data()
    train_pick, val_pick, test_pick, train_rand, test_rand = split_rand_pick(df_log, 'diversity', 'random')

    # experimental matrix, data, preprocessing or not, splitting options
    for predictive_set_key in predictive_dataset.keys():
        for preprocessing_decision in [True, False]:
            if predictive_set_key != 'fp' and preprocessing_decision is False:

                # set X and y
                X = SetXMatrix(predictive_dataset, predictive_set_key, preprocess=preprocessing_decision)
                y = df_log.logS0

                for splitter_option in ['rand', 'pick']:

                    X_train, X_val, X_ext, y_train, y_val, y_ext = return_sets(splitter_option, X, y, train_pick,
                                                                               val_pick, test_pick, train_rand,
                                                                               test_rand)

                    # Random Forest
                    test_rf = regression_test(5, 10, 50, 'rf_', X_train, X_val, X_ext, y_train, y_val, y_ext)
                    # PLS
                    test_pls = regression_test(5, 5, 10, 'pls_', X_train, X_val, X_ext, y_train, y_val, y_ext)
                    # LASSO
                    test_lasso = regression_test(1, 10, 40, 'lasso_', X_train, X_val, X_ext, y_train, y_val, y_ext)
                    # LightGBM
                    test_lgbm = regression_test(5, 10, 50, 'lg_', X_train, X_val, X_ext, y_train, y_val, y_ext)

                    save_result({'LightGBM': test_lgbm,
                                 'LASSO': test_lasso,
                                 'PLS': test_pls,
                                 'RandomForest': test_rf},
                                str(predictive_set_key),
                                str(splitter_option),
                                str(preprocessing_decision),
                                path='../results_collector_dict/')
