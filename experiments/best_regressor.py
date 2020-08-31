import sys
sys.path.append('..')
from src.model_support import *
from src.models import *
from src.utils import *


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # load data and set indices
    df_log, df_fp, df_desc, df_all, predictive_dataset = load_data()
    train_pick, val_pick, test_pick, train_rand, test_rand = split_rand_pick(df_log, 'diversity', 'random')

    # experimental matrix, data, preprocessing or not, splitting options
    for predictive_set_key in predictive_dataset.keys():
        for preprocessing_decision in [True, False]:
            if predictive_set_key != 'fp' and preprocessing_decision is False:

                # set X and y
                X = set_x_matrix(predictive_dataset, predictive_set_key, preprocess=preprocessing_decision)
                y = df_log.logS0

                for splitter_option in ['rand', 'pick']:

                    X_train, X_val, X_ext, y_train, y_val, y_ext = return_sets(splitter_option, X, y, train_pick,
                                                                               val_pick, test_pick, train_rand,
                                                                               test_rand)
                    print(X_train.shape, X_val.shape, X_ext.shape, y_train.shape, y_val.shape, y_ext.shape)

                    # Random Forest
                    test_rf = RegressorTest(1, 1, 1, 'rf_', X_train, X_val, X_ext, y_train, y_val, y_ext)
                    # PLS
                    test_pls = RegressorTest(1, 1, 1, 'pls_', X_train, X_val, X_ext, y_train, y_val, y_ext)
                    # LASSO
                    test_lasso = RegressorTest(1, 1, 1, 'lasso_', X_train, X_val, X_ext, y_train, y_val, y_ext)
                    # LightGBM
                    test_lgbm = RegressorTest(1, 1, 1, 'lg_', X_train, X_val, X_ext, y_train, y_val, y_ext)
                    #print(test_rf)

                    save_result({'LightGBM': test_lgbm,
                                 'LASSO': test_lasso,
                                 'PLS': test_pls,
                                 'RandomForest': test_rf},
                                str(predictive_set_key),
                                str(splitter_option),
                                str(preprocessing_decision),
                                path='../results/')
