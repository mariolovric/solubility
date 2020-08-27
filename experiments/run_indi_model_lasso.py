import sys
sys.path.append('..')
from src.model_support import *
from src.utils import *
import warnings
with warnings.catch_warnings(): warnings.filterwarnings("ignore")

# winning model parameters
# ==================
# regressor_result_key-all_test-rand_preprocess-False.pickle
# {'train_score': 0.4723311329495703, 'mean_score': 0.8299982962282464, 'validation_score': 0.9376601479621512,
# 'test_score': 0.7223364444943414}

predictive_set_key = 'all'
splitter_option = 'rand'
preproc_decision = False
random_seed = 42
alpha = 0.0150523354781535
params = {'normalize': False, 'fit_intercept': False, 'max_iter': 1000,
          'alpha': float(alpha), 'random_state': 42, 'tol': 0.01}

if __name__ == '__main__':
    df_log, df_fp, df_desc, df_all, predictive_dataset = load_data()
    train_pick, val_pick, test_pick, train_rand, test_rand = split_rand_pick(df_log, 'diversity', 'random')

    X = SetXMatrix(predictive_dataset, predictive_set_key, preprocess=preproc_decision)
    y = df_log.logS0

    X_train, X_val, X_ext, y_train, y_val, y_ext = return_sets(splitter_option, X, y, train_pick,
                                                               val_pick, test_pick, train_rand,
                                                               test_rand)
    model = Lasso(**params)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_ext)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    val_score = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_score = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_score = np.sqrt(mean_squared_error(y_ext, y_pred_test))

    scores = {'Train': train_score, 'Validation': val_score, 'Test': test_score}
    print(scores)
    pd.Series(y_pred_test, index=y_ext.index, name='lasso').to_csv('../results/test_lasso_predicted.csv')
    y_ext.to_csv('../results/test_true.csv')
