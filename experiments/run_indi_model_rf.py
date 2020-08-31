import sys

sys.path.append('..')

from src.model_support import *
from src.utils import *
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

# winning model parameters
# ==================
# regressor_result_key-all_test-rand_preprocess-False.pickle
# {'train_score': 0.4723311329495703, 'mean_score': 0.8299982962282464, 'validation_score': 0.9376601479621512,
# 'test_score': 0.7223364444943414}

features = ['Molecular_properties_AMR', 'Constitutional_indices_SCBO', 'Ring_descriptors_D/Dtr06',
            'Molecular_properties_MLOGP', 'Molecular_properties_BLTD48', 'Molecular_properties_BLTF96',
            'Ring_descriptors_Rperim', 'Constitutional_indices_C%', 'Constitutional_indices_nCsp2',
            'Functional_group_counts_nCar', 'Molecular_properties_BLTA96', 'fp4582', 'Topological_indices_ICR',
            'Functional_group_counts_nROH', 'Topological_indices_MAXDN', 'Molecular_properties_TPSA(Tot)',
            'Constitutional_indices_nC']

predictive_set_key = 'all'
splitter_option = 'rand'
preproc_decision = False
random_seed = 42

params = {'bootstrap': True, 'max_depth': int(18.52203732903717), 'max_samples': float(0.5907891410962993),
          'min_samples_split': int(3.1474161594606516), 'n_estimators': int(247.25842336056604), 'random_state': 42}

if __name__ == '__main__':
    df_log, df_fp, df_desc, df_all, predictive_dataset = load_data()
    train_pick, val_pick, test_pick, train_rand, test_rand = split_rand_pick(df_log, 'diversity', 'random')

    X = set_x_matrix(predictive_dataset, predictive_set_key, preprocess=preproc_decision)
    y = df_log.logS0
    X_train, X_val, X_ext, y_train, y_val, y_ext = return_sets(splitter_option, X, y, train_pick,
                                                               val_pick, test_pick, train_rand,
                                                               test_rand)

    model = RandomForestRegressor(**params)
    model.fit(X_train[features], y_train)

    y_pred_test = model.predict(X_ext[features])
    y_pred_train = model.predict(X_train[features])
    y_pred_val = model.predict(X_val[features])

    val_score = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_score = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_score = np.sqrt(mean_squared_error(y_ext, y_pred_test))

    scores = {'Train': train_score, 'Validation': val_score, 'Test': test_score}
    print(scores)
    pd.Series(y_pred_test, index=y_ext.index, name='rf').to_csv('../results/test_rf_predicted.csv')
