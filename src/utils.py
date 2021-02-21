from src import *


def load_data(solubility_data: str = target_path,
              fp_data: str = fingerprint_path,
              descriptor_data: str = descriptor_path):
    """

    :param solubility_data:
    :param fp_data:
    :param descriptor_data:
    :return:
    """
    df_log = pd.read_csv(solubility_data, index_col=0).apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df_fp = pd.read_csv(fp_data, index_col=0).apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df_desc = pd.read_csv(descriptor_data, index_col=0).apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df_all = pd.concat([df_desc, df_fp], axis=1)
    # map predictive datasets in dictionary
    predictive_datasets = {'desc': df_desc, 'fp': df_fp, 'all': df_all}
    return df_log, df_fp, df_desc, df_all, predictive_datasets


def split_rand_pick(df, splitting_column1, splitting_column2, splitting_column3):
    """

    :param df: Target dataframe
    :param splitting_column1: Column in df with split indices
    :param splitting_column2: Column in df with split indices
    :param splitting_column3: Column in df with split indices
    :return:
    """
    train_pick = df[df[splitting_column1] == 1].index
    val_pick = df[df[splitting_column1] == 0].index

    test_pick = df[df[splitting_column1] == 2].index
    train_rand = df[df[splitting_column2] == 0].index
    test_rand = df[df[splitting_column2] == 1].index

    train_pca = df[df[splitting_column3] == 0].index
    val_pca = df[df[splitting_column3] == 1].index

    return train_pick, val_pick, test_pick, train_rand, test_rand, train_pca, val_pca


def split_with_index(x_matrix, y_vector, indexer):
    """

    :param x_matrix:
    :param y_vector:
    :param indexer:
    :return:
    """

    x_indexed, y_indexed = x_matrix.loc[indexer], y_vector.loc[indexer]

    return x_indexed, y_indexed


def return_sets(sets, x, y, train_pick, val_pick, test_pick, train_rand, test_rand, train_pca, test_pca):
    """

    :param sets:
    :param x:
    :param y:
    :param train_pick:
    :param val_pick:
    :param test_pick:
    :param train_rand:
    :param test_rand:
    :param train_pca:
    :param test_pca:
    :return:
    """
    if sets == 'rand':

        x_train_set, y_train_set = split_with_index(x, y, train_rand)
        x_train_set, x_val_set, y_train_set, y_val_set = train_test_split(x_train_set, y_train_set, test_size=0.2,
                                                                          random_state=42)
        x_ext_set, y_ext_set = split_with_index(x, y, test_rand)

        return x_train_set, x_val_set, x_ext_set, y_train_set, y_val_set, y_ext_set

    elif sets == 'pick':

        x_train_set, y_train_set = split_with_index(x, y, train_pick)
        x_val_set, y_val_set = split_with_index(x, y, val_pick)
        x_ext_set, y_ext_set = split_with_index(x, y, test_pick)

        return x_train_set, x_val_set, x_ext_set, y_train_set, y_val_set, y_ext_set

    elif sets == 'pca_split':

        x_train_set, y_train_set = split_with_index(x, y, train_pca)
        x_train_set, x_val_set, y_train_set, y_val_set = train_test_split(x_train_set, y_train_set, test_size=0.2,
                                                                          random_state=42)
        x_ext_set, y_ext_set = split_with_index(x, y, test_pca)

        return x_train_set, x_val_set, x_ext_set, y_train_set, y_val_set, y_ext_set


    else:

        print('error')
        return None, None, None, None, None, None


def save_result(results_collector_dict: dict, predictive_set_key: str, splitter_option: str,
                preprocessing_decision: str, path='../results_collector_dict/'):
    """

    :param results_collector_dict:
    :param predictive_set_key:
    :param splitter_option:
    :param preprocessing_decision:
    :param path:
    :return:
    """
    import pickle

    with open(path + f'regressor_result_key-{predictive_set_key}_test-{splitter_option}_preprocess-{preprocessing_decision}.pickle',
              'wb') as handle:
        pickle.dump(results_collector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
