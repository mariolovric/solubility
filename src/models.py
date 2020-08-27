from src.configs import param_space
from src.model_support import *


def rmse(y_actual, y_predicted):
    """

    :param y_actual:
    :param y_predicted:
    :return:
    """
    rms_result = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms_result


def regression_test(runs, iters, inits, test, x_train, x_val, x_test, y_train, y_val, y_test):
    """
    :param runs: number of runs
    :param iters: iterations in BayesOpt
    :param inits: initial points of parameter space in BayesOpt
    :param test: algorithm, can be: rf_ , lg_ , pls_ , lasso_
    :param x_train: train set
    :param x_val: validation set
    :param x_test: splitter_option set
    :param y_train: train set
    :param y_val: validation set
    :param y_test: splitter_option set
    :return:
    """

    results_dictionary = {}
    features_dictionary = {}
    parameters_dictionary = {}
    final_result = {}

    for i in range(runs):
        try:
            print('\n===== RUN:', i, '=====', )

            if i == 0:
                # run 0, first time regression, no feat sel
                regressor_instance = RegressorEvaluatorModule(x_train, x_val, x_test, y_train, y_val, y_test)
                features_list = x_train.columns

            else:
                # iterative feature sel, if run is not 0
                print('**** started feature selection')
                features_list = regressor_instance.feat_selector_module(best_model_from_bo)

                # update
                features_dictionary.update({test + str(i): features_list})
                print('len features_list:', len(features_list))

                # train with feature selection
                regressor_instance = RegressorEvaluatorModule(x_train[features_list], x_val[features_list],
                                                              x_test[features_list], y_train, y_val, y_test)

            regressor_object = {'rf_': regressor_instance.rf_eval, 'lg_': regressor_instance.lgbm_eval,
                                'pls_': regressor_instance.pls_eval, 'lasso_': regressor_instance.lasso_eval}

            # call optimizer, special case with pls because n_components can be lower than len(features)
            if test == 'pls_':
                try:
                    bo = BayesianOptimization(regressor_object[test], param_space[test])
                    bo.maximize(n_iter=iters, init_points=inits)

                except:
                    # reducing n_components
                    pls_space = {'n_components': (2, len(features_list) - 1)}
                    bo = BayesianOptimization(regressor_object[test], pls_space)
                    bo.maximize(n_iter=iters, init_points=inits)
            else:
                # for other algorithms than PLS
                bo = BayesianOptimization(regressor_object[test], param_space[test])
                bo.maximize(n_iter=iters, init_points=inits)

            # extract best params from the bayesian optimizer
            best_params_ = bo.max["params"]
            print('\n** Results from run **', )

            # return model w best params
            best_model_from_bo = regressor_object[test](**best_params_, return_model=True,
                                                        print_res=True, exp_=test + str(i))

            # if first run, lasso special case for coefficients
            if i == 0:
                if test == 'lasso_':
                    print('\n********\ncoef')

                    lasso_coefficients_list = pd.Series(best_model_from_bo.coef_, name='coef',
                                                        index=x_train.columns.tolist())

                    features_list = lasso_coefficients_list[lasso_coefficients_list != 0].index.tolist()

                features_dictionary.update({test + str(i): features_list})

            # store results_collector_dict
            results_dictionary.update(regressor_instance.results_dictionary)

            parameters_dictionary.update({test + str(i): best_params_})

        except:
            pass
    final_result.update({test: results_dictionary, 'features': features_dictionary, 'params': parameters_dictionary})

    return final_result


class RegressorEvaluatorModule:
    """
    Evaluation of 4 regressors packed in modules: lasso_eval, pls_eval, rf_eval, lgbm_eval

    """

    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test):
        """

        :param x_train:
        :param x_val:
        :param x_test:
        :param y_train:
        :param y_val:
        :param y_test:
        """
        self.X_train = x_train
        self.y_train = y_train
        self.X_val = x_val
        self.y_val = y_val
        self.X_test = x_test
        self.y_test = y_test
        self.results_dictionary = {}

    def lasso_eval(self, alpha, return_model=False, print_res=False):
        """
        Evaluator Lasso Regression, arguments are algorithm specific
        :param alpha:
        :param return_model:
        :param print_res:
        :return:
        """

        params = {'normalize': False, 'fit_intercept': False, 'max_iter': 1000, 'alpha': float(alpha),
                  'random_state': 42}

        model = Lasso(**params)
        model.fit(self.X_train, self.y_train)

        validation_score, result = self.internal_validator(model, print_res)

        return -validation_score, result if return_model else -validation_score

    def pls_eval(self, n_components, return_model=False, print_res=False):
        """
        Evaluator PLS Regression, arguments are algorithm specific
        :param n_components:
        :param return_model:
        :param print_res:
        :return:
        """

        params = {'scale': True, 'n_components': int(n_components), 'max_iter': 1000}
        try:
            model = PLSRegression(**params)
            model.fit(self.X_train, self.y_train)

            validation_score, result = self.internal_validator(model, print_res)

        except:
            params = {'scale': True, 'n_components': int(n_components - 1), 'max_iter': 2000}
            model = PLSRegression(**params)
            model.fit(self.X_train, self.y_train)

            validation_score, result = self.internal_validator(model, print_res)

        return -validation_score, result if return_model else -validation_score

    def rf_eval(self, max_depth, n_estimators, min_samples_split, max_samples,
                return_model=False, print_res=False):
        """
        Evaluator RF Regression, arguments are algorithm specific
        :param max_depth:
        :param n_estimators:
        :param min_samples_split:
        :param max_samples:
        :param return_model:
        :param print_res:
        :return:
        """

        params = {'bootstrap': True, 'max_depth': int(max_depth), 'n_estimators': int(n_estimators),
                  'max_samples': float(max_samples), 'min_samples_split': int(min_samples_split),
                  'random_state': 42}

        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train.ravel())

        validation_score, result = self.internal_validator(model, print_res)

        return -validation_score, result if return_model else -validation_score

    def lgbm_eval(self, num_leaves, max_depth, lambda_l2, lambda_l1, min_data_in_leaf,
                  return_model=False, print_res=False):
        """
        Evaluator LightGBM Regression, arguments are algorithm specific
        :param num_leaves:
        :param max_depth:
        :param lambda_l2:
        :param lambda_l1:
        :param min_data_in_leaf:
        :param return_model:
        :param print_res:
        :return:
        """
        params = {
            "objective": "regression", "metric": "rmse", "num_leaves": int(num_leaves), "max_depth": int(max_depth),
            "lambda_l2": lambda_l2, "lambda_l1": lambda_l1, 'min_data_in_leaf': int(min_data_in_leaf),
            "learning_rate": 0.03, "subsample_freq": 5, "bagging_seed": 42, "verbosity": -1, 'random_state': 42}

        model = LGBMRegressor(**params)
        model.fit(self.X_train, self.y_train)

        validation_score, result = self.internal_validator(model, print_res)

        return -validation_score, result if return_model else -validation_score

    def internal_validator(self, model, print_res=False):
        """
        Internal validator used in the models.
        :param model: pass model from xxx_eval
        :param print_res: if True, will print the results_collector_dict
        :return: validation score and results_collector_dict dictionary
        """

        # Do predictions Train
        y_pred_train = model.predict(self.X_train)

        train_score = np.sqrt(mean_squared_error(self.y_train, y_pred_train))

        # Do predictions Val
        y_pred_val = model.predict(self.X_val)
        validation_score = np.sqrt(mean_squared_error(self.y_val, y_pred_val))
        # Do predictions Test
        y_pred_test = model.predict(self.X_test)
        test_score = np.sqrt(mean_squared_error(self.y_test, y_pred_test))

        mean_score = np.mean([test_score, validation_score])
        if print_res is True:
            print("--- Train ---")
            print(f"RMSE: {train_score}")
            print("--- Validation ---")
            print(f"RMSE: {validation_score}")
            print("--- Test ---")
            print(f"RMSE: {test_score}")
            print("--- Train Val mean ---")
            print(f"RMSE: {mean_score}")
        result = {'train_score': train_score, 'mean_score': mean_score,
                  'validation_score': validation_score, 'test_score': test_score}

        return validation_score, result

    def feat_selector_module(self, model):
        """
        Feature selection based on permutation importance from the eli5 library.
        Validation set is used for evaluation.
        :param model:
        :return: list with selected features
        """

        permutation_importance_object = PermutationImportance(model, random_state=42).fit(self.X_val, self.y_val)
        explained_weight_df = explain_weights_df(estimator=permutation_importance_object,
                                                 feature_names=self.X_val.columns.tolist()).sort_values('weight',
                                                                                                        ascending=False)
        # feature with weight > 0.001 or 1/3 of feature length
        one_third_of_feature_count = int(len(explained_weight_df) / 3)
        selected_features_list = explained_weight_df[explained_weight_df.weight > 0.001]

        if len(selected_features_list) == 0:
            # if there are no pos.weights
            print('error in selection', len(self.X_val.columns.tolist()), ', or all features selected')
            return self.X_val.columns.tolist()

        elif len(selected_features_list) > one_third_of_feature_count:
            # reduce to one third
            return selected_features_list.iloc[0:one_third_of_feature_count].feature.tolist()

        else:
            return selected_features_list.feature.tolist()
