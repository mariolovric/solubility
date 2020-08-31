from src import *


# separate floats from cats
class PreprocessorClass:

    def __init__(self, df):

        self.df = df
        self.proc_df = None
        self.dummy_disc_df = None
        self.bool_disc_df = None
        self.float_cols_org = []
        self.int_cols_org = []
        self.bool_cols = []
        self.bool_cols_from_int = []
        self.float_cols_from_int = []
        self.disc_cols = []
        self.all_floats = []
        self.dropped_corr = []
        self.likely_cat = {}
        self.all_float_df = {}

    def run_preprocessor(self):

        # 1) Remove nans and correlated
        self.proc_df = self.p1_pre_clean(self.df)

        # 2) Initial split floats and integers, saves them globally to self
        self.float_cols_org, self.int_cols_org = self.p2_sep_disc_float_initial(self.proc_df)

        # 3) Splits columns to bool, discrete, float based on likelyhood, unique values
        self.bool_cols_from_int, self.disc_cols, self.float_cols_from_int = self.p3_sep_types_likely(
            self.proc_df[self.int_cols_org])

        # 4) Takes disc_cols from #3 converts disc to dummies (binary), uses two different functions for the conversion
        self.dummy_disc_df = self.p4_disc_converter(self.proc_df, self.disc_cols)

        # 5) Takes list of bool cols from #3, dataframe from #1 and remove columns with few 1's
        self.bool_cols = self.remove_rare_bools(self.proc_df[self.bool_cols_from_int])

        # 6) Collect floats from #2 and #3
        self.all_floats = self.float_cols_org + self.float_cols_from_int

        # 7) Create dataframe of floats
        self.all_float_df = self.proc_df[self.all_floats]

        # 8) Create boolean dataframe, from #4 and #5
        self.bool_disc_df = pd.concat([self.dummy_disc_df, self.proc_df[self.bool_cols]], axis=1)

        # 9) Concat all floats and bools
        preprocessed_df = pd.concat([self.all_float_df, self.bool_disc_df], axis=1)

        # Assert that #9 doesn't have NaNs
        preprocessed_df.dropna(how='any', axis=0, inplace=True)
        assert (preprocessed_df.isnull().sum().any() is False), 'Preprocessed DataFrame has NaNs'
        return preprocessed_df

    # Utils func section
    @staticmethod
    def kick_corr(in_df, corr_th=0.85):
        """
        Removes correlated featurs above 0.9
        :param in_df: Dataframe of features
        :param corr_th: Correlation Threshold
        :return: Dataframe without correlated features
        """
        corr_matrix = in_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_th)]
        no_corr_df = in_df.drop(to_drop, axis=1)
        return no_corr_df

    @staticmethod
    def check_floats_on_integers(in_df):
        """
        Just double checking the floats
        :return: True if no float is actually an integer
        """
        return in_df.apply(lambda x: x.apply(float.is_integer).all()).any()

    @staticmethod
    def get_bool_df(in_df):
        """'
        Get boolean columns from a df.
        :param in_df: dataframe from which to extract boolean columns
        :return: boolean columns list
        """
        bool_cols = [col for col in in_df if
                     in_df[col].dropna().value_counts().index.isin([0, 1]).all()]
        return bool_cols

    @staticmethod
    def remove_rare_bools(in_df, bin_thresh=11):
        """
        Function for removing columns from a DF with less True/1 values than given threshold as an apsolute number.
        :param in_df: dataframe of bools for reduction
        :param bin_thresh: 11 like 1% being that DF as 1100 instances appr
        :return: list of columns who passed the criteria
        """
        # Assertion for bool cols
        assert in_df.apply(lambda x: x.value_counts().index.isin([0, 1]).all()).all() is True
        left_over_cols = in_df.sum()[in_df.sum() > bin_thresh].index.tolist()
        return left_over_cols

    @staticmethod
    def cut_n_dummy(in_df, var):
        """
        Cuts variables in 4 bins and return 4 dummy columns.
        :param in_df: dataframe with discretes
        :param var: variable to be cut and dummified
        :return: dataframe of cutted dummies
        """
        cut_var = in_df[var].replace(0, np.nan)
        to_dummy = pd.cut(cut_var, bins=4, labels=['c1', 'c2', 'c3', 'c4'], right=False)
        outg_df = pd.get_dummies(to_dummy, prefix=var)
        return outg_df

    @staticmethod
    def simple_dummy(in_df, var):
        """
        Will be used for discrete columns with less than 4 unique values.
        :param in_df: dataframe base for creating dummies
        :param var: variable to be dummified
        :return: dataframe with dummy columns
        """
        to_dummy = in_df[var].replace(0, np.nan)
        outg_df = pd.get_dummies(to_dummy, prefix=var)
        return outg_df

    # Main func
    def p1_pre_clean(self, in_df):
        """
        Bad value in X is -999, this function converts them to nan and drops columns with nans.
        Then if removes correlated using the "kick_corr" function.
        :param in_df:
        :return:
        """
        rem_na = in_df.replace(-999, np.nan).dropna(how='any', axis=1)
        rem_corr = self.kick_corr(rem_na)
        return rem_corr

    @staticmethod
    def p2_sep_disc_float_initial(in_df):
        """
        Tries to convert data to integer. Separates floats from ints.
        :return: list of floats, list of ints
        """
        sep_df = in_df.apply(lambda x: pd.to_numeric(x, downcast='integer'))
        floats_ = sep_df.select_dtypes(include='float').columns.tolist()
        # Double check the floats
        # Assert self.check_floats_on_integers(in_df[floats_])==True
        ints_ = sep_df.select_dtypes(include='integer').columns.tolist()

        return floats_, ints_

    def p3_sep_types_likely(self, in_df, thresh=0.05):
        """
        Separates columns on variable types based on likelyhood of belongin to a certain type.
        It checks whether there are more unique values then given by threshold, here 5%.
        :param in_df: dataframe with mixed columns to be separated
        :param thresh: threshold (in decimal) for percentage of uniques in column
        :return: three lists, boolean columns, disrete columns, float columns
        """
        for var in in_df.columns:
            self.likely_cat[var] = str(1. * in_df[var].nunique() / in_df[var].count() < thresh)
        # set df with categorical columns true false
        cat_df = pd.DataFrame.from_dict(self.likely_cat, orient='index', columns=['tf'])
        # diff floats bools ints
        float_cols_x = cat_df[cat_df.tf == 'False'].index.tolist()
        ints_x = cat_df[cat_df.tf == 'True'].index.tolist()
        bool_cols_x = self.get_bool_df(in_df[ints_x])
        disc_cols_x = [col for col in ints_x if col not in bool_cols_x]

        return bool_cols_x, disc_cols_x, float_cols_x

    def p4_disc_converter(self, df, cols):
        """
        Converts give columns in a DataFrame to dummies.
        Use "cut_n_dummy" for columns with more than 4 unique values.
        :param df: DataFrame, full
        :param cols: discrete columns in the given DF
        :return: returns DataFrame of dummy vectors
        """
        collector_df = pd.DataFrame()
        for col in cols:
            if df[col].nunique() > 4:
                new_bool_df = self.cut_n_dummy(in_df=df, var=col)
                collector_df = pd.concat([collector_df, new_bool_df], axis=1)
            else:
                new_bool_df = self.simple_dummy(in_df=df, var=col)
                collector_df = pd.concat([collector_df, new_bool_df], axis=1)
        assert collector_df.apply(lambda x: x.value_counts()).index.isin([0, 1]).all() is True
        return collector_df


def set_x_matrix(set_predictive_matrices: dict, dictionary_key_predictve: str, preprocess=True):

    """
    :param set_predictive_matrices: dict of predictive matrices
    :param dictionary_key_predictve:
    :param preprocess: boolean, True or False
    :return:
    """

    df = set_predictive_matrices[dictionary_key_predictve]

    if preprocess:
        processor_object = PreprocessorClass(df)
        final_x_matrix = processor_object.run_preprocessor()

    else:
        final_x_matrix = df.dropna(how='any', axis=1)

    print('Preprocessing ', df.shape, ' to ', final_x_matrix.shape)
    return final_x_matrix
