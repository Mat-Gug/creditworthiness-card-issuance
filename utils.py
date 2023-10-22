import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler
)
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    fbeta_score
)

class DataPreprocessor:
    def __init__(self, standardize=True, remove_one_hot=True):
        """
        Initialize a DataPreprocessor instance.

        This class is designed to preprocess input data for machine learning tasks.
        It performs a series of operations on the dataset variables, including
        encoding categorical features, standardizing numeric features, and optionally removing 
        1 column for each one-hot encoded feature to prepare the data for training machine learning models.

        Args:
            standardize (bool, optional): 
                Whether to standardize numeric columns. Default is True.
            remove_one_hot (bool, optional): Whether to remove certain one-hot encoded columns. Default is True.
                If set to True, 1 column for each one-hot encoded feature is removed from the dataset
                to avoid multicollinearity issues.

        Attributes:
            preprocessor (sklearn.compose.ColumnTransformer): 
                The fitted scikit-learn ColumnTransformer object used for
                feature encoding and transformation.
            final_columns (list): 
                The final list of column names after preprocessing,
                which includes both encoded and unencoded columns.
            standardize (bool): 
                Indicates whether numeric feature standardization is enabled.
            remove_one_hot (bool): 
                Indicates whether removal of specific one-hot encoded columns is enabled.
            to_remove_indices (list): 
                Indices of one-hot encoded columns that are removed from the transformed dataset.
            numeric_indices (list):
                Indices of numeric columns in the transformed dataset.
            education_index (int):
                Index of 'EDUCATION_TYPE' column in the transformed dataset.
        """
        self.preprocessor = None
        self.final_columns = None
        self.standardize = standardize
        self.remove_one_hot = remove_one_hot
        self.to_remove_indices = None
        self.numeric_indices = None
        self.education_index = None

    def fit(self, data_frame: pd.DataFrame):
        """
        Fit the DataPreprocessor to the input data.

        This method performs the following operations on the input data:
        - Encodes categorical features, including 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
          'CODE_GENDER', and 'EDUCATION_TYPE'.
        - Encodes 'INCOME_TYPE', 'FAMILY_STATUS', and 'HOUSING_TYPE' as one-hot features.
        - Optionally standardizes numeric features if self.standardize is True.
        - Creates the preprocessor and final column list based on the specified operations.

        Args:
            data_frame (pd.DataFrame): 
                Input data as a pandas DataFrame.

        Returns:
            DataPreprocessor: 
                The fitted DataPreprocessor instance.

        Note:
            Before calling this method, ensure that the DataFrame
            contains the necessary columns mentioned above.
        """
        education_order = ['Lower secondary',
                           'Secondary / secondary special',
                           'Incomplete higher',
                           'Higher education',
                           'Academic degree']
        flag_mapping = ['N', 'Y']
        gender_mapping = ['F', 'M']
        flag_columns = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
        gender_column = ['CODE_GENDER']
        ordinal_column = ['EDUCATION_TYPE']
        one_hot_columns = ['INCOME_TYPE', 'FAMILY_STATUS', 'HOUSING_TYPE']
        transformers = [
            ('flags', OrdinalEncoder(categories=[flag_mapping, flag_mapping]), flag_columns),
            ('gender', OrdinalEncoder(categories=[gender_mapping]), gender_column),
            ('ordinal', OrdinalEncoder(categories=[education_order]), ordinal_column),
            ('one_hot', OneHotEncoder(), one_hot_columns)
        ]
        if self.standardize:
            numeric_columns = ['CNT_CHILDREN',
                               'AMT_INCOME_TOTAL',
                               'AGE',
                               'YEARS_EMPLOYED',
                               'CNT_FAM_MEMBERS']
            transformers.append(('standardizer', StandardScaler(), numeric_columns))
            transformed_columns = flag_columns+gender_column+ordinal_column+one_hot_columns+numeric_columns
        else:
            transformed_columns = flag_columns+gender_column+ordinal_column+one_hot_columns
        passthrough_columns = [col for col in data_frame.columns if col not in transformed_columns]
        self.preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        self.preprocessor.fit(data_frame)
        one_hot_encoder = self.preprocessor.named_transformers_['one_hot']
        one_hot_columns = one_hot_encoder.get_feature_names_out()
        if self.standardize:
            self.final_columns = (
                flag_columns +
                gender_column +
                ordinal_column +
                one_hot_columns.tolist() +
                numeric_columns +
                passthrough_columns
            )
        else:
            self.final_columns = (
                flag_columns +
                gender_column +
                ordinal_column +
                one_hot_columns.tolist() +
                passthrough_columns
            )
        if self.remove_one_hot:
            to_remove = ['INCOME_TYPE_Commercial associate',
                        'FAMILY_STATUS_Civil marriage',
                        'HOUSING_TYPE_Co-op apartment']
            self.to_remove_indices = [self.final_columns.index(col) for col in to_remove]
            for col in to_remove:
                self.final_columns.remove(col)
        self.education_index = self.final_columns.index(ordinal_column[0])
        if self.standardize:
            self.numeric_indices = [
                self.final_columns.index(col) for col in numeric_columns
            ]
        return self

    def transform(self, data_frame: pd.DataFrame):
        """
        Transform input data using the fitted DataPreprocessor.

        Args:
            data_frame (pd.DataFrame): 
                Input data as a pandas DataFrame.

        Returns:
            np.ndarray: 
                Transformed data as a NumPy array.

        Note:
            This method should be called after the DataPreprocessor
            has been fitted using the fit method.
        """
        preprocessed_data = self.preprocessor.transform(data_frame)
        if self.remove_one_hot:
            preprocessed_data = np.delete(preprocessed_data,
                                          self.to_remove_indices,
                                          axis=1)
        return preprocessed_data
    
    def inverse_transform(self, X):
        """
        Inverse transform the preprocessed 'EDUCATION_TYPE' and numeric columns
        to their original form.
        This method takes the preprocessed data and reverses the transformations applied during
        the preprocessing step, including the inverse encoding of the 'EDUCATION_TYPE' column
        and, if standardization was applied, the inverse standardization of numeric columns.
        The result is a DataFrame with data in its original, interpretable form.

        Args:
            X (np.ndarray): 
                Preprocessed data as a NumPy array.

        Returns:
            pd.DataFrame:
                DataFrame containing the inverse-transformed data.
        """
        X_df = pd.DataFrame(X, columns=self.final_columns)
        education_ordinal_encoder = self.preprocessor.named_transformers_['ordinal']
        X_df[X_df.columns[self.education_index]] = \
            education_ordinal_encoder.inverse_transform(
                X_df[X_df.columns[self.education_index]].values.reshape(-1,1)
            ).reshape(-1)
        if self.standardize:
            standardizer = self.preprocessor.named_transformers_['standardizer']
            X_df[X_df.columns[self.numeric_indices]] = \
                standardizer.inverse_transform(
                    X_df[X_df.columns[self.numeric_indices]]
                ).round(1)
        return X_df

def _obs_window_analysis(credit: pd.DataFrame, command: str):
    """
    Calculate observe window.

    Args:
        credit (pd.DataFrame): 
            Credit data as a pandas DataFrame.
        command (str): 
            Python command used for data manipulation.

    Returns:
        pd.Series: A pandas Series containing calculated rates.
    """
    id_sum = len(set(credit['ID']))
    credit['status'] = 0
    exec(command)
    credit['month_on_book'] = credit['MONTHS_BALANCE'] - credit['open_month']
    # minimum value of `month_on_book` such that status==1 for each ID:
    minagg = credit[credit['status'] == 1].groupby('ID')['month_on_book'].min()
    minagg = pd.DataFrame(minagg)
    minagg['ID'] = minagg.index
    rates_df = pd.DataFrame({'month_on_book': range(0,credit["window"].max()+1), 'rate': None})
    lst = []
    for i in range(0,credit["window"].max()+1):
        # select the IDs with first time having status=1 in month_on_book==i:
        due = list(minagg[minagg['month_on_book']  == i]['ID'])
        lst.extend(due)
        rates_df.loc[rates_df['month_on_book'] == i, 'rate'] = len(set(lst)) / id_sum 
    return rates_df['rate']

def plot_cumulative_percentages(credit: pd.DataFrame):
    """
    Plot cumulative percentages of bad customers over time.

    Args:
        credit (pd.DataFrame): 
            Credit data as a pandas DataFrame.
    """
    command = "credit.loc[(credit['STATUS'] == '1') | (credit['STATUS'] == '2') | " \
        "(credit['STATUS'] == '3' ) | (credit['STATUS'] == '4' ) | (credit['STATUS'] == '5'), 'status'] = 1"   
    morethan30 = _obs_window_analysis(credit, command)
    command = "credit.loc[(credit['STATUS'] == '2') | (credit['STATUS'] == '3' ) | " \
        " (credit['STATUS'] == '4' ) | (credit['STATUS'] == '5'), 'status'] = 1"
    morethan60 = _obs_window_analysis(credit, command)
    command = "credit.loc[(credit['STATUS'] == '3' ) | (credit['STATUS'] == '4' ) | " \
        " (credit['STATUS'] == '5'), 'status'] = 1"
    morethan90 = _obs_window_analysis(credit, command)
    command = "credit.loc[(credit['STATUS'] == '4' ) | (credit['STATUS'] == '5'), 'status'] = 1"
    morethan120 = _obs_window_analysis(credit, command)
    command = "credit.loc[(credit['STATUS'] == '5'), 'status'] = 1"
    morethan150 = _obs_window_analysis(credit, command)
    labels = [
        'More than 30 days past due',
        'More than 60 days past due',
        'More than 90 days past due',
        'More than 120 days past due',
        'More than 150 days past due'
    ]
    data = [morethan30, morethan60, morethan90, morethan120, morethan150]
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    for i in range(len(data)):
        plt.plot(data[i], label=labels[i])
    plt.title('Cumulative % of Bad Customers', fontsize=18)
    plt.xlabel('Month on Book', fontsize=14)
    plt.ylabel('Cumulative %', fontsize=14)
    plt.legend()
    plt.show()

def get_performance_window(credit: pd.DataFrame, command: str, ratio=0.9):
    """
    Get the performance window based on the given command and ratio.

    Args:
        credit (pd.DataFrame): 
            Credit data as a pandas DataFrame.
        command (str): 
            Python command used for data manipulation.
        ratio (float, optional): 
            The threshold ratio for performance window calculation. Default is 0.9.

    Returns:
        int: The calculated performance window.
    """
    rates = _obs_window_analysis(credit, command)
    max_value = max(rates)
    threshold = ratio * max_value
    # search the index (i.e. the number of months from the current one)
    # of the value greater than or equal to the threshold:
    performance_window = next((i for i, rate in enumerate(rates) if rate >= threshold), None)
    return performance_window

def convert_days_to_years(data_frame: pd.DataFrame, column_name: str, new_column_name: str):
    """
    Convert a column representing days to years in a DataFrame.

    Args:
        data_frame (pd.DataFrame):
            The DataFrame containing the data.
        column_name (str):
            The name of the column to be converted from days to years.
        new_column_name (str):
            The new name to replace the existing column name.
    """
    data_frame[column_name] = data_frame[column_name].apply(lambda x: 0 if x>0 else x)
    data_frame[column_name] = \
        (abs(data_frame[column_name]) / 365.25).round().astype(int)
    data_frame.rename(columns={column_name: new_column_name}, inplace=True)

def _plot_confusion_matrix(model, data: tuple, title="Train", subplot=121):
    """
    Plot confusion matrix for a classification model.

    Args:
        model: 
            The classification model.
        data (tuple): 
            Tuple containing X and y.
        title (str, optional): 
            Title for the confusion matrix plot. Default is "Train".
        subplot (int, optional): 
            Subplot position for plotting. Default is 121.
    """
    X, y = data
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f2 = fbeta_score(y, y_pred, beta=2)
    
    df_cm = pd.DataFrame(cm,
                        index=["Negative", "Positive"],
                        columns=["Predicted Negative", "Predicted Positive"]
                        )
    
    plt.subplot(subplot)
    ax = plt.gca()
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
    x = 0.5
    ax.text(1, -0.22, title, ha='center', fontsize=18)
    ax.text(x, -0.12, f"Precision={precision:.3f}", ha='center', fontsize=14)
    ax.text(x, -0.03, f"Recall={recall:.3f}", ha='center', fontsize=14)
    ax.text(x + 1, -0.12, f"Balanced Accuracy={balanced_accuracy:.3f}", ha='center', fontsize=14)
    ax.text(x + 1, -0.03, f"F2 Score={f2:.3f}", ha='center', fontsize=14)

def plot_confusion_matrices(model, data: tuple):
    """
    Plot confusion matrices for a classification model on both training and test data.

    Args:
        model: 
            The classification model.
        data (tuple): 
            Tuple containing X_train, X_test, y_train, and y_test.
    """
    X_train, X_test, y_train, y_test = data
    plt.figure(figsize=(16, 7))
    _plot_confusion_matrix(model, (X_train, y_train), title="Train", subplot=121)
    _plot_confusion_matrix(model, (X_test, y_test), title="Test", subplot=122)
    plt.tight_layout()

def train_and_test_lr(data, resampling, model_name, Cs, existing_df=None):
    """
    Train and test a logistic regression model with a given
    resampling method applied on the train set.

    Args:
        data (tuple): 
            Tuple containing X_train, X_test, y_train, and y_test.
        resampling: 
            Resampling technique(s) or object.
        model_name (str): 
            Name for the model.
        Cs (list): 
            List of regularization parameter values.
        existing_df (pd.DataFrame, optional): 
            Existing DataFrame to append results. Default is None.

    Returns:
        pd.DataFrame: Results of model training and testing.
    """
    X_train, X_test, y_train, y_test = data
    if isinstance(resampling, tuple) and len(resampling)>1:
        for method in resampling:
            X_train, y_train = method.fit_resample(X_train, y_train)
    else:
        X_train, y_train = resampling.fit_resample(X_train, y_train)
    print(f"Number of train samples after {model_name}: {len(y_train)}")
    
    results = []
    
    for C in Cs:
        lr = LogisticRegression(class_weight='balanced', C=C, max_iter=600)
        lr.fit(X_train, y_train)
        
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        
        model_name_ = model_name + "_C_" + str(C)
        balanced_accuracy_train = round(balanced_accuracy_score(y_train, y_pred_train), 3)
        balanced_accuracy_test = round(balanced_accuracy_score(y_test, y_pred_test), 3)
        precision_train = round(precision_score(y_train, y_pred_train), 3)
        precision_test = round(precision_score(y_test, y_pred_test), 3)
        recall_train = round(recall_score(y_train, y_pred_train), 3)
        recall_test = round(recall_score(y_test, y_pred_test), 3)
        f2_train = round(fbeta_score(y_train, y_pred_train, beta=2), 3)
        f2_test = round(fbeta_score(y_test, y_pred_test, beta=2), 3)
        results.append({
            'model': model_name_,
            'balanced_acc_train': balanced_accuracy_train,
            'balanced_acc_test': balanced_accuracy_test,
            'precision_train': precision_train,
            'precision_test': precision_test,
            'recall_train': recall_train,
            'recall_test': recall_test,
            'f2_train': f2_train,
            'f2_test': f2_test
        })
    df = pd.DataFrame(results)
    if existing_df is not None:
        df = pd.concat([existing_df, df], ignore_index=True)
    return df

def custom_grid_search(data, model, param_grid, k=5, random_state=None):
    """
    Perform custom grid search for model hyperparameter tuning.

    Args:
        data (tuple): 
            Tuple containing X and y.
        model: 
            The machine learning model.
        param_grid (dict): 
            Dictionary of hyperparameter values to search.
        k (int, optional): 
            Number of cross-validation folds. Default is 5.
        random_state (int, optional): 
            Random state for reproducibility. Default is None.

    Returns:
        pd.DataFrame: 
            Results of the custom grid search.
    """
    X, y = data
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    models_df = pd.DataFrame()
    
    for grid in tqdm(list(ParameterGrid(param_grid)), 
                     desc="Grid Search Progress",
                     unit="configuration"):

        balanced_acc_train_scores = []
        balanced_acc_test_scores = []
        f2_train_scores = []
        f2_test_scores = []

        for train_index, test_index in kf.split(X, y):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            preprocessor = DataPreprocessor()
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)

            model.set_params(**grid)
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            balanced_acc_train_scores.append(balanced_accuracy_score(y_train, y_pred_train))
            balanced_acc_test_scores.append(balanced_accuracy_score(y_test, y_pred_test))
            f2_train_scores.append(fbeta_score(y_train, y_pred_train, beta=2))
            f2_test_scores.append(fbeta_score(y_test, y_pred_test, beta=2))
        
        scores = {
            'balanced_acc_train' : np.array(balanced_acc_train_scores),
            'balanced_acc_test' : np.array(balanced_acc_test_scores),
            'f2_train' : np.array(f2_train_scores),
            'f2_test' : np.array(f2_test_scores)
        }
        
        model_dict = {
            'mean_train_balanced_acc' : round(np.mean(scores['balanced_acc_train']),3),
            'mean_test_balanced_acc' : round(np.mean(scores['balanced_acc_test']),3),
            'std_test_balanced_acc' : round(np.std(scores['balanced_acc_test']),3),
            'mean_train_f2' : round(np.mean(scores['f2_train']),3),
            'mean_test_f2' : round(np.mean(scores['f2_test']),3),
            'std_test_f2' : round(np.std(scores['f2_test']),3)
        }
        
        model_dict.update(grid)
        model_df = pd.DataFrame([model_dict])
        models_df = pd.concat([models_df, model_df], ignore_index=True)
    
    return models_df


def get_best_configs(models_df):
    """
    Get the best model configurations based on performance.

    Args:
        models_df (pd.DataFrame): 
            DataFrame containing model performance results.

    Returns:
        pd.DataFrame: 
            Best model configurations based on performance.
    """
    top_acc_rows = models_df.nlargest(3, 'mean_test_balanced_acc')
    top_recall_rows = models_df.nlargest(3, 'mean_test_f2')
    top_rows = pd.concat([top_acc_rows, top_recall_rows], ignore_index=True)
    best_configs_df = top_rows[~top_rows.index.duplicated(keep='first')]
    
    return best_configs_df

def get_parameters(data_frame: pd.DataFrame, row_index: int):
    """
    Extract parameters from a DataFrame row.

    Args:
        data_frame (pd.DataFrame):
            The DataFrame containing the parameter data.
        row_index (int):
            The index of the row from which to extract the parameters.

    Returns:
        dict: A dictionary containing the best parameters.
    """
    std_test_recall_index = data_frame.columns.get_loc('std_test_f2')
    parameter_cols = data_frame.iloc[:, std_test_recall_index+1:]
    best_params_dict = parameter_cols.iloc[row_index].to_dict()
    for key, value in best_params_dict.items():
        if isinstance(value, float) and value.is_integer():
            best_params_dict[key] = int(value)
    
    return best_params_dict