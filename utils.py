import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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
    f1_score
)

class DataPreprocessor:
    def __init__(self, standardize=True, remove_one_hot=True):
        self.preprocessor = None
        self.final_columns = None
        self.standardize = standardize
        self.remove_one_hot = remove_one_hot
        self.to_remove_indices = None

    def fit(self, data_frame: pd.DataFrame):
        education_order = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']
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
            numeric_columns = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']
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
        to_remove = ['INCOME_TYPE_Commercial associate',
                     'FAMILY_STATUS_Civil marriage',
                     'HOUSING_TYPE_Co-op apartment']
        self.to_remove_indices = [self.final_columns.index(col) for col in to_remove]
        for col in to_remove:
            self.final_columns.remove(col)
        return self

    def transform(self, data_frame: pd.DataFrame):
        preprocessed_data = self.preprocessor.transform(data_frame)
        if self.remove_one_hot:
            preprocessed_data = np.delete(preprocessed_data,
                                          self.to_remove_indices,
                                          axis=1)
        return preprocessed_data

def _obs_window_analysis(credit, command):
    """"
    calculate observe window
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

def plot_cumulative_percentages(credit):

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

def get_performance_window(credit, command, ratio=0.9):
    rates = _obs_window_analysis(credit, command)
    max_value = max(rates)
    threshold = ratio * max_value
    # search the index (i.e. the number of months from the current one)
    # of the value greater than or equal to the threshold:
    performance_window = next((i for i, rate in enumerate(rates) if rate >= threshold), None)
    return performance_window

def _plot_confusion_matrix(model, data, title="Train", subplot=121):
    X, y = data
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
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
    ax.text(x + 1, -0.03, f"F1 Score={f1:.3f}", ha='center', fontsize=14)

def plot_confusion_matrices(model, data):
    X_train, X_test, y_train, y_test = data
    plt.figure(figsize=(16, 7))
    _plot_confusion_matrix(model, (X_train, y_train), title="Train", subplot=121)
    _plot_confusion_matrix(model, (X_test, y_test), title="Test", subplot=122)
    plt.tight_layout()

def train_and_test_lr(data, resampling, model_name, Cs, existing_df=None):
    
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
        results.append({
            'model': model_name_,
            'balanced_acc_train': balanced_accuracy_train,
            'balanced_acc_test': balanced_accuracy_test,
            'precision_train': precision_train,
            'precision_test': precision_test,
            'recall_train': recall_train,
            'recall_test': recall_test
        })
    df = pd.DataFrame(results)
    if existing_df is not None:
        df = pd.concat([existing_df, df], ignore_index=True)
    return df