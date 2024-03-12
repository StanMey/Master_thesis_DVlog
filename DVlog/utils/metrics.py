import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


## Performance measures
def calculate_performance_measures(y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> Tuple[float, float, float, float]:
    """_summary_

    :param y_true: _description_
    :type y_true: list[np.ndarray]
    :param y_pred: _description_
    :type y_pred: list[np.ndarray]
    :return: _description_
    :rtype: Tuple[float, float, float, float]
    """

    # flatten the predictions and ground truths
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    # 
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    return accuracy, precision, recall, fscore


## The fairness measures
def calculate_fairness_measures(y_true: list[np.ndarray], y_pred: list[np.ndarray], protected: list[str], unprivileged: str) -> Tuple[float, float, float]:
    """_summary_

    :param y_true: _description_
    :type y_true: list[np.ndarray]
    :param y_pred: _description_
    :type y_pred: list[np.ndarray]
    :param protected: _description_
    :type protected: list[str]
    :param unprivileged: _description_
    :type unprivileged: str
    :return: _description_
    :rtype: Tuple[float, float, float]
    """
    
    # flatten the predictions and ground truths
    y_pred = np.argmax(np.concatenate(y_pred), axis=1)
    y_true = np.argmax(np.concatenate(y_true), axis=1)
    protected = np.concatenate(protected)

    # put it all in a dataframe for easy handling and filtering
    df_fairness = pd.DataFrame(np.column_stack([y_pred, y_true, protected]),
                               columns=["y_pred", "y_true", "group"])

    # compute the fairness measures
    unpriv_acc, unpriv_tpr, unpriv_fpr = fairness_metrics(df_fairness[df_fairness["group"] == unprivileged])
    priv_acc, priv_tpr, priv_fpr = fairness_metrics(df_fairness[df_fairness["group"] != unprivileged])
    
    # calculate the ratios
    eq_oppor = unpriv_tpr / priv_tpr
    eq_acc = unpriv_acc / priv_acc

    # for equalized odds we use the approach by fairlearn (https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html#equalized-odds)
    # "The smaller of two metrics: `true_positive_rate_ratio` and `false_positive_rate_ratio`."
    eq_odds = min((unpriv_tpr / priv_tpr), (unpriv_fpr / priv_fpr))

    return eq_odds, eq_oppor, eq_acc


def fairness_metrics(df: pd.DataFrame) -> Tuple[float, float, float]:
    """_summary_

    :param df: _description_
    :type df: pd.DataFrame
    :return: _description_
    :rtype: Tuple[float, float, float]
    """

    # calculate the confusion matrix
    cm = confusion_matrix(df["y_true"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # calculate the rates
    N = TP + FP + FN + TN  # total population
    ACC = (TP + TN) / N  # accuracy
    TPR = TP / (TP + FN)  # True Positive Rate (sensitivity)
    FPR = FP / (FP + TN)  # False Positive Rate

    return ACC, TPR, FPR