import numpy as np
import pandas as pd

from fairlearn.metrics import equalized_odds_ratio
from itertools import chain
from typing import Tuple
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support
)


## Performance measures
def calculate_performance_measures(y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> Tuple[float, float, float, float, float]:
    """Function to calculate all the performance measures.

    :param y_true: A 2d list with the model's predictions
    :type y_true: list[np.ndarray]
    :param y_pred: A 2d list with the ground truths
    :type y_pred: list[np.ndarray]
    :return: Returns the calculated performance measures
    :rtype: Tuple[float, float, float, float, float]
    """

    # flatten the predictions and ground truths
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    # 
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    w_precision, w_recall, w_fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    _, _, m_fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    return accuracy, w_precision, w_recall, w_fscore, m_fscore


### Gender-based performance measures
def calculate_gender_performance_measures(y_true: list[np.ndarray], y_pred: list[np.ndarray], protected: list[str]) -> list[Tuple[str, float, float, float]]:
    """_summary_

    :param y_true: A 2d list with the model's predictions
    :type y_true: list[np.ndarray]
    :param y_pred: A 2d list with the ground truths
    :type y_pred: list[np.ndarray]
    :return: _description_
    :rtype: Tuple[float, float, float, float]
    """
    # first flatten the predictions and ground truths
    y_pred = np.argmax(np.concatenate(y_pred), axis=1)
    y_true = np.argmax(np.concatenate(y_true), axis=1)
    y_protected = np.array(list(chain.from_iterable(protected)))
    
    metrics = []
    for gender_value in np.unique(y_protected):

        # check the precision, recall, F1-score for each gender
        indices = np.where(y_protected == gender_value)
        y_pred_subset = y_pred[indices]
        y_true_subset = y_true[indices]

        precision, recall, fscore, _ = precision_recall_fscore_support(y_true_subset, y_pred_subset, average="weighted", zero_division=0)
        metrics.append((gender_value, precision, recall, fscore))

    return metrics


## The fairness measures
def calculate_fairness_measures(y_true: list[np.ndarray], y_pred: list[np.ndarray], protected: list[str], unprivileged: str):
    """Calculates the actual fairness measures.

    :param y_true: A 2d list with the model's predictions
    :type y_true: list[np.ndarray]
    :param y_pred: A 2d list with the ground truths
    :type y_pred: list[np.ndarray]
    :param protected: A 2d list with the sensitive features
    :type protected: list[str]
    :param unprivileged: The value of the unprivileged group
    :type unprivileged: str
    :return: Returns the calculated fairness measures
    :rtype: Tuple[float, float, float, Tuple[float, float], Tuple[float, float]]
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

    # for the equalized odds we use the approach by fairlearn (https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html#equalized-odds)
    # "The smaller of two metrics: `true_positive_rate_ratio` and `false_positive_rate_ratio`."
    fairl_eq_odds = equalized_odds_ratio(y_true, y_pred, sensitive_features=protected)
    # eq_odds = min((unpriv_tpr / priv_tpr), (unpriv_fpr / priv_fpr))

    return eq_oppor, eq_acc, fairl_eq_odds, (unpriv_tpr, unpriv_fpr), (priv_tpr, priv_fpr)


def fairness_metrics(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Calculates the prediction rates.

    :param df: dataframe holding the predictions and ground truths
    :type df: pd.DataFrame
    :return: Returns the accuracy, TPR and FPR
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