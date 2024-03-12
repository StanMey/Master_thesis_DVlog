import numpy as np

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
def calculate_fairness_measures(y_true: list[np.ndarray], y_pred: list[np.ndarray], protected: list[str]) -> Tuple[float, float, float]:
    ...
    eq_odds = 1
    eq_oppor = 1
    eq_acc = 1
    return eq_odds, eq_oppor, eq_acc