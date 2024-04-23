import numpy as np
import pandas as pd

from itertools import chain
from typing import Tuple
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import (
    equalized_odds_ratio
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
)


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
    w_precision, w_recall, w_fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    _, _, m_fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    return accuracy, w_precision, w_recall, w_fscore, m_fscore


### Gender-based performance measures
def calculate_gender_performance_measures(y_true: list[np.ndarray], y_pred: list[np.ndarray], protected: list[str]) -> list[Tuple[str, float, float, float]]:

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

    # for the equalized odds we use the approach by fairlearn (https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html#equalized-odds)
    # "The smaller of two metrics: `true_positive_rate_ratio` and `false_positive_rate_ratio`."
    print((unpriv_tpr / priv_tpr), (unpriv_fpr / priv_fpr))
    eq_odds = equalized_odds_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=protected)
    print(eq_odds)

    return eq_odds, eq_oppor, eq_acc


def calculate_fairness_measures2(y_true: list[np.ndarray], y_pred: list[np.ndarray], protected: list[str], unprivileged: str, verbose: bool=False) -> dict:
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
    print(unpriv_fpr, unpriv_tpr)
    print(priv_fpr, priv_tpr)
    
    # calculate the ratios
    # eq_oppor = unpriv_tpr / priv_tpr
    # eq_acc = unpriv_acc / priv_acc

    # for the equalized odds we use the approach by fairlearn (https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html#equalized-odds)
    # "The smaller of two metrics: `true_positive_rate_ratio` and `false_positive_rate_ratio`."
    print((unpriv_tpr / priv_tpr), (unpriv_fpr / priv_fpr))
    eq_odds = equalized_odds_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=protected, method="to_overall")
    print(eq_odds)

    return eq_odds, 1, 1


def print_metrics():
    ...

    # print out all the metrics
# print(f"Model: {SAVED_MODEL_WEIGHTS}\n----------")
# print(f"(weighted) -->\nAccuracy: {accuracy}\nPrecision: {w_precision}\nRecall: {w_recall}\nF1-score: {w_fscore}")
# print(f"(macro) -->\nF1-score: {m_fscore}\n----------")
# print(f"Equal odds: {eq_odds}\nEqual opportunity: {eq_oppor}\nEqual accuracy: {eq_oppor}\n----------")

# # print out the gender-based metrics
# print("Gender-based metrics:\n----------")
# gender_metrics = calculate_gender_performance_measures(y_labels, predictions, protected)
# for gender_metric in gender_metrics:
#     print("Metrics for label {0}:\n---\nPrecision: {1}\nRecall: {2}\nF1-score: {3}\n----------".format(*gender_metric))


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