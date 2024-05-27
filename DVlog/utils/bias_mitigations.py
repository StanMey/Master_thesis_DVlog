import pandas as pd
import numpy as np
import random

from itertools import combinations


def apply_oversampling(annotations_df: pd.DataFrame, seed: int=42) -> pd.DataFrame:
    """Apply the oversampling bias mitigation method on the dataset.
    Calculates the difference between the minority and majority class and uses this difference to randomly oversample the minority class

    :param annotations_file: Dataframe consisting of the columns 'gender', and 'depression'.
    :type annotations_file: pd.DataFrame
    """
    # get the minority and majority group
    samples_dist = annotations_df.value_counts("gender").sort_values()
    labels, samples = samples_dist.index.tolist(), samples_dist.tolist()
    minority_label, minority_count, majority_count = labels[0], samples[0], samples[1]

    # get all the samples for the minority class
    minority_df = annotations_df[annotations_df["gender"] == minority_label]
    sample_diff = majority_count - minority_count

    # randomly select from the dataframe
    sample_df = minority_df.sample(n=sample_diff, replace=True, random_state=seed)

    # concatenate this sampled dataframe to the original annotation set
    annotations_df = pd.concat([annotations_df, sample_df])
    return annotations_df


def apply_mixfeat_oversampling(annotations_df: pd.DataFrame, mixfeat_type: str, dataset: str="train", seed: int=42) -> pd.DataFrame:
    """Setup the mixfeat features following the 'Fairness for a Small Dataset of Multi-modal Dyadic Mental Well-being Coaching' paper.

    :param annotations_df: Dataframe consisting of (at least) the columns 'gender', and 'depression'.
    :type annotations_df: pd.DataFrame
    :param mixfeat_type: _description_
    :type mixfeat_type: str
    :param dataset: _description_, defaults to "train"
    :type dataset: str, optional
    :param seed: _description_, defaults to 42
    :type seed: int, optional
    :return: _description_
    :rtype: pd.DataFrame
    """
    # select the appropriate sampling approach
    if mixfeat_type == "group_upsample":
         # upsample the minority group (gender)
         ...
    elif mixfeat_type == "class_upsample":
         # upsample the minority class (depression) in a mixedgender approach
         ...
    elif mixfeat_type == "":
         ...
    
    # get the minority and majority group
    samples_dist = annotations_df.value_counts("gender").sort_values()
    labels, samples = samples_dist.index.tolist(), samples_dist.tolist()
    minority_label, minority_count, majority_count = labels[0], samples[0], samples[1]

    # get all the samples for the minority class and divide them in positive and negative samples
    pos_minority_vids = annotations_df[(annotations_df["gender"] == minority_label) & (annotations_df["label"] == 1)]["video_id"].tolist()
    neg_minority_vids = annotations_df[(annotations_df["gender"] == minority_label) & (annotations_df["label"] == 0)]["video_id"].tolist()
    sample_diff = majority_count - minority_count

    # randomly select combinations of video_ids for both labels
    pos_combinations = list(combinations(pos_minority_vids, 2))
    pos_samples_choice = random.sample(pos_combinations, sample_diff // 2)

    neg_combinations = list(combinations(neg_minority_vids, 2))
    neg_samples_choice = random.sample(neg_combinations, sample_diff // 2)

    # select the beta's for loading in the data (the inverse of the probs will be calculated later)
    beta_probs = map(tuple, np.random.rand(sample_diff, 1))
    
    # expand the original annotations dataframe with the mixfeat column
    annotations_df["mixfeat"] = None
    annotations_df["mixfeat_probs"] = None

    # build the dataframe that is to be concatenated to the original dataframe which contains the chosen mixfeat combinations
    mixfeat_df = pd.DataFrame(
        {
            "video_id": [None] * sample_diff,
            "label": [1 for _ in range(len(pos_samples_choice))] + [0 for _ in range(len(neg_samples_choice))],
            "gender": [minority_label] * sample_diff,
            "dataset": [self.dataset] * sample_diff,
            "mixfeat": pos_samples_choice + neg_samples_choice,
            "mixfeat_probs": beta_probs
        }
    )

    # concatenate both dataframes
    annotations_df = pd.concat([annotations_df, mixfeat_df])
    return annotations_df


def setup_mixfeat_mixedgender(annotations_df: pd.DataFrame, seed: int=42) -> pd.DataFrame:
        """Setup the mixfeat features following the 'Fairness for a Small Dataset of Multi-modal Dyadic Mental Well-being Coaching' paper.

        :param annotations_file: _description_
        :type annotations_file: Path
        """
        # get the minority and majority group
        samples_dist = annotations_df.value_counts("label").sort_values()
        labels, samples = samples_dist.index.tolist(), samples_dist.tolist()
        minority_label, minority_count, majority_count = labels[0], samples[0], samples[1]

        # get all the samples for the minority class and divide them in positive and negative samples
        pos_minority_vids = annotations_df[(annotations_df["gender"] == minority_label) & (annotations_df["label"] == 1)]["video_id"].tolist()
        neg_minority_vids = annotations_df[(annotations_df["gender"] == minority_label) & (annotations_df["label"] == 0)]["video_id"].tolist()
        sample_diff = majority_count - minority_count

        # randomly select combinations of video_ids for both labels
        pos_combinations = list(combinations(pos_minority_vids, 2))
        pos_samples_choice = random.sample(pos_combinations, sample_diff // 2)

        neg_combinations = list(combinations(neg_minority_vids, 2))
        neg_samples_choice = random.sample(neg_combinations, sample_diff // 2)

        # select the beta's for loading in the data (the inverse of the probs will be calculated later)
        beta_probs = map(tuple, np.random.rand(sample_diff, 1))
        
        # expand the original annotations dataframe with the mixfeat column
        annotations_df["mixfeat"] = None
        annotations_df["mixfeat_probs"] = None


if __name__ == "__main__":
    df_annotations = pd.read_csv(r"./dataset/dvlog_labels_v2.csv")

    # filter for the dataset
    df_annotations = df_annotations[df_annotations["dataset"] == "train"]

    # run the function
    setup_mixfeat_mixedgender(df_annotations)