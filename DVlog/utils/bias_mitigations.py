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


def apply_mixfeat_oversampling(annotations_df: pd.DataFrame, mixfeat_type: str, n_modalities: int, seed: int=42) -> pd.DataFrame:
    """Setup the mixfeat features following the 'Fairness for a Small Dataset of Multi-modal Dyadic Mental Well-being Coaching' paper.

    :param annotations_df: Dataframe consisting of (at least) the columns 'gender', and 'depression'.
    :type annotations_df: pd.DataFrame
    :param mixfeat_type: Specifies which mixfeat experiment is ran, can be either ['group_upsample', 'mixgender_upsample', 'subgroup_upsample', 'pure_synthetic']
    :type mixfeat_type: str
    :param n_modalities: _description_
    :type n_modalities: int
    :param seed: _description_, defaults to 42
    :type seed: int, optional
    :return: _description_
    :rtype: pd.DataFrame
    """
    # select the appropriate sampling approach
    if mixfeat_type == "group_upsample":
        # upsample the minority group (which is the 'male' group)
        mixfeat_df = setup_mixfeat_group_upsample(annotations_df, n_modalities, seed)

    elif mixfeat_type == "mixgender_upsample":
        # upsample the minority class (which is the 'non-depressed' class) using the mixgender approach
        mixfeat_df = setup_mixfeat_mixgender_upsample(annotations_df, n_modalities, seed)

    elif mixfeat_type == "subgroup_upsample":
        # upsample the minority class-minority gender subgroup (which are the 'depressed male' subgroup)
        mixfeat_df = setup_mixfeat_subgroup_upsample(annotations_df, n_modalities, seed)

    elif mixfeat_type == "synthetic":
        # Use only the mixfeat samples from the 'depression' class.
        mixfeat_df = setup_mixfeat_synthetic_only(annotations_df, n_modalities, seed)
    
    # setup and expand the original annotations dataframe with the mixfeat columns
    annotations_df.drop("dataset", axis=1, inplace=True)
    annotations_df["mixfeat"] = None
    annotations_df["mixfeat_probs"] = None

    if mixfeat_type == "synthetic":
        # with the synthetic we have to remove all the existing depression examples before concatenating it.
        annotations_df = annotations_df[annotations_df["label"] == 0]
        annotations_df = pd.concat([annotations_df, mixfeat_df])
    
    else:
        # here we have to concatenate the mixfeat dataframe with the original dataframe
        annotations_df = pd.concat([annotations_df, mixfeat_df])

    return annotations_df


def setup_mixfeat_group_upsample(annotations_df: pd.DataFrame, n_modalities: int, seed: int) -> pd.DataFrame:
    """Here we upsample the minority group using mixfeat.
    For the D-Vlog this will mean that we upsample the 'male' group

    :param annotations_file: the Dataframe consisting of the samples.
    :type annotations_file: Path
    """
    # get the minority and majority group
    samples_dist = annotations_df.value_counts("gender").sort_values()
    labels, samples = samples_dist.index.tolist(), samples_dist.tolist()
    minority_label, minority_count, majority_count = labels[0], samples[0], samples[1]

    # get all the samples for the minority class and divide them in positive and negative samples
    pos_minority_vids = annotations_df[(annotations_df["gender"] == minority_label) & (annotations_df["label"] == 1)]["video_id"].tolist()
    neg_minority_vids = annotations_df[(annotations_df["gender"] == minority_label) & (annotations_df["label"] == 0)]["video_id"].tolist()
    sample_diff = majority_count - minority_count
    n_pos_samples = sample_diff // 2

    # randomly select combinations of video_ids for both labels
    pos_combinations = list(combinations(pos_minority_vids, 2))
    pos_samples_choice = random.sample(pos_combinations, n_pos_samples)

    neg_combinations = list(combinations(neg_minority_vids, 2))
    neg_samples_choice = random.sample(neg_combinations, (sample_diff - n_pos_samples))

    # select the beta's for loading in the data (the inverse of the probs will be calculated later)
    beta_probs = list(map(tuple, np.random.rand(sample_diff, n_modalities)))

    mixfeat_df = pd.DataFrame(
        {
            "video_id": [None] * sample_diff,
            "label": [1] * n_pos_samples + [0] * len(neg_samples_choice),
            "gender": [minority_label] * sample_diff,
            "mixfeat": pos_samples_choice + neg_samples_choice,
            "mixfeat_probs": beta_probs
        }
    )
    return mixfeat_df


def setup_mixfeat_mixgender_upsample(annotations_df: pd.DataFrame, n_modalities: int, seed: int) -> pd.DataFrame:
    """Here we upsample the depression class using mixfeat in a mixgendered way.
    For the D-Vlog this means that the depression class gets upsampled by .
    
    :param annotations_file: the Dataframe consisting of the samples.
    :type annotations_file: Path
    """
    # get the minority and majority group
    samples_dist = annotations_df.value_counts("label").sort_values()
    labels, samples = samples_dist.index.tolist(), samples_dist.tolist()
    minority_label, minority_count, majority_count = labels[0], samples[0], samples[1]

    # get all the samples for the minority class and divide them in male and female samples
    male_minority_vids = annotations_df[(annotations_df["gender"] == "m") & (annotations_df["label"] == minority_label)]["video_id"].tolist()
    female_minority_vids = annotations_df[(annotations_df["gender"] == "f") & (annotations_df["label"] == minority_label)]["video_id"].tolist()
    sample_diff = majority_count - minority_count

    # randomly select ids for both genders to combine later
    male_samples_choice = random.sample(male_minority_vids, sample_diff)
    female_samples_choice = random.sample(female_minority_vids, sample_diff)
    final_samples = list(zip(male_samples_choice, female_samples_choice))

    # select the beta's for loading in the data (the inverse of the probs will be calculated later)
    beta_probs = list(map(tuple, np.random.rand(sample_diff, n_modalities)))

    mixfeat_df = pd.DataFrame(
        {
            "video_id": [None] * sample_diff,
            "label": [minority_label] * sample_diff,
            "gender": ["both"] * sample_diff,
            "mixfeat": final_samples,
            "mixfeat_probs": beta_probs
        }
    )
    
    return mixfeat_df


def setup_mixfeat_subgroup_upsample(annotations_df: pd.DataFrame, n_modalities: int, seed: int) -> pd.DataFrame:
    """Here we upsample minority class-minority gender subgroup using mixfeat.
    For the D-Vlog this will mean that we upsample the 'depressed male' group

    :param annotations_file: the Dataframe consisting of the samples.
    :type annotations_file: Path
    """
    # get the minority and majority group
    annotations_df = annotations_df[annotations_df["label"] == 1]
    samples_dist = annotations_df.value_counts("gender").sort_values()
    labels, samples = samples_dist.index.tolist(), samples_dist.tolist()
    minority_label, minority_count, majority_count = labels[0], samples[0], samples[1]

    # get all the samples for the minority subgroup
    minority_vids = annotations_df[annotations_df["gender"] == minority_label]["video_id"].tolist()
    sample_diff = majority_count - minority_count

    # randomly select combinations of video_ids
    all_combinations = list(combinations(minority_vids, 2))
    samples_choice = random.sample(all_combinations, sample_diff)

    # select the beta's for loading in the data (the inverse of the probs will be calculated later)
    beta_probs = list(map(tuple, np.random.rand(sample_diff, n_modalities)))

    mixfeat_df = pd.DataFrame(
        {
            "video_id": [None] * sample_diff,
            "label": [1] * sample_diff,
            "gender": [minority_label] * sample_diff,
            "mixfeat": samples_choice,
            "mixfeat_probs": beta_probs
        }
    )
    return mixfeat_df


def setup_mixfeat_synthetic_only(annotations_df: pd.DataFrame, n_modalities: int, seed: int) -> pd.DataFrame:
    """Here we upsample minority class-minority gender subgroup using mixfeat.
    For the D-Vlog this will mean that we upsample the 'depressed male' group

    :param annotations_file: the Dataframe consisting of the samples.
    :type annotations_file: Path
    """
    # get the minority and majority group
    annotations_df = annotations_df[annotations_df["label"] == 1]
    samples_dist = annotations_df.value_counts("gender").sort_values()
    labels, samples = samples_dist.index.tolist(), samples_dist.tolist()
    minority_label, minority_count, majority_label, majority_count = labels[0], samples[0], labels[1], samples[1]
    sample_diff = minority_count + majority_count

    # get all the samples for the minority class and divide them in positive and negative samples
    min_depressed_vids = annotations_df[annotations_df["gender"] == minority_label]["video_id"].tolist()
    maj_depressed_vids = annotations_df[annotations_df["gender"] == majority_label]["video_id"].tolist()

    # randomly select combinations of video_ids for both genders
    min_combinations = list(combinations(min_depressed_vids, 2))
    min_samples_choice = random.sample(min_combinations, minority_count)

    maj_combinations = list(combinations(maj_depressed_vids, 2))
    maj_samples_choice = random.sample(maj_combinations, majority_count)

    # select the beta's for loading in the data (the inverse of the probs will be calculated later)
    beta_probs = list(map(tuple, np.random.rand(sample_diff, n_modalities)))

    mixfeat_df = pd.DataFrame(
        {
            "video_id": [None] * sample_diff,
            "label": [1] * sample_diff,
            "gender": [minority_label] * minority_count + [majority_label] * majority_count,
            "mixfeat": min_samples_choice + maj_samples_choice,
            "mixfeat_probs": beta_probs
        }
    )
    return mixfeat_df


if __name__ == "__main__":
    df_annotations = pd.read_csv(r"./dataset/dvlog_labels_v2.csv")

    # filter for the dataset
    df_annotations = df_annotations[df_annotations["dataset"] == "train"]

    # run the function
    # apply_mixfeat_oversampling(df_annotations, "group_upsample", 1)
    # apply_mixfeat_oversampling(df_annotations, "mixgender_upsample", 1)
    # apply_mixfeat_oversampling(df_annotations, "subgroup_upsample", 1)
    df = apply_mixfeat_oversampling(df_annotations, "synthetic", 1)
    print(df.head())