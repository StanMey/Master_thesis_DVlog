import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset
from pathlib import Path


class BaseDVlogDataset(Dataset):
    def __init__(self, annotations_file: Path, data_dir: Path, dataset: str, sequence_length: int = 596):
        """Loads in the base DVlog dataset.

        :param annotations_file: The path to the file holding the annotations and names of the video features
        :type annotations_file: Path
        :param data_dir: The path to the stored extracted video features
        :type data_dir: Path
        :param dataset: Which type of dataset needs to be loaded in
        :type dataset: str
        :param sequence_length: The length of timestamps, defaults to 596
        :type sequence_length: int, optional
        """
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_labels = self.retrieve_dataset_labels(annotations_file)
        self.seq_length = sequence_length

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        video_id = self.data_labels.iloc[idx, 0]
        label = self.data_labels.iloc[idx, 1]

        visual_path = os.path.join(self.data_dir, str(video_id), f"{video_id}_visual.npy")
        acoustic_path = os.path.join(self.data_dir, str(video_id), f"{video_id}_acoustic.npy")

        visual = np.load(visual_path)
        acoustic = np.load(acoustic_path)

        if self.seq_length > visual.shape[0]:
            # the sequence is to short, so apply the padding to the timesteps (https://stackoverflow.com/questions/73444621/padding-one-numpy-array-to-achieve-the-same-number-os-columns-of-another-numpy-a)
            diff_timesteps = self.seq_length - visual.shape[0]

            padded_visual = np.concatenate((visual, np.zeros((diff_timesteps, visual.shape[1]))), axis=0)
            padded_acoustic = np.concatenate((acoustic, np.zeros((diff_timesteps, acoustic.shape[1]))), axis=0)

        else:
            # the sequence is to long, so apply the truncate
            padded_visual = visual[:self.seq_length]
            padded_acoustic = acoustic[:self.seq_length]
        
        # format the label as a class
        class_label = np.zeros(2)
        class_label[label] = 1

        return padded_visual, padded_acoustic, class_label
    
    def retrieve_dataset_labels(self, annotations_file: Path):
        """Filter the annotation dataset for the specific dataset we want to use.

        :param annotations_file: _description_
        :type annotations_file: Path
        """
        df_annotations = pd.read_csv(annotations_file)

        # filter and return
        return df_annotations[df_annotations["dataset"] == self.dataset]


