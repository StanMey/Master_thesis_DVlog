import numpy as np
import pandas as pd
import os
import torch

from torch.utils.data import Dataset
from pathlib import Path


class BaseDVlogDataset(Dataset):
    def __init__(self, annotations_file: Path, data_dir: Path, dataset: str, sequence_length: int = 596, to_tensor: bool = False, with_protected: bool = False):
        """Loads in the base DVlog dataset.

        :param annotations_file: The path to the file holding the annotations and names of the video features
        :type annotations_file: Path
        :param data_dir: The path to the stored extracted video features
        :type data_dir: Path
        :param dataset: Which type of dataset needs to be loaded in
        :type dataset: str
        :param sequence_length: The length of timestamps, defaults to 596
        :type sequence_length: int, optional
        :param to_tensor: , defaults to False
        :type to_tensor: bool, optional
        :param with_protected: Whether the protected attribute should be returned (for the evaluation part), defaults to False
        :type with_protected: bool, optional
        """
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_labels = self.retrieve_dataset_labels(annotations_file)
        self.seq_length = sequence_length
        self.to_tensor = to_tensor
        self.with_protected = with_protected

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        video_id = self.data_labels.iloc[idx, 0]
        label = self.data_labels.iloc[idx, 1]
        protected = self.data_labels.iloc[idx, 2]

        visual_path = os.path.join(self.data_dir, str(video_id), f"{video_id}_visual.npy")
        acoustic_path = os.path.join(self.data_dir, str(video_id), f"{video_id}_acoustic.npy")

        visual = np.load(visual_path).astype(np.float32)
        acoustic = np.load(acoustic_path).astype(np.float32)

        if self.seq_length > visual.shape[0]:
            # the sequence is to short, so apply the padding to the timesteps (https://stackoverflow.com/questions/73444621/padding-one-numpy-array-to-achieve-the-same-number-os-columns-of-another-numpy-a)
            visual_diff_timesteps = self.seq_length - visual.shape[0]
            acoustic_diff_timesteps = self.seq_length - acoustic.shape[0]

            padded_visual = np.concatenate((visual, np.zeros((visual_diff_timesteps, visual.shape[1]))), axis=0)
            padded_acoustic = np.concatenate((acoustic, np.zeros((acoustic_diff_timesteps, acoustic.shape[1]))), axis=0)

        else:
            # the sequence is to long, so apply the truncate
            padded_visual = visual[:self.seq_length]
            padded_acoustic = acoustic[:self.seq_length]
        
        # format the label as a class
        class_label = np.zeros(2)
        class_label[label] = 1

        if self.to_tensor:
            # convert the tuple to a tensor
            output_item =  torch.Tensor(padded_visual), torch.Tensor(padded_acoustic), torch.Tensor(class_label)
        else:
            output_item = (padded_visual, padded_acoustic, class_label)

        if self.with_protected:
            # also add the protected label to the output
            output_item = (*output_item, protected)
        
        return output_item


    def retrieve_dataset_labels(self, annotations_file: Path):
        """Filter the annotation dataset for the specific dataset we want to use.

        :param annotations_file: _description_
        :type annotations_file: Path
        """
        df_annotations = pd.read_csv(annotations_file)

        # filter and return
        return df_annotations[df_annotations["dataset"] == self.dataset]


class UnimodalEmbeddingsDataset(Dataset):
    def __init__(self, annotations_file: Path, data_dir: Path, dataset: str, sequence_length: int = 596, to_tensor: bool = False, with_protected: bool = False):
        """Loads in any kind of temporal features or embeddings for the model.

        :param annotations_file: The path to the file holding the annotations and names of the video features
        :type annotations_file: Path
        :param data_dir: The path to the stored extracted video features
        :type data_dir: Path
        :param dataset: Which type of dataset needs to be loaded in
        :type dataset: str
        :param sequence_length: The amount of embeddings to load in, defaults to 596
        :type sequence_length: int, optional
        :param to_tensor: , defaults to False
        :type to_tensor: bool, optional
        :param with_protected: Whether the protected attribute should be returned (for the evaluation part), defaults to False
        :type with_protected: bool, optional
        """
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_labels = self.retrieve_dataset_labels(annotations_file)
        self.seq_length = sequence_length
        self.to_tensor = to_tensor
        self.with_protected = with_protected

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        video_id = self.data_labels.iloc[idx, 0]
        label = self.data_labels.iloc[idx, 1]
        protected = self.data_labels.iloc[idx, 2]

        embeddings_path = os.path.join(self.data_dir, str(video_id), f"{video_id}.npy")
        embeddings = np.load(embeddings_path).astype(np.float32)

        if self.seq_length > embeddings.shape[0]:
            # the sequence is to short, so apply the padding to the timesteps (https://stackoverflow.com/questions/73444621/padding-one-numpy-array-to-achieve-the-same-number-os-columns-of-another-numpy-a)
            embeddings_diff_timesteps = self.seq_length - embeddings.shape[0]
            padded_embeddings = np.concatenate((embeddings, np.zeros((embeddings_diff_timesteps, embeddings.shape[1]))), axis=0)

        else:
            # the sequence is to long, so apply the truncate
            padded_embeddings = embeddings[:self.seq_length]
        
        # format the label as a class
        class_label = np.zeros(2)
        class_label[label] = 1

        if self.to_tensor:
            # convert the tuple to a tensor
            output_item =  torch.Tensor(padded_embeddings), torch.Tensor(class_label)
        else:
            output_item = (padded_embeddings, class_label)

        if self.with_protected:
            # also add the protected label to the output
            output_item = (*output_item, protected)
        
        return output_item


    def retrieve_dataset_labels(self, annotations_file: Path):
        """Filter the annotation dataset for the specific dataset we want to use.

        :param annotations_file: _description_
        :type annotations_file: Path
        """
        df_annotations = pd.read_csv(annotations_file)

        # filter and return
        return df_annotations[df_annotations["dataset"] == self.dataset]


