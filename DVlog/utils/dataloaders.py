import numpy as np
import pandas as pd
import os
import torch

from torch.utils.data import Dataset
from pathlib import Path

from util import ConfigDict


class MultimodalEmbeddingsDataset(Dataset):
    def __init__(self, dataset: str, train_config: ConfigDict, to_tensor: bool = False, with_protected: bool = False):
        """Loads in the (multiple) temporal features or embeddings for the model.

        :param dataset: Which type of dataset needs to be loaded in (train, test, or val)
        :type dataset: str
        :param train_config: 
        :type train_config: ConfigDict
        :param to_tensor: , defaults to False
        :type to_tensor: bool, optional
        :param with_protected: Whether the protected attribute should be returned (for the evaluation part), defaults to False
        :type with_protected: bool, optional
        """
        self.dataset = dataset
        self.config = train_config

        # check the input
        self.annotations_file = self.config.annotations_file
        assert os.path.exists(self.annotations_file), "Annotations file could not be found"

        self.data_labels = self.retrieve_dataset_labels(self.annotations_file)
        self.seq_length = self.config.sequence_length
        self.to_tensor = to_tensor
        self.with_protected = with_protected

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        video_id = self.data_labels.iloc[idx, 0]
        label = self.data_labels.iloc[idx, 1]
        protected = self.data_labels.iloc[idx, 2]

        # load in the first embedding
        embeddings = (self._retrieve_embedding(video_id, self.config.encoder1_data_dir, self.config.encoder1_feature_name), )

        if self.config.n_modalities >= 2:
            # load in the additional embedding
            embeddings = (*embeddings, self._retrieve_embedding(video_id, self.config.encoder2_data_dir, self.config.encoder2_feature_name))

        if self.config.n_modalities >= 2:
            # load in the last embedding
            raise NotImplementedError("Not yet implemented")
            embeddings = (*embeddings, self._retrieve_embedding(video_id, self.config.encoder3_data_dir, self.config.encoder3_feature_name))

        # apply the padding over all the embeddings
        padded_embeddings = [self._pad_sequences(embedding, self.seq_length) for embedding in embeddings]
        
        # format the label as a class
        class_label = np.zeros(2)
        class_label[label] = 1

        if self.to_tensor:
            # convert the tuple to a tensor
            output_item =  (*tuple([torch.Tensor(p_embedding) for p_embedding in padded_embeddings]), torch.Tensor(class_label))
        else:
            output_item = (*padded_embeddings, class_label)

        if self.with_protected:
            # also add the protected label to the output
            output_item = (*output_item, protected)

        return output_item
    
    def _retrieve_embedding(self, idx: int, data_dir: Path, feature_name: str) -> np.ndarray:
        """Loads in the numpy features.
        """
        # check if we have to update the feature name according to the DVlog features
        # (since DVlog features have the index in front of the feature name themselves)
        feature_name = feature_name.replace("#i", f"{str(idx)}")

        embeddings_path = os.path.join(data_dir, str(idx), f"{feature_name}.npy")
        embedding = np.load(embeddings_path).astype(np.float32)
        return embedding

    def _pad_sequences(self, embedding: np.ndarray, seq_length: int) -> np.ndarray:
        """Applies the padding for each embedding.
        """
        if seq_length > embedding.shape[0]:
            # the sequence is to short, so apply the padding to the timesteps (https://stackoverflow.com/questions/73444621/padding-one-numpy-array-to-achieve-the-same-number-os-columns-of-another-numpy-a)
            embeddings_diff_timesteps = seq_length - embedding.shape[0]
            padded_embeddings = np.concatenate((embedding, np.zeros((embeddings_diff_timesteps, embedding.shape[1]))), axis=0)

        else:
            # the sequence is to long, so apply the truncate
            padded_embeddings = embedding[:seq_length]
        
        return padded_embeddings


    def retrieve_dataset_labels(self, annotations_file: Path):
        """Filter the annotation dataset for the specific dataset we want to use.

        :param annotations_file: _description_
        :type annotations_file: Path
        """
        df_annotations = pd.read_csv(annotations_file)

        # filter and return
        return df_annotations[df_annotations["dataset"] == self.dataset]
