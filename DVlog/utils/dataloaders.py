import numpy as np
import pandas as pd
import os
import torch
import json

from torch.utils.data import Dataset
from pathlib import Path

from utils.util import ConfigDict
from utils.bias_mitigations import apply_oversampling, apply_mixfeat_oversampling


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

        if self.config.n_modalities == 3:
            # load in the last embedding
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
            # also add the protected label + the video_id itself to the output (since this will be needed for the evaluation part)
            output_item = (*output_item, protected, video_id)

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


class SyncedMultimodalEmbeddingsDataset(Dataset):
    def __init__(self, dataset: str, train_config: ConfigDict, to_tensor: bool = False, with_protected: bool = False):
        """Loads in the (multiple) temporal features or embeddings for the model.
        Synchronises the feature with the rest of the features using the supplied sync_file
        This means that when the span of a sentence is from 1.2 to 3.4 seconds, we take the average of the other features of the same span.

        :param dataset: Which type of dataset needs to be loaded in (train, test, or val)
        :type dataset: str
        :param train_config: 
        :type train_config: ConfigDict
        :param to_tensor: , defaults to False
        :type to_tensor: bool, optional
        :param with_protected: Whether the protected attribute and the video_id should be returned (for the evaluation part), defaults to False
        :type with_protected: bool, optional
        """
        self.dataset = dataset
        self.config = train_config

        # check the input
        self.annotations_file = self.config.annotations_file
        self.sync_file = self.config.sync_file
        assert os.path.exists(self.annotations_file), "Annotations file could not be found"
        assert os.path.exists(self.sync_file), "Sync file could not be found"

        self.data_labels = self.retrieve_dataset_labels(self.annotations_file)
        self.sync_dict = self.load_sync_file(self.sync_file)
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
            embeddings = (*embeddings, self._retrieve_synced_embedding(video_id, self.config.encoder2_data_dir, self.config.encoder2_feature_name, self.sync_dict))

        if self.config.n_modalities == 3:
            # load in the last embedding
            embeddings = (*embeddings, self._retrieve_synced_embedding(video_id, self.config.encoder3_data_dir, self.config.encoder3_feature_name, self.sync_dict))

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
            # also add the protected label + the video_id itself to the output (since this will be needed for the evaluation part)
            output_item = (*output_item, protected, video_id)

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

    def _retrieve_synced_embedding(self, idx: int, data_dir: Path, feature_name: str, sync_dict: dict) -> np.ndarray:
        """Retrieves the synced embedding averages based on the sync_dict.
        """
        # check if we have to update the feature name according to the DVlog features
        # (since DVlog features have the index in front of the feature name themselves)
        feature_name = feature_name.replace("#i", f"{str(idx)}")

        # load in the embeddings
        embeddings_path = os.path.join(data_dir, str(idx), f"{feature_name}.npy")
        embedding = np.load(embeddings_path).astype(np.float32)

        # apply the synchronisation
        embeddings = []
        for (start_t, end_t) in sync_dict.get(str(idx)):
            sliced_embedding = embedding[int(np.floor(start_t)):int(np.floor(end_t))+1,:]
            if sliced_embedding.size == 0:
                # apparently for one of the input features the synced_file gets empty slices, so mitigate this by just returning zeros
                avg_embedding = np.zeros(embedding.shape[1], dtype=np.float32)
            else:
                avg_embedding = np.mean(sliced_embedding, axis=0)
            embeddings.append(avg_embedding)
        return np.array(embeddings)

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


    def load_sync_file(self, sync_file: Path):
        """Load in the sync file.

        :param sync_file: _description_
        :type sync_file: Path
        """
        with open(sync_file) as user_file:
            parsed_json = json.load(user_file)
        
        return parsed_json


class BiasMitMultimodalEmbeddingsDataset(Dataset):
    def __init__(self, dataset: str, train_config: ConfigDict, to_tensor: bool = False, with_protected: bool = False):
        """Loads in the (multiple) temporal features or embeddings for the model.

        :param dataset: Which type of dataset needs to be loaded in (train, test, or val)
        :type dataset: str
        :param train_config: 
        :type train_config: ConfigDict
        :param to_tensor: , defaults to False
        :type to_tensor: bool, optional
        :param with_protected: Whether the protected attribute and the video_id should be returned (for the evaluation part), defaults to False
        :type with_protected: bool, optional
        """
        self.dataset = dataset
        self.config = train_config

        # check the input
        self.annotations_file = self.config.annotations_file
        assert os.path.exists(self.annotations_file), "Annotations file could not be found"

        # load in the dataset labels (take into account the type of bias mitigation that has to be performed)
        self.n_modalities = self.config.n_modalities
        self.bias_mit_type = self.config.bias_mit
        self.data_labels = self.retrieve_dataset_labels(self.annotations_file, self.bias_mit_type)
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
        if self.bias_mit_type == "mixfeat":
            mixfeat_ids = self.data_labels.iloc[idx, 4]
            mixfeat_probs = self.data_labels.iloc[idx, 5]

            if mixfeat_ids:
                # we have a synthetic example, so use the mixfeat approach
                embeddings = (self._retrieve_mixfeat_embeddings(mixfeat_ids, mixfeat_probs[0], self.config.encoder1_data_dir, self.config.encoder1_feature_name),)

            else:
                # we have a normal embedding, so just retrieve it the normal way
                embeddings = (self._retrieve_embedding(video_id, self.config.encoder1_data_dir, self.config.encoder1_feature_name),)
        
        else:
            embeddings = (self._retrieve_embedding(video_id, self.config.encoder1_data_dir, self.config.encoder1_feature_name),)

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
            # also add the protected label + the video_id itself to the output (since this will be needed for the evaluation part)
            output_item = (*output_item, protected, video_id)

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
    
    def _retrieve_mixfeat_embeddings(self, idxs: tuple[int, int], prob: float, data_dir: Path, feature_name: str) -> np.ndarray:
        """Loads in the embeddings using the mixfeat approach.
        """
        idx1, idx2 = idxs

        # load in the embeddings
        embeddings_path = os.path.join(data_dir, str(idx1), f"{feature_name}.npy")
        embedding1 = np.load(embeddings_path).astype(np.float32)

        embeddings_path = os.path.join(data_dir, str(idx2), f"{feature_name}.npy")
        embedding2 = np.load(embeddings_path).astype(np.float32)

        # take the shortest embedding and apply the probabilities on that one
        if embedding1.shape[0] <= embedding2.shape[0]:
            # first embedding is shorter
            final_embedding = (embedding1 * prob) + (embedding2[:embedding1.shape[0]] * (1 - prob))

        else:
            # second embedding is shorter
            final_embedding = (embedding1[:embedding2.shape[0]] * prob) + (embedding2 * (1 - prob))

        return final_embedding

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

    def retrieve_dataset_labels(self, annotations_file: Path, bias_mit_type: str):
        """Filter the annotation dataset for the specific dataset we want to use.

        :param annotations_file: _description_
        :type annotations_file: Path
        """
        df_annotations = pd.read_csv(annotations_file)

        # filter for the dataset
        df_annotations = df_annotations[df_annotations["dataset"] == self.dataset]

        if bias_mit_type == "mixfeat":
            df_annotations = self._setup_mixfeat_dataset(df_annotations)
        elif bias_mit_type == "oversample":
            df_annotations = self._setup_oversample_dataset(df_annotations)

        return df_annotations