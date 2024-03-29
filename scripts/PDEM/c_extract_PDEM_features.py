import audeer
import audonnx
import pandas as pd
import audinterface
import os

from preprocessing.utils import rename_file
from pathlib import Path


# Below is a git-hub repository link of PDEM with the documentation for using PDEM
# https://github.com/audeering/w2v2-how-to?tab=readme-ov-file

def extract_embedding_pdem(root_dir: Path, index_file_path: Path):
    
    # setup the model paths
    path_pdem_cache = os.path.join(root_dir, "pdem_cache")
    path_model_cache = os.path.join(root_dir, "model_cache")
    model_cache_root = audeer.mkdir(path_pdem_cache)
    model_root = audeer.mkdir(path_model_cache)

    cache_root = os.path.join(root_dir, "features")  # Set the output files' root directory (feature embed scores)

    # Step 0: Download the PDEM model to the model_root dir and Load the PDEM model, this has been done in b_extract_vad

    # Step 1: load model and set sampling rate
    model = audonnx.load(model_root)
    sampling_rate = 16000  # sampling rate of DAIC-WOZ is 16000

    # Define model interface, output is PDEM embedding. This format is fixed defined by audinterface package.
    hidden_states = audinterface.Feature(
        model.labels('hidden_states'), process_func=model,
        process_func_args={'outputs': 'hidden_states'},
        sampling_rate=sampling_rate, resample=True,
        num_workers=8, verbose=True)

    # Step 2: Load index file
    dvlog_dataset = pd.read_csv(index_file_path)

    # Setting the index to 'file_path' column: where audio recording is
    dvlog_dataset.set_index('file_path', inplace=True)
    dvlog_dataset.index = dvlog_dataset.index.astype(str)

    # Step 3: Extracting VAD scores using PDEM model for each wav
    """ Development set """
    hidden_states.process_index(daic_woz_dev.index, root=root_dir,
                                cache_root=audeer.path(cache_root, 'pdem_wav2vec/'))

    path_to_pdem_folder = os.path.join(cache_root, 'pdem_wav2vec/')
    files = os.listdir(path_to_pdem_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:5] == 'pdem_wav2vec':  # check if the filename is a random number
            rename_file(os.path.join(path_to_pdem_folder, file), 'embedding_dev.pkl')

    """ Training set """
    hidden_states.process_index(daic_woz_train.index, root=root_dir,
                                cache_root=audeer.path(cache_root, 'pdem_wav2vec/'))

    files = os.listdir(path_to_pdem_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:5] == 'pdem_wav2vec':
            rename_file(os.path.join(path_to_pdem_folder, file), 'embedding_train.pkl')

    """ Test set """
    hidden_states.process_index(daic_woz_test.index, root=root_dir,
                                cache_root=audeer.path(cache_root, 'pdem_wav2vec/'))

    files = os.listdir(path_to_pdem_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:5] == 'pdem_wav2vec':
            rename_file(os.path.join(path_to_pdem_folder, file), 'embedding_test.pkl')

    return


if __name__ == "__main__":

    root_dir = Path(r"E:/master/data/PDEM/model")
    index_file_path = Path(r"E:/master/data/PDEM/pdem_index_file.csv")

    extract_embedding_pdem(root_dir, index_file_path)