import audeer
import audonnx
import pandas as pd
import audinterface
import os

# from preprocessing.utils import rename_file
from pathlib import Path


# Below is a git-hub repository link of PDEM with the documentation for using PDEM
# https://github.com/audeering/w2v2-how-to?tab=readme-ov-file

def extract_vad_pdem(root_dir: Path, index_file_path: Path):

    # setup the model paths
    path_pdem_cache = os.path.join(root_dir, "pdem_cache")
    path_model_cache = os.path.join(root_dir, "model_cache")
    model_cache_root = audeer.mkdir(path_pdem_cache)
    model_root = audeer.mkdir(path_model_cache)

    cache_root = os.path.join(root_dir, "features")  # Set the output files' root directory (pdem_vad scores)

    # Step 0: Download the PDEM model to the model_root dir and Load the PDEM model
    url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    archive_path = audeer.download_url(url, model_cache_root, verbose=True)
    audeer.extract_archive(archive_path, model_root)

    # Step 1: load model and set sampling rate
    model = audonnx.load(model_root)
    sampling_rate = 16000  # sampling rate of DAIC-WOZ is 16000

    # Define model interface, output is valence, arousal and dominance scores. This format is fixed
    logits = audinterface.Feature(
        model.labels('logits'), process_func=model,
        process_func_args={'outputs': 'logits'},
        sampling_rate=sampling_rate, resample=True,
        num_workers=8, verbose=True)  # optional to change num_workers: higher = faster

    # Step 2: Load index file
    dvlog_dataset = pd.read_csv(index_file_path, delimiter=";")

    # Setting the index to 'file_path' column: where audio recording is
    dvlog_dataset.set_index('file_path', inplace=True)
    dvlog_dataset.index = dvlog_dataset.index.astype(str)

    # Step 3: Extracting VAD scores using PDEM model for each wav
    logits.process_index(dvlog_dataset.index, root=root_dir,
                         # The output files will be saved here. File string_functionals will be a random number ending with '.pkl'
                         cache_root=audeer.path(cache_root, 'pdem_vad/'))

    # path_to_vad_folder = os.path.join(cache_root, 'pdem_vad/')
    # files = os.listdir(path_to_vad_folder)
    # for file in files:
    #     if file.endswith('.pkl') and not file[:3] == 'pdem_vad':  # check if the filename is a random number
    #         rename_file(os.path.join(path_to_vad_folder, file), 'vad_dev.pkl')

    return


if __name__ == "__main__":

    root_dir = Path(r"E:/master/data/PDEM/model")
    index_file_path = Path(r"E:/master/data/PDEM/pdem_index_file.csv")

    extract_vad_pdem(root_dir, index_file_path)
