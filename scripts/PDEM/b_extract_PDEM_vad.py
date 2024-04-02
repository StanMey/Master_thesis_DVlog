import audeer
import audonnx
import pandas as pd
import audinterface
import os

# from preprocessing.utils import rename_file
from pathlib import Path


def rename_file(folder_path, old_name, new_name):

    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)
    
    # Renaming the file
    os.rename(old_path, new_path)


# Below is a git-hub repository link of PDEM with the documentation for using PDEM
# https://github.com/audeering/w2v2-how-to?tab=readme-ov-file

def extract_vad_pdem(root_dir: Path, index_file_path: Path):

    filename, _ = os.path.splitext(index_file_path)
    dataset_type = filename.split("_")[-1]

    assert os.path.exists(index_file_path), "index file path or directory does not exist"

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
    # model = audonnx.load(model_root, device='cuda')
    model = audonnx.load(model_root)
    # sampling_rate = 16000  # sampling rate of DAIC-WOZ is 16000
    sampling_rate = 44100  # sampling rate of .wav file is 44100 Hz

    # Define model interface, output is the concatenation of PDEM embeddings, valence, arousal and dominance scores. This format is fixed
    logits = audinterface.Feature(
        model.labels(),
        process_func=model,
        process_func_args={'concat': 'true'},
        sampling_rate=sampling_rate, resample=True,
        num_workers=10, verbose=True)  # optional to change num_workers: higher = faster

    # Step 2: Load index file/directory
    if os.path.isfile(index_file_path):
        datasets = [index_file_path]
    else:
        datasets = [os.path.join(index_file_path, f) for f in os.listdir(index_file_path) if os.path.isfile(os.path.join(index_file_path, f))]

    for dataset_path in datasets:

        # first check if we already have extracted the dataset
        filename, _ = os.path.splitext(dataset_path)
        output_file_name = f"vad_features_{filename.split('file_')[-1]}.pkl"
        output_file_path = os.path.join(cache_root, 'pdem_vad/', output_file_name)

        if os.path.exists(output_file_path):
            print(f"File: {output_file_name} already exists")
            continue

        else:
            print(f"Extracting File: {output_file_name}")
            dvlog_dataset = pd.read_csv(dataset_path, delimiter=";")

            # Setting the index to 'file_path' column: where audio recording is
            dvlog_dataset.set_index('file_path', inplace=True)
            dvlog_dataset.index = dvlog_dataset.index.astype(str)

            # Step 3: Extracting VAD scores using PDEM model for each wav
            logits.process_index(dvlog_dataset.index, root=root_dir,
                                # The output files will be saved here. File string_functionals will be a random number ending with '.pkl'
                                cache_root=audeer.path(cache_root, 'pdem_vad/'))

            path_to_vad_folder = os.path.join(cache_root, 'pdem_vad/')
            files = os.listdir(path_to_vad_folder)
            for file in files:
                if file.endswith('.pkl') and not "vad_features" in file:  # check if the filename is a random number
                    rename_file(path_to_vad_folder, file, output_file_name)


if __name__ == "__main__":

    root_dir = Path(r"../PDEM/model")
    index_file_path = Path(r"../PDEM/index_files/")

    extract_vad_pdem(root_dir, index_file_path)
