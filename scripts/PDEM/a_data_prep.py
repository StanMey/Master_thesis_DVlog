import os
import csv
import math
import pandas as pd

from pathlib import Path
from pydub import AudioSegment


# Make sure that the folder path in the function below only contains transcript files for each participant

def get_split_audio_by_transcript(transcripts_path: Path, audio_path: Path, output_path: Path) -> None:
    """_summary_

    Args:
        transcripts_path (Path): _description_
        audio_path (Path): _description_
        output_path (Path): _description_
    """
    # get all files from the subset
    files = os.listdir(transcripts_path)

    for file in files:

        # 
        filename, _ = os.path.splitext(file)
        subject = filename.split("_")[0]
        print(f'Processing for {subject}')

        # find audio recording of subject
        audio_file_path = os.path.join(audio_path, f"{filename}.wav")  # path of audio file
        transcript_file_path = os.path.join(transcripts_path, file)  # path of recording file
        subject_output_folder_path = os.path.join(output_path, subject)  # path to save split audio

        # make folder for each subject separately
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        if not os.path.exists(subject_output_folder_path):
            os.makedirs(subject_output_folder_path)
        
        # Read audio file
        audio = AudioSegment.from_wav(audio_file_path)

        with open(transcript_file_path, 'r', encoding='utf-8') as csv_file:
            nr = 0  # for numbering the turns/splits of a participant

            reader = csv.reader(csv_file, delimiter=';')
            next(reader)  # ignore title row
            for row in reader:

                # extract the information
                start_t = float(row[0])
                end_t = float(row[1])

                formatted_nr = "{:04d}".format(nr)  # generate split audio string_functionals
                output_filename = f"{subject}_AUDIO_{str(formatted_nr)}_{start_t}_{end_t}.wav"
                output_file_path = os.path.join(subject_output_folder_path, output_filename)

                if not os.path.isfile(output_file_path):
                    # save audio segment
                    segment = audio[start_t * 1000:end_t * 1000]  # get split audio from participant
                    segment.export(output_file_path, format="wav")

                nr += 1

            print(f'Done for {subject}')

    return


def make_index_file(dataset_type: str, transcripts_path: Path, output_audio_path: Path, output_index_path: Path, chunks: int=1) -> None:
    """_summary_

    Args:
        dataset_type (str): _description_
        transcripts_path (Path): _description_
        output_audio_path (Path): _description_
        output_index_path (Path): _description_
        chunks (int, optional): _description_. Defaults to 1.
    """

    output_index_file_path = os.path.join(output_index_path, f"pdem_index_file_{dataset_type}.csv")

    # get all files from the subset
    files = os.listdir(transcripts_path)
    rows = []

    for file in files:

        # 
        filename, _ = os.path.splitext(file)
        subject = filename.split("_")[0]

        transcript_file_path = os.path.join(transcripts_path, file)  # path of recording file
        subject_output_folder_path = os.path.join(output_audio_path, subject)  # path to save split audio


        with open(transcript_file_path, 'r', encoding='utf-8') as csv_file:
            nr = 0  # for numbering the turns/splits of a participant

            reader = csv.reader(csv_file, delimiter=';')
            next(reader)  # ignore title row
            for row in reader:

                # extract the information
                start_t = float(row[0])
                end_t = float(row[1])
                text = row[2]
                audio_duration = int(end_t * 1000 - start_t * 1000)  # get the duration

                formatted_nr = "{:04d}".format(nr)  # generate split audio string_functionals
                output_audio_filename = f"{subject}_AUDIO_{str(formatted_nr)}_{start_t}_{end_t}.wav"
                output_audio_file_path = os.path.join(subject_output_folder_path, output_audio_filename)

                rows.append((subject, output_audio_filename, output_audio_file_path, text, audio_duration))
                nr += 1

    columns = ['Participant_ID', 'saved_file_name', 'file_path', 'text', 'duration']
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_index_file_path, sep=";", index=False)

    if chunks > 1:
        # 
        n_chunk = math.ceil(len(df) / chunks)
        with pd.read_csv(output_index_file_path, sep=";", chunksize=n_chunk) as reader:
            for i, chunk in enumerate(reader):
                output_chunk_file_path = os.path.join(output_index_path, f"pdem_index_file_{dataset_type}_{i}.csv")
                chunk.to_csv(output_chunk_file_path, sep=";", index=False)


if __name__ == "__main__":

    # setup the appropriate paths
    transcripts_path = Path(r"../PDEM/transcripts")
    audio_path = Path(r"D:/master/data/dvlog_audio_wav")
    audio_output_path = Path(r"../PDEM/outputs")
    index_file_output_path = Path(r"../PDEM")

    # 
    for dataset_dir in os.listdir(transcripts_path):
        dataset_transcripts_path = os.path.join(transcripts_path, dataset_dir)

        # extract all the audio samples
        get_split_audio_by_transcript(dataset_transcripts_path, audio_path, audio_output_path)

        # build the index file
        make_index_file(dataset_dir, dataset_transcripts_path, audio_output_path, index_file_output_path, chunks=20)
