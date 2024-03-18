# This script has been drafted to use the whisper_timestamped library to extract the transcripts from collected audio files.
# github link -> https://github.com/linto-ai/whisper-timestamped
# 

import click
import json
import torch
import os
import whisper_timestamped as whisper

from pathlib import Path


@click.command()
@click.argument("audio_dir", default=Path(r"/data/dvlog_audio_mp3"), type=click.Path(exists=True))
@click.argument("text_dir", default=Path(r"/data/dvlog_text"), type=click.Path(exists=True))
def main(audio_dir, text_dir):

    # setup whisper
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = whisper.load_model("base.en", device=device)
    
    # go over every video and extract the audio
    for audio_file in os.listdir(audio_dir):
        
        # retrieve the filename and setup the read and write paths
        filename, _ = os.path.splitext(audio_file)
        audio_path = os.path.join(audio_dir, audio_file)
        text_path = os.path.join(text_dir, f"{filename}.json")

        if os.path.isfile(text_path):
            print(f"File: {filename} already exists")
        
        else:
            # load in the audio file
            audio = whisper.load_audio(audio_path)

            # do the transcribing
            result = whisper.transcribe(model, audio, language="en",
                                        vad="auditok", beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

            # store it away
            with open(text_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()