import click
import os

from moviepy.editor import VideoFileClip
from pathlib import Path


@click.command()
@click.argument("video_dir", default=Path(r"E:/Master/dvlog_videos"), type=click.Path(exists=True))
@click.argument("audio_dir", default=Path(r"E:/Master/dvlog_audio"), type=click.Path(exists=True))
def main(video_dir, audio_dir):
    audio_dir = Path(r"C:/Users/stan_/Documents/Master/thesis/dvlog-dataset/audio")
    
    # go over every video and extract the audio
    for video_file in os.listdir(video_dir):
        convert_video_to_audio_moviepy(video_file, video_dir, audio_dir)


def convert_video_to_audio_moviepy(video_file: str, video_dir: Path, audio_dir: Path, output_ext="mp3"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood
    """
    # retrieve the filename and setup the read and write paths
    filename, ext = os.path.splitext(video_file)
    clip_path = os.path.join(video_dir, video_file)
    audio_path = os.path.join(audio_dir, f"{filename}.{output_ext}")

    if os.path.isfile(audio_path):
        print(f"File: {filename} already exists")

    else:
        # extract the clip from the video file
        clip = VideoFileClip(clip_path)

        # extract the audio and write it into the audio_dir
        clip.audio.write_audiofile(audio_path)


if __name__ == "__main__":
    main()