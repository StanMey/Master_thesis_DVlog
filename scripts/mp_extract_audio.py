import click
import os

from moviepy.editor import VideoFileClip
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple


def convert_video_to_audio_moviepy(data: Tuple[str, Path, Path, str]):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood
    """
    # extract the input tuple
    video_file, video_dir, audio_dir, output_ext = data

    # retrieve the filename and setup the read and write paths
    filename, _ = os.path.splitext(video_file)
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
    # setup the paths
    video_dir = Path(r"E:/Master/dvlog_videos")
    audio_dir = Path(r"E:/Master/dvlog_audio")
    output_ext = "mp3"
    num_processes = 4

    # check the paths
    assert video_dir.is_dir(), f"video_dir is not valid: {video_dir}"
    assert audio_dir.is_dir(), f"audio_dir is not valid: {audio_dir}"

    # go over every video and setup the data list
    extraction_list = []
    for video_file in os.listdir(video_dir):
        extraction_list.append((video_file, video_dir, audio_dir, output_ext))
    
    # apply the function over 
    with Pool(num_processes) as p:
        extractions = list(p.map(convert_video_to_audio_moviepy, extraction_list))