import pandas as pd

from typing import Tuple
from multiprocessing import Pool
from pytube import YouTube


def download_video(video_info: Tuple[int, str]):
    """"""
    video_id, video_url = video_info
    link = f'https://www.youtube.com/watch?v={video_url}'
    output_path = f"./data/dvlog_videos/"

    try:
        youtubeObject = YouTube(link)
        streamObject = youtubeObject.streams.get_by_resolution("720p")

        if streamObject is None:
            streamObject = youtubeObject.streams.get_highest_resolution()

        # download the video
        streamObject.download(output_path=output_path, filename=f"{video_id}_{video_url}.mp4")
        print(f"{video_id} - {video_url}")

    except Exception as e:
        print(f"{video_url}: An error has occurred")

    return video_url


if __name__ == "__main__":
    df_dvlog = pd.read_csv("./data/dvlog.csv")
    video_ids = list(zip(df_dvlog["video_id"], df_dvlog["key"]))

    with Pool(4) as p:
        print(p.map(download_video, video_ids))
