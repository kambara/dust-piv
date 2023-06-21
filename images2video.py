import re
from pathlib import Path

import cv2
import ffmpeg

output_video_path = "output/wipe01.mp4"
fps = 30
temp_video_path = "_temp.avi"
images_dir = Path("images")


def main():
    print("Create Motion JPEG")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    first_image = cv2.imread(str(images_dir.joinpath("frame-0.png")))
    size = (first_image.shape[1], first_image.shape[0])
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, round(fps), size)
    for filepath in sorted(images_dir.glob("*.png"), key=natural_keys):
        print(filepath)
        image = cv2.imread(str(filepath))
        video_writer.write(image)
    video_writer.release()
    print("Convert Motion JPEG to mp4")
    ffmpeg.input(temp_video_path).output(output_video_path, vcodec="libx264").run(
        overwrite_output=True
    )


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(path):
    text = str(path)
    return [atoi(c) for c in re.split(r"(\d+)", text)]


main()
