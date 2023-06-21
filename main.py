from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openpiv import filters, pyprocess, validation

# Path
input_video_path = "./input/wipe/wipe01.mp4"
images_dir = "images"

# PIV
winsize = 64  # pixels, interrogation window size in frame A
searchsize = winsize  # pixels, search in frame B
overlap = int(winsize / 2)  # pixels, 50% overlap
dt = 0.03  # sec, time interval between pulses

# Quiver
arrow_scale = 6  # 小さいほど長い矢印


def binarize(color_image):
    threshold = 128
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    th, bin_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_OTSU)
    # th, bin_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return bin_image


def main():
    video = cv2.VideoCapture(input_video_path)
    success, frame_a = video.read()
    count = 0
    U = deque([], 10)
    V = deque([], 10)
    while success:
        print(f"Frame {count}")
        success, frame_b = video.read()
        if success:
            binary_frame_a = binarize(frame_a)
            binary_frame_b = binarize(frame_b)

            # PIV
            u0, v0, sig2noise = pyprocess.extended_search_area_piv(
                binary_frame_a.astype(np.int32),
                binary_frame_b.astype(np.int32),
                window_size=winsize,
                overlap=overlap,
                dt=dt,
                search_area_size=searchsize,
                sig2noise_method="peak2peak",
            )
            x, y = pyprocess.get_coordinates(
                image_size=binary_frame_a.shape,
                search_area_size=searchsize,
                overlap=overlap,
            )
            invalid_mask = validation.sig2noise_val(sig2noise, threshold=1.05)
            u2, v2 = filters.replace_outliers(
                u0,
                v0,
                invalid_mask,
                method="localmean",
                max_iter=3,
                kernel_size=3,
            )

            # Mean of last several frames
            U.append(u2)
            V.append(v2)
            Umean = np.mean(np.stack(U), axis=0)
            Vmean = np.mean(np.stack(V), axis=0)

            # Plot arrows
            fig, ax = plt.subplots(figsize=(16, 9))
            # ax.imshow(binary_frame_a, alpha=0.7)  # Binary
            ax.imshow(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB), alpha=0.7)  # Original
            ax.quiver(
                x,
                y,
                Umean,
                Vmean,
                color="blue",
                scale=arrow_scale,
                scale_units="xy",
                angles="xy",
            )
            # plt.show()

            # Save image
            plt.savefig(f"{images_dir}/frame-{count}.png")
            plt.clf()
            plt.close()

            # Next frame
            frame_a = frame_b.copy()
            count += 1
    video.release()


main()
