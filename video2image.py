import os

import cv2
from PIL import Image


def extract_frames(video_path, output_folder, save_as_png=False):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is returned (end of video)

        # Save each frame as an image file
        if save_as_png:
            output_filename = os.path.join(output_folder, f"{frame_count:05d}.png")
        else:
            output_filename = os.path.join(output_folder, f"{frame_count:05d}.jpg")

        Image.fromarray(frame[..., ::-1]).save(output_filename, optimize=True)
        # cv2.imwrite(output_filename, frame)
        frame_count += 1

    cap.release()
    print(f'Done! {frame_count} frames extracted and saved in "{output_folder}".')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument(
        "--output_folder", type=str, help="Path to the output folder to save frames"
    )
    parser.add_argument(
        "--png", action="store_true", help="Save frames as PNG images instead of JPG"
    )
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder, args.png)
