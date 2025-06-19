import cv2
import os
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim

def extract_frames_from_mp4(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_number = 0
    previous_frame = None

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        # Read the next frame
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale for comparison
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_frame is not None:
            # Calculate Structural Similarity Index (SSIM) to compare frames
            similarity = ssim(previous_frame, current_frame)
            if similarity > 0.98:
                # Skip saving the frame if it is more than 98% similar to the previous one
                continue

        # Save the current frame as a separate image
        frame_path = os.path.join(output_folder, f"{video_id}-{frame_number}-{total_frames}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_number += 1
        previous_frame = current_frame

    video.release()
    return total_frames

def get_video_duration(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Get the frame rate and number of frames
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    # Calculate duration in seconds
    duration = frame_count / fps if fps > 0 else 0
    return duration

def record_metadata(output_jsonl_path, metadata):
    with open(output_jsonl_path, 'a') as f:
        f.write(json.dumps(metadata) + "\n")

if __name__ == "__main__":
    video_path = "/home/pauline/GIT/crawl/video/AmazeFileManager/1467.mp4"  # Replace with your MP4 path
    output_folder = "/home/pauline/GIT/1467"  # Replace with your desired output folder
    output_jsonl_path = os.path.join(output_folder, "output.jsonl")

    # Extract frames and get total frames
    total_frames = extract_frames_from_mp4(video_path, output_folder)
    print(f"Frames extracted to folder: {output_folder}")

    # Get video duration
    video_duration = get_video_duration(video_path)
    print(f"Video duration: {video_duration} seconds")

    # Record metadata
    metadata = {
        "id": "1467",
        "app": "AmazeFileManager",
        "description": "Example description for the video.",
        "episode_length": total_frames
    }
    record_metadata(output_jsonl_path, metadata)
