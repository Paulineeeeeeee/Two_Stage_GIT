import cv2
import os
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def extract_frames_from_gif(gif_path, output_folder):
    # Open the GIF file
    gif = Image.open(gif_path)
    frame_number = 0
    previous_frame = None

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_frames = gif.n_frames
    gif_id = os.path.splitext(os.path.basename(gif_path))[0]

    for frame_index in range(total_frames):
        # Select the current frame
        gif.seek(frame_index)
        frame = np.array(gif.convert("RGB"))

        # Convert the frame to grayscale for comparison
        current_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if previous_frame is not None:
            # Calculate Structural Similarity Index (SSIM) to compare frames
            similarity = ssim(previous_frame, current_frame)
            if similarity > 0.98:
                # Skip saving the frame if it is more than 98% similar to the previous one
                continue

        # Save the current frame as a separate image
        frame_path = os.path.join(output_folder, f"{gif_id}-{frame_number}-{total_frames}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_number += 1
        previous_frame = current_frame

    return total_frames

def get_gif_duration(gif_path):
    # Open the GIF file
    gif = Image.open(gif_path)
    # Get duration in milliseconds per frame and total frames
    frame_duration_ms = gif.info.get("duration", 0)
    total_frames = gif.n_frames
    # Calculate total duration in seconds
    duration = (frame_duration_ms * total_frames) / 1000.0 if frame_duration_ms > 0 else 0
    return duration

def record_metadata(output_jsonl_path, metadata):
    with open(output_jsonl_path, 'a') as f:
        f.write(json.dumps(metadata) + "\n")


if __name__ == "__main__":
    gif_path = "/data/pauline/dataset/segments/2243.gif"  # Replace with your GIF path
    output_folder = "/home/pauline/GIT/2243"  # Replace with your desired output folder
    output_jsonl_path = os.path.join(output_folder, "output.jsonl")

    # Extract frames and get total frames
    total_frames = extract_frames_from_gif(gif_path, output_folder)
    print(f"Frames extracted to folder: {output_folder}")

    # Get GIF duration
    gif_duration = get_gif_duration(gif_path)
    print(f"GIF duration: {gif_duration} seconds")

    # Record metadata
    metadata = {
        "id": "2243",
        "app": "AmazeFileManager",
        "description": "Example description for the GIF.",
        "episode_length": total_frames
    }
    record_metadata(output_jsonl_path, metadata)
