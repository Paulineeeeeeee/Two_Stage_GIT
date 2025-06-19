import os
import json
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import torchvision.transforms as transforms
import torch

# Replace the mean and std with the ones used in trsfm
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

def main(image_path, output_dir="output", split="VALID", image_size=(224, 224)):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    # Load and save original image
    image = Image.open(image_path).convert('RGB')
    image.save(os.path.join(output_dir, "step_0_original.jpg"))

    # Define initial transform
    if split == 'TRAIN':
        initial_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, interpolation=Image.BICUBIC),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.75, 1)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
        ])
    else:
        initial_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
        ])

    # Apply initial transform and save
    transformed = initial_transform(image)
    transformed.save(os.path.join(output_dir, "step_1_transformed.jpg"))

    # Perform OCR
    ocr_results = ocr.ocr(np.array(transformed), cls=True)
    # Process OCR results
    ocr_info = []
    if ocr_results:
        detections = ocr_results[0] if isinstance(ocr_results[0], list) else ocr_results
        for line in detections:
            if line and len(line) > 1 and line[1]:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                box = line[0]
                ocr_info.append({"text": text, "box": box})

    # Save OCR info
    with open(os.path.join(output_dir, "step_2_ocr.json"), "w", encoding="utf-8") as f:
        json.dump(ocr_info, f, ensure_ascii=False, indent=2)

    # Define final transform (normalize)
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Apply final transform
    tensor = final_transform(transformed)
    final_transform = transforms.ToPILImage()(tensor)
    final_transform.save(os.path.join(output_dir, "step_3_transformed.jpg"))

if __name__ == "__main__":
    # User should modify the image_path as needed
    IMAGE_PATH = "/data/pauline/llamatouch_dataset/general/trace_18/2.png"
    main(IMAGE_PATH)
