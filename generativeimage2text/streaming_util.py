import jsonlines
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import BertTokenizer
from torchvision.transforms import functional as F
import numpy as np
import pandas as pd
import torchvision.transforms as transforms


def write_log(args,im):
    with open(f"./log/{args['exp_name']}_bs{args['bs']}_lr{args['lr']}_im{im}_log.txt","a") as f:
        f.write(f"bs = {args['bs']}, num_epoch = {args['epoch']},\n")
        f.write(f"lr = {args['lr']}, wd = {args['wd']},\n")


def get_llama_images(id, data_path, length, num_images,trsfm):
    # num_images default is 6
    # f"{des}{id}-{i}-{len(episode)}.jpg"
    path = data_path
    images = []
    if length >=num_images:
        for i in range(num_images):
            # get last six images
            image_path = f"{path}/{length-num_images+i}.png"
            images.append(trsfm(Image.open(image_path).convert('RGB')))
          
    else:
        for i in range(length):
            # get last six images
            image_path = f"{path}/{i}.png"            
            images.append(trsfm(Image.open(image_path).convert('RGB')))
        shape = images[0].shape
        for i in range(num_images-length):
            images.append(torch.zeros(shape))
        # for i in range(num_images-length):
        #     images.append()
    # Image.open(image_path).convert('RGB')
    images_stacked = torch.stack(images, dim=0)
    return images_stacked

def get_auto_images(id, data_path, length, num_images,trsfm):
    # num_images default is 6
    # f"{des}{id}-{i}-{len(episode)}.jpg"
    path = data_path
    images = []
    if length >=num_images:
        for i in range(num_images):
            # get last six images
            image_path = f"{path}/frame_{length-num_images+i}.jpg"
            images.append(trsfm(Image.open(image_path).convert('RGB')))
          
    else:
        for i in range(length):
            # get last six images
            image_path = f"{path}/frame_{i}.jpg"            
            images.append(trsfm(Image.open(image_path).convert('RGB')))
        shape = images[0].shape
        for i in range(num_images-length):
            images.append(torch.zeros(shape))
        # for i in range(num_images-length):
        #     images.append()
    # Image.open(image_path).convert('RGB')
    images_stacked = torch.stack(images, dim=0)
    return images_stacked

# def trsfm(image_size = (224,224),split = 'VALID'):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    if split == 'TRAIN':
        ret = v2.Compose(
                [
                    # v2.RandomResizedCrop(size=image_size, scale=(0.5,1.0), interpolation=InterpolationMode.BICUBIC),
                    v2.RandomResizedCrop(size=image_size, interpolation=InterpolationMode.BICUBIC),
                    v2.RandomAffine(degrees=0,translate=(0.2, 0.2), scale=(0.75, 1)),                    
                    v2.ColorJitter(brightness=0.5, contrast=0.5),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean, std)
                ]
            )
    else:
        ret = v2.Compose(
                [
                    v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean, std)
                ]
            )
    return ret

# def trsfm_ocr(image_path, image_size=(224, 224), split='VALID', ocr=None):
    image = Image.open(image_path).convert('RGB')
    # Initialize PaddleOCR
    
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    
    # Define the initial transformations (random crop, affine, color jitter)
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
    
    # Apply the initial transformations
    transformed_image = initial_transform(image)
    
    # Convert the transformed image to a numpy array for OCR
    transformed_image_np = np.array(transformed_image)
    
    # Perform OCR on the transformed image using PaddleOCR
    ocr_results = ocr.ocr(transformed_image_np, cls=True)
    if ocr_results is None or not isinstance(ocr_results, list) or len(ocr_results) == 0:
        ocr_info = []  # 沒有 OCR 結果，返回空列表
    else:
        try:
            # 檢查是否為多圖片的情況（最外層為一個列表，其第一個元素包含了實際的 OCR 結果）
            if isinstance(ocr_results[0], list):
                detections = ocr_results[0]
            else:
                detections = ocr_results

            ocr_info = []
            for line in detections:
                # 檢查該筆結果是否有效，且至少包含邊界框和文字信息
                if line and len(line) > 1 and line[1]:
                    # 提取文字：如果 line[1] 為列表或元組，則第一個元素為文字
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                    # 提取邊界框：假設 line[0] 就是邊界框資訊
                    box = line[0]
                    ocr_info.append({
                        "text": text,
                        "box": box
                    })
        except Exception as e:
            # 若解析過程出現異常，返回空列表（你也可以在此處記錄錯誤信息）
            ocr_info = []

    
    # Convert the transformed image to a tensor and normalize it
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    normalized_image = final_transform(transformed_image)
    
    return normalized_image, ocr_info

# class AITW_Dataset(Dataset):
    def __init__(self, data_path,data_name,split,tokenizer,transform, num_images=8 , model = 'CLIPViT_B_16'):
        self.data = []
        self.num_images = num_images
        self.data_path = data_path
        self.data_name = data_name
        self.transform = transform
        self.tokenizer = tokenizer
        self.model = model
        
        if self.data_name == 'AITW': 
            category = ['google_apps','general','install','web_shopping']
        elif self.data_name == 'bugs' :
            category = ['bugs']
        else:
            # exception
            print('data_name not found')
            return
        
        # Read data from JSONL file
        for cat in category:
            jsonl_file = f'/home/pauline/GIT/util/no_miss_{cat}_train.jsonl'
            with jsonlines.open(jsonl_file) as reader:
                size = sum(1 for i in reader)
            
            with jsonlines.open(jsonl_file) as reader:
                if split == 'TRAIN':
                    limit = int(size*0.8)
                    for i,lines in enumerate(reader):
                        if i < limit:
                            self.data.append(lines)
                        else:
                            break
                if split == 'VALID':
                    # limit = int(size*0.99999)
                    limit = int(size * 0.8)
                    for i,lines in enumerate(reader):
                        if i >= limit:
                            self.data.append(lines)

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, samples):
        max_length = 0
        for x in samples:
            max_length = max(max_length,x['caption_tokens'].shape[0])
            x['image'] = self.model.process_image_features(x['image'])
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            b1 = torch.zeros(max_length, dtype=t.dtype, device=t.device)
            b2 = torch.zeros(max_length, dtype=n.dtype, device=n.device)
            b1[:t.shape[0]] = t
            b2[:n.shape[0]] = n
            x['caption_tokens'] = b1
            x['need_predict'] = b2
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        image_length = torch.Tensor([x['image_length'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
            'image_length': image_length
        }
        return data

    def __getitem__(self, idx):
        # {"id": "18375519518960921438", "goal_info": "Is it going to rain this weekend?", "episode_length": 8}
        # 10002872452831025023-0-14.jpg
        item = self.data[idx]
        # category = ['google_apps','general','install','web_shopping']

        path = self.data_path + item['category']+'/'
        
        images = get_n_images(item['id'],path,item['episode_length'],self.num_images,self.transform)
        image_length = item['episode_length']
        
        max_text_len = 40 #from train.py
        target = item['goal_info']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']

        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        # Convert other fields to tensors as needed
        data = {
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'image': images,
            'image_length': image_length
            # 'rect' field can be fed in 'caption', which tells the bounding box
            # region of the image that is described by the caption. In this case,
            # we can optionally crop the region.
            # 'caption': {},
            # this iteration can be used for crop-size selection so that all GPUs
            # can process the image with the same input size
            # 'iteration': 0
        }
        
        return data


def tokenize_and_concatenate(texts, bbox ,tokenizer, max_total_length=40):

    ocr_token = []
    for text_per_image in texts:
        
        # 1. Tokenize 文本與 box 座標
        text_box_encoding = [tokenizer.cls_token_id]

        # if text_per_image is None or not isinstance(text_per_image, list) or len(text_per_image) == 0:
        # make sure the last component is [SEP] to represent the end of the text
        if text_per_image is None or len(text_per_image) == 0:
            text_box_encoding += [tokenizer.sep_token_id]
        # else text_per_image is a list of dictionaries
        for item in text_per_image:
            # 1.a Tokenize text
            text_encoding = tokenizer(
                item['text'],
                add_special_tokens=False,
                padding=False,
                truncation=False
            )
            text_ids = text_encoding["input_ids"]  # list of int
            
            # 1.b Tokenize box
            # covert box to string
            box_str = " ".join([str(int(coord)) for point in item['box'] for coord in point])
            # tokenize box
            box_encoding = tokenizer(
                box_str,
                add_special_tokens=False,
                padding=False,
                truncation=False
            )
            box_ids = box_encoding["input_ids"]
            
            # 1.c 連接 text 與 box 的 token 序列，加上 SEP token
            if bbox :
                # combine text and box token
                combined = text_ids + box_ids + [tokenizer.sep_token_id]
            else:
                # only use text token
                combined = text_ids + [tokenizer.sep_token_id]

            text_box_encoding += combined
        # 2. Padding
        if len(text_box_encoding) > max_total_length:
            # only the first max_total_length tokens are used
            text_box_encoding = text_box_encoding[:max_total_length - 1]
            text_box_encoding += [tokenizer.sep_token_id]
        elif len(text_box_encoding) < max_total_length:
            text_box_encoding += [tokenizer.pad_token_id] * (max_total_length - len(text_box_encoding))
        # 3. append text box encoding to ocr token
        ocr_token.append(text_box_encoding)

    ocr_token = torch.tensor(ocr_token) # (num_images, max_total_length)
    return ocr_token

def trsfm_ocr(image_path, image_size=(224, 224), split='VALID', ocr=None):
    image = Image.open(image_path).convert('RGB')
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    # Define the initial transformations (random crop, affine, color jitter)
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

    # Apply the initial transformations
    transformed_image = initial_transform(image)

    # Convert the transformed image to a numpy array for OCR
    transformed_image_np = np.array(transformed_image)

    # Perform OCR on the transformed image using PaddleOCR

    # if ocr is None, which means that we didn't use OCR, we will return None
    if ocr is not None:
        ocr_results = ocr.ocr(transformed_image_np, cls=True)
        if ocr_results is None or not isinstance(ocr_results, list) or len(ocr_results) == 0:
            ocr_info = []  # 沒有 OCR 結果，返回空列表
        else:
            try:
                # 檢查是否為多圖片的情況（最外層為一個列表，其第一個元素包含了實際的 OCR 結果）
                if isinstance(ocr_results[0], list):
                    detections = ocr_results[0]
                else:
                    detections = ocr_results

                ocr_info = []
                for line in detections:
                    # 檢查該筆結果是否有效，且至少包含邊界框和文字信息
                    if line and len(line) > 1 and line[1]:
                        # 提取文字：如果 line[1] 為列表或元組，則第一個元素為文字
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                        # 提取邊界框：假設 line[0] 就是邊界框資訊
                        box = line[0]
                        ocr_info.append({
                            "text": text,
                            "box": box
                        })
            except Exception as e:
                # 若解析過程出現異常，返回空列表（你也可以在此處記錄錯誤信息）
                ocr_info = []
    else:
        ocr_info = None


    # Convert the transformed image to a tensor and normalize it
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    normalized_image = final_transform(transformed_image)

    return normalized_image, ocr_info

def _load_images_and_optional_ocr(data_name, id, data_path, episode_length, num_images, image_transform, use_ocr, ocr_instance, streaming=False):
    """
    Loads images from an episode based on strategy (first N, last N, or all)
    and optionally performs OCR.

    Args:
        data_name (str): Name of the dataset ('AITW', 'bugs', or 'llamatouch').
        id (str): Episode ID.
        data_path (str): Base path to image data.
        episode_length (int): Total number of frames in the episode.
        num_images (int): Number of images to load when not streaming.
        image_transform (callable): Image transformation pipeline (applied after loading).
        use_ocr (bool): If True, perform OCR on each loaded image.
        ocr_instance: The PaddleOCR instance (required if use_ocr is True).
        load_last_n (bool): If True, load the last num_images frames (when not streaming).
                            If False, load the first num_images frames (when not streaming).
        streaming (bool): If True, load all frames from the episode, ignoring num_images and load_last_n.

    Returns:
        tuple: (torch.Tensor of stacked transformed images,
                list of OCR results per image or None if use_ocr is False)
               OCR results per image is a list of dicts [{"text": ..., "box": ...}, ...]
    """
    images_tensor_list = []
    texts_ocr_list = [] if use_ocr else None # Initialize list only if OCR is needed

    # Determine the range of indices to load based on strategy
    if streaming:
        # Strategy 2: Load all frames
        start_idx = 0
        end_idx = episode_length
        # num_images and load_last_n are ignored in this mode
        frames_to_process_count = max(episode_length, num_images) # We expect this many outputs
    else:
        # Strategy 1: Load a fixed number (num_images), either first or last
        # Load last num_images frames
        start_idx = max(0, episode_length - num_images)
        end_idx = episode_length

        frames_to_process_count = num_images # We expect this many outputs (after padding)


    # Load images and optionally perform OCR
    for i in range(start_idx, end_idx):
        # Assuming image path format is {data_path}/{id}-{frame_idx}-{episode_length}.jpg
        # Note: Original get_llama_images/get_auto_images used different paths.
        # This assumes the AITW/bugs format. Adjust if needed for other datasets.
        if data_name == 'llamatouch':
            image_path = f"{data_path}/{i}.png"
        else:
            image_path = f"{data_path}{id}-{i}-{episode_length}.jpg"
        try:
            # tranform image and get the ocr information
            img, ocr_info = image_transform(image_path, ocr = ocr_instance)
            if use_ocr:            
                texts_ocr_list.append(ocr_info)
            images_tensor_list.append(img)

        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Appending zero tensor and empty OCR.")

    # Pad with placeholder tensors/lists if fewer than the expected number were loaded/processed
    # This padding is only needed when we expect a fixed number (num_images) and not streaming
    while len(images_tensor_list) < frames_to_process_count: # frames_to_process_count is num_images here
        # Need a placeholder shape for the zero tensor
        shape = images_tensor_list[0].shape
        images_tensor_list.append(torch.zeros(shape))
        
        if use_ocr:
            # Need to pad texts_ocr_list as well
            texts_ocr_list.append([]) # Append empty list for padding

    images_stacked = torch.stack(images_tensor_list, dim=0)
    

    return images_stacked, texts_ocr_list

class AITW_Dataset(Dataset):
    def __init__(self, data_name, split, tokenizer, model,num_images=8,
                 use_ocr=False, ocr_instance=None, use_bbox=True,
                 streaming=False, # New parameter for loading all frames
                 use_paddle_processing=False, per_frame_image_processing=True):
        """
        Unified AITW Dataset for different modalities and processing strategies.

        Args:
            data_path (str): Base path to dataset images.
            data_name (str): Name of the dataset ('AITW' or 'bugs').
            split (str): Dataset split ('TRAIN' or 'VALID').
            tokenizer: Tokenizer instance (e.g., BertTokenizer).
            model: The model instance (expected to have process_batch,
                   process_image_features, preprocess_features methods).
            image_transform (callable): Standard image transformation pipeline.
            num_images (int): Number of images to load per sample when not streaming.
            use_ocr (bool): If True, load and process OCR text.
            ocr_instance: PaddleOCR instance (required if use_ocr is True).
            use_bbox (bool): If True, include bounding box tokens with OCR (only if use_ocr).
            load_last_n_images (bool): If True, load the last num_images frames (when not streaming).
                                       If False, load the first num_images frames (when not streaming).
            streaming (bool): If True, load all frames from the episode, ignoring num_images and load_last_n.
            use_paddle_processing (bool): If True, use model.preprocess_features (multimodal).
                                          If False, use model.process_image_features (image-only).
            per_frame_image_processing (bool): Strategy for image-only processing
                                               when downsampling is needed (only if use_paddle_processing is False).
        """
        self.data = []
        self.num_images = num_images
        # self.data_path = data_path
        self.data_name = data_name
        self.split = split
        self.tokenizer = tokenizer
        self.model = model

        self.use_ocr = use_ocr
        self.ocr_instance = ocr_instance
        self.use_bbox = use_bbox

        self.streaming = streaming # Store the new flag
        self.use_paddle_processing = use_paddle_processing
        self.per_frame_image_processing = per_frame_image_processing # true : process each image separately, false: process all images at once

        if self.use_ocr and self.ocr_instance is None:
             raise ValueError("ocr_instance must be provided if use_ocr is True.")
        # Add a check for conflicting flags if necessary, e.g., streaming and load_last_n_images
        # if self.streaming and self.load_last_n_images:
        #     print("Warning: Both streaming and load_last_n_images are True. streaming will take precedence.")


        if self.data_name in ['AITW', 'bugs']:
            self.data_path = '/data/pauline/no-miss-AITW/'
            category = ['google_apps', 'general', 'install', 'web_shopping'] if self.data_name == 'AITW' else ['bugs']
            # Read data from JSONL file (keep existing logic)
            for cat in category:
                jsonl_file = f'/home/pauline/GIT/util/no_miss_{cat}_train.jsonl'
                try:
                    with jsonlines.open(jsonl_file) as reader:
                        size = sum(1 for _ in reader)

                    with jsonlines.open(jsonl_file) as reader:
                        if split == 'TRAIN':
                            limit = int(size * 0.8)
                            for i, lines in enumerate(reader):
                                if i < limit:
                                    self.data.append(lines)
                                else:
                                    break
                        elif split == 'VALID':
                            limit = int(size * 0.8)
                            for i, lines in enumerate(reader):
                                if i >= limit:
                                    self.data.append(lines)
                        else:
                            raise ValueError(f"split '{split}' not supported ('TRAIN' or 'VALID').")
                except FileNotFoundError:
                    print(f"Warning: JSONL file not found at {jsonl_file}. Skipping category {cat}.")
                except Exception as e:
                    print(f"Warning: Error reading JSONL file {jsonl_file}: {e}. Skipping category {cat}.")


            print(f"Loaded {len(self.data)} samples for {data_name} {split} split.")
        elif self.data_name == 'llamatouch':
            self.data_path = '/data/pauline/llamatouch_dataset/'
            df = pd.read_csv('/data/pauline/llamatouch_dataset/llamatouch_task_metadata.tsv',
                sep='\t')
            df = df[df['category'] == 'generated']
            self.data = df.iloc[:] # just for testing, so we load all data
            

        else:
            raise ValueError(f"data_name '{data_name}' not supported.")

        print(f"Dataset initialized with {len(self.data)} samples from {data_name} ({split} split).")

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """
        Collates samples into a batch, processing features using the model.
        """

        # Find max length for padding text tokens
        max_caption_length = 0
        for x in samples:
            max_caption_length = max(max_caption_length, x['caption_tokens'].shape[0])

        # Process features using the model (image + optional text)
        # This is where the model's preprocess_features or process_image_features is called
        # process_batch is called once per batch before processing samples
        self.model.process_batch() # Reset valid masks etc. in the model

        processed_features_list = []
        for x in samples:
            # preprocessing the features before collating to reduce the peak GPU requeirments
            processed_features = self.model.preprocess_features(x['image'], x.get('ocr_tokens', None), per_frame=self.per_frame_image_processing)
            processed_features_list.append(processed_features)

        # Pad and stack caption tokens and need_predict masks
        caption_batch = []
        predict_batch = []
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            # Pad caption_tokens
            b1 = torch.zeros(max_caption_length, dtype=t.dtype, device=t.device)
            b1[:t.shape[0]] = t
            caption_batch.append(b1)
            # Pad need_predict
            b2 = torch.zeros(max_caption_length, dtype=n.dtype, device=n.device)
            b2[:n.shape[0]] = n
            predict_batch.append(b2)

        # Stack the processed features, padded captions, and other data
        image_batch = torch.stack(processed_features_list, dim=0) # Stack the output of model processing
        caption_batch = torch.stack(caption_batch, dim=0)
        predict_batch = torch.stack(predict_batch, dim=0)
        image_length = torch.tensor([x['image_length'] for x in samples], dtype=torch.long) # Use long for lengths

        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch, # This now contains the processed features
            'image_length': image_length # This is the original episode length
        }

        return data

    def __getitem__(self, idx):
        """
        Retrieves a single sample (images, optional OCR, goal text).
        """
        if self.data_name in ['AITW', 'bugs']:
            item = self.data[idx] # List of dicts
            item_id = item['id']
            episode_length = item['episode_length']
            episode_sub_path = item['category'] + '/' # Sub-path for image folder
            target = item['goal_info']
        elif self.data_name == 'llamatouch':
            item = self.data.iloc[idx] # Pandas DataFrame row
            item_id = str(item['episode']) # Use episode ID as item_id
            episode_length = item['nsteps'] # Use nsteps as episode_length
            episode_sub_path = item['path'] + '/' # Sub-path for image folder
            target = item['description'] # Use description as target

        else:
            raise ValueError(f"data_name '{self.data_name}' not supported.")
        

                # Construct the full path to the episode folder
        episode_folder_path = self.data_path + episode_sub_path

        # Load images and optionally OCR using the fused helper function
        # Pass the trsfm_ocr function itself, not its output
        images_tensor, texts_ocr_list = _load_images_and_optional_ocr(
            self.data_name, # Pass data_name
            item_id, # Pass item_id (used for AITW/bugs path)
            episode_folder_path, # Pass the full episode folder path
            episode_length,
            self.num_images,
            trsfm_ocr, # Pass the trsfm_ocr function
            self.use_ocr,
            self.ocr_instance,
            streaming=self.streaming
        )

        # Tokenize OCR text and boxes if OCR was used
        ocr_tokens_tensor = None
        max_text_len = 40 # Assuming this is a fixed parameter


        if self.use_ocr:
            # texts_ocr_list will be a list of lists of dicts, one inner list per frame loaded
            ocr_tokens_tensor = tokenize_and_concatenate(
                texts_ocr_list, 
                self.use_bbox, 
                self.tokenizer, 
                max_total_length= max_text_len
            )
            # ocr_tokens_tensor shape will be (num_frames_loaded, max_ocr_len)


        # Process goal_info into caption_tokens and need_predict (keep existing logic)
        prefix = '' # Assuming prefix is always empty based on original code
        
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len
        )

        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len
        )

        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])

        if len(payload) > max_text_len - 2:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]

        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]

        caption_tokens = torch.tensor(input_ids, dtype=torch.long)
        need_predict_tensor = torch.tensor(need_predict, dtype=torch.long)

        data = {
            'caption_tokens': caption_tokens,
            'need_predict': need_predict_tensor,
            'image': images_tensor, # This is the stacked tensor of transformed images
            'image_length': episode_length # Original episode length
        }

        if self.use_ocr:
            data['ocr_tokens'] = ocr_tokens_tensor # Add OCR tokens if used

        return data

class llava_dataset(Dataset):
    def __init__(self, jsonl_file, data_path,split,tokenizer,transform, num_images=6 ):
        self.data = []
        self.num_images = num_images
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Read data from JSONL file
        with jsonlines.open(jsonl_file) as reader:
            size = sum(1 for i in reader)
        
        with jsonlines.open(jsonl_file) as reader:
            if split == 'TRAIN':
                limit = int(size*0.8)
                for i,lines in enumerate(reader):
                    if i < limit:
                        self.data.append(lines)
                    else:
                        break
            if split == 'VALID':
                limit = int(size*0.8)
                for i,lines in enumerate(reader):
                    if i >= limit:
                        self.data.append(lines)

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, samples):
        max_length = 0
        for x in samples:
            max_length = max(max_length,x['caption_tokens'].shape[0])
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            b1 = torch.zeros(max_length, dtype=t.dtype, device=t.device)
            b2 = torch.zeros(max_length, dtype=n.dtype, device=n.device)
            b1[:t.shape[0]] = t
            b2[:n.shape[0]] = n
            x['caption_tokens'] = b1
            x['need_predict'] = b2
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
        }
        return data

    def __getitem__(self, idx):
        # {"id": "18375519518960921438", "goal_info": "Is it going to rain this weekend?", "episode_length": 8}
        # 10002872452831025023-0-14.jpg
        
        item = self.data[idx]

        path = f"{self.data_path}/{item['id']}"
        images = []
        image_path = f"{path}.jpg"            
        images.append(self.transform(Image.open(image_path).convert('RGB')))
        shape = images[0].shape
        for i in range(7):
            images.append(torch.zeros(shape))
        images_stacked = torch.stack(images, dim=0)


        max_text_len = 40 #from train.py
        target = item['goal_info']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
        # print('get_data')
        # print(prefix)
        # print(target)
        # print(need_predict)
        # print(payload)
        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        # Convert other fields to tensors as needed
        data = {
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'image': images_stacked,
            # 'rect' field can be fed in 'caption', which tells the bounding box
            # region of the image that is described by the caption. In this case,
            # we can optionally crop the region.
            # 'caption': {},
            # this iteration can be used for crop-size selection so that all GPUs
            # can process the image with the same input size
            # 'iteration': 0
        }
        
        return data

class animation(Dataset):
    def __init__(self, data_path,split,tokenizer,transform, num_images = 6):
        self.data = []
        self.num_images = num_images
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer
        # Read the tsv file, shuffle the data, and split it into train and validation
        # Read the tsv file
        df = pd.read_csv(f"{data_path}/output.tsv", sep='\t')

        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)

        # Split into train and validation
        train_df = df[:int(0.8 * len(df))]
        valid_df = df[int(0.8 * len(df)):]

        # Save train and validation data to self.data
        self.data = train_df if split == 'TRAIN' else valid_df

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, samples):
        max_length = 0
        for x in samples:
            max_length = max(max_length,x['caption_tokens'].shape[0])
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            b1 = torch.zeros(max_length, dtype=t.dtype, device=t.device)
            b2 = torch.zeros(max_length, dtype=n.dtype, device=n.device)
            b1[:t.shape[0]] = t
            b2[:n.shape[0]] = n
            x['caption_tokens'] = b1
            x['need_predict'] = b2
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
        }
        return data

    def __getitem__(self, idx):
        # episode	        category path	            description	        nsteps	app
        # 84143002711104077	general	 general/trace_11	Open the settings	4	    Settings
        item = self.data.iloc[idx]
        path = f"{self.data_path}/{item['episode']}/"
        images = get_auto_images(item['episode'],path,item['nsteps'],self.num_images,self.transform)

        max_text_len = 40 #from train.py
        target = item['description']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
        # print('get_data')
        # print(prefix)
        # print(target)
        # print(need_predict)
        # print(payload)
        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        # Convert other fields to tensors as needed
        data = {
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'image': images,
            # 'rect' field can be fed in 'caption', which tells the bounding box
            # region of the image that is described by the caption. In this case,
            # we can optionally crop the region.
            # 'caption': {},
            # this iteration can be used for crop-size selection so that all GPUs
            # can process the image with the same input size
            # 'iteration': 0
        }
        
        return data

class LlamaTouch(Dataset):
    def __init__(self, data_path,split,tokenizer,transform, num_images = 6):
        self.data = []
        self.num_images = num_images
        self.data_path = '/data/pauline/llamatouch_dataset/'
        self.transform = transform
        self.tokenizer = tokenizer
        # Read the tsv file, shuffle the data, and split it into train and validation
        # Read the tsv file
        df = pd.read_csv('/data/pauline/llamatouch_dataset/llamatouch_task_metadata.tsv', sep='\t')

        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)

        # Split into train and validation
        train_df = df[:int(0.8 * len(df))]
        valid_df = df[int(0.8 * len(df)):]

        # Save train and validation data to self.data
        self.data = train_df if split == 'TRAIN' else valid_df

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, samples):
        max_length = 0
        for x in samples:
            max_length = max(max_length,x['caption_tokens'].shape[0])
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            b1 = torch.zeros(max_length, dtype=t.dtype, device=t.device)
            b2 = torch.zeros(max_length, dtype=n.dtype, device=n.device)
            b1[:t.shape[0]] = t
            b2[:n.shape[0]] = n
            x['caption_tokens'] = b1
            x['need_predict'] = b2
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
        }
        return data

    def __getitem__(self, idx):
        # episode	        category path	            description	        nsteps	app
        # 84143002711104077	general	 general/trace_11	Open the settings	4	    Settings
        item = self.data.iloc[idx]
        path = self.data_path + item['path']+'/'
        images = get_llama_images(item['episode'],path,item['nsteps'],self.num_images,self.transform)

        max_text_len = 40 #from train.py
        target = item['description']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']

        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        # Convert other fields to tensors as needed
        data = {
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'image': images,
        }
        
        return data

def main():
    annotations_file = '/local/pauline/GIT/preprocessing/no_miss_general_train.jsonl'
    data_path = '/home/pauline/no-miss-AITW/general/'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    dataset = AITW_Dataset(annotations_file,data_path,'TRAIN',tokenizer,transform = trsfm())
    for i in range(5):
        print(dataset[i])


if __name__ == '__main__':
    main()
    # print_demo_images()
