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
import os
import logging

os.environ['GLOG_minloglevel'] = '3'
os.environ['FLAGS_log_level'] = '3'
logging.getLogger('ppocr').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)


def write_log(args,im):
    with open(f"./log/{args['exp_name']}_bs{args['bs']}_lr{args['lr']}_im{im}_log.txt","a") as f:
        f.write(f"bs = {args['bs']}, num_epoch = {args['epoch']},\n")
        f.write(f"lr = {args['lr']}, wd = {args['wd']},\n")

def get_images(id, data_path, length, num_images, split , ocr):
    # num_images default is 6
    # f"{des}{id}-{i}-{len(episode)}.jpg"
    path = data_path+id
    images = []
    texts = []
    if length >=num_images:
        for i in range(num_images):
            # get last six images
            image_path = f"{path}-{length-num_images+i}-{length}.jpg"
            image, text = trsfm(image_path, split=split, ocr = ocr)
            images.append(image)
            texts.append(text)
            # images.append(trsfm(Image.open(image_path).convert('RGB')))
 
    else:
        for i in range(length):
            # get last six images
            image_path = f"{path}-{i}-{length}.jpg"            
            image, text = trsfm(image_path,split= split,ocr = ocr)
            images.append(image)
            texts.append(text)
        shape = images[0].shape
        for i in range(num_images-length):
            images.append(torch.zeros(shape))
            texts.append('')

    images_stacked = torch.stack(images, dim=0)
    return images_stacked, texts

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

def trsfm(image_path, image_size=(224, 224), split='VALID', ocr=None):
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
def tokenize_and_concatenate(texts,tokenizer, max_total_length=40):

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
            # box_str = " ".join([str(int(coord)) for point in item['box'] for coord in point])
            # # tokenize box
            # box_encoding = tokenizer(
            #     box_str,
            #     add_special_tokens=False,
            #     padding=False,
            #     truncation=False
            # )
            # box_ids = box_encoding["input_ids"]
            
            # 1.c 連接 text 與 box 的 token 序列，加上 SEP token
            # combined = text_ids + box_ids + [tokenizer.sep_token_id]
            combined = text_ids +[tokenizer.sep_token_id]
            text_box_encoding += combined
        # 2. Padding
        if len(text_box_encoding) > max_total_length:
            # only the first max_total_length tokens are used
            text_box_encoding = text_box_encoding[:max_total_length]
        elif len(text_box_encoding) < max_total_length:
            text_box_encoding += [tokenizer.pad_token_id] * (max_total_length - len(text_box_encoding))
        # 3. append text box encoding to ocr token
        ocr_token.append(text_box_encoding)

    ocr_token = torch.tensor(ocr_token) # (num_images, max_total_length)
    return ocr_token

class AITW_Dataset(Dataset):
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
        ocr_batch = torch.stack([x['ocr_tokens'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
            'ocr_tokens': ocr_batch,
        }
        return data

    def __getitem__(self, idx):
        # {"id": "18375519518960921438", "goal_info": "Is it going to rain this weekend?", "episode_length": 8}
        # 10002872452831025023-0-14.jpg
        item = self.data[idx]
        images = get_images(item['id'],self.data_path,item['episode_length'],self.num_images,self.transform)

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



class AITW_Dataset_V2(Dataset):
    def __init__(self ,data_path,split, tokenizer, ocr , num_images=6 ):
        self.data = []
        self.num_images = num_images
        self.data_path = data_path
        self.split = split
        self.tokenizer = tokenizer
        self.ocr = ocr
        category = ['google_apps','general','install','web_shopping']

        # Read data from JSONL file
        for cat in category:
            jsonl_file = f'/home/pauline/GIT/util/no_miss_{cat}_train.jsonl'
            with jsonlines.open(jsonl_file) as reader:
                size = sum(1 for i in reader)
            
            with jsonlines.open(jsonl_file) as reader:
                if split == 'TRAIN':
                    # limit = int(1)
                    limit = int(size*0.8)
                    for i,lines in enumerate(reader):
                        if i < limit:
                            self.data.append(lines)
                        else:
                            break
                if split == 'VALID':
                    # limit = int(size*0.99999)
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
        ocr_batch = torch.stack([x['ocr_tokens'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        image_length = torch.Tensor([x['image_length'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
            'ocr_tokens': ocr_batch,
            'image_length': image_length
        }
        return data

    def __getitem__(self, idx):
        # {"id": "18375519518960921438", "goal_info": "Is it going to rain this weekend?", "episode_length": 8}
        # 10002872452831025023-0-14.jpg
        item = self.data[idx]
        path = self.data_path + item['category']+'/'
        
        images, texts = get_images(item['id'],path,item['episode_length'],self.num_images, self.split, ocr = self.ocr)
        image_length = item['episode_length']

        max_text_len = 40 #from train.py
        
        # Tokenize the OCR text and concatenate them
        texts_stack = tokenize_and_concatenate(texts, tokenizer = self.tokenizer)
        
        target = item['goal_info']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, 
            max_length=max_text_len)
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
            'ocr_tokens': texts_stack,
            'image_length': image_length
        }
        
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

def main():
    # test trsfm
    image_path = '/data/pauline/no-miss-AITW/general/1877204021578832713-5-11.jpg'
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    # transform image and text
    image, text = trsfm(image_path, image_size=(224, 224), split='TRAIN',ocr = ocr)
    # annotations_file = '/local/pauline/GIT/preprocessing/no_miss_general_train.jsonl'
    # data_path = '/home/pauline/no-miss-AITW/general/'
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # dataset = AITW_Dataset(annotations_file,data_path,'TRAIN',tokenizer,transform = trsfm())
    # for i in range(5):
    #     print(dataset[i])


if __name__ == '__main__':
    main()
    # print_demo_images()
