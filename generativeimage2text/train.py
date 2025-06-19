import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#hide error tf message 
from .common import Config
import json
import os.path as op
from .common import qd_tqdm as tqdm
from .common import json_dump
from .common import pilimg_from_base64
from .torch_common import recursive_to_device
from .tsv_io import TSVFile, tsv_writer, tsv_reader
from .common import write_to_file
import torch
import PIL
from pprint import pformat
import logging
from transformers import BertTokenizer
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from azfuse import File

from .common import init_logging
from .common import parse_general_args
from .tsv_io import load_from_yaml_file
from .torch_common import torch_load
from .torch_common import load_state_dict
from .torch_common import resize_2d_pos_embed
from .layers.CLIP import clip
from .layers.decoder import (TransformerDecoderTextualHead,
                             AutoRegressiveBeamSearch, GeneratorWithBeamSearch)
from .layers.decoder import CaptioningModel
from .process_image import load_image_by_pil
from .data_layer.transform import RenameKey, SelectTransform
from .data_layer.transform import ImageTransform2Dict
from .data_layer.transform import get_inception_train_transform
from .data_layer.builder import collate_fn
from .model import get_git_model

from transformers import get_cosine_schedule_with_warmup
import jsonlines
from torch.utils.data import Dataset,DataLoader
from .util import AITW_Dataset_V2 ,trsfm, write_log,LlamaTouch, llava_dataset,AITW_Dataset_1000, animation
from .scorer import Scorers

def get_data(image_file, prefix, target, tokenizer, image_transform):
    max_text_len = 40
    prefix_encoding = tokenizer(
        prefix, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    target_encoding = tokenizer(
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
    input_ids = [tokenizer.cls_token_id] + payload + [tokenizer.sep_token_id]
    need_predict = [0] + need_predict + [1]

    im = load_image_by_pil(image_file)

    print(im)
    data = {
        'caption_tokens': torch.tensor(input_ids),
        #'caption_lengths': len(input_ids),
        'need_predict': torch.tensor(need_predict),
        'image': im,
        # 'rect' field can be fed in 'caption', which tells the bounding box
        # region of the image that is described by the caption. In this case,
        # we can optionally crop the region.
        'caption': {},
        # this iteration can be used for crop-size selection so that all GPUs
        # can process the image with the same input size
        'iteration': 0,
    }
    data = image_transform(data)
    # print(image_transform)
    # print(data['image'].shape)
    return data

def get_image_transform(cfg):
    return get_multi_scale_image_transform(cfg, is_train=True)

def get_default_mean():
    return [0.485, 0.456, 0.406]

def get_default_std():
    return [0.229, 0.224, 0.225]

def get_transform_image_norm(cfg, default=None):
    if cfg.data_normalize == 'default':
        normalize = transforms.Normalize(
            mean=get_default_mean(), std=get_default_std())
    elif cfg.data_normalize == 'clip':
        # clip model
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    else:
        raise NotImplementedError(cfg.data_normalize)
    return normalize

def get_transform_vit_default(cfg, is_train):
    default_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = get_transform_image_norm(cfg, default_normalize)
    transform = get_inception_train_transform(
        bgr2rgb=True,
        crop_size=cfg.train_crop_size,
        normalize=normalize,
        small_scale=cfg.input_small_scale,
        no_color_jitter=cfg.no_color_jitter,
        no_flip=cfg.no_flip,
        no_aspect_dist=cfg.no_aspect_dist,
        resize_crop=cfg.resize_crop,
        max_size=cfg.train_max_size,
        interpolation=cfg.interpolation or Image.BILINEAR,
    )
    return transform

def get_transform_image(cfg, is_train):
    train_transform = cfg.train_transform
    if train_transform == 'vitp':
        transform = get_transform_vit_default(
            cfg, is_train=is_train)
    else:
        raise NotImplementedError(train_transform)
    return transform

class ImageTransform2Images(object):
    def __init__(self, sep_transform, first_joint=None):
        self.image_transform = sep_transform
        self.first_joint = first_joint

    def __call__(self, imgs):
        if self.first_joint is not None:
            imgs = self.first_joint(imgs)
        return [self.image_transform(im) for im in imgs]

    def __repr__(self):
        return 'ImageTransform2Images(image_transform={})'.format(
            self.image_transform,
        )

def get_transform_images(cfg, is_train):
    trans = get_transform_image(cfg, is_train)
    trans = ImageTransform2Images(trans)
    return trans

def trans_select_for_crop_size(
    data, train_crop_sizes,
    iteration_multi=0,
):
    if iteration_multi <= 0:
        if len(train_crop_sizes) == 1:
            idx = 0
        else:
            idx = data['iteration'] % len(train_crop_sizes)
    elif data['iteration'] <= iteration_multi:
        idx = data['iteration'] % len(train_crop_sizes)
    else:
        idx = -1
    return idx

def get_multi_scale_image_transform(cfg, is_train, get_one=get_transform_image):
    def get_multi_res_transform(s):
        old = cfg.train_crop_size if is_train else cfg.test_crop_size
        all_t = []
        multi_res_factors = cfg.multi_res_factors or []
        for i, f in enumerate(multi_res_factors):
            if is_train:
                cfg.train_crop_size = s // f
            else:
                cfg.test_crop_size = s // f
            key = 'image_{}'.format(i)
            all_t.append(RenameKey({'image': key}, not_delete_origin=True))
            t = get_one(cfg, is_train)
            t = ImageTransform2Dict(t, key=key)
            all_t.append(t)
        # get_one depends on train_crop_size
        if is_train:
            cfg.train_crop_size = s
        else:
            cfg.test_crop_size = s
        t = get_one(cfg, is_train)
        t = ImageTransform2Dict(t)
        all_t.append(t)
        if is_train:
            cfg.train_crop_size = old
        else:
            cfg.test_crop_size = old
        return transforms.Compose(all_t)

    if is_train:
        if cfg.min_size_range32 is None:
            train_crop_sizes = [cfg.train_crop_size]
        else:
            train_crop_sizes = list(range(
                cfg.min_size_range32[0],
                cfg.min_size_range32[1] + cfg.patch_size - 1, cfg.patch_size,
            ))
    else:
        train_crop_sizes = [cfg.test_crop_size]

    crop_trans = []
    for s in train_crop_sizes:
        t = get_multi_res_transform(s)
        crop_trans.append(t)
    iteration_multi = 0
    image_transform = SelectTransform(
        crop_trans,
        lambda d: trans_select_for_crop_size(
            d, train_crop_sizes, iteration_multi))
    return image_transform

def forward_backward_example(image_files, captions, prefixs=None):
    if prefixs is None:
        prefixs = [''] * len(captions)
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 224,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [160, 224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        'patch_size': 16, #six images' size ranges in (160,176,192,208,224)
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    all_data = []
    # print(image_files)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    image_transform = get_image_transform(cfg)
    for image_file, prefix, target in zip(image_files, prefixs, captions):
        # print(image_file, prefix, target)
        data = get_data(image_file, prefix, target,# see above
                        tokenizer, image_transform)
        all_data.append(data)
    print('before collate_fn')
    print(all_data)
    data = collate_fn(all_data) #locate any types of data
    print(data)
    # print('in train.py: ',data)  #1,3,160,160
    # logging.info(image_transform)
    data = recursive_to_device(data, 'cuda')

    # return ######################
    param = {}
    model = get_git_model(tokenizer, param)
    model.train()
    model.cuda()
    # print('before training')
    # print(model.img_temperal_embedding[2].grad)
    loss_dict = model(data)
    print(loss_dict)
    loss = sum(loss_dict.values())
    loss.backward()
    # print('congrad')
    # print(model.img_temperal_embedding[2].grad)

    logging.info(loss)

def speed_test_forward_backward():
    duplicate = 32
    image_files = ['aux_data/images/1.jpg', 'aux_data/images/2.jpg'] * duplicate
    captions = ['a couple of boats in a large body of water.', 'a view of a mountain with a tree'] * duplicate

    prefixs = [''] * len(captions)
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 224,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [160, 224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        'patch_size': 16,
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    all_data = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    image_transform = get_image_transform(cfg) #see 158
    for image_file, prefix, target in zip(image_files, prefixs, captions):
        data = get_data(image_file, prefix, target,
                        tokenizer, image_transform)
        all_data.append(data)
    data = collate_fn(all_data)
    logging.info(image_transform)
    data = recursive_to_device(data, 'cuda')
    data['image'] = data['image'].to(torch.float16)

    param = {}
    model = get_git_model(tokenizer, param)
    model.train()
    model.cuda()
    model.half()

    # warmup
    for _ in range(2):
        loss_dict = model(data)
        loss = sum(loss_dict.values())
        loss.backward()

    import time
    start = time.time()
    for iteration in range(1000):
        loss_dict = model(data)
        loss = sum(loss_dict.values())
        loss.backward()
        if (iteration % 10) == 0:
            end = time.time()
            speed = data['image'].shape[0] * 100 / (end - start)
            if iteration > 0:
                logging.info('speed = {mytrain}'.format(speed))
            start = time.time()

    logging.info(loss)

def mytrain(param,args):
    '''
    'num_workers':2
    'use_dif_lr': False
    'wd':0.1,     weight decay
    'lr':1e-5,    learning rate
    'epoch':2,    training epoch
    'bs':32 ,     batch size
    'acc_step':4, accumulation steps
    'pat':5,      patience for early stop, but may not use
    'load_path':'/data/poyang/checkpoint/TryLowWD_lr1e-05_wd0.01_im6.ckpt',
    'ckpt_path':'/data/cv/poyang/' path to save model
    'exp_name'
    '''
    args['lr'] = float(args['lr'])
    
    logfile = f"./log/{args['exp_name']}_lr{args['lr']}_wd{args['wd']}_im{param.get('num_image_with_embedding')}_log.txt"
    # 確認目錄是否存在，如果不存在則創建
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    # 確認文件是否存在，如果不存在則創建
    if not os.path.exists(logfile):
        with open(logfile, 'w') as f:
            f.write('')  # 創建文件並寫入空內容

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    image_size = ((288,160) if args['Pix2Struct'] else (224,224))

    # param = {'num_image_with_embedding':6}
    device = torch.device(args.get('cuda','cuda') if torch.cuda.is_available() else "cpu")
    model = get_git_model(tokenizer, param)   
    model.to(device)

    loaded_image = param['num_image_with_embedding']
    
    if args['exp_name'] == 'auto':
        data_path_V2 = '/data/pauline/animation/'
        TrainDataset = animation(data_path_V2,'TRAIN',tokenizer,transform = trsfm(image_size=image_size,split='TRAIN')
                                ,num_images=loaded_image)
        TrainLoader = DataLoader(TrainDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=True, collate_fn = TrainDataset.collate_fn)    
        
        ValidDataset = animation(data_path_V2,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
                                ,num_images=loaded_image)
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=False, collate_fn = ValidDataset.collate_fn)
    elif args['exp_name'] == 'LLAMATOUCH':
        data_path_V2 = '/data/pauline/llamatouch_dataset/'
        TrainDataset = LlamaTouch(data_path_V2,'TRAIN',tokenizer,transform = trsfm(image_size=image_size,split='TRAIN')
                                ,num_images=loaded_image)
        TrainLoader = DataLoader(TrainDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=True, collate_fn = TrainDataset.collate_fn)    
        
        ValidDataset = LlamaTouch(data_path_V2,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
                                ,num_images=loaded_image)
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=False, collate_fn = ValidDataset.collate_fn)
    elif args['exp_name'] == 'AITW' or args['exp_name'] == 'bugs':
        data_path_V2 = '/data/pauline/no-miss-AITW/'

        TrainDataset = AITW_Dataset_V2(data_path_V2,'TRAIN',tokenizer,transform = trsfm(image_size=image_size,split='TRAIN')
            ,num_images=loaded_image)
        TrainLoader = DataLoader(TrainDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=True, collate_fn = TrainDataset.collate_fn)

        ValidDataset = AITW_Dataset_V2(data_path_V2,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
            ,num_images=loaded_image)
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=False, collate_fn = ValidDataset.collate_fn)
    elif args['exp_name'] == 'AITW_1000':
        data_path_V2 = '/data/pauline/no-miss-AITW/'
        jsonl_file = f"/home/pauline/GIT/preprocessing/{args['exp_name']}.jsonl"
        TrainDataset = AITW_Dataset_1000(jsonl_file,data_path_V2,'TRAIN',tokenizer,transform = trsfm(image_size=image_size,split='TRAIN')
            ,num_images=loaded_image)
        TrainLoader = DataLoader(TrainDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=True, collate_fn = TrainDataset.collate_fn)
        ValidDataset = AITW_Dataset_1000(jsonl_file,data_path_V2,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
            ,num_images=loaded_image)
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=False, collate_fn = ValidDataset.collate_fn)
    elif args['exp_name'] == 'llava':
        annotations_file = '/data/pauline/llava_detail/detail.jsonl'
        data_path = '/data/pauline/llava_detail/train2017'
        TrainDataset = llava_dataset(annotations_file,data_path,'TRAIN',tokenizer,transform = trsfm(image_size=image_size,split='TRAIN')
        ,num_images=loaded_image)
        TrainLoader = DataLoader(TrainDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'], \
            shuffle=True, collate_fn = TrainDataset.collate_fn)
        ValidDataset = llava_dataset(annotations_file,data_path,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
            ,num_images=loaded_image)
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'], \
            shuffle=False, collate_fn = ValidDataset.collate_fn)
        
    num_training_steps = len(TrainLoader) / args['acc_step'] * args['epoch'] 
    num_warmup_steps = int(0.1 * num_training_steps)

    if args['use_dif_lr']:
        optimizer_img = torch.optim.AdamW(params=model.image_encoder.parameters(), lr=args['lr'], weight_decay=args['wd'])
        optimizer_tex = torch.optim.AdamW(params=model.textual.parameters(), lr=args['lr']*5, weight_decay=args['wd'])
        optimizer_emb = torch.optim.AdamW(params=model.img_temperal_embedding.parameters(), lr=args['lr']*5, weight_decay=args['wd'])

        scheduler_img = get_cosine_schedule_with_warmup(optimizer_img, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        scheduler_tex = get_cosine_schedule_with_warmup(optimizer_tex, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        scheduler_emb = get_cosine_schedule_with_warmup(optimizer_emb, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args['lr'], weight_decay=args['wd'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    if 'load_path' in args.keys():
        load_path = '/data/pauline/checkpoint/' + args['load_path']
        print(f"load model from {load_path}")
        if args['use_dif_lr']:
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model'])
            optimizer_img.load_state_dict(checkpoint['optimizer_img'])
            optimizer_tex.load_state_dict(checkpoint['optimizer_tex'])
            optimizer_emb.load_state_dict(checkpoint['optimizer_emb'])
            scheduler_img.load_state_dict(checkpoint['scheduler_img'])
            scheduler_tex.load_state_dict(checkpoint['scheduler_tex'])
            scheduler_emb.load_state_dict(checkpoint['scheduler_emb'])
        else:
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])


    # batch = next(iter(TrainLoader))
    # data = recursive_to_device(data, 'cuda')
    # return ######################
    # print('before training')
    # print(model.img_temperal_embedding[2].grad)
    stale = 0
    best_score = 0.0
    with open(logfile, "a") as f:
        checkpoint_status = f"checkpoint = {args.get('load_path', 'none')}, train_data = {args['exp_name']}\n"
        train_settings = (f"bs = {args['bs']}, num_epoch = {args['epoch']},\n"
                          f"lr = {args['lr']}, wd = {args['wd']},\n"
                          f"use_dif_lr = {args['use_dif_lr']}, img = {param['num_image_with_embedding']}, "
                          f"Pix = {args.get('Pix2Struct', False)}\n")
        f.write(checkpoint_status + train_settings)

    for epoch in range(args['epoch']):

        train_loss = []
        model.train()
        for index, batch in enumerate(tqdm(TrainLoader)):
            batch['image'] = batch['image'].to(device)
            batch['caption_tokens'] = batch['caption_tokens'].to(device)
            batch['need_predict'] = batch['need_predict'].to(device)
            loss_dict = model(batch)
            # print(loss_dict)
            loss = sum(loss_dict.values()) / args['acc_step']
            # print(loss)
            # return
            loss.backward()
            if (index + 1) % args['acc_step'] == 0 or (index+1) == len(TrainLoader):
                # Clip the gradient norms for stable training.
                # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                if args['use_dif_lr']:
                    optimizer_img.step()
                    optimizer_tex.step()
                    optimizer_emb.step()
                    scheduler_img.step()
                    scheduler_tex.step()
                    scheduler_emb.step()
                    optimizer_img.zero_grad()
                    optimizer_tex.zero_grad()
                    optimizer_emb.zero_grad()
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            train_loss.append(loss.item())
        # des = '/data/cv/poyang/checkpoint/'
        # torch.save(model.state_dict(),\
        # f"{des}{args['exp_name']}_bs{args['bs']}_lr{args['lr']}_im{param.get('num_image_with_embedding')}_e{epoch}.ckpt")
        total_loss = sum(train_loss) / len(train_loss)
        with open(logfile,"a") as f:
            f.write(f"[ Train | {epoch + 1:03d}/{args['epoch']:03d} ] loss = {total_loss:.5f}\n")
        print(f"[ Train | {epoch + 1:03d}/{args['epoch']:03d} ] loss = {total_loss:.5f}")
        
        if args['use_dif_lr']:
            checkpoint = {
                'model':model.state_dict(),
                'optimizer_img':optimizer_img.state_dict(),
                'optimizer_tex':optimizer_tex.state_dict(),
                'optimizer_emb':optimizer_emb.state_dict(),
                'scheduler_img':scheduler_img.state_dict(),
                'scheduler_tex':scheduler_tex.state_dict(),
                'scheduler_emb':scheduler_emb.state_dict(),
            }    
        else:
            checkpoint = {
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
            }
        if 'load_path' in args.keys():
            pretrain = "pretrain_"
        else:
            pretrain = ""
        checkpoint_name = f"{args['ckpt_path']}{pretrain}{args['exp_name']}_ep{epoch + 1}_lr{args['lr']}_wd{args['wd']}_im{param.get('num_image_with_embedding')}.ckpt"
        torch.save(checkpoint, checkpoint_name)
        print(f"Saved checkpoint to {checkpoint_name}")

        # if 'load_path' in args.keys():
        #     torch.save(checkpoint,
        #     f"{args['ckpt_path']}{args['load_path']}_{args['exp_name']}_ep{epoch+1}_lr{args['lr']}_wd{args['wd']}_im{param.get('num_image_with_embedding')}.ckpt")
        #     print(f"save {args['ckpt_path']}_{args['load_path']}_{args['exp_name']}_ep{epoch+1}_lr{args['lr']}_wd{args['wd']}_im{param.get('num_image_with_embedding')}.ckpt")
        # else:
        #     torch.save(checkpoint,
        #     f"{args['ckpt_path']}{args['exp_name']}_ep{epoch+1}_lr{args['lr']}_wd{args['wd']}_im{param.get('num_image_with_embedding')}.ckpt")
        #     # f"{args['ckpt_path']}cuda{args.get('cuda')}.ckpt")
        #     print(f"save {args['ckpt_path']}{args['exp_name']}_ep{epoch+1}_lr{args['lr']}_wd{args['wd']}_im{param.get('num_image_with_embedding')}.ckpt")
        
        continue
        
        ################################################################################################
        ## validation
        model.eval()
        caption_predictions = []
        caption_references = []
        for i,batch in enumerate(tqdm(ValidLoader)):
            with torch.no_grad():
                batch['image'] = batch['image'].to(device)
                result = model(batch)
            # return a list of text: ['a b c','i am a dog']
            cap = tokenizer.batch_decode(result['predictions'], skip_special_tokens=True)
            ref = tokenizer.batch_decode(batch['caption_tokens'], skip_special_tokens=True)
            ref = [[r] for r in ref]
            caption_predictions += cap # [ 1 , 2 , 3 ]
            caption_references  += ref # [[1],[2],[3]]
        print('predictions')
        print(caption_predictions[:5])
        print('reference')
        print(caption_references[:5])
        total_score = Scorers(caption_predictions, caption_references).compute_scores()
        print(f" bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}")
        
        # save score
        valid_score = total_score['bleu'][3] + total_score['CIDEr']
        if valid_score > best_score:
            with open(logfile,"a") as f:
                f.write(f"[ Valid | {epoch + 1:03d}/{args['epoch']:03d} ] bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f} -> best\n")
            
        else:
            with open(logfile,"a") as f:
                f.write(f"[ Valid | {epoch + 1:03d}/{args['epoch']:03d} ] bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}\n")
        # save models
        if valid_score > best_score:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            if args['use_dif_lr']:
                checkpoint = {
                    'model':model.state_dict(),
                    'optimizer_img':optimizer_img.state_dict(),
                    'optimizer_tex':optimizer_tex.state_dict(),
                    'optimizer_emb':optimizer_emb.state_dict(),
                    'scheduler_img':scheduler_img.state_dict(),
                    'scheduler_tex':scheduler_tex.state_dict(),
                    'scheduler_emb':scheduler_emb.state_dict(),
                }
            else:
                checkpoint = {
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),
                }
            torch.save(checkpoint,
            f"{args['ckpt_path']}{args['exp_name']}_lr{args['lr']}_wd{args['wd']}_im{param.get('num_image_with_embedding')}.ckpt")
            best_score = valid_score
            stale = 0
        else:
            stale += 1
            if stale >= args['pat']:
                print(f"No improvment {args['pat']} consecutive epochs, early stopping at {epoch+1}")
                break
    print('--------------------------------------------end-----------------------------------------')
    print(logfile)

def myinfer(param,args):

    logfile = f"./valid_log/{args['exp_name']}_lr{args['lr']}_wd{args['wd']}_im{param.get('num_image_with_embedding')}_log.txt"
    # 確認目錄是否存在，如果不存在則創建
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    # 確認文件是否存在，如果不存在則創建
    if not os.path.exists(logfile):
        with open(logfile, 'w') as f:
            f.write('')  # 創建文件並寫入空內容
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    image_size = ((288,160) if args['Pix2Struct'] else (224,224))
    device = torch.device(args.get('cuda','cuda') if torch.cuda.is_available() else "cpu")
    model = get_git_model(tokenizer, param)   
    model.to(device)
    checkpoint = torch.load(args['load_path'])
    print(f"load model from {args['load_path']}")
    model.load_state_dict(checkpoint['model'])

    if args['exp_name'] == 'LLAMATOUCH':
        data_path_V2 = '/data/pauline/llamatouch_dataset/'
        ValidDataset = LlamaTouch(data_path_V2,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
                                ,num_images=param['num_image_with_embedding'])
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=False, collate_fn = ValidDataset.collate_fn)

    elif args['exp_name'] == 'AITW' or args['exp_name'] == 'bugs':
        data_path_V2 = '/data/pauline/no-miss-AITW/'
        ValidDataset = AITW_Dataset_V2(data_path_V2,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
            ,num_images=param['num_image_with_embedding'])
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=False, collate_fn = ValidDataset.collate_fn)
    
    elif 'AITW' in args['exp_name']:
        data_path_V2 = '/data/pauline/no-miss-AITW/'
        jsonl_file = f"/home/pauline/GIT/util/{args['exp_name']}.jsonl"

        ValidDataset = AITW_Dataset_1000(jsonl_file,data_path_V2,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
            ,num_images=param['num_image_with_embedding'])
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=False, collate_fn = ValidDataset.collate_fn)
    
    elif args['exp_name'] == 'llava':
        annotations_file = '/data/pauline/llava_detail/detail.jsonl'
        data_path = '/data/pauline/llava_detail/train2017'

        ValidDataset = llava_dataset(annotations_file,data_path,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
            ,num_images=param['num_image_with_embedding'])
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'], \
            shuffle=False, collate_fn = ValidDataset.collate_fn)
    if args['exp_name'] == 'auto':
        data_path_V2 = '/data/pauline/animation/'

        ValidDataset = animation(data_path_V2,'VALID',tokenizer,transform = trsfm(image_size=image_size,split='VALID')
                                ,num_images=param['num_image_with_embedding'])
        ValidLoader = DataLoader(ValidDataset, batch_size=int(args['bs']/args['acc_step']), num_workers=args['num_workers'],shuffle=False, collate_fn = ValidDataset.collate_fn)
    
        
    with open(logfile,"a") as f:
        f.write(f"checkpoint = {args['load_path']}, train_data = {args['exp_name']}"+"\n")
        f.write(f"bs = {args['bs']}, num_epoch = {args['epoch']},\n")
        f.write(f"lr = {args['lr']}, wd = {args['wd']},\n")
        f.write(f"2lr = {args['use_dif_lr']}, img = {param['num_image_with_embedding']}, Pix = {args['Pix2Struct']}\n")
  
    model.eval()
    caption_predictions = []
    caption_references = []
    for i,batch in enumerate(tqdm(ValidLoader)):
        with torch.no_grad():
            batch['image'] = batch['image'].to(device)
            result = model(batch)
        # return a list of text: ['a b c','i am a dog']
        cap = tokenizer.batch_decode(result['predictions'], skip_special_tokens=True)
        ref = tokenizer.batch_decode(batch['caption_tokens'], skip_special_tokens=True)
        ref = [[r] for r in ref]
        caption_predictions += cap # [ 1 , 2 , 3 ]
        caption_references  += ref # [[1],[2],[3]]
    print('predictions')
    print(caption_predictions[:5])
    print('reference')
    print(caption_references[:5])
    total_score = Scorers(caption_predictions, caption_references).compute_scores()
    print(f" bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}")
    with open(logfile,"a") as f:
        f.write(f"bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}\n")

    output_data = {
    "caption_predictions": caption_predictions,
    "caption_references": caption_references
    }

    # Define the output file name
    output_file = f"infer/infer.txt"

    # Write the data to a JSON file
    if not os.path.exists('infer'):
        os.makedirs('infer')

    with open(output_file,"a") as f:
        f.write(f"checkpoint : {args['load_path']}\n")
        f.write(f"train data : {args['exp_name']}\n")
        f.write(f"bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}\n")
        f.write(output_data)

    print(f"Caption predictions and references have been saved to {output_file}")

if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

