from .torch_common import resize_2d_pos_embed
import torch
from .layers.CLIP import clip
from .layers.bert import modeling_bert
from .layers.multi_decoder import CaptioningModel
from .layers.multi_decoder import (TransformerDecoderTextualHead,
                             AutoRegressiveBeamSearch, GeneratorWithBeamSearch)

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

def get_git_model(tokenizer, param):
    image_encoder = get_image_encoder(
        param.get('image_encoder_type', 'CLIPViT_B_16'),
        input_resolution=param.get('test_crop_size', 224),
    )

    text_decoder = TransformerDecoderTextualHead(
        visual_feature_size=param.get('visual_feature_size', 768),
        vocab_size=30522,
        hidden_size=768,
        num_layers=6,
        attention_heads=12,
        feedforward_size=768* 4,
        max_caption_length=1024,
        mask_future_positions=True,
        padding_idx=0,
        decoder_type='bert_en',
        visual_projection_type='linearLn',
    )
    #decoder = AutoRegressiveBeamSearch(
        #eos_index=tokenizer.sep_token_id,
        #max_steps=40,
        #beam_size=1,
        #per_node_beam_size=1,
        #fix_missing_prefix=True,
    #)
    decoder = GeneratorWithBeamSearch(
        eos_index=tokenizer.sep_token_id,
        max_steps=param.get('max_step',40),
        # max_steps=1024,
        beam_size=4,
        length_penalty=0.6,
    )

    #from .trie_decoder import TrieAutoRegressiveBeamSearch, get_trie
    #decoder = TrieAutoRegressiveBeamSearch(
        #eos_index=tokenizer.sep_token_id,
        #max_steps=1022,
        #beam_size=1,
        #trie=get_trie(tokenizer),
    #)
    model = CaptioningModel(
        image_encoder,
        text_decoder,
        decoder=decoder,
        sos_index=tokenizer.cls_token_id,
        eos_index=tokenizer.sep_token_id,
        context_not_share_embedding = param.get('context_not_share_embedding'),
        tokenizer=tokenizer,
        use_history_for_infer=True,
        loss_type='smooth',
        num_image_with_embedding=param.get('num_image_with_embedding'),
        streaming=param.get('streaming'),
        paddle=param.get('paddle'),
    )
    return model


def get_image_encoder(encoder_type, input_resolution=224):
    name_map = {
        'CLIPViT_B_16': 'ViT-B/16',
        'CLIPViT_L_14': 'ViT-L/14',
    }
    name_in_clip = name_map[encoder_type]
    model, _ = clip.load(name_in_clip, device='cpu', jit=False)
    model = model.train()
    ret = model.visual
    ret.to(torch.float32)
    ret.output_grid = True
    ret.grid_after_ln = True
    if ret.input_resolution != input_resolution:
        if encoder_type in ['CLIPViT_B_16', 'CLIPViT_L_14']:
            pos = ret.positional_embedding
            patch_size = ret.conv1.kernel_size[0]
        else:
            pos = ret.attnpool.positional_embedding
            patch_size = 32
        p2 = resize_2d_pos_embed(pos,
                            ret.input_resolution,
                            patch_size,
                            input_resolution)
        ret.input_resolution = input_resolution
        if encoder_type in ['CLIPViT_B_16', 'CLIPViT_L_14']:
            ret.positional_embedding = torch.nn.Parameter(p2)
        else:
            ret.attnpool.positional_embedding = torch.nn.Parameter(p2)
    
    return ret

# construct a main to test Test encoder and image encoder

from transformers import BertTokenizer, BertModel
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 输入句子
    sentence = "This is a sample sentence."

    # 将句子分词并转换为 BERT 输入格式
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # 模型前向传播
    outputs = model(**inputs)

    # 提取输出
    last_hidden_state = outputs.last_hidden_state  # 所有 token 的最后一层表示
    pooler_output = outputs.pooler_output  # [CLS] 的句子表示

    # 输出形状
    print("Input IDs shape:", inputs["input_ids"].shape)
    print("Last Hidden State shape:", last_hidden_state.shape)
    print("Pooler Output shape:", pooler_output.shape)
    print("Pooler Output:", pooler_output)

    image_encoder = get_image_encoder("CLIPViT_B_16")
    image = torch.randn(2, 3, 224, 224)
    image_encoder(image)
    # print shape of image embedding
    print("image shape", image_encoder(image).shape)


