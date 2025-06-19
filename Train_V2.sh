# python -m generativeimage2text.streaming_train -p "{'type': 'mytrain', \
# 'param':{ \
#       'num_image_with_embedding':8,\
#       'streaming':True,\
#       'paddle':False \
#       }, \
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':3,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'load_path':'AITW_ep3_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW'\
#       }}" 


# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1 python -m generativeimage2text.streaming_train -p "{'type': 'mytrain', \
# 'param':{ \
#       'num_image_with_embedding':8,\
#       'streaming':True,\
#       'paddle':True, \
#       'context_not_share_embedding' : True \
#       }, \
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':3,    \
#       'bs': 32,    \
#       'acc_step':8, \
#       'pat':2,      \
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'load_path':'text_encoder_multi_AITW_ep3_lr1e-05_wd0.0001_im8_paddle_True_streaming_False_with_bbox.ckpt',\
#       'exp_name' :'AITW',\
#       'bbox' : True ,\
#       'per_frame' : False\
#       }}"


# python -m generativeimage2text.multi_train -p "{'type': 'mytrain', \
# 'param':{ \
#       'num_image_with_embedding':8,\
#       'streaming':False,\
#       'paddle':True,\
#       'context_not_share_embedding' : True \
#       }, \
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':3,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'exp_name' :'AITW'\
#       }}" 

# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':True, 'streaming':True},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_with_text_encoder_multi_AITW_ep3_lr1e-05_wd0_and_temporal_embedding_and_without_bbox_AITW_ep2_lr1e-05_wd0.0001_im8_paddle_True_streaming_True.ckpt',\
#       'exp_name' :'bugs'\
#       }}" 

# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1  python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':True, 'streaming':True, 'context_not_share_embedding': True},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_with_text_encoder_multi_AITW_ep3_lr1e-05_wd0_and_temporal_embedding_and_with_bbox_AITW_ep2_lr1e-05_wd0.0001_im8_paddle_True_streaming_True.ckpt',\
#       'exp_name' :'llamatouch',\
#       'bbox' : True ,\
#       'per_frame' : False\
#       }}" 

# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1  python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':True, 'streaming':True, 'context_not_share_embedding': True},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_with_text_encoder_multi_AITW_ep3_lr1e-05_wd0_and_temporal_embedding_and_without_bbox_AITW_ep2_lr1e-05_wd0.0001_im8_paddle_True_streaming_True.ckpt',\
#       'exp_name' :'llamatouch',\
#       'bbox' : False ,\
#       'per_frame' : False\
#       }}" 

# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1  python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding': False},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_with_AITW_ep3_lr1e-05_wd0_and_temporal_embedding_and_feauture_from_each_image_AITW_ep2_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'llamatouch',\
#       'bbox' : False ,\
#       'per_frame' : True\
#       }}" 


# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1  python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding': False},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_with_AITW_ep3_lr1e-05_wd0_and_temporal_embeddingAITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'llamatouch',\
#       'bbox' : False ,\
#       'per_frame' : False\
#       }}" 

# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1  python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':False, 'context_not_share_embedding': False},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'llamatouch',\
#       'bbox' : False ,\
#       'per_frame' : False\
#       }}" 



CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1  python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
'param':{'num_image_with_embedding':8,'paddle': True , 'streaming':False, 'context_not_share_embedding': True},\
'args' :{ \
      'num_workers':0, \
      'Pix2Struct':False,\
      'use_dif_lr': True,\
      'wd':0.0001,     \
      'lr':1e-5,    \
      'epoch':1,    \
      'bs':32,     \
      'acc_step':8, \
      'pat':2,      \
      'load_path':'/data/pauline/checkpoint/text_encoder_multi_AITW_ep3_lr1e-05_wd0.0001_im8_paddle_True_streaming_False_with_bbox.ckpt',\
      'exp_name' :'llamatouch',\
      'bbox' : True ,\
      'per_frame' : False\
      }}" 

CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1  python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
'param':{'num_image_with_embedding':8,'paddle': True , 'streaming':False, 'context_not_share_embedding': True},\
'args' :{ \
      'num_workers':0, \
      'Pix2Struct':False,\
      'use_dif_lr': True,\
      'wd':0.0001,     \
      'lr':1e-5,    \
      'epoch':1,    \
      'bs':32,     \
      'acc_step':8, \
      'pat':2,      \
      'load_path':'/data/pauline/checkpoint/text_encoder_multi_AITW_ep3_lr1e-05_wd0.0001_im8_paddle_True_streaming_False_without_bbox.ckpt',\
      'exp_name' :'llamatouch',\
      'bbox' : False ,\
      'per_frame' : False\
      }}" 

# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1 python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':False, 'context_not_share_embedding':False },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW',\
#       'per_frame' : False, \
#       'bbox' : False\
#       }}" 

# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1 python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding':False },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'AITW', \
#       'per_frame' : True, \
#       'bbox' : False\
#       }}" 

# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding':False },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_with_AITW_ep3_lr1e-05_wd0_and_temporal_embedding_and_feauture_from_each_image_AITW_ep2_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'bugs', \
#       'per_frame' : True\
#       }}" 

# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding':False },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_with_AITW_ep3_lr1e-05_wd0_and_temporal_embeddingAITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'bugs', \
#       'per_frame' : False\
#       }}" 

# python -m generativeimage2text.streaming_train -p "{'type': 'few_shot_finetune_and_evaluate', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding':False },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'pretrain_with_AITW_ep3_lr1e-05_wd0_and_temporal_embeddingAITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'exp_name' :'bugs', \
#       'per_frame' : False,\
#       'bbox' : False\
#       }}" 

# python -m generativeimage2text.streaming_train -p "{'type': 'few_shot_finetune_and_evaluate', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding':True },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':True,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':9,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'pretrain_with_text_encoder_multi_AITW_ep3_lr1e-05_wd0_and_temporal_embedding_and_with_bbox_AITW_ep2_lr1e-05_wd0.0001_im8_paddle_True_streaming_True.ckpt',\
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'exp_name' :'bugs', \
#       'per_frame' : False,\
#       'bbox' : True\
#       }}"

# python -m generativeimage2text.streaming_train -p "{'type': 'few_shot_finetune_and_evaluate', \
# 'param':{'num_image_with_embedding':8,'paddle':True, 'streaming':True, 'context_not_share_embedding':True },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':True,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':9,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'pretrain_with_text_encoder_multi_AITW_ep3_lr1e-05_wd0_and_temporal_embedding_and_without_bbox_AITW_ep2_lr1e-05_wd0.0001_im8_paddle_True_streaming_True.ckpt',\
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'exp_name' :'bugs', \
#       'per_frame' : False,\
#       'bbox' : False\
#       }}"

# python -m generativeimage2text.streaming_train -p "{'type': 'few_shot_finetune_and_evaluate', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding':False },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':True,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':9,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'pretrain_with_AITW_ep3_lr1e-05_wd0_and_temporal_embedding_and_feauture_from_each_image_AITW_ep2_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'exp_name' :'bugs', \
#       'per_frame' : True,\
#       'bbox' : False\
#       }}"


# python -m generativeimage2text.streaming_train -p "{'type': 'few_shot_finetune_and_evaluate', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True, 'context_not_share_embedding':False },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':True,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':9,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'pretrain_with_AITW_ep3_lr1e-05_wd0_and_temporal_embeddingAITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'exp_name' :'bugs', \
#       'per_frame' : False,\
#       'bbox' : False\
#       }}"

# python -m generativeimage2text.streaming_train -p "{'type': 'few_shot_finetune_and_evaluate', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':False, 'context_not_share_embedding':False },\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':True,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':9,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'AITW_ep3_lr1e-05_wd0.0001_im8.ckpt',\
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'exp_name' :'bugs', \
#       'per_frame' : False,\
#       'bbox' : False\
#       }}"

# python -m generativeimage2text.streaming_train -p "{'type': 'mytrain', \
# 'param':{'num_image_with_embedding':8,'paddle':True, 'streaming':True},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/text_encoder_multi_AITW_ep3_lr1e-05_wd0.0001_im8_paddle_True_streaming_False.ckpt',\
#       'exp_name' :'AITW'\
#       }}" 


# python -m generativeimage2text.multi_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':True, 'streaming':False, 'context_not_share_embedding' : True},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/text_encoder_multi_AITW_ep3_lr1e-05_wd0.0001_im8_paddle_True_streaming_False_without_bbox.ckpt',\
#       'exp_name' :'AITW'\
#       }}" 