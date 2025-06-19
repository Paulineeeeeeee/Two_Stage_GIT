# # for main model
# python -m generativeimage2text.streaming_train -p "{'type': 'mytrain', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':9,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'exp_name' :'LLAMATOUCH'\
#       }}" 
#       # 'load_path':'/data/cv/poyang/checkpoint/final_2lr_8img_ep1_lr1e-05_wd1e-05_im8.ckpt',\
#       # 'exp_name' :'1000_2lr_8img_lowWD' \

# python -m generativeimage2text.train -p "{'type': 'mytrain', \
# 'param':{ \
#       'num_image_with_embedding':6,\
#       'streaming':False,\
#       'paddle':False,\
#       'context_not_share_embedding': False}, \
# 'args' :{ \
#       'num_workers':4, \
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

# python -m generativeimage2text.train -p "{'type': 'mytrain', \
# 'param':{ \
#       'num_image_with_embedding':7,\
#       'streaming':False,\
#       'paddle':False,\
#       'context_not_share_embedding': False}, \
# 'args' :{ \
#       'num_workers':4, \
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


# python -m generativeimage2text.train -p "{'type': 'mytrain', \
# 'param':{ \
#       'num_image_with_embedding':8,\
#       'streaming':True,\
#       'paddle':False}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':9,    \
#       'bs':16 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'load_path':'AITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'bugs'\
#       }}" 

# python -m generativeimage2text.train -p "{'type': 'mytrain', \
# 'param':{ \
#       'num_image_with_embedding':8,\
#       'streaming':True,\
#       'paddle':False}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':9,    \
#       'bs':16 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'ckpt_path':'/data/pauline/checkpoint/', \
#       'load_path':'AITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'auto'\
#       }}" 

# python -m generativeimage2text.streaming_train -p "{'type': 'mytrain', \
# 'param':{ \
#       'num_image_with_embedding':8,\
#       'streaming':True,\
#       'paddle':False}, \
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
#       'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW'\
#       }}" 
# CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1 python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':7},\
# 'args' :{ \
#       'num_workers':0, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':16,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im7.ckpt',\
#       'exp_name' :'AITW'\
#       }}" 

CUDA_VISIBLE_DEVICES=1  CUDA_LAUNCH_BLOCKING=1 python -m generativeimage2text.train -p "{'type': 'myinfer', \
'param':{'num_image_with_embedding':6,},\
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
      'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im6.ckpt',\
      'exp_name' :'AITW'\
      }}" 