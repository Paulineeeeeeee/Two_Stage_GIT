# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW'\
#       }}" 


# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':3}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/LLAMATOUCH_ep9_lr1e-05_wd0.0001_im3.ckpt',\
#       'exp_name' :'LLAMATOUCH'\
#       }}"

# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':1}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/llava_ep9_lr1e-05_wd0.0001_im1.ckpt',\
#       'exp_name' :'llava'\
#       }}" 


# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_llava_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'llava'\
#       }}" 

# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_llava_ep9_lr1e-05_wd0.0001_im8.ckpt_AITW_1000_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW_1000'\
#       }}" 

# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_llava_ep9_lr1e-05_wd0.0001_im8.ckpt_LLAMATOUCH_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'LLAMATOUCH'\
#       }}" 
# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/streaming_LLAMATOUCH_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'LLAMATOUCH'\
#       }}" 

# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/streaming_AITW_1000_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW_<=8_1000'\
#       }}"
# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/streaming_AITW_1000_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW_>8_and_<=16_1000'\
#       }}" 
# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/streaming_AITW_1000_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW_>16_1000'\
#       }}"  
# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_llava_ep9_lr1e-05_wd0.0001_im8.ckpt_AITW_1000_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW_>16_1000'\
#       }}" 
# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8}, \
# 'args' :{ \
#       'num_workers':4, \
#       'Pix2Struct':False,\
#       'use_dif_lr': True,\
#       'wd':0.0001,     \
#       'lr':1e-5,    \
#       'epoch':1,    \
#       'bs':32 ,     \
#       'acc_step':8, \
#       'pat':2,      \
#       'load_path':'/data/pauline/checkpoint/pretrain_llava_ep9_lr1e-05_wd0.0001_im8.ckpt_AITW_1000_ep9_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW_<=8_1000'\
#       }}" 
# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True},\
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
#       'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'auto'\
#       }}" 

# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':False},\
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
#       'load_path':'/data/pauline/checkpoint/_dataAITW_ep3_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'auto'\
#       }}"

# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True},\
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
#       'load_path':'/data/pauline/checkpoint/AITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'AITW'\
#       }}" 

# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':False},\
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
#       'load_path':'/data/pauline/checkpoint/_dataAITW_ep3_lr1e-05_wd0.0001_im8.ckpt',\
#       'exp_name' :'AITW'\
#       }}"

# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True},\
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
#       'load_path':'/data/pauline/checkpoint/pretrain_bugs_ep9_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'bugs'\
#       }}" 

# python -m generativeimage2text.train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':False},\
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
#       'load_path':'/data/pauline/checkpoint/pretrain_bugs_ep9_lr1e-05_wd0.0001_im8_paddle_False_streaming_False.ckpt',\
#       'exp_name' :'bugs'\
#       }}" 

# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True},\
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
#       'load_path':'/data/pauline/checkpoint/pretrain_AITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'AITW'\
#       }}" 
# python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
# 'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True},\
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
#       'load_path':'/data/pauline/checkpoint/pretrain_AITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
#       'exp_name' :'bugs'\
#       }}" 

python -m generativeimage2text.streaming_train -p "{'type': 'myinfer', \
'param':{'num_image_with_embedding':8,'paddle':False, 'streaming':True},\
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
      'load_path':'/data/pauline/checkpoint/pretrain_AITW_ep3_lr1e-05_wd0.0001_im8_paddle_False_streaming_True.ckpt',\
      'exp_name' :'AITW'\
      }}" 
