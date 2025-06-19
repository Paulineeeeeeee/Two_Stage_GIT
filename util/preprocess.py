import os
import tensorflow as tf
import pickle
import jsonlines
from tqdm import tqdm
from PIL import Image

def _decode_image(example):
    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )
    image_height = example.features.feature['image/height'].int64_list.value[0]
    image_width = example.features.feature['image/width'].int64_list.value[0]
    image_channels = example.features.feature['image/channels'].int64_list.value[0]

    height = tf.cast(image_height, tf.int32)
    width = tf.cast(image_width, tf.int32)
    n_channels = tf.cast(image_channels, tf.int32)

    return tf.reshape(image, (height, width, n_channels)).numpy()

def find_miss_file(dataset, name):
    count = {}
    total = {}
    miss_img_set = set()

    for d in tqdm(dataset):
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        step_id = ex.features.feature['step_id'].int64_list.value[0]
        ep_length = ex.features.feature['episode_length'].int64_list.value[0]

        if ep_id not in count:
            count[ep_id] = 1
            total[ep_id] = ep_length
        else:
            count[ep_id] += 1

    for k, v in count.items():
        if v != total[k]:
            miss_img_set.add(k)

    with open(name, 'wb') as handle:
        pickle.dump(miss_img_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_images(dataset, des_no_miss, des_miss, miss_set, jsonl_name, category):
    with jsonlines.open(jsonl_name, mode='a') as writer:
        for d in tqdm(dataset):
            ex = tf.train.Example()
            ex.ParseFromString(d)
            ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
            step_id = ex.features.feature['step_id'].int64_list.value[0]
            ep_length = ex.features.feature['episode_length'].int64_list.value[0]
            image = _decode_image(ex)

            if ep_id in miss_set:
                img_path = f"{des_miss}{ep_id}-{step_id}-{ep_length}.jpg"
            else:
                img_path = f"{des_no_miss}{ep_id}-{step_id}-{ep_length}.jpg"

                if step_id == 0:
                    goal_info = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
                    writer.write({"category": category, "id": ep_id, "goal_info": goal_info, "episode_length": ep_length})

            Image.fromarray(image).save(img_path)

def main():
    dir_list = ['general', 'google_apps', 'install', 'web_shopping']
    
    for category in dir_list:
        data_src = f'/home/pauline/android-in-the-wild/android-in-the-wild/{category}/'
        des_no_miss = f'/home/pauline/no-miss-AITW/{category}/'
        des_miss = f'/home/pauline/miss-AITW/{category}/'
        jsonl_name = f'no_miss_{category}_train.jsonl'
        pkl_name = f'AiTW_Miss_Img_ID_{category}.pickle'

        # 创建存储目录
        os.makedirs(des_no_miss, exist_ok=True)
        os.makedirs(des_miss, exist_ok=True)

        files_in_directory = os.listdir(data_src)
        train_files = [(data_src + i) for i in files_in_directory]

        raw_dataset = tf.data.TFRecordDataset(train_files, compression_type='GZIP').as_numpy_iterator()

        if not os.path.exists(pkl_name):
            find_miss_file(raw_dataset, pkl_name)

        with open(pkl_name, 'rb') as handle:
            loaded_set = pickle.load(handle)
        
        save_images(raw_dataset, des_no_miss, des_miss, loaded_set, jsonl_name, category)


if __name__ == '__main__':
    main()
