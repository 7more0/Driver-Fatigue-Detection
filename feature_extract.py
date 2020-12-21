import numpy as np
import os
from keras.models import Model
from data_proc import clip_generator, read_data
import pickle
from model import Inception_model
from ModelConfig import ModelConfig
'''
    Extract frame feature with retrained Inception model.
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
InceptionModel = Inception_model()
InceptionModel.load_weights('./out/Inception_weight.h5')
feature_model = Model(inputs=InceptionModel.input, outputs=InceptionModel.get_layer('feature_out').output)

# train data path config
model_config = ModelConfig()
read_path = './Datasets/YawDD/train/lstm/train/video'
frame_path = './Datasets/YawDD/train/lstm/train/train_frames'
frames_per_video = model_config.lstm_step       # input dimension of lstm node and represents the length of video representation learning
pro_batch_size = 8
labels = model_config.labels

clips = read_data(read_path, frame_path, frames_per_video, sample_ratio=0.5)
sample_num = np.sum(np.array([len(clip_list) for cls, clip_list in clips.items()]))
clip_data_generator = clip_generator(clips, frame_path, labels, batch_size=pro_batch_size, img_size=model_config.cnn_input_shape)

# extract all clips' feature
step_count = 0
clip_feature_sample = {'normal':[], 'yawning':[]}
while step_count*pro_batch_size<=sample_num:
    # (batch_size, 90, 299, 299, 3), (batch_size, 2)
    clip_batch, clip_labels = next(clip_data_generator)
    for clip_ind in range(pro_batch_size):
        # (90, 1024)
        clip_feature = feature_model.predict_on_batch(clip_batch[clip_ind])
        if list(clip_labels[clip_ind]).index(1)==0:
            clip_feature_sample['normal'].append(clip_feature)
        else:
            clip_feature_sample['yawning'].append(clip_feature)
    step_count += 1

with open('./out/train_clip_feature_sample_10.pkl', 'wb') as file:
    pickle.dump(clip_feature_sample, file)
    print('train data:')
    print('normal:', len(clip_feature_sample['normal']))
    print('yawning:', len(clip_feature_sample['yawning']))
    file.close()


# val data path config
read_path = './Datasets/YawDD/train/lstm/val/video'
frame_path = './Datasets/YawDD/train/lstm/val/val_frames'

clips = read_data(read_path, frame_path, frames_per_video, sample_ratio=0.5)
sample_num = np.sum(np.array([len(clip_list) for cls, clip_list in clips.items()]))
clip_data_generator = clip_generator(clips, frame_path, labels, batch_size=pro_batch_size, img_size=model_config.cnn_input_shape)

# extract all clips' feature
step_count = 0
clip_feature_sample = {'normal':[], 'yawning':[]}
while step_count*pro_batch_size<=sample_num:
    # (batch_size, 90, 299, 299, 3), (batch_size, 2)
    clip_batch, clip_labels = next(clip_data_generator)
    for clip_ind in range(pro_batch_size):
        # (90, 1024)
        try:
            clip_feature = feature_model.predict_on_batch(clip_batch[clip_ind])
            if list(clip_labels[clip_ind]).index(1)==0:
                clip_feature_sample['normal'].append(clip_feature)
            else:
                clip_feature_sample['yawning'].append(clip_feature)
        except:
            continue
    step_count += 1

with open('./out/val_clip_feature_sample_10.pkl', 'wb') as file:
    pickle.dump(clip_feature_sample, file)
    print('val data:')
    print('normal:', len(clip_feature_sample['normal']))
    print('yawning:', len(clip_feature_sample['yawning']))
    file.close()

