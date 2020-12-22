from model import Inception_model, LSTM_model
import numpy as np
from keras.models import Model
from data_proc import clip_generator, read_data
from ModelConfig import ModelConfig
import time
import os

model_config = ModelConfig()
# detector config
test_frame_path = './Datasets/YawDD/test/test_frames'
test_video_path = './Datasets/YawDD/test/test_videos'
clip_length = model_config.lstm_step
batch_size = model_config.test_sample_batch
labels = model_config.labels
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load data
clips = read_data(test_video_path, test_frame_path, clip_length=clip_length)
sample_num = np.sum(np.array([len(clip_list) for cls, clip_list in clips.items()]))
clip_gen = clip_generator(clips, test_frame_path, labels, batch_size=batch_size)

# model build
InceptionModel = Inception_model()
InceptionModel.load_weights('./out/Inception_weight.h5')
feature_model = Model(inputs=InceptionModel.input, outputs=InceptionModel.get_layer('feature_out').output)
lstmModel = LSTM_model(model_config.lstm_step)
lstmModel.load_weights('./out/lstm_weight.h5')

# test
test_step = 0
tp_sample = 0
p_sample = 0
sample_count = 0
total_time = 0
start_time = time.time()
while sample_count<=sample_num:
    clip_batch, clip_batch_label = next(clip_gen)
    clip_feature_batch = []

    batch_start_time = time.time()
    for clip in clip_batch:
        clip_feature = feature_model.predict_on_batch(clip)
        clip_feature_batch.append(clip_feature)
    clip_feature_batch = np.stack(clip_feature_batch, axis=0)
    lstm_predict = lstmModel.predict_on_batch(clip_feature_batch)
    for clip_index, clip in enumerate(lstm_predict):
        predict_label = max([i for i in range(len(labels))], key=lambda i: lstm_predict[clip_index, i])
        ground_truth = max([i for i in range(len(labels))], key=lambda i: clip_batch_label[clip_index, i])
        if predict_label==ground_truth:
            tp_sample += 1
            sample_count += 1
            print(1)
        else:
            sample_count += 1
            print(0)
    batch_end_time = time.time()

    total_time += batch_end_time-batch_start_time

end_time = time.time()
test_acc = tp_sample/sample_count
print('hit: {}; total: {}'.format(tp_sample, sample_count))
print('Test accuracy: {:.2f}%'.format(test_acc*100))
fps = (sample_count*model_config.lstm_step)/total_time
fps1 = (sample_count*model_config.lstm_step)/(end_time-start_time)
print('Model processing speed: {:.2f} fps'.format(fps1))
print('Average clip processing time: {:.2f} s'.format(model_config.lstm_step/fps1))



