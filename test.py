from model import Inception_model, LSTM_model
import numpy as np
from keras.models import Model
from data_proc import clip_generator, read_data

# detector config
test_frame_path = './Datasets/YawDD/test/test_frames'
test_video_path = './Datasets/YawDD/test/test_videos'
clip_length = 90
batch_size = 4
labels = labels = {'normal': np.array([1, 0], dtype='float32'),
                   'yawning': np.array([0, 1], dtype='float32')}
# load data
clips = read_data(test_video_path, test_frame_path, clip_length=clip_length)
sample_num = np.sum(np.array([len(clip_list) for cls, clip_list in clips.items()]))
clip_gen = clip_generator(clips, test_frame_path, labels, batch_size=batch_size)

# model build
InceptionModel = Inception_model()
InceptionModel.load_weights('./out/Inception_weight.h5')
feature_model = Model(inputs=InceptionModel.input, outputs=InceptionModel.get_layer('feature_out').output)
lstmModel = LSTM_model()
lstmModel.load_weights('./out/lstm_weight.h5')

# test
test_step = 0
tp_sample = 0
p_sample = 0
sample_count = 0
while sample_count<=sample_num:
    clip_batch, clip_batch_label = next(clip_gen)
    clip_feature_batch = []
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

    sample_count += batch_size

test_acc = tp_sample/sample_count
print('Test accuracy: {:.2f}%'.format(test_acc*100))



