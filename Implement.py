from model import End_to_end_model
from data_proc import clip_generator, read_data
from ModelConfig import ModelConfig
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import os
from keras.utils import multi_gpu_model

# training config
model_config = ModelConfig()
train_read_path = './Datasets/YawDD/train/lstm/train/video'
train_frame_path = './Datasets/YawDD/train/lstm/train/train_frames'
val_read_path = './Datasets/YawDD/train/lstm/val/video'
val_frame_path = './Datasets/YawDD/train/lstm/val/val_frames'
feature_batch_size = model_config.feature_batchsize
clip_feature_size = (model_config.lstm_step, model_config.cnn_feature_dim)  # (lstm input node dim, feature dim)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
callback_list = [
    ModelCheckpoint(filepath='./out/eteModel_weight.h5', monitor='val_loss', save_best_only=True),
    TensorBoard(log_dir='./out/tensorboard/' + 'eteModel' + datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1,
                embeddings_freq=1)
]


# data load
clips = read_data(train_read_path, train_frame_path, model_config.lstm_step, sample_ratio=model_config.sample_ratio)
train_clip_generator = clip_generator(clips, train_frame_path, labels=model_config.labels,
                                      batch_size=model_config.feature_batchsize,
                                      img_size=model_config.cnn_input_shape)
clips = read_data(val_read_path, val_frame_path, model_config.lstm_step, sample_ratio=model_config.sample_ratio)
val_clip_generator = clip_generator(clips, val_frame_path, labels=model_config.labels,
                                    batch_size=model_config.feature_batchsize,
                                    img_size=model_config.cnn_input_shape)

# model initiate
EndToEndModel = End_to_end_model(model_config.lstm_step, model_config.cnn_feature_dim, model_config.output_classes)
# EndToEndModel.compile(optimizer=sgd(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# EndToEndModel.summary()
parallel_model = multi_gpu_model(EndToEndModel, gpus=2)
parallel_model._get_distribution_strategy = lambda: None
parallel_model.compile(optimizer=sgd(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# train
parallel_model.fit_generator(train_clip_generator,
                             steps_per_epoch=8,
                             epochs=128,
                             validation_data=val_clip_generator,
                             validation_steps=2,
                             class_weight='auto',
                             callbacks=callback_list
                             )
