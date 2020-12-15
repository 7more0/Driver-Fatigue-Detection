from model import Inception_model, LSTM_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adagrad, Adam
from keras.applications.inception_v3 import preprocess_input
import os
import numpy as np
from keras.models import Model
from data_proc import clip_generator, read_data
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# data read and preprocess
# train_data = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.3,
#     shear_range=0.1,
#     zoom_range=0.3,
#     horizontal_flip=True,
# )
# val_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.3,
#     shear_range=0.1,
#     zoom_range=0.3,
#     horizontal_flip=True,
# )

# train_generator = train_data.flow_from_directory(directory='../../Datasets/YawDD/Train/CNN/train',
#                                                  target_size=(299, 299),
#                                                  batch_size=64)
# val_generator = val_datagen.flow_from_directory(directory='../../Datasets/YawDD/Train/CNN/val',
#                                                 target_size=(299, 299),
#                                                 batch_size=64)
# model initialization
# Inception model
# InceptionModel = Inception_model()
# InceptionModel.summary()
# # plot_model(InceptionModel, './Inception.png')
# InceptionModel.compile(optimizer=adam(lr=0.01, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
callback_list = [
    ModelCheckpoint(filepath='./out/Inception_weight.h5', monitor='val_loss', save_best_only=True),
    TensorBoard(log_dir='./out/tensorboard', histogram_freq=1, embeddings_freq=1)
]
# if 'Inception_weight.h5' in os.listdir('./out/'):
#     InceptionModel.load_weights('./out/Inception_weight.h5')
#     print('Load weights from former training.')
# train Inception net
# InceptionModel._get_distribution_strategy = lambda: None
# Inception_model_history = InceptionModel.fit_generator(generator=train_generator,
#                                                        steps_per_epoch=128,
#                                                        epochs=64,
#                                                        validation_data=val_generator,
#                                                        validation_steps=32,
#                                                        class_weight='auto',
#                                                        shuffle=True,
#                                                        callbacks=callback_list
#                                                        )
# InceptionModel.save('./out/Inception_weight_final.h5')
# with open('./out/inception_history.pkl', 'wb') as file:
#     pickle.dump(Inception_model_history.history, file)
#     file.close()

# LSTM model
lstmModel = LSTM_model()
# lstmModel.compile(optimizer=adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

# lstm train
InceptionModel = Inception_model()
InceptionModel.load_weights('./out/Inception_weight.h5')
feature_model = Model(inputs=InceptionModel.input, outputs=InceptionModel.get_layer('feature_out').output)
# feature_model.compile(optimizer=adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
read_path = 'E:/System/Desktop/fatigue-drive-yawning-detection-master/dataset/train_lst/train_video/train/video/'
frame_path = 'E:/System/Desktop/fatigue-drive-yawning-detection-master/dataset/train_lst/train_video/train/train_frames/'
clips = read_data(read_path, frame_path, 90)
labels = {'normal': np.array([1, 0]),
          'yawning': np.array([0, 1])}
clip_data_generator = clip_generator(clips, frame_path, feature_model, labels, batch_size=2, img_size=(299, 299))

lstmModel.fit_generator(
                        clip_data_generator,
                        steps_per_epoch=1,
                        epochs=2,
                        callbacks=callback_list
                        )


# sample_clip, clip_label = clip_generator()
# feature = feature_model.predict_on_batch(sample_clip)
# lstm_loss = lstmModel.train_on_batch(feature, clip_label)






pass
