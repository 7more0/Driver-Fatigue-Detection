from model import Inception_model, LSTM_model
import os
import numpy as np
from keras.models import Model
from data_proc import clip_generator, read_data
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
callback_list = [
    ModelCheckpoint(filepath='./out/lstm_weight.h5', monitor='val_loss', save_best_only=True),
    TensorBoard(log_dir='./out/tensorboard', histogram_freq=1, embeddings_freq=1)
]

# LSTM model
lstmModel = LSTM_model()
# lstmModel.compile(optimizer=adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

# lstm train
InceptionModel = Inception_model()
InceptionModel.load_weights('./out/Inception_weight.h5')
feature_model = Model(inputs=InceptionModel.input, outputs=InceptionModel.get_layer('feature_out').output)
# feature_model.compile(optimizer=adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
# initiate train data generator
train_read_path = '../Dataset/YawDD/lstm/train/video'
train_frame_path = '../Dataset/YawDD/lstm/train/train_frames'
clips = read_data(train_read_path, train_frame_path, 90)
labels = {'normal': np.array([1, 0]),
          'yawning': np.array([0, 1])}
clip_data_generator = clip_generator(clips, train_frame_path, feature_model, labels, batch_size=16, img_size=(299, 299))
# initiate val data generator
val_read_path = '../Dataset/YawDD/lstm/val/video'
val_frame_path = '../Dataset/YawDD/lstm/val/val_frames'
val_clips = read_data(val_read_path, val_frame_path, 90)
val_clip_data_generator = clip_generator(val_clips, val_frame_path, feature_model, labels, batch_size=16)
if 'lstm_weight.h5' in os.listdir('./out/'):
    InceptionModel.load_weights('./out/lstm_weight.h5')
    print('Load weights from former training.')
lstmModel_history = lstmModel.fit_generator(
                                            clip_data_generator,
                                            steps_per_epoch=8,
                                            validation_data=val_clip_data_generator,
                                            validation_steps=2,
                                            epochs=32,
                                            callbacks=callback_list
                                            )
lstmModel.save('./out/lstm_weight_final.h5')
with open('./out/lstmModel_history.pkl', 'wb') as file:
    pickle.dump(lstmModel_history.history, file)
    file.close()
