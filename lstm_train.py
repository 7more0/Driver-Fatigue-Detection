from model import Inception_model, LSTM_model
import os
import numpy as np
from keras.optimizers import adam
from data_proc import clip_generator, read_data, feature_generator
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
'''
    train lstm model for sequential feature learning.
'''
# training config
train_feature_path = './out/train_clip_feature_sample.pkl'      # feature file path
val_feature_path = './out/val_clip_feature_sample.pkl'
feature_batch_size = 8
clip_feature_size = (90, 1024)      # (lstm input node dim, feature dim)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

callback_list = [
    ModelCheckpoint(filepath='./out/lstm_weight.h5', monitor='val_loss', save_best_only=True),
    TensorBoard(log_dir='./out/tensorboard', histogram_freq=1, embeddings_freq=1)
]
# load data
with open(train_feature_path, 'rb') as file:
    train_feature_data = pickle.load(file)
    file.close()
with open(val_feature_path, 'rb') as file:
    val_feature_data = pickle.load(file)
    file.close()
labels = {'normal': np.array([1, 0]),
          'yawning': np.array([0, 1])}
train_feature_generator = feature_generator(train_feature_data, labels=labels, batch_size=feature_batch_size,
                                            clip_feature_shape=clip_feature_size)
val_feature_generator = feature_generator(val_feature_data, labels, feature_batch_size, clip_feature_shape=clip_feature_size)
# LSTM model
lstmModel = LSTM_model()
lstmModel._get_distribution_strategy = lambda: None
lstmModel.compile(optimizer=adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

# lstm train
lstmModel_history = lstmModel.fit_generator(
    train_feature_generator,
    steps_per_epoch=16,
    validation_data=val_feature_generator,
    validation_steps=4,
    epochs=64,
    callbacks=callback_list
)
lstmModel.save('./out/lstm_weight_final.h5')
with open('./out/lstmModel_history.pkl', 'wb') as file:
    pickle.dump(lstmModel_history.history, file)
    file.close()
