from model import Inception_model, LSTM_model
import os
from datetime import datetime
from keras.optimizers import adam
from data_proc import clip_generator, read_data, feature_generator
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
from ModelConfig import ModelConfig
'''
    train lstm model for sequential feature learning.
'''
# training config
model_config = ModelConfig()
train_feature_path = './out/train_clip_feature_sample_10.pkl'      # feature file path
val_feature_path = './out/val_clip_feature_sample_10.pkl'
feature_batch_size = model_config.feature_batchsize
clip_feature_size = (model_config.lstm_step, model_config.cnn_feature_dim)      # (lstm input node dim, feature dim)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

callback_list = [
    ModelCheckpoint(filepath='./out/lstm_weight_10.h5', monitor='val_loss', save_best_only=True),
    TensorBoard(log_dir='./out/tensorboard/'+ 'lstm_10_'+ datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, embeddings_freq=1)
]
# load data
with open(train_feature_path, 'rb') as file:
    train_feature_data = pickle.load(file)
    file.close()
with open(val_feature_path, 'rb') as file:
    val_feature_data = pickle.load(file)
    file.close()
labels = model_config.labels
train_feature_generator = feature_generator(train_feature_data, labels=labels, batch_size=feature_batch_size,
                                            clip_feature_shape=clip_feature_size)
val_feature_generator = feature_generator(val_feature_data, labels, batch_size=feature_batch_size, clip_feature_shape=clip_feature_size)
# LSTM model
lstmModel = LSTM_model(model_config.lstm_step)
lstmModel.summary()
lstmModel._get_distribution_strategy = lambda: None
lstmModel.compile(optimizer=adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

# lstm train
lstmModel_history = lstmModel.fit_generator(
    train_feature_generator,
    steps_per_epoch=4,
    validation_data=val_feature_generator,
    validation_steps=1,
    epochs=64,
    callbacks=callback_list
)
# lstmModel.save('./out/lstm_weight_final_2.h5')
# with open('./out/lstmModel_history_2.pkl', 'wb') as file:
#     pickle.dump(lstmModel_history.history, file)
#     file.close()
