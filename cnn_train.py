from model import Inception_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from keras.applications.inception_v3 import preprocess_input
import os
from datetime import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
from ModelConfig import ModelConfig
'''
    train Inception model for frame feature extraction.
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# data read and preprocess
model_config = ModelConfig()
train_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=model_config.data_augment['rotation_range'],
    width_shift_range=model_config.data_augment['width_shift_range'],
    height_shift_range=model_config.data_augment['height_shift_range'],
    shear_range=model_config.data_augment['shear_range'],
    zoom_range=model_config.data_augment['zoom_range'],
    horizontal_flip=model_config.data_augment['horizontal_flip'],
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=model_config.data_augment['rotation_range'],
    width_shift_range=model_config.data_augment['width_shift_range'],
    height_shift_range=model_config.data_augment['height_shift_range'],
    shear_range=model_config.data_augment['shear_range'],
    zoom_range=model_config.data_augment['zoom_range'],
    horizontal_flip=model_config.data_augment['horizontal_flip'],
)

train_generator = train_data.flow_from_directory(directory='./Datasets/YawDD/Train/CNN/train',
                                                 target_size=model_config.cnn_input_shape,
                                                 batch_size=model_config.cnn_frame_data_batchsize)
val_generator = val_datagen.flow_from_directory(directory='./Datasets/YawDD/Train/CNN/val',
                                                target_size=model_config.cnn_input_shape,
                                                batch_size=model_config.cnn_frame_data_batchsize)

# Inception model
InceptionModel = Inception_model()
InceptionModel.summary()
# plot_model(InceptionModel, './Inception.png')
InceptionModel.compile(optimizer=adam(lr=0.01, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
callback_list = [
    ModelCheckpoint(filepath='./out/Inception_weight.h5', monitor='val_loss', save_best_only=True),
    TensorBoard(log_dir='./out/tensorboard/'+ 'cnn'+ datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, embeddings_freq=1)
]
if 'Inception_weight.h5' in os.listdir('./out/'):
    InceptionModel.load_weights('./out/Inception_weight.h5')
    print('Load weights from former training.')
# train Inception net
InceptionModel._get_distribution_strategy = lambda: None
Inception_model_history = InceptionModel.fit_generator(generator=train_generator,
                                                       steps_per_epoch=128,
                                                       epochs=64,
                                                       validation_data=val_generator,
                                                       validation_steps=32,
                                                       class_weight='auto',
                                                       shuffle=True,
                                                       callbacks=callback_list
                                                       )
InceptionModel.save('./out/Inception_weight_final.h5')
with open('./out/inception_history.pkl', 'wb') as file:
    pickle.dump(Inception_model_history.history, file)
    file.close()


