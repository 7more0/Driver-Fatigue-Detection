from keras import layers
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import sgd
import numpy as np

def fine_tune_setup(base_model, fixed_layers=100):
    # fine tune model from Inception block 6
    GAP_LAYER = fixed_layers
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True

def transfer_learning_setup(base_model):
    for layer in base_model.layers:
        layer.trainable = False

def Inception_model():
    inception = InceptionV3(weights='imagenet',include_top=False)

    inception_out = inception.output
    x = layers.GlobalAveragePooling2D()(inception_out)
    x = layers.Dense(1024, activation='relu', name='feature_out')(x)
    predictions = layers.Dense(2, activation='sigmoid')(x)
    model = Model(inputs=inception.input, outputs=predictions)

    # transfer_learning_setup(inception)
    fine_tune_setup(inception)
    return model

def LSTM_model(input_dim=90, feature_dim=1024, output_dim=2):
    lstm_input = layers.Input(shape=(input_dim, feature_dim), name='feature_input')
    lstm = layers.LSTM(input_dim)(lstm_input)
    cls_out = layers.Dense(output_dim, activation='sigmoid')(lstm)
    # lstm.add(layers.Dense(2, activation='sigmoid'))
    lstmModel = Model(inputs=lstm_input, outputs=cls_out)
    # lstmModel.compile(optimizer=Adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    return lstmModel


def End_to_end_model(time_step, feature_dim, output_dim):
    frames_input = layers.Input(shape=(None, 299, 299, 3), dtype='float32', name='frame_input')

    InceptionModel = Inception_model()
    InceptionModel.load_weights('./out/Inception_weight.h5')
    feature_model = Model(inputs=InceptionModel.input, outputs=InceptionModel.get_layer('feature_out').output)
    # apply feature model to each time step of input
    feature_output = layers.TimeDistributed(feature_model, input_shape=(299, 299, 3))(frames_input)

    # apply lstm model to extracted feature
    lstmModel = LSTM_model(time_step, feature_dim, output_dim)
    # mask padding value
    # feature_output = layers.Masking(mask_value=0., input_shape=(time_step, feature_dim))(feature_output)
    model_output = lstmModel(inputs=feature_output)

    end_to_end_model = Model(inputs=frames_input, outputs=model_output)
    end_to_end_model._get_distribution_strategy = lambda: None

    return end_to_end_model


if __name__ == '__main__':
    End_to_End_Model = End_to_end_model(90, 1024, 2)
    End_to_End_Model.compile(optimizer=sgd(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])





