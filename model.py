from keras import layers
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import adam

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

