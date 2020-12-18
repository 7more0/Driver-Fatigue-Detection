# Driver-Fatigue-Detection

Driver fatigue detection model using Inception v3 and LSTM. Train and test on YawDD dataset.  
This project contains a deep neural network model for driver drowsiness detection using video and facial feature.  
No human face or facial organ detection is required in this project.

## Requirement
    python==3.7
    keras==2.3.1
    tensorflow-gpu==2.1.0
    opencv, numpy

## Data preprocessing
* split original videos with `yawn_split_video.py` in `Datasets/YawDD/seg_list/`.
* divide training data to train and val dataset with tools in `opr_tools.py`.
#### for Inception model training
* split videos into frames with `data_proc.py` functions.

#### for LSTM model training
* run `feature_extract.py` to extract frame features of all training data.


## Training
2. run `cnn_train.py` to train Inception model. In this case, I used only frames from videos with normal/yawning label.
2. run `lstm_train.py` to train LSTM model.

## Notice
* trained model weights and preprocessed data will be write in `./out`
* log path of TensorBoard is `./out/tensorboard`

