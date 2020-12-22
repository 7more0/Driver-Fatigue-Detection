# Driver-Fatigue-Detection

Driver fatigue detection model using Inception v3 and LSTM. Model was trained and tested on YawDD dataset.  
This project contains a deep neural network model for driver drowsiness detection using video and facial feature.  
No human face or facial organ detection is required.

## Requirement
    python==3.7
    keras==2.3.1
    tensorflow-gpu==2.1.0
    opencv, numpy

## Data preprocessing
1. split original videos with `yawn_split_video.py` in `Datasets/YawDD/seg_list/`.
2. divide training data to train and val dataset with tools in `opr_tools.py`.
#### for Inception model training
* split videos into frames with `data_proc.py` functions.
#### for LSTM model training
* run `feature_extract.py` to extract frame features of all training data.
#### preprocessed data structure
    YawDD/
        train/
            CNN/
                train/
                    normal/
                    yawning/
                val/
                    normal/
                    yawning/
            lstm/
                train/
                    train_video/
                        normal/
                        ...
                    train_frame/
                        normal/
                        ...
                val/
                    val_video/
                        normal/
                        ...
                    val_frame/
                        normal/
                        ...
        test/
            test_videos/
                normal/
                ...
            test_frames/
                normal/
                ...
        

## Train
1. run `cnn_train.py` to train Inception model. In this case, I used only frames from videos with normal/yawning label.
2. run `lstm_train.py` to train LSTM model.

## Test
1. run `test.py` to test model on test dataset.

## End to end model
1. to train the end to end model, configure model in `ModelConfig.py` and then run `Implement.py`.

## Notice
* trained model weights and preprocessed data will be writen in `./out`
* log path of TensorBoard is `./out/tensorboard`
* end to end model training will require large amount of computation resources, multi-gpu training is recommended.

