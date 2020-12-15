from opr_tools import video_to_imgs
import os
import shutil
import numpy as np
import cv2
import random
# from keras.utils import Sequence


def video_seg_imgs(folder, out_path):
    for video in os.listdir(folder):
        video_to_imgs(folder, video, out_path, frame_gap=5)

    return True

def read_data(read_path, frame_path, clip_length):
    '''
        sort video data and relate frames to videos
    :param read_path: videos path
    :param frame_path: frames path
    :param clip_length: least frame number of clip
    :return:
        clips: dict{'class1': ['ClipsOfClass1': [ListOfClipFrames]]}
    '''
    clips = {}
    all_class_clips = os.listdir(read_path)
    for class_clips in all_class_clips:
        clip_list = os.listdir(os.path.join(read_path, class_clips))
        frame_list = os.listdir(os.path.join(frame_path, class_clips))
        clips[class_clips] = {}
        for clip in clip_list:
            clips[class_clips][clip] = [frame for frame in frame_list if clip.strip('.avi') in frame]

    return clips

def clip_generator(clips_dict, frame_path, feature_net, labels, batch_size=2, img_size=(299, 299)):
    '''
        yield [frames, label] of one clip a time.
    :return:
    '''
    cls_n = len(clips_dict)
    clip_index = 0
    clip_samples = {}
    for cls_key in clips_dict.keys():
        clip_samples[cls_key] = list(clips_dict[cls_key].keys())
    while True:
        sample_batch = []
        label_batch = []
        for cls, clip_list in clips_dict.items():
            # travel all classes
            # select batch_size/class_num sample clip for class cls sample by index
            cls_clips_count = clip_samples[cls][clip_index: clip_index+(batch_size//cls_n)]
            clip_index += batch_size//cls_n
            if clip_index>=len(clips_dict[cls]):
                clip_index = 0
            for clip in cls_clips_count:
                # extract feature vector for each frame in sample clip
                clip_frames = clip_list[clip]
                frames = []
                for frame in clip_frames:
                    frame = ((cv2.resize(cv2.imread(os.path.join(frame_path, cls, frame)), (299, 299), interpolation=cv2.INTER_LINEAR)/255)-0.5)*2
                    frames.append(frame)
                frames = np.array(frames)
                feature = feature_net.predict_on_batch(frames)
                # feature = np.zeros((90, 1024))
                sample_batch.append(feature)
                label_batch.append(labels[cls])

        sample_batch = np.array(sample_batch)
        label_batch = np.array(label_batch)

        yield sample_batch, label_batch

    return clip_generator


if __name__ == '__main__':
    clip_path = 'E:/System/Desktop/fatigue-drive-yawning-detection-master/dataset/train_lst/train_video/val/yawning/'
    frame_path = 'E:/System/Desktop/fatigue-drive-yawning-detection-master/dataset/train_lst/train_video/val/val_frames/yawning/'
    count = 0
    # for clip in os.listdir(clip_path):
    #     if cv2.VideoCapture(clip_path+clip).get(7)>=90:
    #         video_to_imgs(clip_path, clip, out_path=frame_path, frame_num=90)
    #         count+=1
    # print(count)
    read_path = 'E:/System/Desktop/fatigue-drive-yawning-detection-master/dataset/train_lst/train_video/train/video/'
    frame_path = 'E:/System/Desktop/fatigue-drive-yawning-detection-master/dataset/train_lst/train_video/train/train_frames/'
    clips = read_data(read_path, frame_path, 90)
    labels = {'normal': np.array([1, 0]),
              'yawning': np.array([0, 1])}
    feature_model = 0
    generator = clip_generator(clips, frame_path, feature_model, labels)
    while True:
        test = next(generator)
        print(test[0].shape, test[1].shape)
    # labels = {'normal': np.array([1, 0, 0]),
    #           'talking': np.array([0, 1, 0]),
    #           'yawning': np.array([0, 0, 1])}
    pass