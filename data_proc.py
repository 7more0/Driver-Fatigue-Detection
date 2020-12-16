from opr_tools import video_to_imgs, dataset_divide, video_length_check
import os
import shutil
import numpy as np
import cv2
import random
import pickle

'''
    Data preprocessing.
'''

def video_seg_imgs(folder, out_path):
    '''
        Convert videos in folder to frames. Write frames to out_path.
    '''
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


def clip_generator(clips_dict, frame_path, labels, batch_size=2, img_size=(299, 299)):
    '''
        yield [frames, label] of one clip a time for feature extracting network.
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
            cls_clips_count = clip_samples[cls][clip_index: clip_index + (batch_size // cls_n)]
            clip_index += batch_size // cls_n
            if clip_index >= len(clips_dict[cls]):
                clip_index = 0
            for clip in cls_clips_count:
                # extract feature vector for each frame in sample clip
                clip_frames = clip_list[clip]
                frames = []
                for frame in clip_frames:
                    frame = ((cv2.resize(cv2.imread(os.path.join(frame_path, cls, frame)), (299, 299),
                                         interpolation=cv2.INTER_LINEAR) / 255) - 0.5) * 2
                    frames.append(frame)
                frames = np.array(frames)
                # feature = feature_net.predict_on_batch(frames)
                # feature = np.zeros((90, 1024))
                sample_batch.append(frames)
                label_batch.append(labels[cls])

        sample_batch = np.array(sample_batch)
        label_batch = np.array(label_batch)

        yield sample_batch, label_batch

    return clip_generator


def feature_generator(clips_feature, labels, batch_size=16, clip_feature_shape=(90, 1024)):
    '''
        yield batch size clips in frames' feature form for lstm training.
    :param clips_feature: feature dictionary load from file.
    :param labels: class labels, dict
    :param clip_feature_shape: presumed yielded feature shape as (frames, frame feature shape)
    '''
    cls_n = len(clips_feature)
    clip_index = 0
    clean_feature = {}
    # data cleaning, delete feature data with wrong shape
    for cls, clips_feature_list in clips_feature.items():
        clean_feature[cls] = []
        for clip_feature in clips_feature_list:
            if clip_feature.shape == clip_feature_shape:
                clean_feature[cls].append(clip_feature)
    clips_feature = clean_feature
    while True:
        sample_batch = []
        label_batch = []
        for cls, clip_feature_list in clips_feature.items():
            # travel all classes
            # select batch_size/class_num sample clip for class cls sample by index
            sel_clips_feature = clip_feature_list[clip_index: clip_index + (batch_size // cls_n)]
            clip_index += batch_size // cls_n
            if clip_index >= (len(clips_feature[cls]) - batch_size):
                # yield samples more than dataset samples
                clip_index = 0
            cls_labels = [labels[cls] for i in range(batch_size // cls_n)]

            sample_batch.extend(sel_clips_feature)
            label_batch.extend(cls_labels)

        sample_batch = np.stack(sample_batch, axis=0)
        label_batch = np.stack(label_batch, axis=0)

        yield sample_batch, label_batch
    return feature_generator


if __name__ == '__main__':

    pass
