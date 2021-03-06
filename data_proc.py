from opr_tools import video_to_imgs, dataset_divide, video_length_check, test_video_length_check
import os
import numpy as np
import cv2
import re
import math
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


def read_data(read_path, frame_path, clip_length, sample_ratio=1):
    '''
        sort video data and relate frames to videos
    :param read_path: videos path
    :param frame_path: frames path
    :param clip_length: least frame number of clip
    :param sample_ratio: sample clip frames per 1/sample_ratio frames
    :return:
        clips: dict{'class1': ['ClipsOfClass1': [ListOfClipFrames]]}
    '''
    clips = {}
    sample_gap = int(1//sample_ratio)
    index_pattern = re.compile(r'(\d*$)')
    all_class_clips = os.listdir(read_path)
    for class_clips in all_class_clips:
        clip_list = os.listdir(os.path.join(read_path, class_clips))
        frame_list = os.listdir(os.path.join(frame_path, class_clips))
        clips[class_clips] = {}
        for clip in clip_list:
            if test_video_length_check(os.path.join(os.path.join(read_path, class_clips), clip)):
                sel_frames = [frame for frame in frame_list if clip.strip('.avi') in frame]
                # extract frame index
                frame_index = [int(re.search(index_pattern, frame.strip('.jpg')).group())
                                  for frame in sel_frames]
                # select frame by sample ratio
                sel_frames = [sel_frames[f_index] for f_index in range(len(frame_index))
                              if frame_index[f_index] % sample_gap==0]
                # sort frames of one clip by frame index
                frame_index = [int(re.search(index_pattern, frame.strip('.jpg')).group())
                               for frame in sel_frames]
                clips[class_clips][clip] = sorted(sel_frames,
                                                  key=lambda i: frame_index[sel_frames.index(i)])
                if len(clips[class_clips][clip])>clip_length:
                    # keep clip frame number fixed
                    clips[class_clips][clip] = clips[class_clips][clip][:clip_length]



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
                    frame = ((cv2.resize(cv2.imread(os.path.join(frame_path, cls, frame)), img_size,
                                         interpolation=cv2.INTER_LINEAR) / 255) - 0.5) * 2
                    frames.append(frame)
                frames = np.array(frames)
                # feature = feature_net.predict_on_batch(frames)
                # feature = np.zeros((90, 1024))
                sample_batch.append(frames)
                label_batch.append(labels[cls])

        sample_batch = np.stack(sample_batch, axis=0)
        label_batch = np.stack(label_batch, axis=0)

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
    # train_read_path = './Datasets/YawDD/train/lstm/train/video'
    # train_frame_path = './Datasets/YawDD/train/lstm/train/train_frames'
    #
    # # read_data(train_read_path, train_frame_path, 90, 1/3)
    # #
    # # test = '39-MaleGlasses-Yawning-clip-088.jpg'
    # # index_mantissa_pattern = re.compile(r'(\d$)')
    # # mantissa = re.search(index_mantissa_pattern, test.strip('.jpg')).group(0)

    pass

