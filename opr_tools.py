'''
    processing tools for data processing.
'''
import numpy as np
import random
import cv2
import copy
import math
import os
import shutil

def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

    img = cv2.imread(img_data_aug['filepath'])

    if augment:
        rows, cols = img.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)      # 沿y轴翻转(水平)
            for bbox in img_data_aug['bboxes']:     # 更bbox
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        elif config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)      # 沿x轴翻转
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        elif config.rot_90:       # 角度旋转
            angle = np.random.choice([0, 90, 180, 270], 1)[0]
            if angle == 270:
                img = np.transpose(img, (1, 0, 2))      # 轴旋转
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img


def video_to_imgs(folder_path, name, out_path, frame_gap=1, frame_num=False):
    # video to frames
    path = os.path.join(folder_path, name)
    video = cv2.VideoCapture(path)
    # success, image = video.read()
    count = 1
    success = True
    while success:
        success, image = video.read()
        if success is not True:
            break
        if count % frame_gap==0:
            # file name format video_name-frame_index.jpg
            cv2.imwrite("{}{}{}.jpg".format(out_path, name.strip('.avi'), count),
                    image)
        if cv2.waitKey(10) == 27:
            break
        count += 1
        if frame_num:
            if count>frame_num:
                break
    return count


def frame_to_video(frame_path, out_path, video_name, fps=16):
    # frames to video
    filelist = os.listdir(frame_path)
    for i in range(len(filelist)):
        filelist[i] = int(filelist[i][len(video_name):].strip('.jpg'))
    # fps = 12
    filelist.sort()
    for i in range(len(filelist)):
        filelist[i] = video_name+str(filelist[i])+'.jpg'
    file_path = os.path.join(out_path, video_name) + ".mp4"
    # fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    size = cv2.imread(os.path.join(frame_path, filelist[0])).shape[:2]
    # notice:frame size should be (width, height)
    video = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
    for item in filelist:
        if item:
            item = os.path.join(frame_path, item)
            img = cv2.imread(item)
            video.write(img)
    video.release()


if __name__=='__main__':
    # video_seg_imgs(
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/dash_female_split_output/Yawning/',
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/train_frames/yawning/')
    # video_seg_imgs(
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/dash_male_split_output/Yawning/',
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/train_frames/yawning/')
    # video_seg_imgs(
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/mirror_female_split_output/Yawning/',
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/train_frames/yawning/')
    # video_seg_imgs(
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/mirror_male_split_output/Yawning/',
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/train_frames/yawning/')
    #
    # video_seg_imgs(
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/mirror_female_split_output/Normal/',
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/train_frames/normal/')
    # video_seg_imgs(
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/mirror_male_split_output/Normal/',
    #     'E:/System/Desktop/fatigue-drive-yawning-detection-master/fatigue-drive-yawning-detection-master/dataset/train_lst/train_frames/normal/')

    # path = 'E:/Outpost/Projects/GraduationProject/Datasets/YawDD/Train/Frames/'
    # frame_list = os.listdir(path)
    # for frame in frame_list:
    #     # if 'Normal' in frame:
    #     #     shutil.move(path+frame, path+'normal/'+frame)
    #     if 'Yawning' in frame:
    #         shutil.move(path+frame, path+'yawning/'+frame)
    path = 'E:/System/Desktop/fatigue-drive-yawning-detection-master/dataset/train_lst/train_video/normal/'
    path1 = 'E:/System/Desktop/fatigue-drive-yawning-detection-master/dataset/train_lst/train_video/val/'
    file_list = os.listdir(path)
    sel_file = random.sample(file_list, 28)
    for file in sel_file:
        shutil.move(path + file, path1 + 'normal/' + file)
