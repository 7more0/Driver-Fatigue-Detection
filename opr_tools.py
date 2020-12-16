import numpy as np
import random
import cv2
import copy
import os
import shutil
'''
    processing tools for data processing.
'''
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
    # fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    size = cv2.imread(os.path.join(frame_path, filelist[0])).shape[:2]
    # notice:frame size should be (width, height)
    video = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
    for item in filelist:
        if item:
            item = os.path.join(frame_path, item)
            img = cv2.imread(item)
            video.write(img)
    video.release()


def dataset_divide(data_path, ratio=0.25):
    '''
        divide dataset into training and val dataset.
    :param data_path: path to train data. (format: train_data/classes/data)
    :param ratio: val/train
    '''
    classes = os.listdir(data_path)
    os.mkdir(data_path.strip('/') + '/val')
    os.mkdir(data_path.strip('/') + '/train')
    for cls in classes:
        os.mkdir(data_path.strip('/') + '/val/' + cls)
        file_list = os.listdir(os.path.join(data_path, cls))
        sel_list = random.sample(file_list, int(len(file_list) * ratio))
        for sel_file in sel_list:
            shutil.move('{}/{}'.format(os.path.join(data_path, cls), sel_file),
                        '{}/{}'.format(data_path.strip('/') + '/val/' + cls, '/' + sel_file))
        shutil.move(os.path.join(data_path, cls), os.path.join(os.path.join(data_path, 'train'), cls))

    return True


def video_length_check(path, frame_num=90):
    '''
        check video length and move videos short than threshold to /del folder
    '''
    os.mkdir(os.path.join(path, 'del'))
    for clip in os.listdir(path):
        video = cv2.VideoCapture(os.path.join(path, clip))
        if video.get(7)<90:
            shutil.move('{}/{}'.format(path, clip),
                        '{}/{}'.format(os.path.join(path, 'del'), clip))

    return True


