import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', help='load_path', type=str)
parser.add_argument('--save_path', help='save_path', type=str)
args = parser.parse_args()


def get_path():
    load_path = args.load_path
    save_path = args.save_path
    return load_path, save_path

def get_video_list(load_path):
    video_list = os.listdir(load_path)
    return video_list

def make_dir(save_path):
    save_path = os.path.abspath(save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

def convert_frame(video, load_path, save_path):
    save_path = os.path.abspath(save_path)
    video_save_path = os.path.join(save_path, video)
    make_dir(video_save_path)

    video_path = os.path.join(load_path, video)
    image_list = os.listdir(os.path.join(load_path, video))
    image_list = [os.path.join(video_path, i) for i in image_list]

    for i in range(len(image_list)):
        img = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        os.chdir(video_save_path)
        cv2.imwrite('{}.png'.format(i), img)

def main():
    load_path, save_path = get_path()
    video_list = get_video_list(load_path)
    make_dir(save_path)
    for i in tqdm(range(len(video_list)), mininterval=1):
        video = video_list[i]
        convert_frame(video, load_path, save_path)

    print('전체 완료')


if __name__ == '__main__':
    main()
