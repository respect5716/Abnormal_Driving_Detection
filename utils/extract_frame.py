import os
import cv2
import argparse
import numpy as np
from PIL import Image
from defogging import Defog

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', help='동영상 로드 경로', type=str)
parser.add_argument('--save_path', help='이미지 저장 경로', type=str)
args = parser.parse_args()


def get_video_list(load_path):
    video_list = os.listdir(load_path)
    return video_list


def get_video_frame(video, load_path, save_path):
    save_path = os.path.abspath(save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    video_save_path = os.path.join(save_path, video[:-4])
    video_save_path = os.path.abspath(video_save_path)
    if not os.path.isdir(video_save_path):
        os.mkdir(video_save_path)

    cap = cv2.VideoCapture(os.path.join(load_path, video))
    frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    for i in range(frame_len):
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        os.chdir(video_save_path)
#         df = Defog()
#         df.read_img(frame)
#         df.defog()
#         df.save_img('{}.png'.format(i))

#         fgbg = cv2.createBackgroundSubtractorMOG2()
#         fgmask = fgbg.apply(frame)
#         cv2.imshow('frame',fgmask)
        
        fgmask=fgbg.apply(frame)
        blur = cv2.GaussianBlur(fgmask,(5,5),0)
        
        cv2.imwrite('{}.png'.format(i), blur)
    cap.release()

def main(args):
    video_list = get_video_list(args.load_path)
    i = 0
    for video in video_list:
        i += 1
        get_video_frame(video, args.load_path, args.save_path)

        if i % 10 == 0:
            print('{}% 완료'.format(int((i / len(video_list)) * 100)))
    print('전체 완료')

if __name__ == '__main__':
    main(args)
