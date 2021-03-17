import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

class Dataset:
    def __init__(self, directory, offset_x, offset_y, batch_size, batch_per_video):
        self.directory = directory
        self.videos = [os.path.join(self.directory, i) for i in os.listdir(directory)]
        self.batch_size = batch_size
        self.batch_per_video = batch_per_video
        self.offset_x = offset_x
        self.offset_y = offset_y

    def _load_frame(self, frame_path):
        frame = Image.open(frame_path)
        arr = np.array(frame)
        arr = np.expand_dims(arr, -1)
        arr = arr.astype('float32')
        arr /= 255
        return arr

    def _random_frames(self):
        video_idx = np.random.randint(low=0, high=len(self.videos))
        video = self.videos[video_idx]
        video_len = int(len(os.listdir(video)) / 2)
        frames = [os.path.join(video, '{}.png'.format(i)) for i in range(video_len)]
        frames_fgbg = [os.path.join(video, '{}_fgbg.png'.format(i)) for i in range(video_len)]

        idx_y = np.random.randint(self.offset_y, len(frames), self.batch_per_video)
        idx_x = [[y - self.offset_y + x for x in self.offset_x] for y in idx_y]

        frame_y = np.array([self._load_frame(frames[i]) for i in idx_y])
        frame_fgbg = np.array([self._load_frame(frames_fgbg[i]) for i in idx_y])
        frame_x = []
        for x in zip(*idx_x):
            temp_x= np.array([self._load_frame(frames[i]) for i in x])
            frame_x.append(temp_x)
        return frame_x, frame_y, frame_fgbg

    def train_loader(self):
        while True:
            for i in range(int(self.batch_size/self.batch_per_video)):
                x, y, fgbg = self._random_frames()
                if i == 0:
                    batch_x = x
                    batch_y = y
                    batch_fgbg = fgbg
                else:
                    batch_x = np.concatenate((batch_x, x), axis=1)
                    batch_y = np.concatenate((batch_y, y), axis=0)
                    batch_fgbg = np.concatenate((batch_fgbg, fgbg), axis=0)

            batch_fgbg[batch_fgbg == 1] = 10
            batch_fgbg[batch_fgbg == 0] = 1

            batch_x = list(batch_x)
            batch_x.append(batch_fgbg)
            yield batch_x, batch_y

    def test_loader(self, video_idx):
        video = self.videos[video_idx]
        video_len = int(len(os.listdir(video)) / 2)
        frames = [os.path.join(video, '{}.png'.format(i)) for i in range(video_len)]

        idx_y = np.arange(self.offset_y, len(frames))
        idx_x = [[y - self.offset_y + x for x in self.offset_x] for y in idx_y]

        frame_y = np.array([self._load_frame(frames[i]) for i in idx_y])
        frame_x = []
        for x in zip(*idx_x):
            temp_x= np.array([self._load_frame(frames[i]) for i in x])
            frame_x.append(temp_x)

        fgbg = np.ones_like(temp_x)
        #frame_x.append(fgbg)
        return frame_x, frame_y
