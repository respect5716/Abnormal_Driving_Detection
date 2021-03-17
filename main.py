import argparse
import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

import keras
import keras.layers as L
import keras.backend as K
from keras.callbacks import ModelCheckpoint

import utils
from utils.models import ConvLSTM
from utils.data import Dataset

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='train')
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_path', type=str)

parser.add_argument('--offset_x', nargs='+', type=int, default=[1, 7, 15])
parser.add_argument('--offset_y', type=int, default=30)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--steps_per_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--batch_per_video', type=int, default=1)

parser.add_argument('--init_channel', type=int, default=128)
parser.add_argument('--block_num', type=int, default=2)
args = parser.parse_args()


def train(dataloader, model, epochs, steps_per_epoch, save_path):
    checkpoint = ModelCheckpoint('{}.h5'.format(save_path), monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = model.fit_generator(
        generator=dataloader,
        epochs = epochs,
        steps_per_epoch= steps_per_epoch,
        callbacks=callbacks_list
    )
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('{}.csv'.format(save_path), index=False)

def test(model, x, y, batch_size):
    for i in range(int(len(y) / batch_size)):
        batch_x = [j[i:i+batch_size] for j in x]
        batch_y = y[i:i+batch_size]
        pred = model.predict(batch_x)
        if i == 0:
            result = pred
        else:
            result = np.concatenate((result, pred), axis=0)
    return result

def anomaly_score(pred, real):
    real = real[:len(pred)]
    err = np.abs(pred - real)
    err_mean = err.mean()
    err_std = err.std()
    err_dist = sp.norm(err_mean, err_std)

    err[err < err_mean] = err_mean
    err_pdf = err_dist.pdf(err)
    err_pdf_norm = (err_pdf - err_pdf.min()) / (err_pdf.max() - err_pdf.min())
    abnormal = err_pdf_norm < 0.001
    score = np.mean(abnormal, axis=(1,2))
    return abnormal, score

def main(args):
    dataset = Dataset(args.data_path, args.offset_x, args.offset_y, args.batch_size, args.batch_per_video)
    optimizer = keras.optimizers.Adam(lr=1e-4)
    model = ConvLSTM(optimizer, args.init_channel, args.block_num)

    if args.train == 'train':
        dataloader = dataset.train_loader()
        train(dataloader, model, args.epochs, args.steps_per_epoch, args.save_path)
        utils.save_model(model, args.save_path)
        x, y = next(dataloader)
        pred = model.predict(x)
        utils.make_image(pred, y)


    elif args.train == 'test':
        video_idx = int(input('예측할 동영상 인덱스를 입력하세요.'))
        x, y = dataset.test_loader(video_idx)
        model = utils.load_model(model, args.save_path)
        pred = test(model, x, y, args.batch_size)
        abnormal, score = anomaly_score(pred, y)
        plt.plot(score)
        plt.savefig('anomaly score.png')
        utils.make_video(pred, abnormal)

if __name__ == '__main__':
    main(args)
