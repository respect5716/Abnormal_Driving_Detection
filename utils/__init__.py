import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import model_from_json

from .data import *
from .models import *


def custom_loss(weights):
    def loss(y_true,y_pred):
        return K.mean(K.square(y_pred - y_true) * weights)
    return loss


def save_model(model, save_path):
    model_json = model.to_json()
    with open('{}.json'.format(save_path), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('{}.h5'.format(save_path))
    print('저장 완료')


def load_model(model, load_path):
    try:
        with open('{}.json'.format(load_path), 'r') as f:
            model = model_from_json(f.read())
    except:
        model = model
    model.load_weights('{}.h5'.format(load_path))
    return model


def make_video(pred, abnormal):
    video = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 12, (256, 256), True)

    for i in range(len(pred)):
        frame = pred[i][:,:,0] * 255
        ab = abnormal[i][:,:,0]

        frame[np.where(ab > 0)] = 0
        frame = np.uint8(frame)
        ab = ab * 100
        ab = np.uint8(ab)

        img = cv2.merge((frame, frame, ab))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    print('저장 완료')

def make_image(pred, real):
    fig = plt.figure(figsize=(13, 13))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.imshow(pred[0][:,:,0])
    ax2.imshow(real[0][:,:,0])
    plt.savefig('train.png')
