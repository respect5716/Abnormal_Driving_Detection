from keras.models import Model, Sequential
import keras.layers as L
import utils

def ConvLSTM(optimizer, init_channel, block_num):
    input_shape = (256, 256, 1)
    input_1 = L.Input(shape=input_shape)
    input_2 = L.Input(shape=input_shape)
    input_3 = L.Input(shape=input_shape)
    weights_input = L.Input(shape=input_shape)

    encoder = Sequential(name='encoder')
    for i in range(block_num):
        if i == 0:
            encoder.add(L.Conv2D(init_channel*(2**i), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape))
        else:
            encoder.add(L.Conv2D(init_channel*(2**i), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'))
        encoder.add(L.Conv2D(init_channel*(2**i), (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)
    encoded_3 = encoder(input_3)

    reshape = (1, *encoder.output_shape[1:])
    reshaped_1 = L.Reshape(reshape)(encoded_1)
    reshaped_2 = L.Reshape(reshape)(encoded_2)
    reshaped_3 = L.Reshape(reshape)(encoded_3)

    concat = L.Concatenate(axis=1)([reshaped_1, reshaped_2, reshaped_3])
    convlstm = L.ConvLSTM2D(init_channel*4, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=False)(concat)

    decoder_shape = (i.value for i in convlstm.get_shape()[1:])
    decoder = Sequential(name='decoder')

    for i in range(block_num):
        if i == 0:
            decoder.add(L.Conv2DTranspose(init_channel*(2**(block_num-i)), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=decoder_shape))
        else:
            decoder.add(L.Conv2DTranspose(init_channel*(2**(block_num-i)), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'))
        decoder.add(L.Conv2D(init_channel*(2**(block_num-i)), (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))
    decoder.add(L.Conv2D(1, (3,3), strides=1, activation='sigmoid', padding='same', kernel_initializer='he_normal'))

    output = decoder(convlstm)

    model = Model(inputs=[input_1, input_2, input_3, weights_input], outputs=output)
    model.compile(optimizer=optimizer, loss = utils.custom_loss(weights_input))
    print(model.summary())
    return model
