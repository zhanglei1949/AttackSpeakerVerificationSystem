import logging

import keras.backend as K
import math
from keras import layers
from keras import regularizers
from keras.layers import Input, GRU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda, Dense, RepeatVector
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf

from constants import *
from attack_utils import add_noise, fbank_layer_1, fbank_layer_2
def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_2b')(x)
    x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)


    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x

def identity_block2(input_tensor, kernel_size, filters, stage, block):   # next step try full-pre activation
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
                   kernel_size=1,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_conv1_1')(input_tensor)
    x = BatchNormalization(name=conv_name_base + '_conv1.1_bn')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=1,
               activation=None,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.00001),
               name=conv_name_base + '_conv3')(x)
    x = BatchNormalization(name=conv_name_base + '_conv3_bn')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
                   kernel_size=1,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_conv1_2')(x)
    x = BatchNormalization(name=conv_name_base + '_conv1.2_bn')(x)

    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x

def my_convolutional_model(input_audio_shape = (25840,),    #input_shape(32,32,3)
                        input_threshold_shape = (160, 257), # shape (1, 160, 257)
                        ori_spec_shape = (160, 257),
                        batch_size=BATCH_SIZE * TRIPLET_PER_BATCH , num_frames=NUM_FRAMES):
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input

    # Conv128-s
    # 5*5*128*128/2+128
    # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

    # take 100 ms -> 4 frames.
    # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
    # 8*8 = 64 features.

    # used to share all the layers across the inputs

    # num_frames = K.shape() - do it dynamically after.

    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                       kernel_size=5,
                       strides=2,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.00001), name=conv_name)(inp)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = clipped_relu(o)
        for i in range(3):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    inputs_audio = Input(shape=input_audio_shape)  # TODO the network should be definable without explicit batch shape
    noise_added = add_noise()(inputs_audio)

    inputs_hearing_threshold = Input(shape = input_threshold_shape)

    inputs_ori_audiospec = Input(shape=ori_spec_shape)
    #3. fbank layer
    fbank_feature_1 = fbank_layer_1()(noise_added)
    #print(fbank_feature_1.shape)

    #4.one identity layer output just the input, but change the gradient calculation
    @tf.custom_gradient
    def scale_grad_layer(x):
        def grad(dy):
            #print("in grad", dy.shape)
            #TODO:scaling 
            #return dy
            return tf.multiply(dy, inputs_hearing_threshold)
            
        return x, grad

    fbank_feature_1_scaled = Lambda(lambda x : scale_grad_layer(x), name = 'scaling')(fbank_feature_1) 

    #5. Yet another scaling is applied
    #   Need the spectrogram of orignial audio in the first place. 
    #   D = M - S
    @tf.custom_gradient
    def scale_grad_layer_2(x):
        def grad(dy):
            pspec_diff = K.abs(fbank_feature_1_scaled - inputs_ori_audiospec)
            pspec_diff = 20 * (K.log(pspec_diff / K.max(inputs_ori_audiospec)) / K.log(10.0))
            f = 10
            acceptable_diff = inputs_hearing_threshold - pspec_diff + f # f is nabla, denoting the acceptable cross threshold
            acceptable_diff = K.maximum(acceptable_diff, 0)
            normalied_acceptable_diff = (acceptable_diff - K.min(acceptable_diff)) / (K.max(acceptable_diff) - K.min(acceptable_diff))
            return tf.multiply(dy, normalied_acceptable_diff)
            #return dy
        return x, grad
    fbank_feature_1_1 = Lambda( lambda x : scale_grad_layer_2(x), name = 'scaling2')(fbank_feature_1_scaled)
    fbank_feature_2 = fbank_layer_2()(fbank_feature_1_1)
    # Another try
    #@tf.RegisterGradient("ClipGrad")
    #def _clip_grad(unused_op, grad):
    #    return 100 * grad
    #g = tf.get_default_graph()
    #with g.gradient_override_map({"Identity": "ClipGrad"}):
    #    fbank_feature_scaled = Lambda(lambda x : K.identity(x), name = "identity")(fbank_feature)

    # Yet anther try
    #fbank_feature_scaled = Lambda(lambda x: 10 * x + K.stop_gradient(x - 10 * x), name = "ccc")(fbank_feature)

    #print(fbank_feature_2.shape)
    #x = Lambda(lambda y: K.reshape(y, (batch_size*num_frames,input_shape[1], input_shape[2], input_shape[3])), name='pre_reshape')(inputs)
    x = cnn_component(fbank_feature_2)  # .shape = (BATCH_SIZE , num_frames/16, 64/16, 512)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames/16), 2048)), name='reshape')(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  #shape = (BATCH_SIZE, 512)
    x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE , 512)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs=[inputs_audio, inputs_hearing_threshold, inputs_ori_audiospec], outputs=x, name='convolutional')
#   model = Model(inputs, fbank_feature, name='fbank_features')
    #print(model.summary())
    return model


def convolutional_model(input_shape=(NUM_FRAMES,64, 1),    #input_shape(32,32,3)
                        batch_size=BATCH_SIZE * TRIPLET_PER_BATCH , num_frames=NUM_FRAMES):
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input

    # Conv128-s
    # 5*5*128*128/2+128
    # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

    # take 100 ms -> 4 frames.
    # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
    # 8*8 = 64 features.

    # used to share all the layers across the inputs

    # num_frames = K.shape() - do it dynamically after.

    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                       kernel_size=5,
                       strides=2,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.00001), name=conv_name)(inp)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = clipped_relu(o)
        for i in range(3):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    inputs = Input(shape=input_shape)  # TODO the network should be definable without explicit batch shape
    #x = Lambda(lambda y: K.reshape(y, (batch_size*num_frames,input_shape[1], input_shape[2], input_shape[3])), name='pre_reshape')(inputs)
    x = cnn_component(inputs)  # .shape = (BATCH_SIZE , num_frames/16, 64/16, 512)
    #x = Reshape((-1,2048))(x)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames/16), 2048)), name='reshape')(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  #shape = (BATCH_SIZE, 512)
    x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE , 512)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs, x, name='convolutional')

    #print(model.summary())
    return model

def convolutional_model_simple(input_shape=(NUM_FRAMES,64, 1),    #input_shape(32,32,3)
                        batch_size=BATCH_SIZE * TRIPLET_PER_BATCH , num_frames=NUM_FRAMES):
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input

    # Conv128-s
    # 5*5*128*128/2+128
    # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

    # take 100 ms -> 4 frames.
    # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
    # 8*8 = 64 features.

    # used to share all the layers across the inputs

    # num_frames = K.shape() - do it dynamically after.

    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                       kernel_size=5,
                       strides=2,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.00001), name=conv_name)(inp)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = clipped_relu(o)
        for i in range(3):
            o = identity_block2(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        #x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    inputs = Input(shape=input_shape)  # TODO the network should be definable without explicit batch shape
    x = cnn_component(inputs)  # .shape = (BATCH_SIZE , num_frames/8, 64/8, 512)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames / 8), 2048)), name='reshape')(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  #shape = (BATCH_SIZE, 512)
    x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE , 512)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs, x, name='convolutional')

    return model

def recurrent_model(input_shape=(NUM_FRAMES, 64, 1),
                    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH ,num_frames=NUM_FRAMES):
    inputs = Input(shape=input_shape)
    #x = Permute((2,1))(inputs)
    x = Conv2D(64,kernel_size=5,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    x = BatchNormalization()(x)  #shape = (BATCH_SIZE , num_frames/2, 64/2, 64)
    x = clipped_relu(x)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames / 2), 2048)), name='reshape')(x) #shape = (BATCH_SIZE , num_frames/2, 2048)
    x = GRU(1024,return_sequences=True)(x)  #shape = (BATCH_SIZE , num_frames/2, 1024)
    x = GRU(1024,return_sequences=True)(x)
    x = GRU(1024,return_sequences=True)(x)  #shape = (BATCH_SIZE , num_frames/2, 1024)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x) #shape = (BATCH_SIZE, 1024)
    x = Dense(512)(x)  #shape = (BATCH_SIZE, 512)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs,x,name='recurrent')

    #print(model.summary())
    return model

if __name__ == '__main__':
    convolutional_model()
