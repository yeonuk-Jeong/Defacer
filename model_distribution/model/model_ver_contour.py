#!/usr/bin/env python
# coding: utf-8



import numpy as np

import random
from scipy import ndimage

from keras import backend as K

from keras.models import Model

from keras import layers, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec
from keras.utils import to_categorical



# # Load Data




config = dict() #configuration info



# # 3D Unet Model 




#configure parameter & hyper parameter

config["img_channel"] = 1
config["num_multilabel"] = 5 # the number of label (channel last)
config["noise"] = 0.1
config["batch_size"] = 1 # 3D segmentation learning needs too large GPU memory to increase batch size. # this script is optimized for single batch size 
config["resizing"] = True #True -> resize input image for learning. if you don't have enough GPU memory.
config["input_shape"] = [128, 128, 128, 1] # smaller GPU memory smaller image size



# loss function

def dice_score(y_true, y_pred):
    smooth = 1.
    label_length = y_pred.get_shape().as_list()[-1] #the number of label (channel last)
    
    loss = 0    
    for num_labels in range(label_length):
        y_true_f = K.flatten(y_true[..., num_labels])
        y_pred_f = K.flatten(y_pred[..., num_labels])
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
   
    return loss/label_length 

def dice_loss(y_true, y_pred):
    return 1-dice_score(y_true, y_pred) + 0.01*K.categorical_crossentropy(y_true, y_pred)
'''
Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    
'''


def tversky_loss(beta): # beta : [0,1] the bigger beta, the more penaly on False Positive
    def t_loss(y_true, y_pred):
        loss = 0
        label_length = y_pred.get_shape().as_list()[-1] # the number of label (channel last)
        smooth = 1
        
        for num_labels in range(label_length):
            y_true_f = K.flatten(y_true[..., num_labels])
            y_pred_f = K.flatten(y_pred[..., num_labels])
            numerator = K.sum(y_true_f * y_pred_f)
            denominator = y_true_f * y_pred_f + beta * (1 - y_true_f) * y_pred_f + (1 - beta) * y_true_f * (1 - y_pred_f)
            loss += (numerator+ smooth) / (K.sum(denominator) + smooth)
        return 1-(loss / label_length)
        
    return t_loss



def focal_tversky(y_true,y_pred):
    pt_1 = tversky_loss(0.7)(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


def resize(data, img_dep=config["input_shape"][0], img_cols=config["input_shape"][1], img_rows=config["input_shape"][2]):
    resize_factor = (img_dep/data.shape[0], img_cols/data.shape[1], img_rows/data.shape[2])
    data = ndimage.zoom(data, resize_factor, order=0, mode='nearest')
    return data
'''
ndimage.zoom(어레이 데이터 인풋,축별로zoom factor. 숫자하나면 각축에 적용.,
The order of the spline interpolation. 0-5까지 옵션있음 ,

'''

    
# keras_contrib.layers.InstanceNormalization
# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# import model structure and weight
from keras.models import load_model
model = load_model('./model/model_contour4.h5',custom_objects={'InstanceNormalization':InstanceNormalization,'dice_loss':dice_loss,'dice_score':dice_score})

