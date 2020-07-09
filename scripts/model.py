# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:21:57 2020

@author: cheng
"""


# from keras_multi_head import MultiHeadAttention

from keras.layers import Input, Dense, Lambda, concatenate, LSTM, Activation, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.core import RepeatVector, Dropout
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.losses import mse, CategoricalCrossentropy 
#CategoricalCrossentropy


class CVAE():
    
    def __init__(self, args):
        # Store the hyperparameters
        self.args = args
        self.window_size = args.window_size
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.hidden_size = args.hidden_size
        self.z_dim = args.z_dim
        self.encoder_dim = args.encoder_dim
        self.z_decoder_dim = args.z_decoder_dim
        self.s_drop = args.s_drop
        self.z_drop = args.z_drop
        self.lr = args.lr
        self.beta = args.beta      
        
        #################### MODEL CONSTRUCTION STARTS FROM HERE ####################
        # Construct the condition model
        self.x = Input(shape=(self.window_size, self.num_features), name='x') 
        self.x_conv1d = Conv1D(self.hidden_size//2, kernel_size=3, strides=1, padding='same', name='x_conv1d')(self.x)
        self.x_dense = Dense(self.hidden_size, activation='relu', name='x_dense')(self.x_conv1d )
        self.x_state = LSTM(self.encoder_dim,
                       return_sequences=False,
                       stateful=False,
                       dropout=self.s_drop,
                       name='x_state')(self.x_dense)
        
        # Construct the label model        
        self.y = Input(shape=(self.num_classes, ), name='y')
        self.y_dense = Dense(self.encoder_dim//8, activation='relu', name='y_dense')(self.y)        
        
        # CONSTRUCT THE CVAE ENCODER BY FEEDING THE CONCATENATED X AND Y
        # the concatenated input
        self.inputs = concatenate([self.x_state, self.y_dense], name='inputs') 
        self.xy_encoded_d1 = Dense(self.hidden_size, activation='relu', name='xy_encoded_d1')(self.inputs) 
        self.xy_encoded_d2 = Dense(self.hidden_size//2, activation='relu', name='xy_encoded_d2')(self.xy_encoded_d1)
        self.mu = Dense(self.z_dim, activation='linear', name='mu')(self.xy_encoded_d2)
        self.log_var = Dense(self.z_dim, activation='linear', name='log_var')(self.xy_encoded_d2)        
        
        # THE REPARAMETERIZATION TRICK FOR THE LATENT VARIABLE z
        # sampling function
        z_dim = self.z_dim
        def sampling(params):
            mu, log_var = params
            eps = K.random_normal(shape=(K.shape(mu)[0], z_dim), mean=0., stddev=1.0)
            return mu + K.exp(log_var/2.) * eps
        
        # sampling z
        self.z = Lambda(sampling, output_shape=(self.z_dim,), name='z')([self.mu, self.log_var])
        # concatenate the z and x_encoded_dense
        self.z_cond = concatenate([self.z, self.x_state], name='z_cond')
            
        # CONSTRUCT THE CVAE DECODER
        self.z_decoder1 = Dense(self.hidden_size//2, activation='relu', name='z_decoder1')
        self.z_decoder2 = Dense(self.hidden_size//8, activation='relu', name='z_decoder2')
        self.y_decoder = Dense(self.num_classes, activation='softmax', name='y_decoder') 
        
        # Instantiate the decoder by feeding the concatenated z and x_encoded_dense
        # Reconstrcting y
        self.z_d1 = self.z_decoder1(self.z_cond)
        # self.z_d2 = self.z_decoder2(self.z_d1)
        self.y_prime = self.y_decoder(self.z_d1)
        
        
    def training(self):
        """
        Construct the CVAE model in training time
        Both features and class label are available 
        y is the ground truth class label
        """
        print('Contruct the cvae model for training')
        
        def vae_loss(y, y_prime):
            '''
            This is the customized loss function
            It consists of crossentropy loss and KL loss
            '''
            # reconstruction_loss = K.mean(mse(y, self.y_prime)*self.window_size)
            reconstruction_loss = K.mean(K.categorical_crossentropy(y, self.y_prime))
            kl_loss = 0.5 * K.sum(K.square(self.mu) + K.exp(self.log_var) - self.log_var - 1, axis=-1)
            cvae_loss = K.mean(reconstruction_loss*self.beta + kl_loss*(1-self.beta))
            return cvae_loss
        
        # BUILD THE CVAE MODEL
        cvae = Model([self.x, self.y], 
                     [self.y_prime])
        opt = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=1e-6, amsgrad=False)
        cvae.compile(optimizer=opt, loss=vae_loss)
        return cvae
    
    
    def X_encoder(self):
        """
        Construct the encoder to get the x_encoded_dense, 
        NOTE: 
            In inference phase, ONLY features are availabel
        Returns
        x_encoder : TYPE
            DESCRIPTION.
        """
        print('Construct the X-Encoder for inference')            
        x_encoder = Model(self.x, self.x_state)
        # x_encoder.summary()
        return x_encoder
    
    
    def Decoder(self):           
        # CONSTRUCT THE DECODER
        print('Construct the Decoder for trajectory oreidction')
        decoder_input = Input(shape=(self.z_dim+self.encoder_dim, ), name='decoder_input')
        _z_d1 = self.z_decoder1(decoder_input)
        # _z_d2 = self.z_decoder2(_z_d1)
        _y_prime = self.y_decoder(_z_d1)
        generator = Model(decoder_input, _y_prime)
        return generator
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
