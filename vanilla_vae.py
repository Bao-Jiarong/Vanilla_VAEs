'''
    ------------------------------------
    Author : Bao Jiarong
    Date   : 2020-08-30
    Project: Variational AE (vanilla)
    Email  : bao.salirong@gmail.com
    ------------------------------------
'''

import tensorflow as tf

class VANILLA_VAE(tf.keras.Model):
    #................................................................................
    # Constructor
    #................................................................................
    def __init__(self, image_size = 28, latent_dim = 200):
        super(VANILLA_VAE, self).__init__(name = "VANILLA_VAE")

        self.image_size = image_size  # height and weight of images
        self.latent_dim = latent_dim
        self.my_in_shape= [image_size, image_size, 3]

        # Encoder Layers
        self.flatten = tf.keras.layers.Flatten()

        # Latent Space
        self.la_dense1= tf.keras.layers.Dense(units = self.latent_dim   , name = "la_fc1")
        self.la_dense2= tf.keras.layers.Dense(units = self.latent_dim   , name = "la_fc2")

        # Decoder Layers
        self.dense2  = tf.keras.layers.Dense(units = (image_size ** 2) * 3, name = "de_fc1")
        self.de_act  = tf.keras.layers.Activation("sigmoid")
        self.reshape = tf.keras.layers.Reshape(self.my_in_shape, name = "de_main_out")

    #................................................................................
    # Decoder Space
    #................................................................................
    def encoder(self, x, training = None):
        # Encoder Space
        x = self.flatten(x)
        return x

    #................................................................................
    # Encoder Space
    #................................................................................
    def decoder(self, x, training = None):
        # Decoder Space
        x = self.dense2(x)
        x = self.de_act(x)
        x = self.reshape(x)
        return x

    #................................................................................
    # Latent Space
    #................................................................................
    def latent_space(self,x):
        mu  = self.la_dense1(x)
        std = self.la_dense2(x)
        shape = mu.shape[1:]
        eps = tf.random.normal(shape, 0.0, 1.0)
        x = mu + eps * (tf.math.exp(std/2.0))
        return x

    #................................................................................
    def call(self, inputs, training = None):
        x = self.encoder(inputs, training)
        x = self.latent_space(x)
        x = self.decoder(x, training)
        return x
