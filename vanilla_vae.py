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
        self.dense1  = tf.keras.layers.Dense(units = self.latent_dim, name = "en_fc1")
        self.en_act  = tf.keras.layers.Activation("relu", name = "en_main_out")

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
        # Latent Space
        x = self.dense1(x)
        x = self.en_act(x)
        return x

    #................................................................................
    # Encoder Space
    #................................................................................
    def decoder(self, x, training = None):
        # Encoder Space
        x = self.dense2(x)
        x = self.de_act(x)
        x = self.reshape(x)
        return x

    #................................................................................
    #
    #................................................................................
    def call(self, inputs, training = None):
        # inputs = self.in_layer(inputs)
        self.encoded = self.encoder(inputs, training)

        shape = self.encoded.shape[1:]
        x = tf.random.uniform(shape, minval=0.0, maxval=1.0)
        de_input = self.encoded + x

        self.decoded = self.decoder(de_input, training)
        return self.decoded
