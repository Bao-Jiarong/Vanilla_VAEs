'''
    ------------------------------------
    Author : Bao Jiarong
    Date   : 2020-08-30
    Project: Variational AE (vanilla)
    Email  : bao.salirong@gmail.com
    ------------------------------------
'''

import os
import sys
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import loader
import vanilla_vae

np.random.seed(7)
tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
image_size = 28
latent_dim = 200
model_name = "models/vanilla/digists"
data_path  = "../../data_img/MNIST/train/"

# Step 0: Global Parameters
epochs     = 10
lr_rate    = 1e-4
batch_size = 4

# Step 1: Create Model
model = vanilla_vae.VANILLA_VAE(image_size = image_size, latent_dim = latent_dim)
model.build((None, image_size,image_size,3))

# Step 2: Define Metrics
# print(model.summary())
# sys.exit()

if sys.argv[1] == "train":
    # Step 3: Load data
    X_train, Y_train, X_valid, Y_valid = loader.load_light(data_path,image_size,image_size,True,0.8,True)

    # Step 4: Training
    # model.load_weights(model_name)

    # Define The Optimizer
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_rate) #, beta_1 = 0.5)

    @tf.function
    def ae_loss(y_true, y_pred):
        # de_loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        return tf.keras.losses.MSE(y_true=y_true, y_pred=y_pred)

    # Define The Metrics
    tr_loss = tf.keras.metrics.MeanSquaredError(name = 'tr_loss')
    va_loss = tf.keras.metrics.MeanSquaredError(name = 'va_loss')

    #---------------------
    @tf.function
    def train_step(X, Y_true):
        with tf.GradientTape(persistent=True) as tape:
            Y_pred = model(X, training=True)
            loss   = ae_loss(y_true = Y_true, y_pred = Y_pred )

        # Training Variables
        # all_vars = model.trainable_variables
        en_vars = [x for x in model.trainable_variables if "en_" not in x.name]
        de_vars = [x for x in model.trainable_variables if "de_" not in x.name]
        all_vars= en_vars + de_vars
        gradients= tape.gradient(loss, all_vars)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, all_vars))

        tr_loss.update_state(y_true = Y_true, y_pred = Y_pred )
        del tape

    #---------------------
    @tf.function
    def valid_step(X, Y_true):
        Y_pred= model(X, training=False)
        loss  = ae_loss(y_true = Y_true, y_pred = Y_pred)
        va_loss.update_state(y_true = Y_true, y_pred = Y_pred)

    #---------------------
    tr_loss.reset_states()
    va_loss.reset_states()
    # start training
    L = len(X_train)
    M = len(X_valid)
    steps  = int(L/batch_size)
    steps1 = int(M/batch_size)

    for epoch in range(epochs):
        # Run on training data + Update weights
        for step in range(steps):
            images, _ = loader.get_batch_light(X_train, Y_train, batch_size, image_size, image_size)
            train_step(images,images)

            print(epoch,"/",epochs,step,steps,"tr_loss:",tr_loss.result().numpy(),end="\r")

        # Run on validation data without updating weights
        for step in range(steps1):
            images, _ = loader.get_batch_light(X_valid, Y_valid, batch_size, image_size, image_size)
            valid_step(images, images)

        print(epoch,"/",epochs,step,steps,
              "tr_loss:",tr_loss.result().numpy(),"va_loss:",va_loss.result().numpy())

        # Save the model for each epoch
        model.save_weights(filepath=model_name, save_format='tf')


elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    img = cv2.imread(sys.argv[2])
    image = cv2.resize(img,(image_size,image_size),interpolation = cv2.INTER_AREA)
    images = np.array([image])
    images = loader.scaling_tech(images,method="normalization")

    # Step 5: Predict the class
    preds = my_model.predict(images)
    # print(np.argmax(preds[0]))
    # print(preds[0])
    all_imgs = np.hstack((images[0], preds[0]))
    all_imgs = cv2.resize(all_imgs,(8*image_size,4*image_size),interpolation = cv2.INTER_AREA)
    cv2.imshow("imgs",all_imgs)
    cv2.waitKey(0)

elif sys.argv[1] == "predict_all":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    imgs_filenames = ["../../data_img/MNIST/test/img_2.jpg" , # 0
                      "../../data_img/MNIST/test/img_18.jpg", # 1
                      "../../data_img/MNIST/test/img_1.jpg" , # 2
                      "../../data_img/MNIST/test/img_5.jpg" , # 3
                      "../../data_img/MNIST/test/img_13.jpg", # 4
                      "../../data_img/MNIST/test/img_11.jpg", # 5
                      "../../data_img/MNIST/test/img_35.jpg", # 6
                      "../../data_img/MNIST/test/img_6.jpg" , # 7
                      "../../data_img/MNIST/test/img_45.jpg", # 8
                      "../../data_img/MNIST/test/img_3.jpg" ] # 9
    images = []
    for filename in imgs_filenames:
        img = cv2.imread(filename)
        image = cv2.resize(img,(image_size,image_size),interpolation = cv2.INTER_AREA)
        images.append(image)

    # True images
    images = np.array(images)
    images = loader.scaling_tech(images,method="normalization")

    # Predicted images
    preds = my_model.predict(images)
    preds = (preds - preds.min())/(preds.max() - preds.min())


    true_images = np.hstack(images)
    pred_images = np.hstack(preds)

    images = np.vstack((true_images, pred_images))
    h = images.shape[0]
    w = images.shape[1]
    images = cv2.resize(images,(w << 1, h << 1))

    cv2.imshow("imgs",images)
    cv2.waitKey(0)
