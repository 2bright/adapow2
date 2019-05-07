import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, Conv2DTranspose, Reshape, MaxPooling2D, Flatten, Dropout, LeakyReLU, Lambda
from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy

def create_model(model_type, optimizer):
  return globals()['model_' + model_type](optimizer)

def model_mnist_logreg(optimizer):
  model = tf.keras.models.Sequential([
    Flatten(),
    Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def model_mnist_2_layers(optimizer):
  model = tf.keras.models.Sequential([
    Flatten(),
    Dense(512, activation=tf.nn.relu),
    Dropout(0.2),
    Dense(10, activation=tf.nn.softmax),
  ])
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def model_mnist_2_layers_l2(optimizer):
  model = tf.keras.models.Sequential([
    Flatten(),
    Dense(512, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.00001)),
    Dropout(0.2),
    Dense(10, activation=tf.nn.softmax, kernel_regularizer=regularizers.l2(0.00001))
  ])
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def model_mnist_mlp(optimizer):
  model = tf.keras.models.Sequential([
    Flatten(),
    Dense(1000, activation=tf.nn.relu),
    Dropout(0.2),
    Dense(500, activation=tf.nn.relu),
    Dropout(0.2),
    Dense(100, activation=tf.nn.relu),
    Dropout(0.2),
    Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def model_mnist_2c2d(optimizer):
  X = Input((28, 28))
  Y = X
  
  Y = Reshape((28, 28, 1))(Y)

  Y = Conv2D(32, 5, 1, 'same')(Y)
  Y = Activation('relu')(Y)
  Y = MaxPooling2D(2)(Y)

  Y = Conv2D(64, 5, 1, 'same')(Y)
  Y = Activation('relu')(Y)
  Y = MaxPooling2D(2)(Y)

  Y = Flatten()(Y)
  Y = Dense(1024)(Y)
  Y = Activation('relu')(Y)

  Y = Dense(10)(Y)
  Y = Activation('softmax')(Y)

  model = Model(X, Y)
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def model_mnist_vae(optimizer):
  '''VAE model from keras example.
  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py
  '''
  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon

  image_size = 28
  input_shape = (image_size, image_size)
  kernel_size = 3
  filters = 16
  latent_dim = 3
  epochs = 30
  
  inputs = Input(shape=input_shape)
  x = Reshape((image_size, image_size, 1))(inputs)
  for i in range(2):
      filters *= 2
      x = Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 strides=2,
                 padding='same')(x)
  
  shape = K.int_shape(x)
  
  x = Flatten()(x)
  x = Dense(16, activation='relu')(x)
  z_mean = Dense(latent_dim, name='z_mean')(x)
  z_log_var = Dense(latent_dim, name='z_log_var')(x)
  z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
  encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
  
  latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
  x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
  x = Reshape((shape[1], shape[2], shape[3]))(x)
  for i in range(2):
      x = Conv2DTranspose(filters=filters,
                          kernel_size=kernel_size,
                          activation='relu',
                          strides=2,
                          padding='same')(x)
      filters //= 2
  outputs = Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            activation='sigmoid',
                            padding='same',
                            name='decoder_output')(x)
  decoder = Model(latent_inputs, outputs, name='decoder')
  
  outputs = decoder(encoder(inputs)[2])
  model = Model(inputs, outputs)

  reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
  reconstruction_loss *= image_size * image_size
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  model.add_loss(vae_loss)
  model.compile(optimizer=optimizer)

  return model, encoder, decoder
