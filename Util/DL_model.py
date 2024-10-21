from Util.SM4 import make_train_data

import numpy as np

from pickle import dump

import os

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.regularizers import l2

def NAAF_resnet(length=44, filters=32, kernel_size=5):
    if length % 4 != 0:
        raise ValueError("Length must be a multiple of 4")

    row_elements = length // 4

    input_layer = Input(shape=(length,))

    x = Reshape((4, row_elements))(input_layer)

    x = Permute((2, 1))(x)

    x1 = Conv1D(filters, kernel_size=1, padding='same', strides=1, activation='linear')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv1D(filters, kernel_size=kernel_size, padding='same', strides=1, activation='linear')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv1D(filters, kernel_size=kernel_size, padding='same', strides=1, activation='linear')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x = Add()([x1, x3])

    x = Flatten()(x)

    # first dense layer
    x = Dense(64, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second dense layer
    x = Dense(64, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['loss', 'acc'])

    return model

bs = 5000
wdir = r'./Model/'

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);

#make residual tower of convolutional blocks
def make_resnet(num_blocks=4, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=32, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size * 2,));
  rs = Reshape((2 * num_blocks, word_size))(inp);
  perm = Permute((2,1))(rs);
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  conv0 = BatchNormalization()(conv0);
  conv0 = Activation('relu')(conv0);
  #add residual blocks
  shortcut = conv0;
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    conv1 = BatchNormalization()(conv1);
    conv1 = Activation('relu')(conv1);
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
    conv2 = BatchNormalization()(conv2);
    conv2 = Activation('relu')(conv2);
    shortcut = Add()([shortcut, conv2]);
  #add prediction head
  flat1 = Flatten()(shortcut);
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
  dense1 = BatchNormalization()(dense1);
  dense1 = Activation('relu')(dense1);
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
  dense2 = BatchNormalization()(dense2);
  dense2 = Activation('relu')(dense2);
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
  model = Model(inputs=inp, outputs=out);
  return(model);

def train_sm4_distinguisher(num_epochs, num_rounds=7, depth=1):
    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5);
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);

    #generate training and validation data
    X, Y = make_train_data(10**4,num_rounds);
    X_eval, Y_eval = make_train_data(10**3, num_rounds);
    X = np.array(X)
    Y = np.array(Y)
    X_eval = np.array(X_eval)
    Y_eval = np.array(Y_eval)

    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');

    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));

    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);

    val_acc_path = os.path.join(wdir, 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_val_acc.npy')
    val_loss_path = os.path.join(wdir, 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_val_loss.npy')
    history_path = os.path.join(wdir, 'hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p')

    np.save(val_acc_path, h.history['val_acc'])
    np.save(val_loss_path, h.history['val_loss'])
    with open(history_path, 'wb') as f:
        dump(h.history, f)
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, h);


if __name__ == '__main__':
    # test code for model compling
    model = NAAF_resnet(length=40, filters=32, kernel_size=3)
    model.summary()