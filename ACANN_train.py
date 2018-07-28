# Romain Fournier
# EPFL - C3MP 
# Creation : 11.11.2017
# Last edit : 27.07.2018


###### Import 

from keras.models import Sequential
import random
import sys
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np
import csv
import pandas
from sklearn import metrics
import tensorflow as tf
from keras.callbacks import History,ReduceLROnPlateau
from keras.models import load_model
from keras.callbacks import TensorBoard
###### Database
#Training files
filename_A = "../Database/A_data.csv"
filename_G = "../Database/nl_data.csv"
filename_Aval = "../Database/A_validationset.csv"
filename_Gval = "../Database/nl_validationset.csv"


####### Input arguments
 ## 1st argument is the number of data to consider in the file
database_size=int(sys.argv[1])
 ## 2nd argument is the name used as prefix for the saved files
name=sys.argv[2]
 ## 3rd argument is the number of layers to use
nb_layers=int(sys.argv[3])
## 3th argument is the number of epochs
nb_epochs=int(sys.argv[4])

## argument 5 -> 5+nb_layers-1 correspond to the number of hidden units


## Variable to save the intermediate results
history = History()

########## Creation of the optimizer (adam) and the model 
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = Sequential()
model.add(Dense(units=int(sys.argv[5]),input_dim=64 ))
for ii in range(1,nb_layers):
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.02))
    model.add(Dense(units=int(sys.argv[5+ii])))
model.add(Activation('softmax'))
print("The model has been initiated with Sequential.")

########## Activate TensorBord in order to observe the evolution of the training live
tbCallBack = TensorBoard(log_dir='./TensorBoard/'+str(name), histogram_freq=0, write_graph=True, write_images=True)
########## Reduce the learning rate when the validation does not improve over the 5 last epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=5, min_lr=0.0)

############ Read normalize and prepare the data (training + validation)
n = sum(1 for line in open(filename_A))  #number of records in file (excludes header)
s = database_size; #desired sample size
skip = sorted(random.sample(xrange(0,n),n-1-s)) #the 0-indexed header will not be included in the skip list
df_A = pandas.read_csv(filename_A, skiprows=skip)
df_G = pandas.read_csv(filename_G, skiprows=skip)
x_train=df_G.as_matrix()
y_train=df_A.as_matrix()
y_train=y_train/y_train.sum(axis=1, keepdims=True)

df_Aval = pandas.read_csv(filename_Aval)
df_Gval = pandas.read_csv(filename_Gval)
x_val=df_Gval.as_matrix()
y_val=df_Aval.as_matrix()
print(np.mean(y_val.sum(axis=1, keepdims=True)))
y_val=y_val/y_val.sum(axis=1, keepdims=True)

######### Compile the model
model.compile(loss='kullback_leibler_divergence',
        optimizer=adam,
        metrics=['mae'])
history=model.fit(x_train, y_train, epochs=nb_epochs, initial_epoch=0,batch_size=100,validation_data=(x_val,y_val), callbacks=[tbCallBack,reduce_lr])


######## Save the results
print(history.history.keys())
f_handle = file('Results/'+str(name)+'/mae.csv', 'a')
np.savetxt(f_handle, history.history['mean_absolute_error'])
f_handle.close()
f_handle = file('Results/'+str(name)+'/val_mae.csv', 'a')
np.savetxt(f_handle, history.history['val_mean_absolute_error'])
f_handle.close()

model.save('Results/'+str(name)+'/model.h5')

