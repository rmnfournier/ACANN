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
from keras.callbacks import History
import numpy as np
import pandas

#from keras.callbacks import TensorBoard
###### Database
#Training files
filename_A = "../Database/A_data.csv"
filename_G = "../Database/nl_data.csv"
filename_Aval = "../Database/A_validationset.csv"
filename_Gval = "../Database/nl_validationset.csv"


####### Input arguments
## Find the dataset
prefix = str(sys.argv[1])
filename_A = prefix+"/A_data.csv"
filename_G = prefix+"/nl_data.csv"
filename_Aval = prefix+"/A_validationset.csv"
filename_Gval = prefix+"/nl_validationset.csv"

 ## 2nd argument is the number of data to consider in the file
database_size=int(sys.argv[2])
 ## 3rd argument is the name used as prefix for the saved files
name=sys.argv[3]
 ## 4th argument is the number of layers to use
nb_layers=int(sys.argv[4])
## 5th argument is the number of epochs
nb_epochs=int(sys.argv[5])

## argument 6 -> 6+nb_layers-1 correspond to the number of hidden units


## Variable to save the intermediate results
history = History()

########## Creation of the optimizer (adam) and the model 
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = Sequential()
model.add(Dense(units=int(sys.argv[6]),input_dim=64 ))
for ii in range(1,nb_layers):
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.02))
    model.add(Dense(units=int(sys.argv[6+ii])))
model.add(Activation('softmax'))
print("The model has been initiated with Sequential.")

########## Activate TensorBord in order to observe the evolution of the training live
#tbCallBack = TensorBoard(log_dir='./TensorBoard/'+str(name), histogram_freq=0, write_graph=True, write_images=True)


############ Read normalize and prepare the data (training + validation)
n = sum(1 for line in open(filename_A))  #number of records in file (excludes header)
s = database_size; #desired sample size
skip = sorted(random.sample(range(0,n),n-1-s)) #range -> xrange if python 2
df_A = pandas.read_csv(filename_A, skiprows=skip)
df_G = pandas.read_csv(filename_G, skiprows=skip)
x_train=df_G.as_matrix()
y_train=df_A.as_matrix()
y_train=y_train/y_train.sum(axis=1, keepdims=True)

df_Aval = pandas.read_csv(filename_Aval)
df_Gval = pandas.read_csv(filename_Gval)
x_val=df_Gval.as_matrix()
y_val=df_Aval.as_matrix()
y_val=y_val/y_val.sum(axis=1, keepdims=True)

######### Compile the model
model.compile(loss='kullback_leibler_divergence',
        optimizer=adam,
        metrics=['mae'])
history=model.fit(x_train, y_train, epochs=nb_epochs, initial_epoch=0,batch_size=25000,validation_data=(x_val,y_val)) #, callbacks=[tbCallBack]


######## Save the results
f_handle = open(str(name)+'_mae.csv', 'a')
np.savetxt(f_handle, history.history['mean_absolute_error'])
f_handle.close()
f_handle = open(str(name)+'_val_mae.csv', 'a')
np.savetxt(f_handle, history.history['val_mean_absolute_error'])
f_handle.close()

model.save(str(name)+'_model.h5')
model.save_weights(str(name)+'_weights.h5')
