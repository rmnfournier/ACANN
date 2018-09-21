# Romain Fournier
# EPFL - TP 4
# Creation : 19.03.2018
# Last edit :  10.04


from keras.models import Sequential
import random
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
import numpy as np
import csv
import pandas
from sklearn import metrics
import tensorflow as tf
from keras.callbacks import History
from keras.models import load_model

model_file="model.h5"
names=['ANNGl_sup21','ANNGl_sup22','ANNGl_sup23','ANNGl_sup24','ANNGl_sup25']


model = load_model(model_file)
print("The model has been loaded")


for name in names:
	features_file= name+".csv"
	x_test = pandas.read_csv(features_file,header=None)
	x_matrix= x_test.as_matrix()
	prediction = model.predict(x_matrix)
	np.savetxt('prediction_'+str(name)+'.csv', prediction,delimiter=',')


