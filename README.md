Solving an Analytic Continuation problem with a deep Artificial Neural Network (ANN)

# 1. Problem Description
 We are interested in the analytic continuation of the imaginary Green's function into the real frequency domain. We use a supervised learning approach for efficiently solving this problem. This method has both advantages of regularizing the problem by distilling prior knowledge into the training dataset and being robust against noisy inputs. Interested readers can look at our paper to get further details.

# 2. Getting Started 
 We propose two different implementations of our model. One using PyTorch and the other one TensorFlow, as well as a Matlab file that allows quickly building a dataset. 

## 2.1 Build the dataset
 Getting the imaginary Green's function from the electron density function is a stable process, which allows generating a dataset of training examples by a simple two steps process. 
1. Generate a set of spectral density functions that look like the problem to handle (e.g., spectral density functions containing a quasiparticle peak).
2. Convert them into Green's function in imaginary time. For performance increase, it is better to save the Legendre coefficients of this function instead of the whole function.
 
The folder Generate Data offers a Matlab implementation of this process, used in the previously mentioned paper. 

## 2.2 Train the model
 We offer two ways of training the  Deep Neural Network: TensorFlow and Pytorch.
### 2.2.1 TensorFlow
 The Folder TensorFlow contains the necessary files to train the neural network. It requires Keras to be installed on the machine.  Taping the following commands launches the training :

  'python train_ACANN.py data_location nb_data name nb_layers nb_epochs units' ,
  
where '**data_location**' is the path to the folder containing the files (A_data.csv, nl_data.csv, A_validationset.csv, nl_validationset.csv), 

'**nb_data**' is the number of data to consider in the training dataset, 

'**name**' is the prefix of the output files, 

'**nb_layers**' is the number of dense layers to use, 

'**nb_epochs**' is the number of epochs to train on, 

and '**units**' is the number of weights in the different layers. 

For example : python train_ACANN.py ./Dataset 25000 test 3 100 1024 1024

### 2.2.2 PyTorch
 The Folder PyTorch contains the necessary files to train the neural network. One can modify the first line of 'train_ACANN.py' to use it : 

* *model = ACANN(64,1024,[128,256,512],drop_p=0.09).double()*

ACANN(input_dim,output_dim,list(hiddenunits),dropout rate)

* *train_data = Database(csv_target="../Database/A_training.csv",csv_input="../Database/nl_training.csv",nb_data=100000).get_loader()*

* *validation_data=Database(csv_target="../Database/A_validation.csv",csv_input="../Database/nl_validation.csv",nb_data=1000).get_loader()*

## 2.3 Make predictions
 Once the model is trained, one can use it to make predictions thanks to ACANN.py file.
One can add different filenames to the list 'names' and provide the model to use in the model_file variable.

