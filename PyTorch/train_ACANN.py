from ACANN import ACANN
from Database import Database
from torch.nn.modules.loss import KLDivLoss,L1Loss, SmoothL1Loss
from torch.optim import Adam,Rprop,Adamax, RMSprop,SGD,LBFGS
from torch.utils.data import DataLoader
import torch


print("Starting ACANN")
# Create the network
model = ACANN(64,1024,[128,256,512],drop_p=0.09).double()

print("Model created")
# Import the data
train_data = Database(csv_target="../Database/A_training.csv",csv_input="../Database/nl_training.csv",nb_data=100000).get_loader()
validation_data=Database(csv_target="../Database/A_validation.csv",csv_input="../Database/nl_validation.csv",nb_data=1000).get_loader()

trainloader = DataLoader(train_data,batch_size=2500,shuffle=True)
validationloader = DataLoader(validation_data,batch_size=1000)
print("Data Loaded")


# Define a function for computing the validation score
def validation_score(nn_model):
    nn_model.eval()
    val_error=L1Loss()
    with torch.no_grad():
        G_val,A_val=next(iter(validationloader))
        prediction=nn_model.forward(G_val)
        score=val_error(prediction,A_val)
    nn_model.train()
    return score.item()


#Define the loss
error = L1Loss()
#Define the optimizer
optimizer = Adam(model.parameters())
#RMSPRO 10 - 2e-3
#ADAM 10 - 1.2e-3

# Training parameters
epochs = 1000
step=-1
print_every = 250
print("Starting the training")

# Training
for e in range(epochs):
    model.train()
    #  Load a minibatch
    for G,A in trainloader:
        step+=1
        # restart the optimizer
        optimizer.zero_grad()
        # compute the loss
        prediction = model.forward(G)
        loss = error(prediction,A)
        # Compute the gradient and optimize
        loss.backward()
        optimizer.step()

        # Write the result
        if step % print_every == 0:
            step=0
            print("Epoch {}/{} : ".format(e+1,epochs),
                  "Training MAE = {} -".format(loss.item()),
                  "Validation MAE = {}".format(validation_score(model)))
torch.save(model.state_dict(),'checkpoint_noise.pth')

