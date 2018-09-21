import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ACANN(nn.Module):
    def __init__(self,input_size,output_size,hidden_layers,drop_p=0.05):
        """  Builds ACANN network with arbitrary number of hidden layers.

        Arguments
        ----------
        input_size : integer, size of the input
        output_size : integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers
        drop_p: float in (0,1) , value of the dropout probability
        """
        super().__init__()
        # Add the first layer : input_size into the first hidden layer
        self.layers = nn.ModuleList([nn.Linear(input_size,hidden_layers[0]).to(device)])
        self.normalizations = nn.ModuleList([nn.BatchNorm1d(input_size).to(device)])

        # Add the other layers
        layers_sizes = zip(hidden_layers[:-1],hidden_layers[1:])
        self.layers.extend([nn.Linear(h1,h2).to(device) for h1,h2 in layers_sizes])
        self.normalizations.extend([nn.BatchNorm1d(size).to(device) for size in hidden_layers])

        self.output=nn.Linear(hidden_layers[-1],output_size).to(device)
        self.dropout = nn.Dropout(drop_p).to(device)

    def forward(self,x):
        # pass through each layers
        for layer,normalization in zip(self.layers,self.normalizations):
            x=normalization(x)
            x=F.relu(layer(x))
            x=self.dropout(x)
        x=self.output(x)
        return F.softmax(x,dim=1)
