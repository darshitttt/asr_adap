import torch.nn as nn
import torch

class LinearAdapter(nn.Module):
    '''
    Linear Adapter for whisperx pipeline.
    To be placed between encoder and decoder
    '''

    def __init__(self, input_dim=1280, hidden_dim=640):
        super(LinearAdapter, self).__init__()
        # Define the linear layers
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Apply the first linear layer
        x = self.linear1(x)
        x = torch.relu(x)  # Use ReLU activation
        # Apply the second linear layer
        x = self.linear2(x)
        x = torch.relu(x)  # Use ReLU activation
        # Apply the output layer to ensure the output shape is the same as the input
        x = self.output_layer(x)
        return x