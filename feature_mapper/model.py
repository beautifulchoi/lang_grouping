import torch.nn as nn

class MapperMLP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MapperMLP, self).__init__()
        self.layer1 = nn.Conv2d(input_channels, output_channels//2, kernel_size=1)
        self.relu1 = nn.ReLU()
        
        self.layer2 = nn.Conv2d(output_channels//2, output_channels, kernel_size=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        
        
        return x
