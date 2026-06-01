import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=3, filters = None, kernel_size = 3, dense_neurons = 128, activation = "relu", dropout = 0.0, 
                batch_norm = False, num_classes=10):
        super().__init__()

        if filters is None:
            filters = [32,32,32,32,32]

        layers = []
        in_c = input_channels

        for out_c in filters:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(self.get_activation(activation))
            
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_c = out_c
        
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1] * 7 * 7, dense_neurons),
            self.get_activation(activation),
            nn.Dropout(dropout),

            nn.Linear(dense_neurons,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
    
    def get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu":
            return nn.SiLU()
        elif name == "mish":
            return nn.Mish()
