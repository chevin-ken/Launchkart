import torch
import torch.nn as nn
import torch.nn.functional as F

class MarioKartBCAgent(nn.Module):
    def __init__(self):
        super(MarioKartBCAgent, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=2), 
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2, padding=2), 
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2, padding=2), 
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(kernel_size=2))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Sequential(nn.Linear(768, 100), nn.Dropout(0.2))
        self.fc2 = nn.Sequential(nn.Linear(100, 50), nn.Dropout(0.2))
        self.fc3 = nn.Sequential(nn.Linear(50, 10), nn.Dropout(0.2))
        self.fc4 = nn.Sequential(nn.Linear(10, 5))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)      
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x 

class TensorKartAgent(nn.Module):
  def __init__(self):
    super(TensorKartAgent, self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=2), 
                                   nn.ReLU())
    self.conv2 = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2, padding=2), 
                                nn.ReLU())
    self.conv3 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2, padding=2), 
                                nn.ReLU())
    self.conv4 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1), 
                                nn.ReLU())
    self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
                                nn.ReLU())
    self.fc1 = nn.Sequential(nn.Linear(14400, 1164), nn.Dropout(0.2))
    self.fc2 = nn.Sequential(nn.Linear(1164, 100), nn.Dropout(0.2))
    self.fc3 = nn.Sequential(nn.Linear(100, 50), nn.Dropout(0.2))
    self.fc4 = nn.Sequential(nn.Linear(50, 10), nn.Dropout(0.2))
    self.fc5 = nn.Sequential(nn.Linear(10, 5), nn.Softsign())

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = torch.flatten(x, 1)      
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    x = self.fc5(x)
    return x 

class MarioKartBCAgentV2(nn.Module):
    def __init__(self):
        super(MarioKartBCAgentV2, self).__init__()
        
        #Feature Extraction Module: Conv layers (check out Conv2d, BatchNorm, Relu, Maxpool)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=36, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        
        #Inference Module: Fully connected layers (check out Linear, Dropout, Relu)
        self.fc1 = nn.Sequential(nn.Linear(211200, 1000), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1000, 100), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(100, 25), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(25, 5))
        
    def forward(self, x):
        #Pass x through conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        #Flatten x to prepare for passing into linear
        x = torch.flatten(x, 1) 
        
        #Pass x through linear layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        #Name variable output on last layer
        output = x
        
        return output    # return x for visualization

class MarioKartBCAgentJulia(nn.Module):
    def __init__(self):
        super(MarioKartBCAgentJulia, self).__init__()
        
        #Feature Extraction Module: Conv layers (check out Conv2d, BatchNorm, Relu, Maxpool)
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        
        self.conv3 = nn.Sequential(         
            nn.Conv2d(32, 64, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        
        self.conv4 = nn.Sequential(         
            nn.Conv2d(64, 128, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        
        self.conv5 = nn.Sequential(         
            nn.Conv2d(128, 256, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        #Inference Module: Fully connected layers (check out Linear, Dropout, Relu)
        self.inf1 = nn.Sequential(         
            nn.Linear(3072, 100),
            nn.Dropout(0.2),
            nn.ReLU(),                                    
        )
        self.inf2 = nn.Sequential(         
            nn.Linear(100, 50),
            nn.Dropout(0.2),
            nn.ReLU(),                                    
        )
        self.inf3 = nn.Sequential(         
            nn.Linear(50, 10),
            nn.Dropout(0.2),
            nn.ReLU(),                                    
        )
        self.inf4 = nn.Sequential(nn.Linear(10, 5))
        
    def forward(self, x):
        #Pass x through conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        #Flatten x to prepare for passing into linear
        x = torch.flatten(x, 1) 
        
        #Pass x through linear layers
        x = self.inf1(x)
        x = self.inf2(x)
        x = self.inf3(x)
        x = self.inf4(x)
        
        #Name variable output on last layer
        output = x
        
        return x    # return x for visualization

class MarioKartBCAgentV3(nn.Module):
    def __init__(self):
        super(MarioKartBCAgentV3, self).__init__()
        
        #Feature Extraction Module: Conv layers (check out Conv2d, BatchNorm, Relu, Maxpool)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        
        #Inference Module: Fully connected layers (check out Linear, Dropout, Relu)
        self.fc1 = nn.Sequential(nn.Linear(768, 256), nn.Dropout(0.2), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.Dropout(0.2), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.Dropout(0.2), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(64, 5), nn.Softsign())
        
    def forward(self, x):
        #Pass x through conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        #Flatten x to prepare for passing into linear
        x = torch.flatten(x, 1) 
        
        #Pass x through linear layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        #Name variable output on last layer
        return x

class MarioKartBCAgentWinston(nn.Module):
    def __init__(self):
        super(MarioKartBCAgentWinston, self).__init__()
        
        #Feature Extraction Module: Conv layers (check out Conv2d, BatchNorm, Relu, Maxpool)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=36, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        
        #Inference Module: Fully connected layers (check out Linear, Dropout, Relu)
        self.fc1 = nn.Sequential(nn.Linear(211200, 1000), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1000, 100), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(100, 25), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(25, 5))
        
    def forward(self, x):
        #Pass x through conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        #Flatten x to prepare for passing into linear
        x = torch.flatten(x, 1) 
        
        #Pass x through linear layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        #Name variable output on last layer
        output = x
        
        return output    # return x for visualization