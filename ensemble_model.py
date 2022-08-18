import torch.nn.functional as F
import torch.nn as nn
from os import path
import torch

# The cell block 1 which represents tge base network

class Base(nn.Module):

    def __init__(self):

        super(Base, self).__init__()

        # Layers of Base network
        
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
       
    
    # The base network forward iter function

    def forward(self, input_data):

        layer_rep_1 = self.conv1(input_data)

        batch_1 = self.bn1(layer_rep_1)
       
        output = F.relu(batch_1)

        layer_rep_2 = self.conv2(output)

        batch_2 = self.bn2(layer_rep_2)

        relu_1 = F.relu(batch_2)

        output = self.pool(relu_1)

        layer_rep_3 = self.conv3(output)

        batch_3 = self.bn3(layer_rep_3)

        output = F.relu(batch_3)

        layer_rep_4 = self.conv4(output)

        batch_4 = self.bn4(layer_rep_4)

        output = self.pool(F.relu(batch_4))

        return output


# The ensemble constituting the convolutional branches

class ConvolutionalBranch(nn.Module):
    

    def __init__(self):

        # The layers of the convolutional branch

        super(ConvolutionalBranch, self).__init__()

       
        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, 3, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # The Linear layer for discrete emotion and continuous range prediction

        
        self.fc = nn.Linear(512, 8)
        self.fc_dimensional = nn.Linear(8, 2)

       

    # The forward function for the convolutional branch
        

    def forward(self, input_data):

        layer_rep_1 = self.conv1(input_data)

        batch_1 = self.bn1(layer_rep_1)

        output = F.relu(batch_1)

        layer_rep_2 = self.conv2(output)

        batch_2 = self.bn2(layer_rep_2)

        output = self.pool(F.relu(batch_2))

        layer_rep_3 = self.conv3(output)

        batch_3 = self.bn3(layer_rep_3)

        output = F.relu(batch_3)

        layer_rep_4 = self.conv4(output)

        batch_4 = self.bn4(layer_rep_4)

        output = self.global_pool(F.relu(batch_4))

        output = output.view(-1, 512)

        emot_dis = self.fc(output)
        
        output = F.relu(emot_dis)

        emot_cont = self.fc_dimensional(output)

        return emot_dis, emot_cont


class Net_Model(nn.Module):


    '''The network model is referenced from https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/ml/esr_9.py'''
       

    def __init__(self, device):

        super(Net_Model, self).__init__()


        # The trained networks are saved here

        model_net = "Base_network_model.pt"
        model_file = "ensemble_{}.pt"
        model_saved = "./model/trained_models/"
        
        self.base = Base()

        model_path = path.join(model_saved, model_net)

        self.base.load_state_dict(torch.load(model_path, map_location = device))

        self.base.to(device)

        self.conv_result = []

        for i in range(1, len(self) + 1):

            self.conv_result.append(ConvolutionalBranch())

            mod_path = path.join(model_saved,model_file.format(i))

            self.conv_result[-1].load_state_dict(torch.load(mod_path, map_location=device))
            self.conv_result[-1].to(device)

        self.to(device)

        self.eval()

    # The forward function for prediction

    def forward(self, val):
       
        mod_rep = self.base(val)

        result = []
        affect_result = []

        for branch in self.conv_result:

            emot, aff = branch(mod_rep)

            result.append(emot)

            affect_result.append(aff)

        return result, affect_result

    def __len__(self):

        # Since we have set the number of branches to be 9

        branch = 9
        
        return branch
