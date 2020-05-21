import torch.nn as nn
import torch.nn.functional as F
import torch

class ImageEncoder(nn.Module):
    
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(in_features=128*28*28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)

    def forward(self, x):
        # x -> [-1, 3, 256, 256]
        
        x = F.relu(self.conv1(x))
        # x -> [-1, 32, 254, 254]
        x = F.relu(self.conv2(x))
        # x -> [-1, 32, 252, 252]
        x = F.max_pool2d(x, 2)
        # x -> [-1, 32, 126, 126]
        
        x = F.relu(self.conv3(x))
        # x -> [-1, 64, 124, 124]
        x = F.relu(self.conv4(x))
        # x -> [-1, 64, 122, 122]
        x = F.max_pool2d(x, 2)
        # x -> [-1, 64, 61, 61]

        x = F.relu(self.conv5(x))
        # x -> [-1, 128, 59, 59]
        x = F.relu(self.conv6(x))
        # x -> [-1, 128, 57, 57]
        x = F.max_pool2d(x, 2)
        # x -> [-1, 128, 28, 28]

        x = x.view(-1, 128*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ContextEncoder(nn.Module):

    def __init__(self):
        super(ContextEncoder, self).__init__()
        self.rnn = nn.RNN(input_size=19, hidden_size=128, num_layers=2, batch_first=True)
    
    def forward(self, x, h=None):
        # x -> [-1, seq_size, 19], h -> [num_layer=2,-1, 128]

        if not h:
            h = torch.zeros((2, x.size(0), 128)).cuda()

        x, _ = self.rnn(x, h)
        return x

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(input_size=1024+128, hidden_size=512, num_layers=2, batch_first=True)
        self.l1 = nn.Linear(512, 19)
    
    def forward(self, image_feature, context_feature, on_cuda = False, h = None):
        # image_feature -> [-1, 1024], context_feature -> [-1, seq_size=48, 128], h -> [num_layer=2, -1, 512]
        image_feature = image_feature.unsqueeze(1)
        # image_feature -> [-1, 1, 1024]
        image_feature = image_feature.repeat(1, context_feature.size(1), 1)
        # image_feature -> [-1, seq_size, 1024]
        x = torch.cat((image_feature, context_feature), 2)
        # x -> [-1, seq_size=48, 1024+128]

        if not h:
            h = torch.zeros((2, x.size(0), 512)).cuda()

        x, _ = self.rnn(x, h)
        x = self.l1(x)
        # x = F.softmax(x, dim=1)
        return x

class Pix2Code(nn.Module):

    def __init__(self):
        super(Pix2Code, self).__init__()
        self.image_encoder = ImageEncoder()
        self.context_encoder = ContextEncoder()
        self.decoder = Decoder()

    def forward(self, image, context):
        image_feature = self.image_encoder(image)
        context_feature = self.context_encoder(context)
        output = self.decoder(image_feature, context_feature)
        return output
