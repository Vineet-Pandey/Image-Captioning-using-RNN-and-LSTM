import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,num_layers, batch_first = True)
        
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = embeddings[:,:-1,:]
       # print ('embeddings shape is ', embeddings.shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings),1)
        hiddens,_ = self.lstm(embeddings)
        #hiddens = hiddens[:,1:,:]
        outputs = self.linear(hiddens)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample_id=[]
        for i in range (max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
           # print ('output is :', outputs.max(1))
            _, predicted = outputs.max(1)
            sample_id.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sample_id = torch.stack(sample_id,1)
        sample_id = sample_id.cpu()
        
        sample_id = sample_id.numpy()
        sample_id = sample_id.tolist()
        return sample_id 
        
        
        
        
        
        
            
            
        
        
        
        