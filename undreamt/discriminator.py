import torch.nn as nn
import torch.nn.functional as F
import torch
class LinearDiscriminator(nn.Module):
    def __init__(self, device, hidden_size,linear_size=100,dropout = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.layer_list = [nn.Linear(self.hidden_size,self.linear_size)]
        self.dropout=nn.Dropout(dropout)
        temp=self.linear_size
        self.device = device
        while temp > 1:
            self.layer_list.append(nn.ReLU())
            self.layer_list.append(nn.Linear(temp,temp//10))
            temp = temp//10
        self.layers = nn.Sequential(*self.layer_list)
        self.log_sigmoid = nn.LogSigmoid()
    def forward(self,encoder_output,lengths):
        probs = []
        for i in range(encoder_output.shape[1]):
            sentence = encoder_output[:lengths[i],i,:]
            output = self.layers(sentence)
            log_sigmoid_out = self.log_sigmoid(output)
            probs.append(torch.exp(torch.sum(log_sigmoid_out)))
        return torch.stack(probs)
#further exploration required.Gotta make a structure first
class ConvDiscriminator(nn.Module):
        def __init__(self,device,hidden_size,num_filters,filter_size = 5):
            super().__init__()
            self.hidden_size = hidden_size
            self.filter_size = filter_size
            self.num_filters = num_filters
            self.layer_list = nn.ModuleList([ nn.Conv2d(1,self.num_filters,[hidden_size,x]) for x in range(1,self.filter_size+1) ])
            self.fc1_layer = nn.Linear(self.filter_size*self.num_filters,100)
            self.fc2_layer = nn.Linear(100,1)
            self.sigmoid = nn.Sigmoid()
            self.device = device
        def forward(self,encoder_output,lengths):
            probs = []
            #print("conv_para:-",self.layer_list[0].bias )
            for i in range(encoder_output.shape[1]):
                """print(encoder_output[:lengths[i],i,:].shape)
                print(encoder_output[:lengths[i],i,:].unsqueeze(0).unsqueeze(0))
                print(encoder_output[:lengths[i],i,:].unsqueeze(0).unsqueeze(0))"""
                sentence = encoder_output[:lengths[i],i,:].unsqueeze(0).unsqueeze(0).contiguous().view(1,1,self.hidden_size,-1)  #[1,1,hidden_size,seq_len]
                if self.filter_size > sentence.shape[-1]:
                    dummy = self.device(torch.zeros([1,1,self.hidden_size,self.filter_size - sentence.shape[-1]]))
                    sentence = torch.cat([sentence,dummy], dim = -1)
                conv_out = [conv(sentence) for conv in self.layer_list]  #[1,k,1,Wout]
                relu = [F.relu(conv).squeeze(2) for conv in conv_out]    #[1,k,Wout]
                pool = [F.max_pool1d(x,x.size(2)).squeeze(2) for x in relu]  #[1,k]*filter_size
                sentence = torch.cat(pool,dim=1)
                fc1 = self.fc1_layer(sentence)
                fc2 = self.fc2_layer(fc1)
                sigmoid_out = self.sigmoid(fc2)
                probs.append(sigmoid_out)
            return torch.stack(probs)
