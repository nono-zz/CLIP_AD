from torch import Tensor, nn
import torch
from torch.nn import functional as F
from sklearn.metrics import pairwise
import numpy as np
class TextAdapter(nn.Module):
    def __init__(self, text_embeddings, label=None, beta=5.5):
        super(TextAdapter, self).__init__()
        #self.text_layer = nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0], bias=False).to(text_embeddings.device)
        #self.text_layer.weight = nn.Parameter(text_embeddings)
        text_embeddings = torch.cat((text_embeddings[...,0],text_embeddings[...,1]),dim=0)
        self.ad = torch.nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0])
        #self.n =  nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0], bias=False).to(text_embeddings.device)
        self.text_embeddings = text_embeddings
        #self.weights = nn.Parameter(text_embeddings)
        #self.label = F.one_hot(label.to(torch.int64)).float()
        self.noise_level = 1
        self.mask_ratio = 0.25
        self.beta = beta

    #def init_parameter(self,):
        #self.ad.weight.data = self.text_embeddings
        #self.weights.data = self.text_embeddings

    def adapter(self,img):
        img = img / img.norm(dim=-1, keepdim=True)
        
        affinity = self.ad(img)#F.linear(img, self.weights)#img @ self.text_embeddings.T #(N,C)
        affinity = torch.tanh(affinity)
        #output = affinity
        #affinity = F.softmax(affinity, dim=1)
        #affinity = F.normalize(affinity, p=2, dim=1)
        #text_length = self.text_embeddings.shape[0]
        #logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.label
        output = F.linear(affinity, self.text_embeddings.t())
        #assert 1==2, img.shape
        #logits = (affinity*2/text_length) @ self.label
        #N, H, W, C = img.shape
        #img = 0.5*img.view(N,H*W,C)+0.5*self.ad(img.view(N,H*W,C))
        #output = img.view(N, H, W, C)
        #output = self.ad(img)
        return output

    def mask_aug(self, true_feats):
        N, H, W, C = true_feats.shape

        ids_noise = torch.rand(N, H*W, device=true_feats.device)
        ids_shuffle = torch.argsort(ids_noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_mask = int(H*W * self.mask_ratio)

        noise = torch.normal(0, 0.05 * 1.1**2, true_feats.shape).to(true_feats.device)
        fake_feats = [true_feats]
        noise_masks = []
        for i in range(int(1/self.mask_ratio)):
            mask = torch.zeros([N, H*W], device=true_feats.device)
            if i != int(1/self.mask_ratio):
                mask[:, i*len_mask:(i+1)*len_mask] = 1
            else:
                mask[:, i*len_mask:] = 1
            noise_mask = torch.gather(mask, dim=1, index=ids_restore)
            noise_masks.append(noise_mask)
            fake_feat = true_feats + noise*noise_mask.view(N,H,W,1)
            fake_feats.append(fake_feat)
        return torch.stack(fake_feats,dim=0).view(-1, H, W, C), torch.stack(noise_masks,dim=0).view(-1,H,W,1)

    def aug(self, true_feat):
        N, H, W, C = true_feat.shape
        feat_list = [true_feat]
        for n in range(self.noise_level):
            noise = torch.normal(0, 0.05 * 1.1 ** (n+1), true_feat.shape).to(true_feat.device)
            fake_feat = true_feat + noise
            feat_list.append(fake_feat)
        return torch.stack(feat_list,dim=0).view(-1, H, W, C)
        '''N, L, C = true_feat.shape
        feat_list = [true_feat]
        for n in range(self.noise_level):
            noise = torch.normal(0, 0.05 * 1.1 ** (n+1), true_feat.shape).to(true_feat.device)
            fake_feat = true_feat + noise
            feat_list.append(fake_feat)
        return torch.stack(feat_list,dim=0).view(-1, L, C)'''

    def forward(self, x, is_test=False,scale=0.1):
        if not is_test:
            x = self.aug(x)
        if len(x.shape)==4:
            N, H, W, C = x.shape
            x = 0.5*x.view(N,H*W,C)+0.5*self.adapter(x.view(N,H*W,C))
            x = x.view(N, H, W, C)
        else:
            x = 0.5*x+0.5*self.adapter(x)
        #x = torch.nn.functional.softmax(x, dim=-1)
        #assert 1==2, x
        return x


class Adapter(nn.Module):
    def __init__(self, text_embeddings, label=None, beta=5.5):
        super(Adapter, self).__init__()
        #self.text_layer = nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0], bias=False).to(text_embeddings.device)
        #self.text_layer.weight = nn.Parameter(text_embeddings)
        text_embeddings = torch.cat((text_embeddings[...,0],text_embeddings[...,1]),dim=0)
        self.ad = torch.nn.Linear(text_embeddings.shape[1], text_embeddings.shape[1])
        #self.n =  nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0], bias=False).to(text_embeddings.device)
        #self.text_embeddings = text_embeddings
        #self.weights = nn.Parameter(text_embeddings)
        #self.label = F.one_hot(label.to(torch.int64)).float()
        self.noise_level = 1
        self.mask_ratio = 0.25
        self.beta = beta

    #def init_parameter(self,):
        #self.ad.weight.data = self.text_embeddings
        #self.weights.data = self.text_embeddings

    def adapter(self,img):
        img = img / img.norm(dim=-1, keepdim=True)
        output = self.ad(img)
        return output

    def aug(self, true_feat):
        N, H, W, C = true_feat.shape
        feat_list = [true_feat]
        for n in range(self.noise_level):
            noise = torch.normal(0, 0.05 * 1.1 ** (n+1), true_feat.shape).to(true_feat.device)
            fake_feat = true_feat + noise
            feat_list.append(fake_feat)
        return torch.stack(feat_list,dim=0).view(-1, H, W, C)

    def forward(self, x, is_test=False,scale=0.1):
        if not is_test:
            x = self.aug(x)         # adding noise to the original patch token
        if len(x.shape)==4:
            N, H, W, C = x.shape
            x = 0.5*x.view(N,H*W,C)+0.5*self.adapter(x.view(N,H*W,C))
            x = x.view(N, H, W, C)
        else:
            x = 0.5*x+0.5*self.adapter(x)
        #x = torch.nn.functional.softmax(x, dim=-1)
        #assert 1==2, x
        return x

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model_name, model):
        super(LinearLayer, self).__init__()
        if 'ViT' in model_name:
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        else:
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])
        self.ln = model.visual.ln_post
        self.proj = model.visual.proj

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                #tokens[i] = self.fc[i](tokens[i][:, 1:, :])
                tokens[i] = self.ln(tokens[i][:, 1:, :]) @ self.proj
            else:
                assert 1==2,"Not completed!"
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
    
class ReverseProjection(nn.Module):
    def __init__(self, dim_in, dim_out, k, model_name, model):
        super(ReverseProjection, self).__init__()
        self.ln = model.visual.ln_post
        self.project = model.visual.proj
        self.inverse_project = torch.pinverse(self.project)
    
    def forward(self, text_token):
        text_token = text_token.permute(0, 2, 1)
        text_token = self.ln(text_token @ self.inverse_project)
        text_token = text_token.permute(0, 2, 1)
        # text_token = self.ln(torch.matmul(text_token, self.inverse_project))
        return text_token

class FNN(nn.Module):
    def __init__(self, input_dim=640, hidden_dim=640, dropout_prob=0.5):
        super(FNN, self).__init__()

        # First linear layer with batch normalization, ReLU, and dropout
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        # Second linear layer with batch normalization, ReLU, and dropout
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        # Third linear layer with batch normalization, ReLU, and dropout
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x