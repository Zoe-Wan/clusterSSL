import torch
import torch.nn as nn
import copy


class Phead_with_pseudo(nn.Module):
    def __init__(self, net, dim_in=2048,hidden_mlp=2048, pred_dim=512, pseudo=3000):
        super().__init__()
        # self.net = nn.Sequential(*list(net.children())[:-1])
        self.net = net
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, hidden_mlp,bias=False),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, hidden_mlp,bias=False),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, pseudo),
            nn.BatchNorm1d(pseudo, affine=False)
        )
        self.pfc = nn.Sequential(
            nn.Linear(pseudo, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True), # hidden layer
            nn.Linear(pred_dim, pseudo)) # output layer
        

    def forward(self, x):
        feat = self.net(x)
        emb = self.projection_head(feat)

        py = self.pfc(emb)
        
        return emb,py


class Phead(nn.Module):
    def __init__(self, net, dim_in=2048, dim_feat=128,hidden_mlp=2048, dim_out=1000):
        super().__init__()
        self.net = net
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, dim_feat),
        )


        self.fc = nn.Linear(dim_feat, dim_out)


    def forward(self, x):
        feat = self.net(x)
        emb = self.projection_head(feat)
        y = self.fc(emb)
        return y