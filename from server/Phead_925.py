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
            nn.Linear(hidden_mlp, pseudo,bias=False),
            nn.BatchNorm1d(pseudo, affine=False)
        )
        # self.projection_head[6].bias.requires_grad=False
        self.pfc = nn.Sequential(
            nn.Linear(pseudo, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True), # hidden layer
            nn.Linear(pred_dim, pseudo)) # output layer
        

    def forward(self, x1):
        feat1 = self.net(x1)
        emb1 = self.projection_head(feat1)
        # emb = nn.functional.normalize(emb,dim=-1)
        py1 = self.pfc(emb1)
        # feat2 = self.net(x2)
        # emb2 = self.projection_head(feat2)
        # py2 = self.pfc(emb2)
        return emb1.detach(),py1

class Phead(nn.Module):
    def __init__(self, net, dim_in=2048, hidden_mlp=2048, pred_dim=512, pseudo=3000, linear_fc=False, use_ema=False):
        super().__init__()
        self.net = net
        self.use_ema = use_ema
        if linear_fc:
            self.projection_head = nn.Sequential(
                nn.Linear(dim_in, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, pseudo),
            )


            self.pfc = nn.Linear(pseudo, pseudo)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(dim_in, hidden_mlp,bias = False),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, hidden_mlp,bias = False),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, pseudo,bias=False),
                nn.BatchNorm1d(pseudo, affine=False)
            )
        
            self.pfc = nn.Sequential(
                nn.Linear(pseudo, pred_dim,bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True), # hidden layer
                nn.Linear(pred_dim, pseudo)) # output layer
            if self.use_ema:
                self.net_ema = copy.deepcopy(net)
                self.projection_head_ema = copy.deepcopy(self.projection_head)
                for param_ema in self.net_ema.parameters():
                    param_ema.requires_grad = False
                for param_ema in self.projection_head_ema.parameters():
                    param_ema.requires_grad = False
    @torch.no_grad()
    def momentum_update_key_encoder(self, m):
        for param, param_ema in zip(self.net.parameters(), self.net_ema.parameters()):
            param_ema.data = param_ema.data * m + param.data * (1. - m)
        for param, param_ema in zip(self.projection_head.parameters(), self.projection_head_ema.parameters()):
            param_ema.data = param_ema.data * m + param.data * (1. - m)
   
    def forward(self, x1):
        feat1 = self.net(x1)
        emb1 = self.projection_head(feat1)
        py1 = self.pfc(emb1)
        #feat2 = self.net(x2)
        #emb2 = self.projection_head(feat2)
        #py2 = self.pfc(emb2)
        if self.use_ema:
            feat1_ema = self.net_ema(x1)
            emb1_ema = self.projection_head_ema(feat1_ema)
            return emb1_ema.detach(),py1
        return emb1.detach(),py1
