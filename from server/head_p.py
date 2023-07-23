import torch
import torch.nn as nn
import copy

class Phead_with_pseudo(nn.Module):
    def __init__(self, net, dim_in=2048, dim_feat=128,hidden_mlp=2048, dim_out=10, pseudo=3000,use_ema = False):
        super().__init__()
        # self.net = nn.Sequential(*list(net.children())[:-1])
        self.net = net
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, dim_feat),
        )
        self.use_ema = use_ema
        if self.use_ema:
            self.net_ema = copy.deepcopy(net)
            self.projection_head_ema = copy.deepcopy(self.projection_head)
            for param_ema in self.net_ema.parameters():
                param_ema.requires_grad = False
            for param_ema in self.projection_head_ema.parameters():
                param_ema.requires_grad = False


        self.fc = nn.Linear(dim_feat, dim_out)
        self.pfc = nn.Linear(dim_feat, pseudo)
        # self.l1_norm = l1_norm


    @torch.no_grad()
    def momentum_update_key_encoder(self, m):
        for param, param_ema in zip(self.net.parameters(), self.net_ema.parameters()):
            param_ema.data = param_ema.data * m + param.data * (1. - m)
        for param, param_ema in zip(self.projection_head.parameters(), self.projection_head_ema.parameters()):
            param_ema.data = param_ema.data * m + param.data * (1. - m)

    def forward(self, x):
        feat = self.net(x)
        emb = self.projection_head(feat)   
        if self.use_ema:
            with torch.no_grad():
                feat_ema = self.net_ema(x)
                emb_ema = self.projection_head_ema(feat_ema)
                emb_ema = nn.functional.normalize(emb_ema, dim=1, p=2)
        y = self.fc(emb) 

        emb = nn.functional.normalize(emb, dim=1, p=2)
        py = self.pfc(emb)
        if self.use_ema:
            return emb_ema, y, py
        else:
            return emb,y,py



