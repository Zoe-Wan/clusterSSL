import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.distributed as dist

class Phead(nn.Module):
    def __init__(self, net, dim_in=2048, hidden_mlp=2048, pred_dim=256, num_class=1000, pseudo=3000, alpha=0.5, t = 0.1, t_ema = 0.04, center_momentum=0.9):
        super().__init__()
        self.net = net
        self.register_buffer("center", torch.zeros(1, pseudo))

        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, pred_dim),
        )
        self.pfc = nn.Linear(pred_dim, pseudo)
        self.fc = nn.Linear(pred_dim, num_class)
        self.num_class = num_class
        self.t = t
        self.t_ema = t_ema
        self.alpha = alpha
        self.center_momentum = center_momentum


        self.net_ema = copy.deepcopy(net)
        self.projection_head_ema = copy.deepcopy(self.projection_head)
        for param_ema in self.net_ema.parameters():
            param_ema.requires_grad = False
        for param_ema in self.projection_head_ema.parameters():
            param_ema.requires_grad = False
        for param_ema in self.pfc_ema.parameters():
            param_ema.requires_grad = False

    @torch.no_grad()
    def momentum_update_key_encoder(self, m):
        for param, param_ema in zip(self.net.parameters(), self.net_ema.parameters()):
            param_ema.data = param_ema.data * m + param.data * (1. - m)
        for param, param_ema in zip(self.projection_head.parameters(), self.projection_head_ema.parameters()):
            param_ema.data = param_ema.data * m + param.data * (1. - m)
        for param, param_ema in zip(self.pfc.parameters(), self.pfc_ema.parameters()):
            param_ema.data = param_ema.data * m + param.data * (1. - m)

    @torch.no_grad()
    def update_center(self, py_ema, label_mask):
        batch_center = torch.zeros(ema_py[0].size())
        batch_center[label_mask==1] = torch.sum(ema_py, dim=0, keepdim=True)
        dist.all_reduce(batch_center,op = ReduceOp.SUM)
        batch_center = batch_center / (len(ema_py) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
   
    def forward(self, x1, x2, label):
        feat1 = self.net(x1)
        emb1 = self.projection_head(feat1)
        emb1 = F.normalize(emb1,dim=-1,p=2)
        py1 = self.pfc(emb1)
        y1 = self.fc(emb1)

        feat2_ema = self.net_ema(x2)
        emb2_ema = self.projection_head_ema(feat2_ema)
        emb2_ema = F.normalize(emb2_ema,dim=-1,p=2)
        py2_ema = self.pfc_ema(emb2_ema)
        
        onehot = self.make_one_hot(label,self.num_class,rep=1)
        label_mask = self.make_one_hot(label,self.pseudo,rep=self.pseudo//self.num_class)

        q = F.softmax((py2_ema-self.center)*label_mask/self.t_ema, dim=-1)
        q = q.detach()

        loss1 = -(F.log_softmax(y1,dim=-1)*onehot).sum(1).mean()
        loss2 = -(F.log_softmax(py1/self.t)*q).sum(1).mean()
        return loss1+self.alpha*loss2, target

    @torch.no_grad()
    def make_one_hot(label,classes,rep=1):
        
        n = label.size()[0]
        one_hot = torch.FloatTensor(n, classes).zero_().to(label.device)
        # one_hot = torch.FloatTensor(n, classes*rep).zero_().to(label.device)
        target = one_hot.scatter_(1, label.data, 1)
        target = target.unsqueeze(2).repeat(1,1,rep).view(target.size(0),-1)
        return target



class P(nn.Module):
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
