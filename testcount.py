import torch
import torch.distributed as dist
from src.utils import dist_init

rank,local_rank,world_size = dist_init(12345)
a=[local_rank,local_rank+1,local+2,9]
a=torch.tensor(a)
num_a = torch.bincount(a,minlength=10)
torch.all_reduce(num_a)
if rank==0:
    print(num_a)
