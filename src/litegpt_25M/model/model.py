import torch
import torch.nn.functional as F
import torch.nn as nn
from omegaconf import OmegaConf 

model_cfg = OmegaConf.load("./configs/model/LiteGPT-25M.yaml")




# class LiteGPT(nn.Module):

#     def __init__(self) -> None:

