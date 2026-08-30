"""Small CNN advice model. Architecture novelty is intentionally not the goal."""
from __future__ import annotations
import torch
from torch import nn

class TinyConvSurrogate(nn.Module):
    def __init__(self,width=32,depth=3,dropout=0.05):
        super().__init__()
        if depth<1: raise ValueError("depth must be >= 1")
        layers=[nn.Conv1d(1,width,5,padding=2),nn.GELU()]
        for _ in range(depth-1):
            layers += [nn.Conv1d(width,width,5,padding=2),nn.GELU(),nn.Dropout(dropout)]
        layers.append(nn.Conv1d(width,1,5,padding=2))
        self.net=nn.Sequential(*layers)
    def forward(self,u):
        squeeze=u.ndim==2
        x=u[:,None,:] if squeeze else u
        if x.ndim!=3 or x.shape[1]!=1: raise ValueError("Expected shape (B,N) or (B,1,N).")
        y=x+self.net(x)
        return y[:,0,:] if squeeze else y

@torch.no_grad()
def mc_dropout_predictions(model,u,samples=8):
    if samples<2: raise ValueError("samples must be >= 2")
    was_training=model.training
    model.train()
    preds=torch.stack([model(u) for _ in range(samples)],dim=0)
    model.train(was_training)
    return preds
