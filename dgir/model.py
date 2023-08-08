import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from typing import Type, Union, List, Tuple


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: Type[Union[int, Tuple]],
        output_shape: Type[Union[int, Tuple]],
        hiddens: List[int] = [64, 64],
        activation: nn.Module = nn.LeakyReLU(0.1, inplace=True),
    ) -> None:
        super(MLP, self).__init__()

        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hiddens = hiddens

        layers = []
        prev_h = np.prod(input_shape)
        for h in hiddens + [np.prod(output_shape)]:
            layers.append(nn.Linear(prev_h, h))
            layers.append(activation)
            prev_h = h
        layers.pop() # remove last activation and dropout
        
        self.net = nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x).view(b, *self.output_shape)


class DenseGridNet(nn.Module):
    def __init__(
        self,
        resolution_x: Type[int] = 64,
        resolution_y: Type[int] = 64,
        feature_length: Type[int] = 4,
        mlp_hiddens: List[int] = [64, 64],
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super(DenseGridNet, self).__init__()
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.feature_length = feature_length
        # create and init embedding table
        self.embeddings = nn.Embedding(self.resolution_x*self.resolution_y, self.feature_length)
        nn.init.uniform_(self.embeddings[0].weight, a=-1e-4, b=1e4)

        self.grid_x = torch.linspace(0, 1, self.resolution_x, dtype=float)
        self.grid_y = torch.linspace(0, 1, self.resolution_y, dtype=float)

        id_feature_length = 1
        self.embed_dim = id_feature_length + feature_length
        self.mlp = MLP(
            input_shape=self.embed_dim,
            output_shape=3,
            hiddens=mlp_hiddens,
            activation=activation,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def _apply(self, fn):
        super(DenseGridNet, self)._apply(fn)
        self.grid_x = fn(self.grid_x)
        self.grid_y = fn(self.grid_y)
        return self

    def get_vert_ids(self, uv):
        """
        Inputs:
            uv, coordinate values for x and y axis in range [0, 1]
        Output:
            vertice ids[x0, x1, y0, y1] of the bounding cell, in below order:
                (x0, y0) (x1, y0)
                (x0, y1) (x1, y1)
            weights of distance on x and y axis to (x0, y0)
        """
        w_xy = torch.empty_like(uv)
        
        x0 = (uv[:,0] * self.resolution_x).int()
        x0[x0 == self.resolution_x] = 0
        x1 = x0 + 1
        x1[x1 == self.resolution_x] = self.resolution_x-1
        
        y0 = (uv[:,1] * (self.resolution_y)).int()
        y1 = y0 + 1
        y1[y1 == self.resolution_y] = self.resolution_y-1

        w_xy[:,0] = uv[:,0]*self.resolution_x - x0.float()
        w_xy[:,1] = uv[:,1]*(self.resolution_y) - y0.float()
        
        return torch.stack([x0, x1, y0, y1]).permute(1,0), w_xy
    
    def get_embedding_idx(self, xid, yid):
        return yid*self.resolution_x+xid
    
    def get_grid_features(self, uv):
        vids, wxy = self.get_vert_ids(uv)
        x0, x1, y0, y1 = vids[:,0], vids[:,1], vids[:,2], vids[:,3]
        wx, wy = wxy[:,0][:,None], wxy[:,1][:,None]
        # bilinear interpolation
        v00 = self.embeddings(self.get_embedding_idx(x0, y0))
        v10 = self.embeddings(self.get_embedding_idx(x1, y0))
        v01 = self.embeddings(self.get_embedding_idx(x0, y1))
        v11 = self.embeddings(self.get_embedding_idx(x1, y1))
        
        vup = v00*(1-wx)+v10*wx
        vdown = v01*(1-wx)+v11*wx
 
        v = vup*(1-wy)+vdown*wy
        return v
    
    def forward(self, x) -> Tensor:
        idf = x[:, :1]
        uv = x[:, 1:3]
        dgf = self.get_grid_features(uv)
        return self.sigmoid(self.mlp(torch.hstack([idf, dgf])))