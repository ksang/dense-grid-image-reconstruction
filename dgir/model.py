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
        resolution_finest: List[int] = [512, 512],
        resolution_coarsest: List[int] = [16, 16],
        num_levels: int = 3,
        feature_length: Type[int] = 4,
        mlp_hiddens: List[int] = [64, 64],
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super(DenseGridNet, self).__init__()
        self.resolution_x = torch.linspace(resolution_finest[0], resolution_coarsest[0], num_levels).int()
        self.resolution_y = torch.linspace(resolution_finest[0], resolution_coarsest[0], num_levels).int()
        self.feature_length = feature_length
        self.num_levels = num_levels
        # create and init embedding table for each level
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding((self.resolution_x[lvl]+1)*(self.resolution_y[lvl]+1), self.feature_length)
                for lvl in range(self.num_levels)
            ]
        )
        
        for lvl in range(self.num_levels):
            nn.init.uniform_(self.embeddings[lvl].weight, a=-1e-4, b=1e-4)

        id_feature_length = 1
        self.embed_dim = id_feature_length + feature_length*num_levels
        self.mlp = MLP(
            input_shape=self.embed_dim,
            output_shape=3,
            hiddens=mlp_hiddens,
            activation=activation,
        )

    def _apply(self, fn):
        super(DenseGridNet, self)._apply(fn)
        self.resolution_x = fn(self.resolution_x)
        self.resolution_y = fn(self.resolution_y)
        return self

    def get_vert_ids(self, uv, level=0):
        """
        Inputs:
            uv, coordinate values for x and y axis in range [0, 1]
            level, indicates which resolution level to use
        Output:
            vertice ids[x0, x1, y0, y1] of the bounding cell, in below order:
                (x0, y0) (x1, y0)
                (x0, y1) (x1, y1)
            weights of distance on x and y axis to (x0, y0)
        """
        w_xy = torch.empty_like(uv)
        
        x0 = (uv[:,0] * self.resolution_x[level]).int()
        x1 = x0 + 1
        x1[x1 >= self.resolution_x[level]] = self.resolution_x[level]
        
        y0 = (uv[:,1] * (self.resolution_y[level])).int()
        y1 = y0 + 1
        y1[y1 >= self.resolution_y[level]] = self.resolution_y[level]

        w_xy[:,0] = uv[:,0]*self.resolution_x[level] - x0.float()
        w_xy[:,1] = uv[:,1]*self.resolution_y[level] - y0.float()
        
        return torch.stack([x0, x1, y0, y1]).permute(1,0), w_xy
    
    def get_embedding_idx(self, level, xid, yid):
        return yid*self.resolution_x[level]+xid
    
    def get_grid_features(self, uv):
        all_features = []
        for lvl in range(self.num_levels):
            vids, wxy = self.get_vert_ids(uv, lvl)
            x0, x1, y0, y1 = vids[:,0], vids[:,1], vids[:,2], vids[:,3]
            wx, wy = wxy[:,0][:,None], wxy[:,1][:,None]
            # bilinear interpolation
            v00 = self.embeddings[lvl](self.get_embedding_idx(lvl, x0, y0))
            v10 = self.embeddings[lvl](self.get_embedding_idx(lvl, x1, y0))
            v01 = self.embeddings[lvl](self.get_embedding_idx(lvl, x0, y1))
            v11 = self.embeddings[lvl](self.get_embedding_idx(lvl, x1, y1))
            
            fup = v00*(1-wx)+v10*wx
            fdown = v01*(1-wx)+v11*wx
    
            feature = fup*(1-wy)+fdown*wy
            all_features.append(feature)
        return torch.hstack(all_features)
    
    def forward(self, x) -> Tensor:
        idf = x[:, :1]
        uv = x[:, 1:3]
        dgf = self.get_grid_features(uv)
        return self.mlp(torch.hstack([idf, dgf]))