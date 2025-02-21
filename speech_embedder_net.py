#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018

@author: harry
"""

import torch
import torch.nn as nn

from hparam import hparam as hp
from utils import get_centroids, get_cossim, calc_loss

class SpeechEmbedder(nn.Module):
    
    def __init__(self):
        super(SpeechEmbedder, self).__init__()    
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)
        
    def forward(self, x):
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x

class GE2ELoss(nn.Module):
    
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device
        
    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss
    
    

class GE2ELoss_Contrastive(nn.Module):

    def __init__(self, device):
        super(GE2ELoss_Contrastive, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device


    def compute_loss(self, sim_matrix):
        """
        sim_matrix: (S, U, S)
        """
        S = sim_matrix.shape[0]
        U = sim_matrix.shape[1]
        same_idx = list(range(sim_matrix.size(0)))
        # (S, U)
        pos = torch.sigmoid(sim_matrix[same_idx, :, same_idx])
        # Mask the similarity of the same speaker
        # (S, U, S)
        mask = torch.eye(S).unsqueeze(1).repeat(1, U, 1) == 0
        # (S, U, S-1)
        neg = torch.sigmoid(sim_matrix[mask].view(S, U, S-1))
        # (S, U)
        neg, _ = torch.max(neg, dim=2)
        per_embedding_loss = 1 - pos + neg
        loss = per_embedding_loss.sum()
        return loss, per_embedding_loss

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss, _ = self.compute_loss(sim_matrix)
        return loss


