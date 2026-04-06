import torch
from src.spd_frechet import WishartSPDDataset, FrechetDRNN
ds = WishartSPDDataset(n=10, n_responses=2)
X, Y = ds[0]
print('Dataset shapes:')
print('X:', X.shape)
print('Y:', Y.shape)
model = FrechetDRNN(input_dim=12, n_ref=10, n_responses=2, response_rank=2)
W = model(X.unsqueeze(0))
print('Model output W:', W.shape)