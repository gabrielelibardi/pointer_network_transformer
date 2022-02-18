import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math



#x = torch.arange(3).unsqueeze(-1).unsqueeze(-1).expand(3, 1, 4).float()
seq_len = 3
batch_size = 1
input_dim = 4
dim_model=4
n_heads = 2
dropout = 0.0
n_lyrs = 12


x = torch.rand(3,1,4)
decoder_lyr = nn.TransformerDecoderLayer(input_dim, n_heads, dim_model, dropout)
decoder = nn.TransformerDecoder(decoder_lyr, n_lyrs)


lr = 0.1 # learning rate
optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss(reduction='mean')
criterion = nn.MSELoss()
dummy_labels = torch.rand([seq_len, batch_size, input_dim])

for i in range(1000):
    
    loss = criterion(decoder(dummy_labels, x), dummy_labels)
    #loss = torch.sum((decoder(x, x) - dummy_labels)**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)

print(decoder(x, x))
print(dummy_labels)