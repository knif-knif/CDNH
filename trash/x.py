import torch.nn.functional as F
import torch

a = torch.tensor([[1, 1, 1]]).float()
b = torch.tensor([[1, 1, 1]]).float()

print(F.cosine_similarity(a, b))