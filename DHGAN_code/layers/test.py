from hyperbolic.PoincareManifold import PoincareManifold
import torch
poincare = PoincareManifold(0, 0)
a = torch.tensor([30], dtype=torch.int32)
b = poincare.exp_map_zero(a)
print(b)