import torch


model_path = 'flow.pth'

checkpoint = torch.load(model_path)

for k, v in checkpoint.items():
    print(k, v)
