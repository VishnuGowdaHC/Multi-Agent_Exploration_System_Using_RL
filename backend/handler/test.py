import torch
ckpt = torch.load("backend/handler/checkpoints/framework_v1_final.pt", map_location="cpu", weights_only=False)
print(ckpt["state_dict"]["net.4.weight"].shape)