import torch
torch.backends.cuda.matmul.allow_tf32 = False

theta = torch.pi/3
theta = torch.pi * 0
theta = torch.tensor([theta])
alpha = torch.rand(1) * torch.pi * 2

n = torch.cat([torch.cos(theta),torch.cos(alpha) * torch.sin(theta),torch.sin(alpha) * torch.sin(theta)]).requires_grad_()
theta_pred = torch.atan2(torch.norm(n[1:]), n[0])
theta_pred2 = torch.arccos(n[0])

grad = torch.autograd.grad(theta_pred**2, n)[0]
grad2 = torch.autograd.grad(theta_pred2**2, n)[0]
print(theta_pred, grad)
print(theta_pred2, grad2)
print(grad[1:])
