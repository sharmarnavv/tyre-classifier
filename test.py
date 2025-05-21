import torch, time
x = torch.randn(1024,1024, device='cuda')
t0 = time.time()
for _ in range(100):
    x = torch.mm(x, x)
torch.cuda.synchronize()
print("Elapsed (ms):", (time.time()-t0)*1000)
