import torch


#here you can control the device and dtype used by CUDAcpl (_U_, particularly)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#device = 'cpu'

dtype = torch.float64
torch.set_printoptions(precision=15)
