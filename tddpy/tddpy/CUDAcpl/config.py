import torch


#here you can control the device and dtype used by CUDAcpl (_U_, particularly)

class Config:
    device = 'cpu'
    dtype = torch.float64

    @staticmethod
    def setting_update(device_cuda: bool, double_type: bool) -> None:
        if device_cuda:
            Config.device = 'cuda'
        else:
            Config.device = 'cpu'

        if double_type:
            Config.dtype = torch.float64
        else:
            Config.dtype = torch.float32

#torch.set_printoptions(precision=15)
