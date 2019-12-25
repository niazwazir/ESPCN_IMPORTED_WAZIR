import torch
import torch.nn as nn
import torch.optim as optim
from scripts.nnet_model import ESPCN


def training(cuda_in, seed_in, upscale_in, lr_in, nEpochs_in,
             train_dataset, valid_dataset):

    if cuda_in and not torch.cuda.is_available():
        print("No CUDA supporting GPU found, using CPU")
        cuda_in = False
    
    device = torch.device("cuda" if cuda_in else "cpu")
    
    torch.manual_seed(seed_in)

    model = ESPCN(upscale_in, 1).to(device)  # 1 is for number of channels
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=lr_in)