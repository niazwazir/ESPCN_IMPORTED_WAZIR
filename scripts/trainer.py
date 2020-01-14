import torch
import torch.nn as nn
import torch.optim as optim
from scripts.nnet_model import ESPCN
from math import log10


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

    f1 = open("PSNR_value_list.log", 'w')
    for epoch in range(nEpochs_in):
        epoch_loss = 0
        iteration = 0
        f = open("epoch_%i.log" % epoch, 'w')
        for data in train_dataset:
            input, label = data

            input = input.to(device)
            label = label.to(device)

            loss = criterion(model(input), label)

            optimiser.zero_grad()

            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()
            f.write("Iteration [%i/%i]: Loss: %0.4f" % (iteration, len(train_dataset), loss.item()))
            iteration += 1
        f.write("-"*60)
        f.write("Average Loss: %0.4f" % (epoch_loss/len(train_dataset)))
        f.close()

        avg_psnr = 0
        for data in valid_dataset:
            input, label = data

            input = input.to(device)
            label = label.to(device)

            with torch.no_grad():
                prediction = model(input)
                mse = criterion(prediction, label)
                psnr = 10 * log10(1/mse.item())
                avg_psnr += psnr
        f1.write("Average PSNR of Epoch [%i]: %0.4f dB" % (epoch, (avg_psnr/len(train_dataset))))

        model_name = "epoch_%i_model.pth" % epoch
        torch.save(model, model_name)
        print("Epoch (%i/%i) is done! See root dir for logs and models" % (epoch, len(train_dataset)))
    f1.close()

