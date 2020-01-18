#  //// Data Representation, Reduction and Analysis Project         //////
#  //// Image Super Resolution: Using ESPCN                         //////
#  // Members: Andres Sánchez, Deniz Alp Atun, Ismail Göktuğ Serin  //////
#  ///////////////////////////////////////////////////////////////////////

import os
import argparse
from shutil import rmtree
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from math import log10

import scripts.file_process as fp
from scripts.dataset_maker import set_maker
from scripts.nnet_model_tanh import ESPCN as tanh_ESPCN
from scripts.nnet_model_leaky import ESPCN as leaky_ESPCN
from scripts.nnet_model_relu import ESPCN as relu_ESPCN


def dir_in_dir(path):
    print("Does the zip include images on root or inside folder?")
    check = input("Type 1 for yes, 0 for no: ")
    if check:
        ext_path = input("Type the path until images are seen:\n" )
        return path, ext_path
    else:
        return path, None


def dir_input(root_in, label):
    while True:
        temp = input("Type in the relative path of %s data set: " % label)
        test = os.path.join(root_in, temp)
        print("Path: " + str(test))
        if os.path.isfile(test):
            path_in = test
            break
        else:
            print("Path doesn't end with a file!")
    extr_path, folder_path = dir_in_dir(path_in)
    return extr_path, folder_path


def main():

    print("\n ██████ Training and Validation Data Preparation ██████")
    print("Important! Place everything in the same folder, since this code uses relative paths!")
    print("Path input examples: dataset.zip or folder\\dataset.zip")
    print("-------------------------------------------------------\n")
    root_dir = os.path.realpath('.')
    train_dirs = dir_input(root_dir, "training")
    valid_dirs = dir_input(root_dir, "validation")

    train_dir = fp.prep_files(root_dir, train_dirs[0], train_dirs[1], "train")
    valid_dir = fp.prep_files(root_dir, valid_dirs[0], valid_dirs[1], "valid")

    print("\n ██████ Loading Data into Dataset ██████")

    train_set = set_maker(train_dir, args.cropSize, args.upscale)
    valid_set = set_maker(valid_dir, args.cropSize, args.upscale)
    training_data = DataLoader(dataset=train_set, batch_size=args.trainBatchSize, shuffle=True,
                               num_workers=args.nWorkers)
    validation_data = DataLoader(dataset=valid_set, batch_size=args.validBatchSize, shuffle=True,
                               num_workers=args.nWorkers)

    if args.cuda and not torch.cuda.is_available():
        print("No CUDA supporting GPU found, using CPU")
        cuda_in = False
    else:
        cuda_in = True

    device = torch.device("cuda" if cuda_in else "cpu")

    torch.manual_seed(args.seed)
    if args.func == "leaky":
        model = leaky_ESPCN(args.upscale, 1).to(device)  # 1 is for number of channels
    elif args.func == "relu":
        model = relu_ESPCN(args.upscale, 1).to(device)  # 1 is for number of channels
    else:
        model = tanh_ESPCN(args.upscale, 1).to(device)  # 1 is for number of channels
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    log_path = "logs_scale_%i_crop_%i" % (args.upscale, args.cropSize)
    output_dir = os.path.join(root_dir, log_path)
    try:
        os.mkdir(output_dir)
    except:
        print("An error occured during mkdir")

    psnr_log = os.path.join(output_dir, "PSNR_value_list.log")

    for epoch in range(args.nEpochs):
        epoch_loss = 0
        iteration = 0
        epoch_log_path = "epoch_%i.log" % (epoch+1)
        epoch_log = os.path.join(output_dir, epoch_log_path)
        f1 = open(epoch_log, 'w')
        for data in training_data:
            input, label = data

            input = input.to(device)
            label = label.to(device)

            loss = criterion(model(input), label)

            optimiser.zero_grad()

            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()
            f1.write("Iteration [%i/%i]: Loss: %0.4f \n" % (iteration+1, len(training_data), loss.item()))
            iteration += 1
        f1.write("-----------------------------------\n")
        f1.write("Average Loss: %0.4f \n" % (epoch_loss / len(training_data)))
        f1.close()

        tot_psnr = 0
        f2 = open(psnr_log, 'a+')
        for data in validation_data:
            input, label = data

            input = input.to(device)
            label = label.to(device)

            with torch.no_grad():
                prediction = model(input)
                mse = criterion(prediction, label)
                psnr = 10 * log10(1 / mse.item())
                tot_psnr += psnr
        f2.write("PSNR of Epoch [%i]: %0.4f dB \n" % (epoch+1, tot_psnr))
        f2.close()

        model_name = "epoch_%i_model.pth" % (epoch+1)
        model_name = os.path.join(output_dir, model_name)
        torch.save(model, model_name)
        print("Epoch (%i/%i) is done! See root dir for logs and models" % (epoch+1, args.nEpochs))

    temp = os.path.join(root_dir,"extr")
    rmtree(temp, ignore_errors=True)

    # |-------------------------------------------------------------------------------------------------| #
    # Until this part, it is very similar to PyTorch-SuperResolution Example on Github
    # Link: https://github.com/pytorch/examples/tree/master/super_resolution
    # Changes are the way to acquire data files, their relative directory and how they are
    # passed to the code. Unlike other solutions, data is dynamically passed (by asking for input)
    # Rest of the code shows similarity since deviating from the example either resulted in
    # poor code readability, making an unnecessary wall of text or using additional libraries
    # such as h5py, numpy and etc.
    # |-------------------------------------------------------------------------------------------------| #
    # Next part is modelling a Network with upscale as input, training and validating by using the model,
    # PyTorch ReLU Network (torch.nn), PyTorch Optimiser (torch.optim) and so on...
    # |-------------------------------------------------------------------------------------------------| #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DRRA Project")
    parser.add_argument('--upscale', type=int, required=True, help="upscale factor")
    parser.add_argument('--trainBatchSize', type=int, required=True, help="training batch size")
    parser.add_argument('--validBatchSize', type=int, required=True, help="validation batch size")
    parser.add_argument('--nEpochs', type=int, required=True, help="number of epochs")
    parser.add_argument('--nWorkers', type=int, default=8, help="number of workers")
    parser.add_argument('--lr', type=float, required=True, help="learning rate")
    parser.add_argument('--cropSize', type=int, required=True, default=128, help="crop size")
    parser.add_argument('--func', type=str, required=True, default="tanh", help="choose a activator function: tanh, "
                                                                                "relu, leaky")
    parser.add_argument('--cuda', action='store_true', help="enable cuda?")
    parser.add_argument('--seed', type=int, default=42, help="random seed to use. Default=42")
    args = parser.parse_args()

    print("Input args:" + str(args))

    main()




