#  //// Data Representation, Reduction and Analysis Project         //////
#  //// Image Super Resolution: Using ESPCN                         //////
#  // Members: Andres Sánchez, Deniz Alp Atun, Ismail Göktuğ Serin  //////
#  ///////////////////////////////////////////////////////////////////////

import os.path
import argparse
import scripts.file_process as fp
import torch


def dir_in_dir(path):
    print("Does the zip include images on root or inside folder?")
    check = input("Type 1 for yes, 0 for no: ")
    if check:
        ext_path = input("Type the path until images are seen:\n" )
        return path, ext_path
    else:
        return path, None


def train_dir_input(root_in):
    while True:
        temp = input("Type in the relative path of training data set: ")
        test = os.path.join(root_in, temp)
        print("Path: " + str(test))
        if os.path.isfile(test):
            train_path = test
            break
        else:
            print("Path doesn't end with a file!")
    extr_path, folder_path = dir_in_dir(train_path)
    return extr_path, folder_path


def valid_dir_input(root_in):
    while True:
        temp = input("Type in the relative path of validation data set: ")
        test = os.path.join(root_in, temp)
        print("Path: " + str(test))
        if os.path.isfile(test):
            valid_path = test
            break
        else:
            print("Path doesn't end with a file!")
    extr_path, folder_path = dir_in_dir(valid_path)
    return extr_path, folder_path


def main():

    if args.cuda and not torch.cuda.is_available():
        print("No CUDA supporting GPU found, using CPU")
        args.cuda = False

    device = torch.device("cuda" if args.cuda else "cpu")

    print("\n ██████ Training and Validation Data Preparation ██████")
    print("Important! Place everything in the same folder, since this code uses relative paths!")
    print("Path input examples: dataset.zip or folder\\dataset.zip")
    print("-------------------------------------------------------\n")
    root_dir = os.path.realpath('.')
    train_dirs = train_dir_input(root_dir)
    valid_dirs = valid_dir_input(root_dir)

    train_dir = fp.prep_files(root_dir, train_dirs[0], train_dirs[1], "train")
    valid_dir = fp.prep_files(root_dir, valid_dirs[0], valid_dirs[1], "valid")

    print("\n ██████ Loading Data into Dataset ██████")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DRRA Project")
    parser.add_argument('--upscale', type=int, required=True, help="upscale factor")
    parser.add_argument('--trainBatchSize', type=int, required=True, help="training batch size")
    parser.add_argument('--validBatchSize', type=int, required=True, help="validation batch size")
    parser.add_argument('--nEpochs', type=int, required=True, help="number of epochs")
    parser.add_argument('--rate', type=float, required=True, help="learning rate")
    parser.add_argument('--cuda', action='store_true', help="enable cuda?")
    args = parser.parse_args()

    print("Input args:" + str(args))

    main()




