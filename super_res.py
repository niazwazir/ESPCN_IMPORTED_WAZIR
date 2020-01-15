import argparse
import torch

from PIL import Image
from torchvision.transforms import ToTensor


def main():
    to_upscale = Image.open(args.input_image).convert('YCbCr')
    y, cb, cr = to_upscale.split()

    model_in = torch.load(args.model)
    image_to_tensor = ToTensor()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DRRA Project")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', help="enable cuda?")
    args = parser.parse_args()

    print("Input args:" + str(args))

    main()