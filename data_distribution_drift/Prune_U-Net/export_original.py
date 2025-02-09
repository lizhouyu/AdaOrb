import os
import torch
import torch.nn as nn
import torch.onnx
import argparse

from unet import UNet


def export(model_folder, output_path):
    model_path = os.path.join(model_folder, 'finetuned.pt')
    model_channels_path = os.path.join(model_folder, 'pruned_channels.txt')
    # check device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device: ", device)

    # set model path 
    # model_path = 'checkpoints/original/best.pt'
    # model_path = 'checkpoints/original/prune/finetuned.pt'

    # set output path
    # output_path = 'unet.onnx'
    # output_path = 'checkpoints/original/best.onnx'
    # output_path = 'checkpoints/original/prune/finetuned.onnx'


    # load model
    model = UNet(n_channels=3, n_classes=1, f_channels=model_channels_path)
    # model = UNet(n_channels=3, n_classes=1, f_channels='checkpoints/original/prune/pruned_channels.txt')
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # prepare arguments
    dummy_input = torch.randn(1, 3, 320, 320, device=device)
    input_names = ["in"]
    output_names = ["out"]
    dynamic_axes = {'in': {0: 'batch'}, 'out': {0: 'batch'}}

    # make output folder
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    torch.onnx.export(model, dummy_input, output_path, verbose=False, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('-m', '--model', type=str, help='model folder')
    parser.add_argument('-o', '--output', type=str, help='output path')
    args = parser.parse_args()
    export(args.model, args.output)
