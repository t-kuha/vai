"""deploying .xmodel.
"""
import argparse
import os

import torch
import torchvision
import tqdm
import vgg
from pytorch_nndct.apis import torch_quantizer


if __name__ == '__main__':
    """main processing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['float', 'calib', 'deploy'], help='process mode')
    args = parser.parse_args()

    quant_mode = args.mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch_model_path = 'vgg19.pth'
    model = vgg.VGG('VGG19')
    
    # state_dict = torch.load(torch_model_path, map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    state_dict = torch.load(torch_model_path, map_location=torch.device('cpu'))['net']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join('..', '.dataset'), train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # inference of float model
    if quant_mode == 'float':
        model.eval()
        num_correct = 0
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(testloader):
                outputs = model(inputs.to(device))
                num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
        print(f'accuracy: {num_correct.item() * 100 / 10000} %')

    # quantization
    if quant_mode == 'calib':
        input = torch.randn([1, 3, 32, 32])
        quantizer = torch_quantizer(quant_mode, model, (input), device=torch.device(device))
        # quant_model = quantizer.quant_model
        quantizer.quant_model(input)
        quantizer.export_quant_config()

        num_correct = 0
        for inputs, targets in tqdm.tqdm(testloader):
            outputs = quantizer.quant_model(inputs.to(device))
            num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
        print(f'accuracy: {num_correct.item() * 100 / 10000} %')

    # deploy .xmodel
    if quant_mode == 'deploy':
        input = torch.randn([1, 3, 32, 32])
        quantizer = torch_quantizer('test', model, (input), device=torch.device(device), target='DPUCZDX8G_ISA1_B1600')
        quantizer.quant_model(input)
        quantizer.export_xmodel(deploy_check=True)
