"""deploying .xmodel.
"""
import argparse
import os

import torch
import torchvision
import tqdm
from pytorch_nndct.apis import torch_quantizer


if __name__ == '__main__':
    """main processing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['float', 'calib', 'deploy'], help='process mode')
    args = parser.parse_args()

    quant_mode = args.mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = torchvision.models.vgg11_bn(weights='DEFAULT')
    transform_test = torchvision.models.vgg.VGG11_Weights.IMAGENET1K_V1.transforms()
    testset = torchvision.datasets.ImageNet(
        root=os.path.join('..', '.dataset', 'imagenet'),
        split='val', transform=transform_test)

    # inference of float model
    if quant_mode == 'float':
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
        model.eval()
        num_correct = 0
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(testloader):
                outputs = model(inputs.to(device))
                num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
        print(f'accuracy: {num_correct.item() * 100 / len(testset)} %')

    # quantization
    if quant_mode == 'calib':
        # create batch for calibration
        input = torch.stack([testset[i][0] for i in range(0, len(testset), 1000)])
        quantizer = torch_quantizer(quant_mode, model, (input), device=torch.device(device))
        quantizer.quant_model(input)
        quantizer.export_quant_config()

        calibloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        num_correct = 0
        for inputs, targets in tqdm.tqdm(calibloader):
            outputs = quantizer.quant_model(inputs.to(device))
            num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
        print(f'accuracy: {num_correct.item() * 100 / 50000} %')

    # deploy .xmodel
    if quant_mode == 'deploy':
        input = torch.randn([1, 3, 224, 224])
        quantizer = torch_quantizer('test', model, (input), device=torch.device(device))
        quantizer.quant_model(input)
        quantizer.export_xmodel(deploy_check=True)
        # at this point, .xmodel will be generated as quantize_result/VGG_int.xmodel
