"""deploying .xmodel.
"""
import argparse
import os
import subprocess
import sys
import tqdm
import torch
import torchvision

from trained_models_cifar10.models import vgg_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', type=str,
        choices=['float', 'inspect', 'calib', 'deploy'], help='process mode'
    )
    parser.add_argument(
        '-d', '--dataset_dir', type=str,
        default=os.path.join('.'),
        help='path to dataset directory'
    )
    parser.add_argument(
        '-f', '--fingerprint', type=str,
        default='0x101000016010405',
        help='DPU name or fingerprint value'
    )
    args = parser.parse_args()

    quant_mode = args.mode
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'{device=:}')

    model = vgg_models.VGG('VGG19')

    if quant_mode == 'inspect':
        # inspect model
        from pytorch_nndct.apis import Inspector
        inspector = Inspector(args.fingerprint)
        inspector.inspect(model, torch.randn([1, 3, 32, 32]), torch.device(device))
        sys.exit(0)

    # load dataset when mode != inspect
    normalize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR10(
        root=args.dataset_dir, download=True, train=False, transform=normalize
    )

    net_state_dict = torch.load('trained_models_cifar10/vgg19_cifar10_lr01.pth', map_location=device)['net']
    model.load_state_dict(net_state_dict)
    model.to(device)
    model.eval()

    if quant_mode == 'float':
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        num_correct = 0
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(testloader):
                outputs = model(inputs.to(device))
                num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
            print(outputs)
        print(f'accuracy: {num_correct.item() * 100 / len(testset)} %')

    if quant_mode == 'calib':
        from pytorch_nndct.apis import torch_quantizer

        # create batch for calibration
        input = torch.stack([testset[i][0] for i in range(0, len(testset), 5000)])
        quantizer = torch_quantizer(quant_mode, model, (input), device=torch.device(device))
        quantizer.quant_model(input)
        quantizer.export_quant_config()

        calibloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
        num_correct = 0
        SKIP = 100
        for i, (inputs, targets) in enumerate(tqdm.tqdm(calibloader)):
            if i % SKIP != 0:
                continue
            outputs = quantizer.quant_model(inputs.to(device))
            num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
        print(f'accuracy: {num_correct.item() * 100 / len(testset) * SKIP} %')

    # deploy .xmodel
    if quant_mode == 'deploy':
        from pytorch_nndct.apis import torch_quantizer

        input = torch.randn([1, 3, 32, 32])
        quantizer = torch_quantizer('test', model, (input), device=torch.device(device))
        quantizer.quant_model(input)
        quantizer.export_xmodel(deploy_check=True)
        # at this point, .xmodel will be generated as quantize_result/VGG_int.xmodel

        subprocess.run([
            'xcompiler',
            '-i', 'quantize_result/VGG_int.xmodel',
            '-o', 'vgg19_cifar10.xmodel',
            '-f', {args.fingerprint}
        ])
