"""deploying Vitis-AI model.
"""
import argparse
import os
import subprocess
import sys

import torch
import torchvision
import tqdm


if __name__ == '__main__':
    """main processing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_name', type=str,
        choices=['wide_resnet50', 'wide_resnet101'], help='model name'
    )
    parser.add_argument(
        'mode', type=str,
        choices=['float', 'inspect', 'calib', 'deploy'], help='process mode'
    )
    parser.add_argument(
        '-d', '--dataset_dir', type=str,
        default=os.path.join('..', 'dataset', 'imagenet'),
        help='path to dataset (ImageNet validation) directory'
    )
    parser.add_argument(
        '-f', '--fingerprint', type=str,
        default='0x101000016010405',
        help='DPU name or fingerprint value'
    )
    args = parser.parse_args()

    if args.model_name == 'wide_resnet50':
        model_name = 'wide_resnet50_2'
        weights = torchvision.models.Wide_ResNet50_2_Weights.DEFAULT
    elif args.model_name == 'wide_resnet101':
        model_name = 'wide_resnet101_2'
        weights = torchvision.models.Wide_ResNet101_2_Weights.DEFAULT

    quant_mode = args.mode
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'{device=:}')

    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, weights=weights)

    if quant_mode == 'inspect':
        # inspect model
        from pytorch_nndct.apis import Inspector
        inspector = Inspector(args.fingerprint)
        inspector.inspect(model, torch.randn([1, 3, 224, 224]), torch.device(device))
        sys.exit(0)

    if quant_mode in ['float', 'calib']:
        # load dataset only when necessary
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(232),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        testset = torchvision.datasets.ImageNet(
            root=args.dataset_dir, split='val', transform=transform
        )

    model.to(device)
    model.eval()

    if quant_mode == 'float':
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        num_correct = 0
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(testloader):
                outputs = model(inputs.to(device))
                num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
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

    # deploy .xmodel
    if quant_mode == 'deploy':
        from pytorch_nndct.apis import torch_quantizer

        input = torch.randn([1, 3, 224, 224])
        quantizer = torch_quantizer('test', model, (input), device=torch.device(device))
        quantizer.quant_model(input)
        quantizer.export_xmodel(deploy_check=True)
        # at this point, .xmodel will be generated as quantize_result/*_int.xmodel

        res = subprocess.run([
            'xcompiler',
            '-i', 'quantize_result/ResNet_int.xmodel',
            '-o', f'{args.model_name}.xmodel',
            '-f', args.fingerprint
        ], capture_output=True)
        print(f'{res.returncode=:}')
