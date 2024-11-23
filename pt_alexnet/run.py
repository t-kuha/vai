"""deploying .xmodel.
"""
import argparse
import os

import torch
import torchvision
import tqdm


if __name__ == '__main__':
    """main processing.
    """
    parser = argparse.ArgumentParser()
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


    quant_mode = args.mode
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    )
    if quant_mode != 'inspect':
        assert os.path.exists(args.dataset_dir), f'{args.dataset_dir = :}'
        testset = torchvision.datasets.ImageNet(
            root=args.dataset_dir,
            split='val',
            transform=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1.transforms()
        )

    model.to(device)
    model.eval()
    
    # inference of float model
    if quant_mode == 'float':
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        num_correct = 0
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(testloader):
                outputs = model(inputs.to(device))
                num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
        print(f'accuracy: {num_correct.item() * 100 / len(testset)} %')

    # inspect model
    if quant_mode == 'inspect':
        from pytorch_nndct.apis import Inspector
        inspector = Inspector(args.fingerprint)
        inspector.inspect(model, torch.randn([1, 3, 224, 224]), torch.device(device))

    # quantization
    if quant_mode == 'calib':
        from pytorch_nndct.apis import torch_quantizer

        # create batch for calibration
        input = torch.stack([testset[i][0] for i in range(0, len(testset), 1000)])
        quantizer = torch_quantizer(quant_mode, model, (input), device=torch.device(device))
        quantizer.quant_model(input)
        quantizer.export_quant_config()

        calibloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
        num_correct = 0
        # SKIP = 100
        for i, (inputs, targets) in enumerate(tqdm.tqdm(calibloader)):
            # if i % SKIP != 0:
            #     continue
            outputs = quantizer.quant_model(inputs.to(device))
            num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
        print(f'accuracy: {num_correct.item() * 100 / len(testset):.3f} {num_correct.item()} {len(testset)}%')

    # deploy .xmodel
    if quant_mode == 'deploy':
        from pytorch_nndct.apis import torch_quantizer

        input = torch.randn([1, 3, 224, 224])
        quantizer = torch_quantizer('test', model, (input), device=torch.device(device))
        quantizer.quant_model(input)
        quantizer.export_xmodel(deploy_check=True)
        # at this point, .xmodel will be generated as quantize_result/GoogLeNet_int.xmodel
