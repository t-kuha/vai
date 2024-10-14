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
    parser.add_argument('mode', type=str, choices=['float', 'calib', 'deploy'], help='process mode')
    parser.add_argument(
        '-d', '--dataset_dir', type=str,
        default=os.path.join('..', '.dataset', 'imagenet'), help='path to dataset (ImageNet validation) directory'
    )
    args = parser.parse_args()

    assert os.path.exists(args.dataset_dir), f'{args.dataset_dir = :}'

    quant_mode = args.mode
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else ('cpu')
    
    model = torchvision.models.googlenet(
        weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1
    )
    testset = torchvision.datasets.ImageNet(
        root=args.dataset_dir,
        split='val',
        transform=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1.transforms()
    )
    
    # weight_to_remove = [
    #     'aux1.conv.conv.weight',
    #     'aux1.conv.bn.weight',
    #     'aux1.conv.bn.bias',
    #     'aux1.conv.bn.running_mean',
    #     'aux1.conv.bn.running_var',
    #     'aux1.conv.bn.num_batches_tracked', 
    #     'aux1.fc1.weight',
    #     'aux1.fc1.bias',
    #     'aux1.fc2.weight',
    #     'aux1.fc2.bias',
    #     'aux2.conv.conv.weight',
    #     'aux2.conv.bn.weight',
    #     'aux2.conv.bn.bias',
    #     'aux2.conv.bn.running_mean',
    #     'aux2.conv.bn.running_var',
    #     'aux2.conv.bn.num_batches_tracked',
    #     'aux2.fc1.weight',
    #     'aux2.fc1.bias',
    #     'aux2.fc2.weight',
    #     'aux2.fc2.bias'
    # ]
    # for k in weight_to_remove:
    #     del state_dict[k]

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
        for i, (inputs, targets) in enumerate(tqdm.tqdm(calibloader)):
            outputs = quantizer.quant_model(inputs.to(device))
            num_correct += torch.sum(torch.argmax(outputs, 1) == targets.to(device))
            # if (i > 0) and (i % 5 == 0):
            #     print(num_correct * 100. / ((i + 1) * 16))
        print(f'accuracy: {num_correct.item() * 100 / len(testset)} %')

    # deploy .xmodel
    if quant_mode == 'deploy':
        from pytorch_nndct.apis import torch_quantizer

        input = torch.randn([1, 3, 224, 224])
        quantizer = torch_quantizer('test', model, (input), device=torch.device(device))
        quantizer.quant_model(input)
        quantizer.export_xmodel(deploy_check=True)
        # at this point, .xmodel will be generated as quantize_result/GoogLeNet_int.xmodel
