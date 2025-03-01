"""code for on-device inference.

Usage:
    $ python inference.py <model name>
"""
import argparse
import os
import tqdm
import numpy as np
import torch
import torchvision

import vart
import xir


def get_child_subgraph_dpu(xmodel_path: str) -> list[xir.Subgraph]:
    """obtain dpu subgrah.
    """
    assert os.path.exists(xmodel_path)

    graph = xir.Graph.deserialize(xmodel_path)
    assert graph, '"graph" should not be None.'

    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph, 'Failed to get root subgraph of input Graph object.'
    if root_subgraph.is_leaf:
        return []

    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs
    return [
        cs for cs in child_subgraphs
        if cs.has_attr('device') and cs.get_attr('device').upper() == 'DPU'
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_name', type=str,
        choices=['vgg19', 'resnet', 'xception', 'inceptionv3', 'densenet'], help='model name'
    )
    args = parser.parse_args()

    xmodel_name = f'{args.model_name}_cifar10.xmodel'

    # load .xmodel
    subgraphs = get_child_subgraph_dpu(xmodel_name)
    assert len(subgraphs) == 1  # only one DPU kernel

    print('[INFO] loading dataset...')
    normalize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR10(
        root='.', download=True, train=False, transform=normalize
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    print('[INFO] creating DPU runner...')
    dpu_runner = vart.Runner.create_runner(subgraphs[0], 'run')
    input_fixpos = dpu_runner.get_input_tensors()[0].get_attr('fix_point')
    input_scale = 2**input_fixpos

    inputTensors = dpu_runner.get_input_tensors()
    outputTensors = dpu_runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])

    output_ndim = tuple(outputTensors[0].dims)
    output_fixpos = outputTensors[0].get_attr('fix_point')
    output_scale = 1 / (2**output_fixpos)

    # prepare batch input/output
    inputData = [np.empty(input_ndim, dtype=np.int8, order='C')]
    outputData = [np.empty(output_ndim, dtype=np.int8, order='C')]

    print('[INFO] starting inference...')
    num_correct = 0
    for inputs, targets in tqdm.tqdm(testloader):
        inputData[0] = (inputs * input_scale).to(torch.int8, memory_format=torch.channels_last).numpy()

        # run
        job_id = dpu_runner.execute_async(inputData, outputData)
        dpu_runner.wait(job_id)
        num_correct += (np.argmax(outputData[0]) == targets.item())

    print(f'[INFO] accuracy: {num_correct * 100 / (len(testset)):.2f} %')
