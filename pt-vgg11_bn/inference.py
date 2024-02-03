#!/usr/bin/env python
import os

import numpy as np
import torch
import torchvision
import tqdm
import vart
import xir


def get_child_subgraph_dpu(graph: xir.Graph) -> list[xir.Subgraph]:
    """obtain dpu subgrah.
    """
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


xmodel_path = 'vgg11_bn.xmodel'
assert os.path.exists(xmodel_path)

# dataloader & transoform info
transform_test = torchvision.models.vgg.VGG11_Weights.IMAGENET1K_V1.transforms()
testset = torchvision.datasets.ImageNet(
    root='.', split='val', transform=transform_test
)

testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

g = xir.Graph.deserialize(xmodel_path)
subgraphs = get_child_subgraph_dpu(g)
assert len(subgraphs) == 1  # only one DPU kernel

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

num_correct = 0
for i, (inputs, targets) in tqdm.tqdm(enumerate(testloader)):
    inputData[0] = (inputs * input_scale).to(torch.int8, memory_format=torch.channels_last).numpy()

    # run
    job_id = dpu_runner.execute_async(inputData, outputData)
    dpu_runner.wait(job_id)
    num_correct += (np.argmax(outputData[0]) == targets.item())

    if (i > 0) and (i % 100 == 0):
        print(num_correct * 100. / i)

print(f'accuracy: {num_correct * 100 / (len(testset))} %')
