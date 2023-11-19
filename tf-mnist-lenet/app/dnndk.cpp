#include <iostream>
#include <fstream>
#include <cstdint>
#include <memory>
#include <endian.h>

#include "dnndk/dnndk.h"


#define NODE_NAME_INPUT     "conv2d_Conv2D"
#define NODE_NAME_OUTPUT    "dense_1_MatMul"

int load_mnist(
    const std::string image_path,
    const std::string label_path,
    std::unique_ptr<uint8_t []>& image,
    std::unique_ptr<uint8_t []>& label,
    unsigned int& count
);

uint8_t argmax(const std::unique_ptr<int8_t []>& data);


int main(int argc, char* argv[])
{
    std::cout << "...... MNIST LeNet (DNNDK flow) ......" << std::endl;
    
    int ret = 0;

    // Final result
    std::unique_ptr<uint8_t []> result;

    // Load MNIST test data
    const std::string mnist_image = "t10k-images-idx3-ubyte";
    const std::string mnist_label = "t10k-labels-idx1-ubyte";
    std::unique_ptr<uint8_t []> label;
    std::unique_ptr<uint8_t []> image;
    unsigned int num = 0;

    ret = load_mnist(mnist_image, mnist_label, image, label, num);
    if(ret != 0){
        std::cerr << "[ERROR] load_mnist()" << std::endl;
        return -1;
    }

    // Allocate buffer for final result
    result.reset(new uint8_t [num]);

    // Inference
    DPUKernel* kernel = NULL;
    DPUTask* task = NULL;

    ret = dpuOpen();
    if(ret != 0){
        std::cerr << "[ERROR] dpuOpen()" << std::endl;
        return -1;
    }

    kernel = dpuLoadKernel("lenet_0");
    if(kernel == NULL){
        std::cerr << "[ERROR] dpuLoadKernel()" << std::endl;
        return -1;
    }

    task = dpuCreateTask(kernel, 0);
    if(task == NULL){
        std::cerr << "[ERROR] dpuCreateTask()" << std::endl;
        return -1;
    }

    // Get tensor info
    int in_size = dpuGetInputTensorSize(task, NODE_NAME_INPUT);
    float scale = dpuGetInputTensorScale(task, NODE_NAME_INPUT);

    int out_size = dpuGetOutputTensorSize(task, NODE_NAME_OUTPUT);

    // Store output from DPU
    std::unique_ptr<int8_t []> output(new int8_t [out_size]);

    // Input buffer to DPU
    int8_t* buf = new int8_t [in_size];

    std::cout << "[INFO] Start inference ......" << std::endl;
    for(unsigned int i = 0; i < num; i++){
        for(int j = 0; j < in_size; j++){
            buf[j] = (image.get()[28*28*i + j])/255.0*scale;
        }

        ret = dpuSetInputTensorInHWCInt8(task, NODE_NAME_INPUT, buf, in_size);
        if(ret != 0){
            std::cerr << "[ERROR] dpuSetInputTensorInHWCInt8()" << std::endl;
            return -1;
        }

        ret = dpuRunTask(task);
        if(ret != 0){
            std::cerr << "[ERROR] dpuRunTask()" << std::endl;
            return -1;
        }

        ret = dpuGetOutputTensorInCHWInt8(task, NODE_NAME_OUTPUT, output.get(), out_size);
        if(ret != 0){
            std::cerr << "[ERROR] dpuGetOutputTensorInCHWInt8()" << std::endl;
            return -1;
        }

        result.get()[i] = argmax(output);
    }
    delete [] buf;

    // Show final result
    int num_correct = 0;
    for(unsigned int i = 0; i < num; i++){
        if(label.get()[i] == result.get()[i]){
            num_correct++;
        }
    }
    std::cout << "Accuracy: " << num_correct << " / " << num << std::endl;

    // Release DPU
    ret = dpuDestroyTask(task);
    if(ret != 0){
        std::cerr << "[ERROR] dpuDestroyTask()" << std::endl;
        return -1;
    }

    ret = dpuDestroyKernel(kernel);
    if(ret != 0){
        std::cerr << "[ERROR] dpuDestroyKernel()" << std::endl;
        return -1;
    }

    ret = dpuClose();
    if(ret != 0){
        std::cerr << "[ERROR] dpuClose()" << std::endl;
        return -1;
    }

    return 0;
}

int load_mnist(
    const std::string image_path,
    const std::string label_path,
    std::unique_ptr<uint8_t []>& image,
    std::unique_ptr<uint8_t []>& label,
    unsigned int& count
)
{
    // TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    // [offset] [type]          [value]          [description]
    // 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    // 0004     32 bit integer  10000            number of items
    // 0008     unsigned byte   ??               label
    // 0009     unsigned byte   ??               label
    // ........
    // xxxx     unsigned byte   ??               label
    // The labels values are 0 to 9.
    //
    // TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    // [offset] [type]          [value]          [description]
    // 0000     32 bit integer  0x00000803(2051) magic number
    // 0004     32 bit integer  10000            number of images
    // 0008     32 bit integer  28               number of rows
    // 0012     32 bit integer  28               number of columns
    // 0016     unsigned byte   ??               pixel
    // 0017     unsigned byte   ??               pixel
    // ........
    // xxxx     unsigned byte   ??               pixel

    std::cout << "[INFO] Loading MNIST data ......" << std::endl;

    // label
    int32_t magic = 0, numl = 0;
    std::ifstream labelf(label_path, std::ios::in|std::ios::binary|std::ios::ate);
    if (!labelf.is_open()){
        std::cerr << "[ERROR] Opening label file: " << label_path << std::endl;
        return -1;
    }
    labelf.seekg (0, std::ios::beg);
    labelf.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    labelf.read(reinterpret_cast<char*>(&numl), sizeof(numl));
    magic = be32toh(magic);
    numl = be32toh(numl);

    label.reset(new uint8_t [numl]);
    labelf.read(reinterpret_cast<char*>(label.get()), numl);

    labelf.close();

    if(magic != 2049){
        // Check magic number
        std::cerr << "[ERROR] Bad magic number (label)..." << std::endl;
        return -1;
    }

    // image
    int32_t numi = 0, rows = 0, cols = 0;
    std::ifstream imagef(image_path, std::ios::in|std::ios::binary|std::ios::ate);
    if (!imagef.is_open()){
        std::cerr << "[ERROR] Opening image file: " << label_path << std::endl;
        return -1;
    }
    imagef.seekg (0, std::ios::beg);
    imagef.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    imagef.read(reinterpret_cast<char*>(&numi), sizeof(numi));
    imagef.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    imagef.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    magic = be32toh(magic);
    numi = be32toh(numi);
    rows = be32toh(rows);
    cols = be32toh(cols);

    image.reset(new uint8_t [numi*rows*cols]);
    imagef.read(reinterpret_cast<char*>(image.get()), numi*rows*cols);

    imagef.close();

    if(magic != 2051){
        // Check magic number
        std::cerr << "[ERROR] Bad magic number (image)..." << std::endl;
        return -1;
    }

    if(numl != numi){
        // Check number of items
        std::cerr << "[ERROR] Invalid number of image..." << std::endl;
        return -1;
    }

    count = numi;

    return 0;
}

uint8_t argmax(const std::unique_ptr<int8_t []>& data)
{
    uint8_t idx = 0;
    int8_t val = data.get()[0];
    for(uint8_t i = 1; i < 10; i++){
        if(data.get()[i] > val){
            idx = i;
            val = data.get()[i];
        }
    }

    return idx;
}