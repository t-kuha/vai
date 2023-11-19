// 
// Running MNIST LeNet model using Vitis AI Runtime
// 
// How to compile:
// $CXX vart.cpp -std=c++17 -Wall -I<vart include directory> 
// -L<vart library directory> -lvart-dpu-runner -lvart-runner
#include <iostream>
#include <fstream>
#include <cstdint>
#include <memory>
#include <assert.h>
#include <endian.h>

#include "vitis/ai/dpu_runner.hpp"


int load_mnist(
    const std::string image_path,
    const std::string label_path,
    std::unique_ptr<uint8_t []>& image,
    std::unique_ptr<uint8_t []>& label,
    unsigned int& count
);

uint8_t argmax(const std::unique_ptr<int8_t []>& data);


// Based on:
//  https://github.com/Xilinx/Vitis-AI/blob/master/docs/DPUCADX8G/Vitis-C++API.md
int main(int argc, char* argv[])
{
    std::cout << "...... MNIST LeNet (VART flow) ......" << std::endl;
    
    int ret = 0;

    // Load MNIST test data
    const std::string mnist_image = "t10k-images-idx3-ubyte";
    const std::string mnist_label = "t10k-labels-idx1-ubyte";
    std::unique_ptr<uint8_t []> image;  // Image data
    std::unique_ptr<uint8_t []> label;  // Label
    unsigned int num = 0;               // Number of data (10000 for test data)

    ret = load_mnist(mnist_image, mnist_label, image, label, num);
    if(ret != 0){
        std::cerr << "[ERROR] load_mnist()" << std::endl;
        return -1;
    }

    // Allocate buffer for final result
    std::unique_ptr<uint8_t []> result;
    result.reset(new uint8_t [num]);
    
    auto runners = vitis::ai::DpuRunner::create_dpu_runner(argv[1]);
    auto runner = runners[0].get();

    // Input/Output tensors
    auto tensors_in  = runner->get_input_tensors();
    auto tensors_out = runner->get_output_tensors();

    // Dimensions
    // auto dims_in  = tensors_in[0]->get_dims();
    // auto dims_out = tensors_out[0]->get_dims();

    // TODO: how to get scale for input/output tensors?
    float scale = 64;

    // Get shape info
    int in_size = tensors_in[0]->get_element_num() / tensors_in[0]->get_dim_size(0);
    int out_size = tensors_out[0]->get_element_num() / tensors_out[0]->get_dim_size(0);
    // std::cout << "Input size:  " << in_size << std::endl;
    // std::cout << "Output size: " << out_size << std::endl;
    // std::cout << "Input Data Type:  " << (int)tensors_in[0]->get_data_type() << std::endl;
    // std::cout << "Output Data Type: " << (int)tensors_out[0]->get_data_type() << std::endl;

    // Input/Output buffer to DPU
    // int8_t* buf_in = new int8_t [in_size];
    // int8_t* buf_out = new int8_t [out_size];
    std::unique_ptr<int8_t []> buf_in(new int8_t [in_size]);
    std::unique_ptr<int8_t []> buf_out(new int8_t [out_size]);

    // In/Out tensor refactory for batch input/output
    auto tensor_tmp_in = std::shared_ptr<vitis::ai::Tensor> (
        new vitis::ai::Tensor (
            tensors_in[0]->get_name(),
            tensors_in[0]->get_dims(),
            tensors_in[0]->get_data_type()
        )
    );
    auto cpu_buf_in = vitis::ai::CpuFlatTensorBuffer(buf_in.get(), tensor_tmp_in.get());

    auto tensor_tmp_out = std::shared_ptr<vitis::ai::Tensor> (
        new vitis::ai::Tensor (
            tensors_out[0]->get_name(),
            tensors_out[0]->get_dims(),
            tensors_out[0]->get_data_type()
        )
    );
    auto cpu_buf_out = vitis::ai::CpuFlatTensorBuffer(buf_out.get(), tensor_tmp_out.get());

    // Create Pointers to Inputs and Outputs
    std::vector<vitis::ai::TensorBuffer*> ptr_in;
    std::vector<vitis::ai::TensorBuffer*> ptr_out;

    // Push the input and output tensors to the vector
    ptr_in.clear();
    ptr_out.clear();

    ptr_in.push_back(&cpu_buf_in);
    ptr_out.push_back(&cpu_buf_out);

    std::cout << "[INFO] Start of inference ......" << std::endl;

    std::pair<uint32_t, int> job_id;
    for(unsigned int i = 0; i < num; i++){
        // Scaling before input to DPU
        for(int j = 0; j < in_size; j++){
            buf_in.get()[j] = (image.get()[in_size*i + j])/255.0*scale;
        }

        job_id = runner->execute_async(ptr_in, ptr_out);
        runner->wait(job_id.first, -1 /* <- block forever */);

        result.get()[i] = argmax(buf_out);
    }

    std::cout << "[INFO] End of inference ......" << std::endl;

    // Show final result
    int num_correct = 0;
    for(unsigned int i = 0; i < num; i++){
        if(label.get()[i] == result.get()[i]){
            num_correct++;
        }
    }
    std::cout << "Accuracy: " << num_correct << " / " << num << std::endl;

    std::cout << "----------------------------------------------" << std::endl;

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
