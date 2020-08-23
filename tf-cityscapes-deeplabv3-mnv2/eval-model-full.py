import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot

colormap = np.zeros((256, 3), dtype=np.uint8)
colormap[0] = [128, 64, 128]
colormap[1] = [244, 35, 232]
colormap[2] = [70, 70, 70]
colormap[3] = [102, 102, 156]
colormap[4] = [190, 153, 153]
colormap[5] = [153, 153, 153]
colormap[6] = [250, 170, 30]
colormap[7] = [220, 220, 0]
colormap[8] = [107, 142, 35]
colormap[9] = [152, 251, 152]
colormap[10] = [70, 130, 180]
colormap[11] = [220, 20, 60]
colormap[12] = [255, 0, 0]
colormap[13] = [0, 0, 142]
colormap[14] = [0, 0, 70]
colormap[15] = [0, 60, 100]
colormap[16] = [0, 80, 100]
colormap[17] = [0, 0, 230]
colormap[18] = [119, 11, 32]

img_path = '/Users/kuriharat/Documents/neuralnet/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/jena/jena_000000_000019_leftImg8bit.png'
# os.path.join('..', '_dataset', 'cityscapes', 'leftImg8bit', 'train', 'jena', 'jena_000000_000019_leftImg8bit.png')

model_path = os.path.join('deeplabv3_mnv2_cityscapes_train', 'frozen_inference_graph.pb')

if not os.path.exists(img_path):
    print('Input image cannot be found...')

if not os.path.exists(model_path):
    print('Model file cannot be found...')

# Load image
image = Image.open(img_path)
image = image.convert('RGB')
input_image = np.expand_dims(np.asarray(image), 0)

# Load model
graph_def = tf.compat.v1.GraphDef()
graph_def.ParseFromString(tf.io.gfile.GFile(model_path, "rb").read())

graph = tf.Graph()
graph.as_default()

tf.import_graph_def(graph_def, name="")

input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('ImageTensor:0')
output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('SemanticPredictions:0')
# SemanticPredictions sub_7

# Run inference
with tf.compat.v1.Session() as sess:
    # for i in range(0, iter):
    feed_dict = {input_tensor: input_image}
    output = sess.run([output_tensor], feed_dict)

# Post-processing: convert raw output to segmentation output
pred = output[0][0]
print(pred.shape)

# Visualization
img_map = colormap[pred]

alpha = 0.4
img_disp = np.asarray(image)*alpha + img_map*(1-alpha)

pyplot.imshow(img_disp/255)
Image.fromarray(np.uint8(img_disp/255)).save('result.jpg')

# Accuracy

