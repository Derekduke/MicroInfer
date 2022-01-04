import os , sys
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model, save_model
import numpy as np

PATH = os.path.abspath(os.path.join(os.getcwd(),".."))
sys.path.append(PATH)

from comman import *

model_name = 'mnist_simple_trained_model.h5'

def image_to_cfile(data, label, num_of_image, file='image.h'):
    with open(file, 'w') as f:
        for i in range(num_of_image):
            selected = np.random.randint(0, 1000) # select 10 out of 1000.
            f.write('#define IMG%d {'% (i))
            np.round(data[selected]).flatten().tofile(f, sep=", ", format="%d") # convert 0~1 to 0~127
            f.write('} \n')
            f.write('#define IMG%d_LABLE'% (i))
            f.write(' %d \n \n' % label[selected])
        f.write('#define TOTAL_IMAGE %d \n \n'%(num_of_image))

        f.write('static const int8_t img[%d][%d] = {' % (num_of_image, data[0].flatten().shape[0]))
        f.write('IMG0')
        for i in range(num_of_image -1):
            f.write(',IMG%d'%(i+1))
        f.write('};\n\n')

        f.write('static const int8_t label[%d] = {' % (num_of_image))
        f.write('IMG0_LABLE')
        for i in range(num_of_image -1):
            f.write(',IMG%d_LABLE'%(i+1))
        f.write('};\n\n')

epochs = 2
num_classes = 10
print("test keras tool")
(x_train, y_train_num), (x_test, y_test_num) = mnist.load_data()
# x_train是训练数据，是三维的数字图案方阵(编号，长，宽，灰度像素值)，y_train是训练数据的标签，表示是数字几
print(x_train.shape[0], 'train samples') # 默认60000张训练数据
print(x_test.shape[0], 'test samples') # 默认10000张测试数据
y_train = tf.keras.utils.to_categorical(y_train_num, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_num, num_classes)
#加一个维度，没什么实质的改变，因为本身就是单通道的灰度图，只是方便处理
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print('x_train shape:', x_train.shape)
#归一化操作
x_test = x_test/255
x_train = x_train/255
print("data range", x_test.min(), x_test.max())
#随机挑选十张图片，对归一化后的测试集数值乘10
image_to_cfile(x_test*127, y_test_num, 10, file='image.h')

#从文件系统中加载模型
model = load_model(model_name)
model.summary()
#生成weight.h头文件（模型量化+权重参数代码头文件生成）
generate_model(model, np.vstack((x_train, x_test)), name="weights.h")
#生成应用程序（模型初始化+模型编译+模型运行的API函数）
