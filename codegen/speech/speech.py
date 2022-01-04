import os
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model, save_model
import numpy as np

model_name = 'mnist_simple_trained_model.h5'

def is_shift_layer(layer):
    ''' layer which can change the output encoding'''
    #FIXME: add more which will change the output shift
    if('input' in layer.name or
       'conv2d' in layer.name or
       'conv1d' in layer.name or
       'dense' in layer.name or
       'softmax' in layer.name or
        'sigmoid' in layer.name or
        'tanh' in layer.name or
        ('add' in layer.name and 'zero' not in layer.name) or # the name, zero_padding contains 'add'
        'subtract' in layer.name or
        'multiply' in layer.name or
       ('activation' in layer.name and layer.get_config()['activation'] == 'softmax')or
       ('activation' in layer.name and layer.get_config()['activation'] == 'sigmoid') or
       ('activation' in layer.name and layer.get_config()['activation'] == 'tanh')
    ):
        return True
    return False

def is_shift_fixed(layer):
    ''' layer which shift to a fixed value'''
    #FIXME: add more which will change the output shift
    if('softmax' in layer.name or
        'sigmoid' in layer.name or
        'tanh' in layer.name or
        ('activation' in layer.name and layer.get_config()['activation'] == 'softmax') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'sigmoid') or
        ('activation' in layer.name and layer.get_config()['activation'] == 'tanh')
    ):
        return True
    return  False

def layers_output_ranges(model, x_test, quantize_method='max_min', calibrate_size=1000):
    # limit the test data size
    np.random.shuffle(x_test)
    if(x_test.shape[0] > calibrate_size):
        x_test = x_test[:1000]
    # test, show the output ranges
    shift_list = {}
    # FIXME: only support one input
    if(type(model.layers[0]) != InputLayer):
        L = [model.input] + model.layers
    else:
        L = model.layers
    last_layer = None

    for layer in L: # layer loop
        print(layer.name)
        if("input" in layer.name):
            features = x_test
        else:
            # batch_normalization will need to be handled differently, since we are fusing the weight to its predecessor.
            # sigmoid and tanh are different, their shift is fixed to 7
            if(is_shift_layer(layer) or
                ('batch_normalization' in layer.name)):
                layer_model = Model(inputs=model.input, outputs=layer.output)
                features = layer_model.predict(x_test)
            else:
                # leave the features not changed, so this layer shift will be the same
                # as its inputs
                pass
        #  calculate no saturation shift
        #print(features)
        max_val = features.max()
        min_val = features.min()
        int_bits = int(np.ceil(np.log2(max(abs(max_val), abs(min_val)))))
        dec_bits = 7 - int_bits

        # saturation shift, using KLD method
        # Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        if('kld' in quantize_method and not is_shift_fixed(layer) and "input" not in layer.name and "dense" not in layer.name): # test, also do not use kld in input layer
            import scipy.stats
            abs_max = max(abs(max_val), abs(min_val))
            small_var = 1e-5
            bins = np.arange(-abs_max, abs_max, abs_max/2048*2)
            q_bins = np.arange(-abs_max, abs_max, abs_max/256*2)
            flat_hist = np.histogram(features.flatten(), bins=bins)[0]
            kl_loss = []
            kl_shifts = []
            for shift in range(4):
                t = 2 ** (dec_bits + shift)     # 2-based threshold
                act = np.round(features.flatten() * t)
                act = act / t
                act = np.clip(act, -128/t, 127/t)
                act = np.histogram(act, bins=q_bins)[0]
                act_hist = np.zeros(2047)
                chunk = int(2048/256)
                for i in range(int(255)):
                    none_zero = np.count_nonzero(flat_hist[i*chunk:(i+1)*chunk])
                    if none_zero == 0:
                        continue
                    for j in range(chunk):
                        act_hist[i*chunk+j] = act[i]/none_zero if flat_hist[i*chunk+j] != 0 else 0
                flat_hist[flat_hist==0] = small_var
                act_hist[act_hist==0] = small_var
                kl = scipy.stats.entropy(flat_hist, act_hist)
                kl_loss.append(kl)
                kl_shifts.append(dec_bits + shift)
                """
                ax = plt.subplot(8, 1, shift+1)
                ax.plot(flat_hist)
                ax.plot(act_hist)
                """
            new_dec = kl_shifts[np.argmin(kl_loss)] # set the dec_bit to the KLD results
            #plt.show()
            print("KLD loss", kl_loss)
            print("KLD shift", kl_shifts)
            if(new_dec != dec_bits):
                print(layer.name,"is using KLD method, original shift",dec_bits, "KLD results", new_dec)
                dec_bits = new_dec

        print( layer.name, "max value:", max_val, "min value:", min_val,"dec bit", dec_bits)
        # record the shift
        if(type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
            shift_list[layer.name.split(':')[0]] = dec_bits
        else:
            shift_list[layer.name] = dec_bits
        if ('batch_normalization' in layer.name):
            shift_list[last_layer.name] = dec_bits  # use the bn layer shift to update the last layer.
        last_layer = layer

    LM = {}
    for layer in model.layers:
        LM[layer.name] = layer
    L = [l for l in model.layers[1:]]
    L.reverse()

    def update_previous_layer_shift(layer, Q):
        if(type(layer.input) == list):
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                if('input' in iname):
                    continue
                shift_list[iname] = Qmin
                if(not is_shift_layer(LM[iname])):
                    update_previous_layer_shift(LM[iname], Q)
        else:
            iname = layer.input.name.split('/')[0]
            if('input' in iname):
                return
            shift_list[iname] = Qmin
            if(not is_shift_layer(LM[iname])):
                update_previous_layer_shift(LM[iname], Q)
    for layer in L:
        if(type(layer.input) == list):
            iname = layer.input[0].name.split('/')[0]
            Qmin = shift_list[iname]
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                if(shift_list[iname] < Qmin):
                    Qmin = shift_list[iname]
                if(shift_list[iname] != Qmin):
                    bFlag = True
            for inp in layer.input:
                iname = inp.name.split('/')[0]
                shift_list[iname] = Qmin
                if(not is_shift_layer(LM[iname])):
                    update_previous_layer_shift(LM[iname], Qmin)
            print('set shift', Qmin, 'for the input of', layer.name, ':', [inp.name.split('/')[0] for inp in layer.input])
            if(not is_shift_layer(layer) or Qmin < shift_list[layer.name]): # update current layer's shift only when we cannot change the shift
                shift_list[layer.name] = Qmin
    print("shift list", shift_list)
    return shift_list

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

def convert_to_x4_q7_weights(weights):
    [r, h, w, c] = weights.shape
    weights = np.reshape(weights, (r, h*w*c))
    num_of_rows = r
    num_of_cols = h*w*c
    new_weights = np.copy(weights)
    new_weights = np.reshape(new_weights, (r*h*w*c))
    counter = 0
    for i in range(int(num_of_rows/4)):
      # we only need to do the re-ordering for every 4 rows
      row_base = 4*i
      for j in range(int(num_of_cols/4)):
        # for each 4 entries
        column_base = 4*j
        new_weights[counter]   =  weights[row_base  ][column_base  ]
        new_weights[counter+1] =  weights[row_base+1][column_base  ]
        new_weights[counter+2] =  weights[row_base  ][column_base+2]
        new_weights[counter+3] =  weights[row_base+1][column_base+2]
        new_weights[counter+4] =  weights[row_base+2][column_base  ]
        new_weights[counter+5] =  weights[row_base+3][column_base  ]
        new_weights[counter+6] =  weights[row_base+2][column_base+2]
        new_weights[counter+7] =  weights[row_base+3][column_base+2]

        new_weights[counter+8] =  weights[row_base  ][column_base+1]
        new_weights[counter+9] =  weights[row_base+1][column_base+1]
        new_weights[counter+10] = weights[row_base  ][column_base+3]
        new_weights[counter+11] = weights[row_base+1][column_base+3]
        new_weights[counter+12] = weights[row_base+2][column_base+1]
        new_weights[counter+13] = weights[row_base+3][column_base+1]
        new_weights[counter+14] = weights[row_base+2][column_base+3]
        new_weights[counter+15] = weights[row_base+3][column_base+3]
        counter = counter + 16
      # the remaining ones are in order
      for j in range((int)(num_of_cols-num_of_cols%4), int(num_of_cols)):
        new_weights[counter] = weights[row_base][j]
        new_weights[counter+1] = weights[row_base+1][j]
        new_weights[counter+2] = weights[row_base+2][j]
        new_weights[counter+3] = weights[row_base+3][j]
        counter = counter + 4
    return new_weights

def generate_weights(model, name='weights.h', format='hwc', shift_list=None):
    f = open(name , 'w')
    f.write('#include "microinfer.h"\n\n')
    f.close()

    for curr_idx, layer in enumerate(model.layers):
        if(not layer.weights):
            continue
        weight_dec_shift = 0
        #遍历网络的每层，取参数值
        print('weights for layer:' , layer.name)
        for var in layer.weights:
            var_name = str(var.name)
            if("kernel" in var.name):
                var_values = layer.get_weights()[0]
                print("  weight: " , var_name)
            elif("bias" in var_name):
                var_values = layer.get_weights()[1]
                print("  bias: " , var_name)
            else:
                continue
            print("  original shape: ", var_values.shape)
            #计算层参数的dec_bits
            min_value = np.min(var_values)
            max_value = np.max(var_values)
            print("  weight min: " , min_value  , "  max:" , max_value)
            int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
            dec_bits = 7 - int_bits
            print("  dec bit", dec_bits)
            #计算层输出的shift
            bSameAsKernel = False
            if(is_shift_layer(layer)):
                bSameAsKernel = False
                inp = layer.input.name.replace(':','/').split('/')[0]
                input_encoding = shift_list[inp]
                if ("kernel" in var_name): #如果是卷积或者全连接层的weight参数
                    weight_dec_shift = dec_bits
                else: #如果是其他参数，比如bias
                    print("test layer name:" , var.name)
                    shift = input_encoding+weight_dec_shift-dec_bits #层的量化左移参数不能为负值
                    if(shift < 0):
                        bSameAsKernel = True 
            if(shift_list is None or bSameAsKernel):
                # check if bias shift > weight shift, then reduce bias shift to weight shift	
                if ("kernel" in var_name):
                    weight_dec_shift = dec_bits	
                else:	
                    if(dec_bits > weight_dec_shift):	
                        dec_bits = weight_dec_shift	
                print("  new dec bit", dec_bits)
                       
            # 执行int8量化，convert to [-128,128) or int8
            var_values = np.round(var_values * 2 ** dec_bits)
            var_name = var_name.replace('/', '_')
            var_name = var_name.replace(':', '_')
            with open(name, 'a') as f:
                f.write('#define ' + var_name.upper() + ' {')
            # CHW format，如果是kernel参数，则需要做矩阵转秩；如果是bias参数，则不用转秩
            if ('chw' in format):
                if "dense" in var_name and "kernel" in var_name:
                    transposed_wts = np.transpose(var_values)
                    transposed_wts = convert_to_x4_q7_weights(
                        np.reshape(transposed_wts, (transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                # all other kernels, bias stay the same
                else:
                    transposed_wts = var_values
            # HWC format
            else:
                if (len(var_values.shape) == 3):  # 1D convolution layer weights
                    transposed_wts = np.transpose(var_values, (2, 0, 1))
                elif (len(var_values.shape) == 4):  # 2D convolution layer weights
                    transposed_wts = np.transpose(var_values, (3, 0, 1, 2))
                else:  # fully connected layer weights or biases of any layer
                    # test, use opt weight reorder
                    if "dense" in var_name and "kernel" in var_name:
                        transposed_wts = np.transpose(var_values)
                        transposed_wts = convert_to_x4_q7_weights(np.reshape(transposed_wts ,(transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)))
                    else:
                        transposed_wts = np.transpose(var_values)

            print("  reshape to:",transposed_wts.shape)
            with open(name, 'a') as f:
                transposed_wts.tofile(f, sep=", ", format="%d") #写入处理后的量化参数和参数本身的量化尺度
                f.write('}\n\n')
                if ("bias" in var_name):
                    f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n\n')
                if ("kernel" in var_name ):
                    f.write('#define ' + var_name.upper() + '_SHIFT ' + '(' + str(dec_bits) + ')\n\n')
            

def generate_model(model, x_test, name='weights.h', format='hwc', quantize_method='max_min'):
    #预先遍历所有层的output值，计算每层的输出量化尺度
    shift_list = layers_output_ranges(model, x_test, quantize_method=quantize_method)
    #根据已有模型，对进行解析，量化，计算出新的权重，并且生成量化偏移参数供算子计算
    generate_weights(model, name=name, format=format, shift_list=shift_list)

    if(type(model.layers[0]) != InputLayer):
        L = [model.input] + model.layers
    else:
        L = model.layers
    with open(name,'a') as fp:
        #根据shift_list，生成每层的输出偏移量#define代码
        fp.write('\n/* output enconding for each layer */\n')
        for layer in L:
            if(type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                iname = layer.name.split(':')[0]
            else:
                iname = layer.name
            fp.write('#define %s_OUTPUT_SHIFT %s\n'%(iname.upper(), shift_list[iname]))    
        #根据上一层的输出偏移量、kernel偏移量和本层的输出偏移量计算本层的右移计算参数
        #根据上一层的输出偏移量、bias偏移量和本层的输出偏移量计算本层的左移计算参数
        #根据以上的两个偏移量，可以计算出本层的量化后输出结果以及输出的偏移量（量化尺度）
        for layer in model.layers:
            if(is_shift_layer(layer)):
                iname = layer.name.upper()
                if(len(layer.weights) == 2 and
                   'kernel' in layer.weights[0].name and
                   'bias' in layer.weights[1].name):
                    kname = layer.weights[0].name.upper().replace('/', '_').replace(':', '_')
                    bname = layer.weights[1].name.upper().replace('/', '_').replace(':', '_')
                    inp = layer.input.name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT+{2}_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp, kname))
                    fp.write('#define {0}_BIAS_LSHIFT   ({1}_OUTPUT_SHIFT+{2}_SHIFT-{3}_SHIFT)\n'.format(
                            iname, inp, kname, bname))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                    fp.write('#if {0}_BIAS_LSHIFT < 0\n#error {0}_BIAS_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                # add, sub
                elif ('add' in layer.name or
                    'subtract' in layer.name):
                    # only consider the first, they have been set to same in out_put_range()
                    inp = layer.input[0].name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
                # mult is different, Q3.4 * Q3.4 = Q6.8. if mult out is Q4.3, then shift (Q.4+q.4)-Q.3=5. Am I right?
                elif ('multiply' in layer.name ):
                    inp = layer.input[0].name.replace(':','/').split('/')[0].upper()
                    fp.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT*2-{0}_OUTPUT_SHIFT)\n'.format(
                            iname, inp))
                    fp.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
    
        fp.write('\n/* weights for each layer */\n')
        LI = {}
        ID = 0
        def is_skipable_layer(layer):
            # FIXME: add more that could be skiped
            if('lambda' in layer.name or
               'dropout' in layer.name or
               'batch_normalization' in layer.name or
                ('flatten' in layer.name and 'chw' not in format)): # flatten layer can be skipped in HWC but have to present in CHW
                return True
            return False
        for id,layer in enumerate(L):
            if(is_skipable_layer(layer)):
                inp = layer.input.name.replace(':','/').split('/')[0]
                LI[layer.name] = (LI[inp][0], layer)
            else:
                if(type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                    LI[layer.name.split(':')[0]] = (ID, layer)
                else:
                    LI[layer.name] = (ID, layer)
                ID += 1
            #将已经写入的层权重数组，和层右移参数或左移参数，共同作为层使用的参数结构
            if ('input' in layer.name or not layer.weights):
                continue
            for var in layer.weights:
                var_name = str(var.name).replace('/', '_').replace(':', '_')
                if("kernel" in var_name):
                    fp.write('static const int8_t %s_weights[] = %s;\n'%(layer.name, var_name.upper()))
                    fp.write('static const microinfer_weight_t %s_w = { (const void*)%s_weights, %s_OUTPUT_RSHIFT};\n'%(layer.name,layer.name, layer.name.upper()))
                elif("bias" in var_name):
                    fp.write('static const int8_t %s_bias[] = %s;\n'%(layer.name, var_name.upper()))
                    fp.write('static const microinfer_bias_t %s_b = { (const void*)%s_bias, %s_BIAS_LSHIFT};\n'%(layer.name,layer.name, layer.name.upper()))


        fp.write('\n/* nnom model */\n')
        # FIXME: now only support one input and one output
        sz = 1
        for d in model.input.shape[1:]:
            sz = sz*d
        fp.write('static int8_t nnom_input_data[%d];\n'%(sz))
        sz = 1
        for d in model.output.shape[1:]:
            sz = sz*d
        fp.write('static int8_t nnom_output_data[%d];\n'%(sz))
        fp.write('static nnom_model_t* nnom_model_create(void)\n{\n')
        fp.write('\tstatic nnom_model_t model;\n')
        if(ID>32):
            fp.write('\tnnom_layer_t ** layer = malloc(sizeof(nnom_layer_t *)*%d);\n'%(ID+1))
            fp.write('\tif(NULL == layer) return NULL;\n')
        else:
            fp.write('\tnnom_layer_t* layer[%d];\n'%(ID+1))
        fp.write('\n\tnew_model(&model);\n\n')
        for layer in L:
            if(is_skipable_layer(layer)):
                continue
            #FIXME: need a better solution to seperate the input 'tensor' from other layers
            if (type(model.input) == tf.Tensor and type(model.layers[0]) != InputLayer):
                id,_ = LI[layer.name.split(':')[0]]
            else:
                id,_ = LI[layer.name]

            if('input' in layer.name):
                try:
                    inshape = layer.input_shape[0][1:] # new changes in tf2?
                except:
                    inshape = layer.shape[1:]
                if (len(inshape) == 1):  # 1-D input
                    fp.write('\tlayer[%d] = Input(shape(%d,1,1), nnom_input_data);\n' % (id, inshape[0]))
                elif (len(inshape) == 2):  # 1-D input
                    fp.write('\tlayer[%d] = Input(shape(1,%d,%d), nnom_input_data);\n' % (id, inshape[0], inshape[1]))
                else:
                    fp.write('\tlayer[%d] = Input(shape%s, nnom_input_data);\n' % (id, inshape))

            # convlutional
            elif('conv1d' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('depthwise' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(DW_Conv2D({1}, kernel(1,{2}), stride(1,{3}), dilation(1,{4}), PADDING_{5}, &{6}_w, &{6}_b), layer[{7}]);\n'.format(
                        id, 1, cfg['kernel_size'][0], cfg['strides'][0], cfg['dilation_rate'][0], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
                else:
                    fp.write('\tlayer[{0}] = model.hook(Conv2D({1}, kernel(1,{2}), stride(1,{3}), dilation(1,{4}), PADDING_{5}, &{6}_w, &{6}_b), layer[{7}]);\n'.format(
                        id, cfg['filters'], cfg['kernel_size'][0], cfg['strides'][0], cfg['dilation_rate'][0], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
            elif('conv2d' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if ('depthwise' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(DW_Conv2D({1}, kernel{2}, stride{3}, dilation{4}, PADDING_{5}, &{6}_w, &{6}_b), layer[{7}]);\n'.format(
                        id, 1, cfg['kernel_size'], cfg['strides'], cfg['dilation_rate'], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
                else:
                    fp.write('\tlayer[{0}] = model.hook(Conv2D({1}, kernel{2}, stride{3}, dilation{4}, PADDING_{5}, &{6}_w, &{6}_b), layer[{7}]);\n'.format(
                        id, cfg['filters'], cfg['kernel_size'], cfg['strides'], cfg['dilation_rate'], cfg['padding'].upper(),
                        layer.name, LI[inp][0]))
            # activations
            elif('activation' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if(cfg['activation'] == 'relu'):
                    fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
                if(cfg['activation'] == 'tanh'):
                    fp.write('\tlayer[%s] = model.active(act_tanh(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                if(cfg['activation'] == 'sigmoid'):
                    fp.write('\tlayer[%s] = model.active(act_sigmoid(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                elif(cfg['activation'] == 'softmax'):
                    fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))
            elif('re_lu' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
            # pooling
            elif('max_pooling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if ('global' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(GlobalMaxPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                elif('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(MaxPool(kernel%s, stride%s, PADDING_%s), layer[%d]);\n'%(
                        id, cfg['pool_size'], cfg['strides'], cfg['padding'].upper(), LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(MaxPool(kernel(1,{1}), stride(1,{2}), PADDING_{3}), layer[{4}]);\n'.format(
                        id, cfg['pool_size'][0], cfg['strides'][0], cfg['padding'].upper(), LI[inp][0]))
            elif('average_pooling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if ('global' in layer.name):
                    # a global avg pool before softmax can be replace by sumpool in MCU (recommend)
                    if(layer == model.layers[-2] and 'Softmax' in model.layers[-1].output.name):
                        print(layer.name, 'has been replaced by GlobalSumPool()')
                        fp.write('\tlayer[%s] = model.hook(GlobalSumPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                    else:
                        fp.write('\tlayer[%s] = model.hook(GlobalAvgPool(),  layer[%s]);\n' % (id, LI[inp][0]))
                elif('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(AvgPool(kernel%s, stride%s, PADDING_%s), layer[%d]);\n'%(
                        id, cfg['pool_size'], cfg['strides'], cfg['padding'].upper(), LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(AvgPool(kernel(1,{1}), stride(1,{2}), PADDING_{3}), layer[{4}]);\n'.format(
                        id, cfg['pool_size'][0], cfg['strides'][0], cfg['padding'].upper(), LI[inp][0]))
            elif ('up_sampling' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[%s] = model.hook(UpSample(kernel%s), layer[%d]);\n'%(id, cfg['size'],  LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(UpSample(kernel(1,{1})), layer[{2}]);\n'.format(
                        id,  cfg['size'][0], LI[inp][0]))
            # zero padding
            elif ('zero_padding' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(ZeroPadding(border({1},{2},{3},{4})), layer[{5}]);\n'.format(
                        id,  cfg['padding'][0][0], cfg['padding'][0][1], cfg['padding'][1][0],cfg['padding'][1][1], LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(ZeroPadding(border(0,0,{1},{2})), layer[{3}]);\n'.format(
                        id,  cfg['padding'][0], cfg['padding'][1], LI[inp][0]))
            # Cropping
            elif ('cropping' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                if('2d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(Cropping(border({1},{2},{3},{4})), layer[{5}]);\n'.format(
                        id,  cfg['cropping'][0][0], cfg['cropping'][0][1], cfg['cropping'][1][0],cfg['cropping'][1][1], LI[inp][0]))
                elif('1d' in layer.name):
                    fp.write('\tlayer[{0}] = model.hook(Cropping(border(0,0,{1},{2})), layer[{3}]);\n'.format(
                        id,  cfg['cropping'][0], cfg['cropping'][1], LI[inp][0]))

            # others
            elif('flatten' in layer.name): # flatten is needed in CHW backend but not needed in HWC
                inp = layer.input.name.replace(':', '/').split('/')[0]
                fp.write('\tlayer[%s] = model.hook(Flatten(), layer[%s]);\n'%(id, LI[inp][0]))
            elif('concatenate' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                cfg = layer.get_config()
                fp.write('\tlayer[%s] = model.mergex(Concat(%s), %s%s);\n'%(
                    id, cfg['axis'], len(inps), inX))
            elif('add' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Add(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('subtract' in layer.name):
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Sub(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('multiply' in layer.name):
                warnings.warn("Warning mutiply is under testing")
                inps = [input.name.replace(':','/').split('/')[0] for input in layer.input]
                inX = ''
                for inp in inps:
                    inX += ' ,layer[%d]'%(LI[inp][0])
                fp.write('\tlayer[%s] = model.mergex(Mult(%s_OUTPUT_RSHIFT), %s%s);\n'%(
                    id, layer.name.upper(), len(inps), inX))
            elif('dense' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                cfg = layer.get_config()
                fp.write('\tlayer[{0}] = model.hook(Dense({1}, &{2}_w, &{2}_b), layer[{3}]);\n'.format(
                    id, cfg['units'], layer.name, LI[inp][0]))
            elif('softmax' in layer.name):
                inp = layer.input.name.replace(':','/').split('/')[0]
                fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))
            else:
                raise Exception('unsupported layer', layer.name, layer)
			
            """
            # temporary fixed for activations attached into layers in construction
            def is_activation_attached(layer):
                if(("Softmax" in layer.output.name and "softmax" not in layer.name)or
                ("Relu" in layer.output.name and "re_lu" not in layer.name) or
                ("Sigmoid" in layer.output.name and "sigmoid" not in layer.name) or
                ("Tanh" in layer.output.name and "tanh" not in layer.name)):
                    return True
                return False
            if "input" not in layer.name and is_activation_attached(layer):
                inp = layer.output.name.replace(':', '/').split('/')[0]
                cfg = layer.get_config()
                if(cfg['activation'] == 'relu'):
                    fp.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
                if(cfg['activation'] == 'tanh'):
                    fp.write('\tlayer[%s] = model.active(act_tanh(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                if(cfg['activation'] == 'sigmoid'):
                    fp.write('\tlayer[%s] = model.active(act_sigmoid(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
                elif(cfg['activation'] == 'softmax'):
                    fp.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))
            """
			
        # FIXME, test later.
        if('softmax' in layer.name
           or ('activation' in layer.name and layer.get_config()['activation'] == 'softmax')):
            fp.write('\tlayer[%s] = model.hook(Output(shape(%s,1,1), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], id))
        elif len(layer.output.shape) == 4:
            fp.write('\tlayer[%s] = model.hook(Output(shape%s, nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1:], id))
        elif len(layer.output.shape) == 3:
            fp.write('\tlayer[%s] = model.hook(Output(shape(1,%s,%s), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], layer.output.shape[2], id))
        elif len(layer.output.shape) == 2:
            fp.write('\tlayer[%s] = model.hook(Output(shape(%s,1,1), nnom_output_data), layer[%s]);\n'%(id+1, layer.output.shape[1], id))
        else:
            raise Exception('unsupported output shape of the last layer', layer.name, layer)
        fp.write('\tmodel_compile(&model, layer[0], layer[%s]);\n'%(id+1))
        if(ID>32):
            fp.write('\tfree(layer);\n')
        fp.write('\treturn &model;\n}\n')
    #with open('.shift_list','w') as fp:
    #    fp.write(str(shift_list))




"""
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
"""
x_train = np.zeros((10, 49, 40),  dtype=float, order='C')
x_test =  np.zeros((10, 49, 40),  dtype=float, order='C')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print('x_train shape:', x_train.shape)
#归一化操作
x_test = x_test/255
x_train = x_train/255
print("data range", x_test.min(), x_test.max())
#随机挑选十张图片，对归一化后的测试集数值乘10
#image_to_cfile(x_test*127, y_test_num, 10, file='image.h')

file_pb = "./speech"
file_h5 = 'speech_model.h5'
model = load_model("speech")
loaded_model = tf.keras.models.load_model(file_pb)
tf.keras.models.save_keras_model(loaded_model, file_h5)
loaded_model_from_h5 = tf.keras.models.load_model(file_h5)

model_name = "speech_model.h5"
#从文件系统中加载模型
model = load_model(model_name)
model.summary()
#生成weight.h头文件（模型量化+权重参数代码头文件生成）
generate_model(model, np.vstack((x_train, x_test)), name="weights.h")
#生成应用程序（模型初始化+模型编译+模型运行的API函数）
