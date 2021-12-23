// /*
//  * Change Logs:
//  * Date           Author       Notes
//  * 2021-12-06     derekduke   The first version
//  */

// #include "microinfer.h"
// #include "microinfer_layers.h"
// #include "microinfer_tensor.h"
// #include "microinfer_local.h"
// #include "layers/microinfer_conv2d.h"

// microinfer_status_t conv2d_run(microinfer_layer_t *layer)
// {
//     for(int i=0 ; i<28*28 ; i++)
//         printf("conv2d input: %d\n" , ((int8_t*)layer->in->tensor->p_data)[i]);
//         /*
//         printf("conv2d input: %d\n" , ((int8_t*)layer->in->tensor->p_data)[1]);
//         printf("conv2d input: %d\n" , ((int8_t*)layer->in->tensor->p_data)[2]);
//         printf("conv2d input: %d\n" , ((int8_t*)layer->in->tensor->p_data)[3]);
//         printf("conv2d input: %d\n" , ((int8_t*)layer->in->tensor->p_data)[4]);
//         */
//     microinfer_conv2d_layer_t *cl = (microinfer_conv2d_layer_t *)layer;
//     local_convolve_HWC_q7_nonsquare(
// 					layer->in->tensor->p_data,
// 					layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
// 					cl->weight->p_data, layer->out->tensor->dim[2],
// 					cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h, cl->dilation.w, cl->dilation.h,
// 					cl->bias->p_data, cl->bias_lshift, cl->output_rshift, cl->weight->qtype,
// 					layer->out->tensor->p_data,
// 					layer->out->tensor->dim[1], layer->out->tensor->dim[0], NULL, NULL);
//         printf("conv2d output: %d\n" , ((int8_t*)layer->out->tensor->p_data)[0]);
//         printf("conv2d output: %d\n" , ((int8_t*)layer->out->tensor->p_data)[1]);
//         printf("conv2d output: %d\n" , ((int8_t*)layer->out->tensor->p_data)[2]);
//         printf("conv2d output: %d\n" , ((int8_t*)layer->out->tensor->p_data)[3]);
//         printf("conv2d output: %d\n" , ((int8_t*)layer->out->tensor->p_data)[4]);
// 		return NN_SUCCESS;
// }

// uint32_t conv_output_length(uint32_t input_length, uint32_t filter_size, microinfer_padding_t padding, uint32_t stride, uint32_t dilation)
// {
//     if (input_length == 0)
//         return 0;
//     uint32_t dilated_filter_size = (filter_size - 1) * dilation + 1;
// 	uint32_t output_length;
//     if(padding == PADDING_SAME)
//         output_length = input_length;
//     else
//         output_length = input_length - dilated_filter_size + 1;
//     return (output_length + stride - 1) / stride;
// }

// microinfer_status_t conv2d_build(microinfer_layer_t *layer)
// {
//     microinfer_conv2d_layer_t* cl = (microinfer_conv2d_layer_t*)layer;
//     layer->in->tensor = layer->in->hook.io->tensor;
//     layer->out->tensor = new_tensor(MICROINFER_QTYPE_PER_TENSOR , layer->in->tensor->num_dim, cl->filter_mult);
//     tensor_cpy_attr(layer->out->tensor , layer->in->tensor);
//     layer->out->tensor->q_dec[0] = layer->in->tensor->q_dec[0] + cl->weight->q_dec[0] - cl->output_rshift[0];
//     if(layer->actail)
//     { 
// 		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);
//         layer->actail->tensor = layer->out->tensor;
//     }
//     layer->out->tensor->dim[0] = conv_output_length(layer->in->tensor->dim[0] , cl->kernel.h , cl->padding_type , cl->stride.h , cl->dilation.h);
//     layer->out->tensor->dim[1] = conv_output_length(layer->in->tensor->dim[1] , cl->kernel.w , cl->padding_type , cl->stride.w , cl->dilation.w);
//     layer->out->tensor->dim[2] = cl->filter_mult;
//     if(cl->padding_type == PADDING_SAME)
//     {
//         cl->pad.w = cl->dilation.w * (cl->kernel.w-1)/2;
//         cl->pad.h = cl->dilation.h * (cl->kernel.h-1)/2;
//         cl->pad.c = 0;
//     }
//     return NN_SUCCESS;
// }

// microinfer_layer_t* Conv2D(uint32_t filters , microinfer_3d_shape_t k , microinfer_3d_shape_t s , microinfer_3d_shape_t d , 
//                             microinfer_padding_t pad_type , const microinfer_weight_t* w , const microinfer_bias_t* b)
// {
//     microinfer_conv2d_layer_t* layer;
//     microinfer_buf_t* comp;
//     microinfer_layer_io_t* in , *out;

//     uint32_t mem_size = sizeof(microinfer_conv2d_layer_t)+sizeof(microinfer_layer_io_t)*2+sizeof(microinfer_buf_t);
//     layer = microinfer_mem(mem_size);
//     if(layer == NULL)
//         return NULL;
//     in = (void*)((uint8_t*)layer + sizeof(microinfer_conv2d_layer_t));
//     out = (void*)((uint8_t*)in + sizeof(microinfer_layer_io_t));
//     comp = (void*)((uint8_t*)out + sizeof(microinfer_layer_io_t));

//     layer->super.type = MICROINFER_CONV_2D;
//     in->type = MICROINFER_TENSOR_BUF_TEMP;
//     out->type = MICROINFER_TENSOR_BUF_TEMP;
//     comp->type = MICROINFER_TENSOR_BUF_TEMP;

//     layer->super.in = io_init(layer , in);
//     layer->super.out = io_init(layer, out);

//     layer->super.run = conv2d_run;
//     layer->super.build = conv2d_build;

//     layer->kernel = k;
//     layer->stride = s;
//     layer->dilation = d;
//     layer->filter_mult = filters;
//     layer->padding_type = pad_type;

//     layer->weight = new_tensor(MICROINFER_QTYPE_PER_TENSOR , 4 , filters);
//     layer->bias = new_tensor(MICROINFER_QTYPE_PER_TENSOR , 1 , filters);
//     {
//     microinfer_shape_data_t dim[4] = {k.h , k.w , k.c , filters};
//     *(layer->weight->q_offset) = 0;
//     *(layer->weight->q_dec) = 0;
//     layer->weight->p_data = (void*)w->p_value;
//     layer->weight->bitwidth = 8;
//     layer->weight->qtype = MICROINFER_QTYPE_PER_TENSOR;
//     memcpy(layer->weight->dim , dim , layer->weight->num_dim*sizeof(microinfer_shape_data_t));

//     dim[0] = filters;
//     *(layer->bias->q_offset) = 0;
//     *(layer->bias->q_dec) = 0;
//     layer->bias->p_data = (void*)w->p_value;
//     layer->bias->bitwidth = 8;
//     layer->bias->qtype = MICROINFER_QTYPE_PER_TENSOR;
//     memcpy(layer->bias->dim , dim , layer->weight->num_dim*sizeof(microinfer_shape_data_t));

//     layer->output_rshift = (microinfer_qformat_param_t*)&w->shift;
//     layer->bias_lshift = (microinfer_qformat_param_t*)&w->shift;
//     }
//     return (microinfer_layer_t*)layer;
// }

/*
 * Copyright (c) 2018-2020
 * Jianjia Ma
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-07-23     Jianjia Ma   The first version
 */


#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "microinfer.h"
#include "microinfer_local.h"
#include "microinfer_layers.h"
#include "layers/microinfer_conv2d.h"

#ifdef MICROINFER_USING_CMSIS_NN
#include "arm_math.h"
#include "arm_nnfunctions.h"
#endif

// keras's implementation. 
// source: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/conv_utils.py#L85
uint32_t conv_output_length(uint32_t input_length, uint32_t filter_size, microinfer_padding_t padding, uint32_t stride, uint32_t dilation)
{
    if (input_length == 0)
        return 0;
    uint32_t dilated_filter_size = (filter_size - 1) * dilation + 1;
	uint32_t output_length;
    if(padding == PADDING_SAME)
        output_length = input_length;
    else
        output_length = input_length - dilated_filter_size + 1;
    return (output_length + stride - 1) / stride;
}

microinfer_status_t conv2d_build(microinfer_layer_t *layer)
{
	microinfer_conv2d_layer_t *cl = (microinfer_conv2d_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for the output
	layer->out->tensor = new_tensor(MICROINFER_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, cl->filter_mult);
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);
	
	// calculate the output tensor q format, only support per tensor quantise now
	layer->out->tensor->q_dec[0] = layer->in->tensor->q_dec[0] + cl->weight->q_dec[0] - cl->output_rshift[0]; // need some modification for 16bit. 
	// see if the activation will change the q format
	if(layer->actail) 
    {
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);
        layer->actail->tensor = layer->out->tensor;
    }
	// now we set up the tensor shape, always HWC format
	layer->out->tensor->dim[0] = conv_output_length(layer->in->tensor->dim[0], cl->kernel.h, cl->padding_type, cl->stride.h, cl->dilation.h);
	layer->out->tensor->dim[1] = conv_output_length(layer->in->tensor->dim[1], cl->kernel.w, cl->padding_type, cl->stride.w, cl->dilation.w);
	layer->out->tensor->dim[2] = cl->filter_mult; // channel stays the same
	
	// fill padding
	if (cl->padding_type == PADDING_SAME)
	{
		cl->pad.w = cl->dilation.w * (cl->kernel.w - 1) / 2;
		cl->pad.h = cl->dilation.h * (cl->kernel.h - 1) / 2;
		cl->pad.c = 0;
	}

	#ifdef MICROINFER_USING_CMSIS_NN
	// bufferA size: (1D shape)
	// 2*ch_im_in*dim_kernel*dim_kernel
	layer->comp->size = 2 * 2 * layer->in->tensor->dim[2] * cl->kernel.w * cl->kernel.h;
	#endif
	// computational cost: K x K x Cin x Hour x Wout x Cout
	//layer->stat.macc = cl->kernel.w * cl->kernel.h * layer->in->tensor->dim[2] * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

microinfer_status_t conv2d_run(microinfer_layer_t *layer)
{
	microinfer_conv2d_layer_t *cl = (microinfer_conv2d_layer_t *)layer;


	{

        if(layer->in->tensor->bitwidth == 16) ;
    	// local_convolve_HWC_q15_nonsquare(
		// 		layer->in->tensor->p_data,
		// 		layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
		// 		cl->weight->p_data, layer->out->tensor->dim[2],
		// 		cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h, cl->dilation.w, cl->dilation.h,
		// 		cl->bias->p_data, cl->bias_lshift, cl->output_rshift, cl->weight->qtype,
		// 		layer->out->tensor->p_data,
		// 		layer->out->tensor->dim[1], layer->out->tensor->dim[0], NULL, NULL);
        else
		local_convolve_HWC_q7_nonsquare(
					layer->in->tensor->p_data,
					layer->in->tensor->dim[1], layer->in->tensor->dim[0], layer->in->tensor->dim[2],
					cl->weight->p_data, layer->out->tensor->dim[2],
					cl->kernel.w, cl->kernel.h, cl->pad.w, cl->pad.h, cl->stride.w, cl->stride.h, cl->dilation.w, cl->dilation.h,
					cl->bias->p_data, cl->bias_lshift, cl->output_rshift, cl->weight->qtype,
					layer->out->tensor->p_data,
					layer->out->tensor->dim[1], layer->out->tensor->dim[0], NULL, NULL);
		return NN_SUCCESS;
	}

	return NN_SUCCESS;
}

// Conv2D
// multiplier of (output/input channel),
// shape of kernal, shape of strides, weight struct, bias struct
microinfer_layer_t *Conv2D(uint32_t filters, microinfer_3d_shape_t k, microinfer_3d_shape_t s, microinfer_3d_shape_t d,  microinfer_padding_t pad_type,
					 const microinfer_weight_t *w, const microinfer_bias_t *b)
{
	microinfer_conv2d_layer_t *layer;
	microinfer_buf_t *comp;
	microinfer_layer_io_t *in, *out;
	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(microinfer_conv2d_layer_t) + sizeof(microinfer_layer_io_t) * 2 + sizeof(microinfer_buf_t);
	layer = microinfer_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(microinfer_conv2d_layer_t));
	out = (void *)((uint8_t*)in + sizeof(microinfer_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(microinfer_layer_io_t));

	// set type in layer parent
	layer->super.type = MICROINFER_CONV_2D;
	// set buf state
	in->type = MICROINFER_TENSOR_BUF_TEMP;
	out->type = MICROINFER_TENSOR_BUF_TEMP;
	comp->type = MICROINFER_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	#ifdef MICROINFER_USING_CMSIS_NN
	layer->super.comp = comp;
	#endif
	// set run method & output shape
	layer->super.run = conv2d_run;
	layer->super.build = conv2d_build;

	// get the private parameters
	layer->kernel = k;
	layer->stride = s;
	layer->dilation = d; 	
	layer->filter_mult = filters; 		// for convs, this means filter number
	layer->padding_type = pad_type;

	// create weight and bias tensor
	layer->weight = new_tensor(MICROINFER_QTYPE_PER_TENSOR, 4, filters);
	layer->bias = new_tensor(MICROINFER_QTYPE_PER_TENSOR, 1, filters);

	// configure weight tensor manually to support new tensor based backends. 
	// needs to be very careful
	{
		// config weight 
		microinfer_shape_data_t dim[4] = {k.h, k.w, k.c, filters};
		*(layer->weight->q_offset) = 0;			// we have no support of offset here
		*(layer->weight->q_dec) = 0;		// not using it
		layer->weight->p_data = (void*)w->p_value;
		layer->weight->bitwidth = 8;
		layer->weight->qtype = MICROINFER_QTYPE_PER_TENSOR;
		microinfer_memcpy(layer->weight->dim, dim, layer->weight->num_dim * sizeof(microinfer_shape_data_t));

		// config bias 
		dim[0] = filters;
		*(layer->bias->q_offset) = 0;			// we have no support of offset here
		*(layer->bias->q_dec) = 0;		// not using it
		layer->bias->p_data = (void*) b->p_value;
		layer->bias->bitwidth = 8;
		layer->weight->qtype = MICROINFER_QTYPE_PER_TENSOR;
		microinfer_memcpy(layer->bias->dim, dim, layer->bias->num_dim * sizeof(microinfer_shape_data_t));
		
		// output shift and bias shift
		layer->output_rshift = (microinfer_qformat_param_t *)&w->shift;
		layer->bias_lshift = (microinfer_qformat_param_t *)&b->shift;
	}

	return (microinfer_layer_t *)layer;
}





