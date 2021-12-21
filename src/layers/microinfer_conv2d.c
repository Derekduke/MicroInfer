/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include "microinfer.h"
#include "microinfer_layers.h"
#include "microinfer_tensor.h"

#include "layers/microinfer_conv2d.h"

microinfer_status_t conv2d_run(microinfer_layer_t *layer)
{

}

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
    microinfer_conv2d_layer_t* cl = (microinfer_conv2d_layer_t*)layer;
    layer->in->tensor = layer->in->hook.io->tensor;
    layer->out->tensor = new_tensor(MICROINFER_QTYPE_PER_TENSOR , layer->in->tensor->num_dim, cl->filter_mult);
    tensor_cpy_attr(layer->out->tensor , layer->in->tensor);
    layer->out->tensor->q_dec[0] = layer->in->tensor->q_dec[0] + cl->weight->q_dec[0] - cl->output_rshift[0];
    if(layer->actail) 
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);
    layer->out->tensor->dim[0] = conv_output_length(layer->in->tensor->dim[0] , cl->kernel.h , cl->padding_type , cl->stride.h , cl->dilation.h);
    layer->out->tensor->dim[1] = conv_output_length(layer->in->tensor->dim[1] , cl->kernel.w , cl->padding_type , cl->stride.w , cl->dilation.w);
    layer->out->tensor->dim[2] = cl->filter_mult;
    if(cl->padding_type == PADDING_SAME)
    {
        cl->pad.w = cl->dilation.w * (cl->kernel.w-1)/2;
        cl->pad.h = cl->dilation.h * (cl->kernel.h-1)/2;
        cl->pad.c = 0;
    }
    return NN_SUCCESS;
}

microinfer_layer_t* Conv2D(uint32_t filters , microinfer_3d_shape_t k , microinfer_3d_shape_t s , microinfer_3d_shape_t d , 
                            microinfer_padding_t pad_type , const microinfer_weight_t* w , const microinfer_bias_t* b)
{
    microinfer_conv2d_layer_t* layer;
    microinfer_buf_t* comp;
    microinfer_layer_io_t* in , *out;

    uint32_t mem_size = sizeof(microinfer_conv2d_layer_t)+sizeof(microinfer_layer_io_t)*2+sizeof(microinfer_buf_t);
    layer = microinfer_mem(mem_size);
    if(layer == NULL)
        return NULL;
    in = (void*)((uint8_t*)layer + sizeof(microinfer_conv2d_layer_t));
    out = (void*)((uint8_t*)in + sizeof(microinfer_layer_io_t));
    comp = (void*)((uint8_t*)out + sizeof(microinfer_layer_io_t));

    layer->super.type = MICROINFER_CONV_2D;
    in->type = MICROINFER_TENSOR_BUF_TEMP;
    out->type = MICROINFER_TENSOR_BUF_TEMP;
    comp->type = MICROINFER_TENSOR_BUF_TEMP;

    layer->super.in = io_init(layer , in);
    layer->super.out = io_init(layer, out);

    layer->super.run = conv2d_run;
    layer->super.build = conv2d_build;

    layer->kernel = k;
    layer->stride = s;
    layer->dilation = d;
    layer->filter_mult = filters;
    layer->padding_type = pad_type;

    layer->weight = new_tensor(MICROINFER_QTYPE_PER_TENSOR , 4 , filters);
    layer->bias = new_tensor(MICROINFER_QTYPE_PER_TENSOR , 1 , filters);

    microinfer_shape_data_t dim[4] = {k.h , k.w , k.c , filters};
    *(layer->weight->q_offset) = 0;
    *(layer->weight->q_dec) = 0;
    layer->weight->p_data = (void*)w->p_value;
    layer->weight->bitwidth = 8;
    layer->weight->qtype = MICROINFER_QTYPE_PER_TENSOR;
    memcpy(layer->weight->dim , dim , layer->weight->num_dim*sizeof(microinfer_shape_data_t));

    dim[0] = filters;
    *(layer->bias->q_offset) = 0;
    *(layer->bias->q_dec) = 0;
    layer->bias->p_data = (void*)w->p_value;
    layer->bias->bitwidth = 8;
    layer->bias->qtype = MICROINFER_QTYPE_PER_TENSOR;
    memcpy(layer->bias->dim , dim , layer->weight->num_dim*sizeof(microinfer_shape_data_t));

    layer->output_rshift = (microinfer_qformat_param_t*)&w->shift;
    layer->bias_lshift = (microinfer_qformat_param_t*)&w->shift;

    return (microinfer_layer_t*)layer;
}