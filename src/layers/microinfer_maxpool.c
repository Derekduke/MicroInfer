#include "microinfer.h"
#include "microinfer_local.h"
#include "microinfer_layers.h"
#include "microinfer_tensor.h"
#include "layers/microinfer_maxpool.h"

microinfer_status_t maxpool_run(microinfer_layer_t *layer)
{

}

microinfer_status_t maxpool_build(microinfer_layer_t *layer)
{
	microinfer_maxpool_layer_t *cl = (microinfer_maxpool_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for output
	layer->out->tensor = new_tensor(MICROINFER_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, tensor_get_num_channel(layer->in->tensor));
	// copy then change later. 
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);
	
	// see if the activation will change the q format
	if(layer->actail) 
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);

	// now we set up the tensor shape, always HWC format
	if (cl->padding_type == PADDING_SAME)
	{
		layer->out->tensor->dim[0] = NN_CEILIF(layer->in->tensor->dim[0], cl->stride.h);
		layer->out->tensor->dim[1] = NN_CEILIF(layer->in->tensor->dim[1], cl->stride.w);
		layer->out->tensor->dim[2] = layer->in->tensor->dim[2]; // channel stays the same
	}
	else
	{
		layer->out->tensor->dim[0] = NN_CEILIF(layer->in->tensor->dim[0] - cl->kernel.h + 1, cl->stride.h);
		layer->out->tensor->dim[1] = NN_CEILIF(layer->in->tensor->dim[1] - cl->kernel.w + 1, cl->stride.w);
		layer->out->tensor->dim[2] = layer->in->tensor->dim[2];
	}

	return NN_SUCCESS;
}

microinfer_layer_t *MaxPool(microinfer_3d_shape_t k, microinfer_3d_shape_t s, microinfer_padding_t pad_type)
{
	microinfer_maxpool_layer_t *layer;
	microinfer_buf_t *comp;
	microinfer_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	uint32_t mem_size = sizeof(microinfer_maxpool_layer_t) + sizeof(microinfer_layer_io_t) * 2 + sizeof(microinfer_buf_t);
	layer = microinfer_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(microinfer_maxpool_layer_t));
	out = (void *)((uint8_t*)in + sizeof(microinfer_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(microinfer_layer_io_t));

	// set type in layer parent
	layer->super.type = MICROINFER_MAXPOOL;
	layer->super.run = maxpool_run;
	layer->super.build = maxpool_build;
	// set buf state
	in->type =MICROINFER_TENSOR_BUF_TEMP;
	out->type =MICROINFER_TENSOR_BUF_TEMP;
	comp->type = MICROINFER_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	layer->super.comp = comp;

	// set parameters
	layer->kernel = k;
	layer->stride = s;
	layer->padding_type = pad_type;

	// padding
	if (layer->padding_type == PADDING_SAME)
	{
		layer->pad.h = (k.h - 1) / 2;
		layer->pad.w = (k.w - 1) / 2;
		layer->pad.c = 1; // no meaning
	}
	else
	{
		layer->pad.h = 0;
		layer->pad.w = 0;
		layer->pad.c = 0;
	}
	return (microinfer_layer_t *)layer;
}