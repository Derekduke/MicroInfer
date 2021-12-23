#include "microinfer.h"
#include "microinfer_tensor.h"
#include "microinfer_local.h"
#include "microinfer_layers.h"
#include "layers/microinfer_dense.h"

microinfer_status_t dense_build(microinfer_layer_t *layer)
{
	microinfer_dense_layer_t *cl = (microinfer_dense_layer_t *)layer;

	// get the tensor from last layer's output
	layer->in->tensor = layer->in->hook.io->tensor;

	// create new tensor for output
	layer->out->tensor = new_tensor(MICROINFER_QTYPE_PER_TENSOR, 1, tensor_get_num_channel(layer->in->tensor));
	// setup new tensor
	microinfer_shape_data_t dim[1] = {cl->output_unit};
	tensor_set_attr(layer->out->tensor, cl->weight->q_dec, cl->weight->q_offset, dim, 1, 8); // test, this is not correct

	// calculate the output tensor q format, only support per tensor quantise now
	layer->out->tensor->q_dec[0] = layer->in->tensor->q_dec[0] + cl->weight->q_dec[0] - cl->output_rshift[0];
	// see if the activation will change the q format
	if(layer->actail)
	{ 
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);
		layer->actail->tensor = layer->out->tensor;
	}
	// vec_buffer size: dim_vec (*2, q7->q15) ? I am not sure this is right
	layer->comp->size = tensor_size(layer->in->tensor)*2;

	// computational cost: In * out
	//layer->stat.macc = tensor_size(layer->in->tensor) * tensor_size(layer->out->tensor);
	return NN_SUCCESS;
}

microinfer_status_t dense_run(microinfer_layer_t *layer)
{
	microinfer_status_t result = NN_SUCCESS;
	microinfer_dense_layer_t *cl = (microinfer_dense_layer_t *)(layer);
	microinfer_qformat_param_t bias_shift = cl->bias_lshift[0];			// this is not correct but a temporary fix solution for backward compatibility.
	microinfer_qformat_param_t output_shift = cl->output_rshift[0];
	local_fully_connected_q7_opt(
		layer->in->tensor->p_data,
		cl->weight->p_data,
		tensor_size(layer->in->tensor), layer->out->tensor->dim[0],
		bias_shift, output_shift,
		cl->bias->p_data,
		layer->out->tensor->p_data, (q15_t *)(layer->comp->mem->blk));
	
	return result;
}

microinfer_layer_t *Dense(size_t output_unit, const microinfer_weight_t *w, const microinfer_bias_t *b)
{
	microinfer_dense_layer_t *layer;
	microinfer_buf_t *comp;
	microinfer_layer_io_t *in, *out;
	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(microinfer_dense_layer_t) + sizeof(microinfer_layer_io_t) * 2 + sizeof(microinfer_buf_t);
	layer = microinfer_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(microinfer_dense_layer_t));
	out = (void *)((uint8_t*)in + sizeof(microinfer_layer_io_t));
	comp = (void *)((uint8_t*)out + sizeof(microinfer_layer_io_t));

	// set type in layer parent
	layer->super.type = MICROINFER_DENSE;
	// set buf state
	in->type = MICROINFER_TENSOR_BUF_TEMP;
	out->type = MICROINFER_TENSOR_BUF_TEMP;
	comp->type = MICROINFER_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->super.in = io_init(layer, in);
	layer->super.out = io_init(layer, out);
	layer->super.comp = comp;
	// set run and outshape methods
	layer->super.run = dense_run;
	layer->super.build = dense_build;

	// set parameters
	layer->output_unit = output_unit; // this is no longer needed. the information is contained in the weight tensor. 

	layer->weight = new_tensor(MICROINFER_QTYPE_PER_TENSOR, 2, output_unit);
	layer->bias = new_tensor(MICROINFER_QTYPE_PER_TENSOR, 1, output_unit);

	// configure weight tensor manually to support new tensor-based backends. 
	// needs to be very careful
	{
		// config weight 
		microinfer_shape_data_t dim[2] = {0, output_unit}; // the first dim doesnt matter here. will be file in later. 
		*(layer->weight->q_offset) = 0;			// we have no support of offset here
		*(layer->weight->q_dec) = 0;		// this is not even correct
		layer->weight->p_data = (void*)w->p_value;
		layer->weight->bitwidth = 8;
		layer->weight->qtype = MICROINFER_QTYPE_PER_TENSOR;
		microinfer_memcpy(layer->weight->dim, dim, layer->weight->num_dim * sizeof(microinfer_shape_data_t));

		// config bias 
		dim[0] = output_unit;
		*(layer->bias->q_offset) = 0;			// we have no support of offset here
		*(layer->bias->q_dec) = 0;		// this is not even correct
		layer->bias->p_data = (void*)b->p_value;
		layer->bias->bitwidth = 8;
		layer->weight->qtype = MICROINFER_QTYPE_PER_TENSOR;
		microinfer_memcpy(layer->bias->dim, dim, layer->bias->num_dim * sizeof(microinfer_shape_data_t));
	}

	// set output shifts
	layer->output_rshift = (microinfer_qformat_param_t *)&w->shift;
	layer->bias_lshift = (microinfer_qformat_param_t *)&b->shift;

	return (microinfer_layer_t *)layer;
}
