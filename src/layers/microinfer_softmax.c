/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include "microinfer.h"
#include "microinfer_local.h"
#include "microinfer_layers.h"
#include "layers/microinfer_softmax.h"

microinfer_status_t softmax_build(microinfer_layer_t *layer)
{
	// get the last layer's output as input shape
	layer->in->tensor = layer->in->hook.io->tensor;
	// output tensor
	layer->out->tensor = new_tensor(MICROINFER_QTYPE_PER_TENSOR, layer->in->tensor->num_dim, tensor_get_num_channel(layer->in->tensor));
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);
	// softmax has fixed output dec bit
	layer->out->tensor->q_dec[0] = 7;
	return NN_SUCCESS;
}

microinfer_status_t softmax_run(microinfer_layer_t *layer)
{
	// looks like the new version cause accuracy drop quite a lot. 
//	#ifdef NNOM_USING_CMSIS_NN
//	// temporary fixed for mutiple dimension input. 
//	arm_softmax_q7(layer->in->tensor->p_data, tensor_size(layer->out->tensor), layer->out->tensor->p_data);
//	#else
	//local_softmax_q7(layer->in->tensor->p_data, tensor_size(layer->out->tensor), layer->out->tensor->p_data);
	//#endif
	return NN_SUCCESS;
}

microinfer_layer_t *Softmax(void)
{
	microinfer_layer_t *layer;
	microinfer_layer_io_t *in, *out;

	// apply a block memory for all the sub handles.
	size_t mem_size = sizeof(microinfer_layer_t) + sizeof(microinfer_layer_io_t) * 2;
	layer = microinfer_mem(mem_size);
	if (layer == NULL)
		return NULL;

	// distribut the memory to sub handles.
	in = (void *)((uint8_t*)layer + sizeof(microinfer_layer_t));
	out = (void *)((uint8_t*)in + sizeof(microinfer_layer_io_t));

	// set type in layer parent
	layer->type = MICROINFER_SOFTMAX;
	layer->run = softmax_run;
	layer->build = softmax_build;
	// set buf state
	in->type = MICROINFER_TENSOR_BUF_TEMP;
	out->type = MICROINFER_TENSOR_BUF_TEMP;
	// put in & out on the layer.
	layer->in = io_init(layer, in);
	layer->out = io_init(layer, out);

	return layer;
}