/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */
#include "microinfer.h"
#include "microinfer_layers.h"
#include "microinfer_local.h"
#include "microinfer_tensor.h"
#include "layers/microinfer_output.h"

microinfer_status_t output_run(microinfer_layer_t *layer)
{
	microinfer_input_layer_t *cl = (microinfer_input_layer_t*)layer;
	microinfer_memcpy(cl->buf, layer->in->tensor->p_data, tensor_size(layer->out->tensor)); // in->memory -> user memory
	return NN_SUCCESS;
}

microinfer_status_t default_build(microinfer_layer_t *layer)
{
	// get the last layer's output as input shape
	layer->in->tensor = layer->in->hook.io->tensor;
	// output tensor
	// 1. allocate a new tensor for output
	// 2. set the same dim, qfmt to the new tensor.
	layer->out->tensor = new_tensor(MICROINFER_QTYPE_PER_TENSOR,layer->in->tensor->num_dim, tensor_get_num_channel(layer->in->tensor));
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);
	
	// see if the activation will change the q format
	if(layer->actail) 
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);

	// now this build has passed the input tensors (shapes, formats) to the new tensors. 
	return NN_SUCCESS;
}

microinfer_layer_t *Output(microinfer_3d_shape_t output_shape, void *p_buf)
{
	// they are acturally the same.. expect the type defined
	microinfer_layer_t *layer = Input(output_shape, p_buf);
	if (layer != NULL)
	{
		layer->type = MICROINFER_OUTPUT;
		layer->run = output_run;
		layer->build = default_build;
	}
	return layer;
}
