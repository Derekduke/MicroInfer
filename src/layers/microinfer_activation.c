/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include "microinfer.h"
#include "microinfer_layers.h"
#include "microinfer_tensor.h"
#include "microinfer_local.h"
#include "layers/microinfer_activation.h"

static microinfer_status_t relu_run(microinfer_activation_t* act)
{

    local_relu_q7(act->tensor->p_data, tensor_size(act->tensor));
	return NN_SUCCESS;
}

microinfer_activation_t* act_relu(void)
{
	microinfer_activation_t* act = microinfer_mem(sizeof(microinfer_activation_t));
	act->run = relu_run;
	act->type = ACT_RELU;
	return act;
}

int32_t act_get_dec_bit(microinfer_activation_type_t type, int32_t dec_bit)
{
	switch(type)
	{
		case ACT_RELU:
		case ACT_LEAKY_RELU:
		case ACT_ADV_RELU:
			break;
		case ACT_TANH:
        case ACT_HARD_TANH:
		case ACT_SIGMOID:
        case ACT_HARD_SIGMOID:
			dec_bit = 7;
		default:break;
	}
	return dec_bit;
}