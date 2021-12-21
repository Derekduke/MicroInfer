/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_CONV2D_H__
#define __MICROINFER_CONV2D_H__

#include "microinfer.h"
#include "microinfer_layers.h"

typedef struct _microinfer_dense_layer_t
{
	microinfer_layer_t super;
	size_t output_unit;
	microinfer_tensor_t *weight;
	microinfer_tensor_t *bias;
	microinfer_qformat_param_t *output_rshift;			
	microinfer_qformat_param_t *bias_lshift;
} microinfer_dense_layer_t;

microinfer_layer_t *Dense(size_t output_unit, const microinfer_weight_t *w, const microinfer_bias_t *b);

#endif
