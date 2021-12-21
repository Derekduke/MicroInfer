/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_CONV2D_H__
#define __MICROINFER_CONV2D_H__

#include "microinfer.h"
#include "microinfer_layers.h"

typedef struct _microinfer_conv2d_layer_t
{
    microinfer_layer_t super;
    microinfer_3d_shape_t kernel;
    microinfer_3d_shape_t stride;
    microinfer_3d_shape_t pad;
    microinfer_3d_shape_t dilation;
	microinfer_padding_t padding_type;
	uint32_t filter_mult;

    microinfer_tensor_t* weight;
    microinfer_tensor_t* bias;

	microinfer_qformat_param_t * output_rshift;			
	microinfer_qformat_param_t * bias_lshift;    
}microinfer_conv2d_layer_t;

microinfer_layer_t* Conv2D(uint32_t filters , microinfer_3d_shape_t k , microinfer_3d_shape_t s , microinfer_3d_shape_t d , 
                            microinfer_padding_t pad_type , const microinfer_weight_t* w , const microinfer_bias_t* b);

#endif