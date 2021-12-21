/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_MAXPOOL_H_
#define __MICROINFER_MAXPOOL_H_

#include "microinfer.h"
#include "microinfer_layers.h"

//输入层专有的描述，包含了一般的层结构
typedef struct _microinfer_maxpool_layer_t
{
	microinfer_layer_t super;
	microinfer_3d_shape_t kernel;
	microinfer_3d_shape_t stride;
	microinfer_3d_shape_t pad;
	microinfer_padding_t padding_type;
	int16_t output_shift;			// reserve
} microinfer_maxpool_layer_t;

microinfer_layer_t *MaxPool(microinfer_3d_shape_t k, microinfer_3d_shape_t s, microinfer_padding_t pad_type);
#endif 