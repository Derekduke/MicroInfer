/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_INPUT_H_
#define __MICROINFER_INPUT_H_

#include "microinfer.h"
#include "microinfer_layers.h"

//输入层专有的描述，包含了一般的层结构
typedef struct _microinfer_input_layer
{
    microinfer_layer_t super;
    microinfer_3d_shape_t shape;
    microinfer_qformat_param_t dec_bit;
    void* buf;
}microinfer_input_layer_t;

microinfer_layer_t* Input(microinfer_3d_shape_t input_shape , void* p_buf);
#endif 
