/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_LAYERS_H_
#define __MICROINFER_LAYERS_H_

#include "microinfer.h"

#define NN_CEILIF(x,y) ((x+y-1)/y)

microinfer_3d_shape_t shape(size_t h, size_t w, size_t c);
size_t shape_size(microinfer_3d_shape_t *s);

microinfer_layer_t* Input(microinfer_3d_shape_t input_shape , void* p_buf);
microinfer_layer_t* Conv2D(uint32_t filters , microinfer_3d_shape_t k , microinfer_3d_shape_t s , microinfer_3d_shape_t d , 
                            microinfer_padding_t pad_type , const microinfer_weight_t* w , const microinfer_bias_t* b);
microinfer_layer_t *MaxPool(microinfer_3d_shape_t k, microinfer_3d_shape_t s, microinfer_padding_t pad_type);
microinfer_layer_t *Dense(size_t output_unit, const microinfer_weight_t *w, const microinfer_bias_t *b);
microinfer_layer_t *Softmax(void);
microinfer_layer_t *Output(microinfer_3d_shape_t output_shape, void *p_buf);

microinfer_activation_t* act_relu(void);
int32_t act_get_dec_bit(microinfer_activation_type_t type, int32_t dec_bit);

microinfer_3d_shape_t kernel(size_t h, size_t w);
microinfer_3d_shape_t stride(size_t h, size_t w);
microinfer_3d_shape_t dilation(size_t h, size_t w);

microinfer_layer_io_t* io_init(void* owner_layer , microinfer_layer_io_t* io);

#endif