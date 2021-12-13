/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_LAYERS_H_
#define __MICROINFER_LAYERS_H_

#include "microinfer.h"

microinfer_3d_shape_t shape(size_t h, size_t w, size_t c);
size_t shape_size(microinfer_3d_shape_t *s);

microinfer_layer_t* Input(microinfer_3d_shape_t input_shape , void* p_buf);
microinfer_layer_io_t* io_init(void* owner_layer , microinfer_layer_io_t* io);

#endif