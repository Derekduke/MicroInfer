/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_OUTPUT_H_
#define __MICROINFER_OUTPUT_H_

#include "microinfer.h"
#include "microinfer_layers.h"
#include "microinfer_local.h"
#include "microinfer_tensor.h"
#include "layers/microinfer_input.h"

microinfer_layer_t *Output(microinfer_3d_shape_t output_shape, void *p_buf);

#endif
