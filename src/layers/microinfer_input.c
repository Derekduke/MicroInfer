/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include <microinfer.h>
#include <microinfer_layers.h>
#include <layers/microinfer_input.h>

microinfer_status_t input_run(microinfer_layer_t* layer)
{
    microinfer_input_layer_t* cl = (microinfer_input_layer_t*)layer;
    //microinfer_memcpy(layer->in->tensor->p_data , cl->buf ,);
}

microinfer_layer_t* Input(microinfer_3d_shape_t input_shape , void* p_buf)
{
    microinfer_input_layer_t* layer;
    microinfer_layer_io_t* in, *out;
    //预先分配整个内存
    layer = microinfer_mem(sizeof(microinfer_input_layer_t) + sizeof(microinfer_layer_io_t)*2);
    if(layer == NULL)
        return NULL;
    //设定IO指针的位置
    in = (void*)((uint8_t*)layer + sizeof(microinfer_input_layer_t));
    out = (void*)((uint8_t*)in + sizeof(microinfer_layer_io_t));
    
    layer->super.type = MICROINFER_INPUT;
    layer->super.run = input_run;
    //layer->super.build = input_build;

    layer->super.in = io_init(layer , in);
    layer->super.out = io_init(layer , out);

    layer->shape = input_shape;
    layer->buf = p_buf;

    microinfer_shape_data_t dim[3] = { input_shape.h, input_shape.w, input_shape.c };

    return (microinfer_layer_t*)layer;
}