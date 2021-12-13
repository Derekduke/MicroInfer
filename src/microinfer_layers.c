/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include "microinfer.h"
#include "microinfer_layers.h"

size_t shape_size(microinfer_3d_shape_t *s)
{
	if (s == NULL)
		return 0;
	return s->h * s->w * s->c;
}

microinfer_3d_shape_t shape(size_t h, size_t w, size_t c)
{
	microinfer_3d_shape_t s;
	s.h = h;
	s.w = w;
	s.c = c;
	return s;
}

microinfer_layer_io_t* io_init(void* owner_layer , microinfer_layer_io_t* io)
{
    io->owner = (microinfer_layer_t*)owner_layer;
    return io;
}

