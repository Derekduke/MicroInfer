/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_H_
#define __MICROINFER_H_

#include "microinfer_port.h"

#define MICROINFER_ALIGN    (sizeof(char*))

struct _microinfer_layer_hook_t
{
	microinfer_layer_io_t *io;	    // hooked io
	microinfer_layer_hook_t *next;  // next hook include secondary hooked layer
};
typedef struct _microinfer_layer_hook_t microinfer_layer_hook_t;

struct _microinfer_layer_io_t
{
	microinfer_layer_hook_t hook;		  // for example: (layer->out)--hook--(layer->in)
    microinfer_layer_io_t *next; 		  // point to auxilary I/O (multiple I/O layer)
	microinfer_layer_t *owner;		      // which layer owns this io.
};

typedef struct _microinfer_layer_io_t microinfer_layer_io_t;

struct _microinfer_layer_t
{
	microinfer_layer_io_t *in;	  // IO buff, last*layer, states
	microinfer_layer_io_t *out;   // IO buff, next*layer, states
};
typedef struct _microinfer_layer_t microinfer_layer_t;

struct _microinfer_model_t
{
    microinfer_layer_t* head;
    microinfer_layer_t* tail;
    microinfer_layer_t* (*hook)(microinfer_layer_t* curr , microinfer_layer_t* pre);  
};
typedef struct _microinfer_model_t microinfer_model_t;


#endif