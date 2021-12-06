/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#ifndef __MICROINFER_H_
#define __MICROINFER_H_

#include "microinfer_port.h"

#define MICROINFER_ALIGN    (sizeof(char*))

struct _microinfer_layer
{

};
typedef struct _microinfer_layer microinfer_layer_t;

struct _microinfer_model
{
    microinfer_layer_t* head;
    microinfer_layer_t* tail;
    microinfer_layer_t* (*hook)(microinfer_layer_t* curr , microinfer_layer_t* pre);  
};
typedef struct _microinfer_model microinfer_model_t;

#endif