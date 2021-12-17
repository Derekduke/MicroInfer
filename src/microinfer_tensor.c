/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include "microinfer.h"
#include "microinfer_tensor.h"

uint32_t tensor_size(microinfer_tensor_t* t)
{
    uint32_t size = 0;
    if(t != NULL)
    {
        size = t->dim[0];
        for(int i=1 ; i<t->num_dim ; i++)
        {
            size *= t->dim[i];
        }
    }
    return size;
}

uint32_t tensor_get_num_channel(microinfer_tensor_t* t)
{
    return t->dim[t->num_dim-1];
}

microinfer_tensor_t* tensor_cpy_attr(microinfer_tensor_t* des, microinfer_tensor_t* src)
{
	uint32_t size;
	if(src->qtype != des->qtype || src->num_dim != des->num_dim)
		return NULL;
	
	if(src->qtype == MICROINFER_QTYPE_PER_AXIS)
		size = sizeof(microinfer_qformat_param_t) * tensor_get_num_channel(src);
	else
		size = sizeof(microinfer_qformat_param_t);
	
	// bit
	des->bitwidth = src->bitwidth;
	// copy quantisation parameters
	microinfer_memcpy(des->q_dec, src->q_dec, size);
	microinfer_memcpy(des->q_offset, src->q_offset, size);
	// copy number of dimension
	des->num_dim = src->num_dim;
	microinfer_memcpy(des->dim, src->dim, src->num_dim * sizeof(microinfer_shape_data_t));
	return des;
}

microinfer_tensor_t* new_tensor(microinfer_qtype_t type , uint32_t num_dim , uint32_t num_channel)
{
    microinfer_tensor_t* t = NULL;
    uint32_t q_len;
    if(type == MICROINFER_QTYPE_PER_TENSOR)
    {
        q_len = 1;
    }
    else
    {
        MICROINFER_LOG("ERROR: tensor type error!");
        return NULL;
    }
    t = microinfer_mem(microinfer_alignto(sizeof(microinfer_tensor_t) , MICROINFER_ALIGN)+num_dim*sizeof(microinfer_shape_data_t)+q_len*sizeof(microinfer_qformat_param_t)*2);
    if(t == NULL)
        return t;
    t->dim = (microinfer_shape_data_t*)((uint8_t*)t+sizeof(microinfer_tensor_t));
    t->q_dec = (microinfer_qformat_param_t*)((uint8_t*)t->dim+num_dim*sizeof(microinfer_shape_data_t));
    t->q_offset = (microinfer_qformat_param_t*)((uint8_t*)t->q_dec+q_len*sizeof(microinfer_qformat_param_t));
    t->num_dim = num_dim;
    t->qtype = type;
    return t;
}

void delete_tensor(microinfer_tensor_t* t)
{
	if (t)
		microinfer_free(t);
}

// set tensor by value
microinfer_tensor_t* tensor_set_attr_v(microinfer_tensor_t* t, 
		microinfer_qformat_param_t dec_bit, microinfer_qformat_param_t offset, microinfer_shape_data_t* dim, uint32_t num_dim, uint8_t bitwidth)
{
	// copy dim
	t->num_dim = num_dim;
	microinfer_memcpy(t->dim, dim, sizeof(microinfer_shape_data_t) * num_dim);
	// bitwidth
	t->bitwidth = bitwidth;
	// copy the offset and q format
	*(t->q_dec) = dec_bit;
	*(t->q_offset) = offset;
	return t;
}