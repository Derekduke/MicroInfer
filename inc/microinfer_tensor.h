/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include "microinfer.h"

uint32_t tensor_size(microinfer_tensor_t* t);
uint32_t tensor_get_num_channel(microinfer_tensor_t* t);
microinfer_tensor_t* tensor_cpy_attr(microinfer_tensor_t* des, microinfer_tensor_t* src);
microinfer_tensor_t* new_tensor(microinfer_qtype_t type , uint32_t num_dim , uint32_t num_channel);
void delete_tensor(microinfer_tensor_t* t);
microinfer_tensor_t* tensor_set_attr_v(microinfer_tensor_t* t, 
		microinfer_qformat_param_t dec_bit, microinfer_qformat_param_t offset, microinfer_shape_data_t* dim, uint32_t num_dim, uint8_t bitwidth);
