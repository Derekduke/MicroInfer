/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include "microinfer.h"
#include "microinfer_layers.h"
#include "microinfer_tensor.h"

const char default_layer_names[][12] = DEFUALT_LAYER_NAMES;
const char default_activation_names[][8] = ACTIVATION_NAMES;

uint32_t microinfer_memory_used = 0; //统计推理框架总共占用内存大小
static uint8_t* microinfer_buf = NULL;
static uint32_t microinfer_buf_size = 0;
static uint32_t microinfer_buf_curr = 0;

/*---------------------------------------------------------------------*/
/*推理框架内存管理机制*/

//内存对齐
uint32_t microinfer_alignto(uint32_t value , uint32_t alignment)
{
    if(value % alignment == 0) return value;
    value += alignment - value % alignment; //如果无法整除对齐位数则补全
    return value;
}

//静态内存初始化，设置buf位置和大小
void microinfer_set_buf(void* buf , uint32_t size)
{
    microinfer_buf = buf; //buf起始指针
    microinfer_buf_size = size; //buf总长度
    microinfer_buf_curr = 0; //当前已使用buff大小
}

//在静态内存区上分配内存块
void* microinfer_malloc(uint32_t size)
{
    size = microinfer_alignto(size , MICROINFER_ALIGN);
    if(size + microinfer_buf_curr < microinfer_buf_size) //判断当前静态内存能否满足要求
    {
        uint8_t* new_buf = microinfer_buf + microinfer_buf_curr;//计算新的指针位置，方便下一次分配
        microinfer_buf_curr += size;
        return new_buf;
    }
    else
    {
        if(microinfer_buf_size == 0) //未分配静态内存，提醒需要事先分配
            MICROINFER_LOG("please set static buffer first\n");
        else //计算出仍需要分配的资源大小，提醒需要增加的内存大小
            MICROINFER_LOG("memory (%d) not big enough, please increase buffer size: (%d)\n", (uint32_t)microinfer_buf_size , (uint32_t)(size+microinfer_buf_curr-microinfer_buf_size));
        return NULL;
    }
}

void* microinfer_free(void* p)
{
    //哪里需要释放内存，待定
}
//为推理框架的内存使用，分配空间并初始化
void* microinfer_mem(uint32_t size)
{
    void* ptr = microinfer_malloc(size);
    if(ptr)
    {
        microinfer_memory_used += size;
        microinfer_memset(ptr , 0 , size); //初始化新分配的内存块为0，memset是否需要单独实现，待定
    }
    return ptr;
}
/*---------------------------------------------------------------------*/

static microinfer_layer_hook_t* allocate_hook(microinfer_layer_io_t* io)
{
    microinfer_layer_hook_t* hook;
    if(io == NULL) return NULL;
    hook = &io->hook;
    if(hook->io == NULL)
    {
        return hook;
    }
    else
    {
        while(hook->next != NULL)
        {
            hook = hook->next;
        }
        hook->next = microinfer_mem(sizeof(microinfer_layer_hook_t));
        if(hook->next == NULL) return NULL;
        return hook->next;
    }
}

static microinfer_layer_io_t* allocate_io(microinfer_layer_io_t* io)
{
    if(io == NULL) return NULL;
    if(io->hook.io == NULL)
    {
        return io;
    }
    else
    {
        while(io->next != NULL)
        {
            io = io->next;
        }
        io->next = microinfer_mem(sizeof(microinfer_layer_io_t));
        if(io->next == NULL) return NULL;
        io->next->owner = io->owner;
        return io->next;
    }
}

static microinfer_layer_t* model_hook(microinfer_layer_t* curr , microinfer_layer_t* pre)
{
    microinfer_layer_io_t* curr_in_io;
    microinfer_layer_hook_t* pre_out_io_hook;

    if(curr == NULL || pre == NULL) return NULL;

    pre_out_io_hook = allocate_hook(pre->out); //为上一层的输出IO分配实体hook（因为每个IO可能不止一个hook，为链式结构）
    curr_in_io = allocate_io(curr->in); //为当前层分配实体输入IO

    pre_out_io_hook->io = curr_in_io; //将上一层输出io的hook，指向当前层输入io
    curr_in_io->hook.io = pre->out; //将本层输入io的hook，指向上一层输出io

    return curr;
}

//模型初始化，分配模型描述符的空间，指定操作函数
microinfer_model_t* model_init(microinfer_model_t* model)
{
    microinfer_model_t* m = model;
    if(m == NULL) //判断模型是否已经初始化过（判断描述符是否已经实例化）
    {
        m = microinfer_mem(sizeof(microinfer_model_t));
    }
    else
    {
        microinfer_memset(m , 0 , sizeof(microinfer_model_t)); //若已经实例化过，则清零重新赋值
    }
    m->hook = model_hook;
    return m;
}

static microinfer_mem_block_t *allocate_block(microinfer_mem_block_t *list)
{
	microinfer_mem_block_t *free = NULL;
	uint32_t idx;

	for (idx = 0; idx < MICROINFER_BLOCK_NUM; idx++)
	{
		if (list[idx].owners == 0)
			break;
	}
    if(idx == MICROINFER_BLOCK_NUM)
    {
        MICROINFER_LOG("\nERROR! No enough memory block for parallel buffers, please increase the 'NNOM_BLOCK_NUM' in 'nnom_port.h'\n");
        return NULL;
    }

	free = &list[idx];
	return free;
}

static void release_block(microinfer_mem_block_t *block)
{
	if (block->owners > 0)
		block->owners -= 1;
	if (block->owners == 0)
		block->state = MICROINFER_BUF_EMPTY;
}

static void release_input_mem(microinfer_layer_t *layer)
{
	microinfer_layer_io_t *in;
	// release all input of buf
	in = layer->in;
	while (in != NULL)
	{
		release_block(in->mem);
		in = in->next;
	}
}

static void release_comp_mem(microinfer_layer_t *layer)
{
	// release computational buf if exist
	if (layer->comp != NULL)
	{
		release_block(layer->comp->mem);
	}
}

static uint32_t io_mem_size(microinfer_layer_io_t *io)
{
	uint32_t size = 0;
	if (io != NULL)
	{
		while (io)
		{
			size += tensor_size(io->tensor);
			io = io->next;
		}
	}
	return size;
}

uint32_t mem_analysis_result(microinfer_model_t *m)
{
	uint32_t index;
	uint32_t total_mem = 0;
	MICROINFER_LOG("\n Memory cost by each block:\n ");
	// print size of memory blocks
	for (index = 0; index < MICROINFER_BLOCK_NUM; index++)
	{
		total_mem += m->blocks[index].size;
		MICROINFER_LOG("blk_%d:%d  ", index, (uint32_t)(m->blocks[index].size));
	}
	// size of total memory cost by networks buffer
	MICROINFER_LOG("\n\n Memory will cost by all blocks: %d bytes\n", total_mem);
    
	return total_mem;
}

microinfer_status_t block_mem_set(microinfer_model_t *m, void *buf)
{
	uint32_t index;
	uint32_t mem_offset = 0;

	for (index = 0; index < MICROINFER_BLOCK_NUM; index++)
	{
		if (m->blocks[index].size == 0)
			break;
		m->blocks[index].blk = (void *)((uint8_t*)buf + mem_offset);
		mem_offset += m->blocks[index].size;
	}
	return NN_SUCCESS;
}

microinfer_status_t tensor_mem_set(microinfer_model_t *m)
{
	microinfer_layer_t *layer = m->head;
	microinfer_layer_io_t *io;
	while (layer)
	{
		io = layer->in;
		while (io)
		{
			io->tensor->p_data = io->mem->blk;
			io = io->next;
		}

		io = layer->out;
		while (io)
		{
			io->tensor->p_data = io->mem->blk;
			io = io->next;
		}
		if (layer->out->hook.io == NULL)
			return NN_SUCCESS;
        layer = layer->out->hook.io->owner;
	}
	return NN_SUCCESS;
}

static void print_layer_info(microinfer_layer_t *layer, uint32_t layer_count)
{
    uint32_t in_size = io_mem_size(layer->in);
    uint32_t out_size = io_mem_size(layer->out);
    uint32_t comp_size;
    if(layer->comp != NULL)
        comp_size = layer->comp->size;
    else
        comp_size = 0;
    MICROINFER_LOG("#%-3d %-10s - ", layer_count,  default_layer_names[layer->type]);
	// activations
	if (layer->actail != NULL)
		MICROINFER_LOG("%-8s - ", default_activation_names[layer->actail->type]);
	else
		MICROINFER_LOG("         - ");

	MICROINFER_LOG("(");
	for (int i = 0; i < 3; i++)
	{
		if (layer->out->tensor->num_dim > i)
			MICROINFER_LOG("%4d,", layer->out->tensor->dim[i]);
		else 
			MICROINFER_LOG("     ");
	}
	MICROINFER_LOG(")  ");

    MICROINFER_LOG("- (%4d,%4d,%4d,)", (uint32_t)in_size, (uint32_t)out_size,(uint32_t) comp_size);
}

static void print_memory_block_info(microinfer_mem_block_t *block_pool)
{
	// show the memory blocks's lifetime (number of owners)
	MICROINFER_LOG("   ");
	for (int i = 0; i < MICROINFER_BLOCK_NUM; i++)
	{
		if (i % 4 == 0)
			MICROINFER_LOG(" ");
		if (block_pool[i].owners)
			MICROINFER_LOG("%d ", block_pool[i].owners);
		else
			MICROINFER_LOG("- ");
	}
	MICROINFER_LOG("\n");
}

microinfer_status_t compile_layers(microinfer_layer_t* first, microinfer_layer_t *curr, microinfer_mem_block_t *block_pool, uint32_t *layer_count)
{
    
    uint32_t mem_size = 0;
    microinfer_layer_t* layer = curr;
    microinfer_layer_io_t* in;
    microinfer_layer_io_t* out;
    microinfer_layer_hook_t* hook;

    microinfer_mem_block_t* in_blk;
    microinfer_mem_block_t* out_blk;

    uint32_t local_layer_count = 1;
	if(layer_count == NULL)
		layer_count = &local_layer_count;

    in = layer->in;
    out = layer->out;

    while(layer)
    {
        in = layer->in;
        if(in->hook.io == NULL)
        {
            if(in->mem == NULL)
            {
                in_blk = allocate_block(block_pool);
                in_blk->owners += 1;
                mem_size = microinfer_alignto(tensor_size(in->tensor) , MICROINFER_ALIGN);
                in_blk->size = mem_size > in_blk->size ? mem_size : in_blk->size;
                in->mem = in_blk;
                in->mem->state = MICROINFER_BUF_FILLED;
            }
        }
        else
        {
            while(in != NULL)
            {
                in->mem = in->hook.io->mem;
                in = in->next;
            }
        }
        
        layer->build(layer);

        if(layer->comp != NULL)
        {
            layer->comp->mem = allocate_block(block_pool);
            layer->comp->mem->owners += 1;
            layer->comp->mem->state = MICROINFER_BUF_FILLED;
            mem_size = microinfer_alignto(layer->comp->size , MICROINFER_ALIGN);
            layer->comp->mem->size = mem_size>layer->comp->mem->size ? mem_size:layer->comp->mem->size;
        }
        print_layer_info(layer, (*layer_count)++);

        if(layer->out == NULL)
            return NN_SUCCESS;
        
        if(layer->out->next == NULL && layer->out->hook.next == NULL)
        {
            if(layer->in->type == MICROINFER_TENSOR_BUF_NULL || layer->out->type == MICROINFER_TENSOR_BUF_NULL)
            {
                layer->out->mem = layer->in->mem;
                print_memory_block_info(block_pool);
            }
            else
            {
                out_blk = allocate_block(block_pool);
                if(out_blk == NULL)
                    return NN_NO_MEMORY;
                out_blk->owners = 1;
                out_blk->state = MICROINFER_BUF_FILLED;
                mem_size = microinfer_alignto(tensor_size(layer->out->tensor) , MICROINFER_ALIGN);
                out_blk->size = mem_size > out_blk->size ? mem_size : out_blk->size;
                layer->out->mem = out_blk;

                print_memory_block_info(block_pool);
				release_input_mem(layer);
				release_comp_mem(layer);                
            }
        }
		if (layer->out->hook.io == NULL)
			return NN_SUCCESS;
        layer = layer->out->hook.io->owner;
    }
    return NN_SUCCESS;
}

microinfer_status_t model_compile(microinfer_model_t* m , microinfer_layer_t* input , microinfer_layer_t* output)
{
    uint32_t buf_size;
    uint8_t* buf;
    uint32_t layer_num = 1;
    
    m->head = input;
    m->tail = output;

    MICROINFER_LOG("MicroInfer Version %d.%d.%d\n" , MICROINFER_MAJORVERSION, MICROINFER_SUBVERSION, MICROINFER_REVISION);
    MICROINFER_LOG("Static memory size set to: %d bytes\n", (uint32_t)microinfer_buf_size);

	MICROINFER_LOG("Start compiling model...\n\n");
	MICROINFER_LOG("Layer(#)         Activation    output shape    mem(in, out, middle)   mem blk lifetime\n");
	MICROINFER_LOG("--------------------------------------------------------------------------------------\n");
    compile_layers(m->head , m->head , m->blocks , &layer_num);
	MICROINFER_LOG("--------------------------------------------------------------------------------------\n");

    buf_size =  mem_analysis_result(m);
	buf = microinfer_mem(buf_size);
	if (buf == NULL)
	{
		MICROINFER_LOG("ERROR: No enough memory for network buffer, required %d bytes\n", (uint32_t)buf_size);
		return NN_NO_MEMORY;
	}
    MICROINFER_LOG("\n Memory already cost by MircoInfer: %d bytes\n", microinfer_buf_curr);
    block_mem_set(m, buf);
    tensor_mem_set(m);
    return NN_SUCCESS;
}