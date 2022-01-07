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
    return NULL;
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
//为指定层的in或者out IO 去分配实体的hook，考虑同一个IO存在多个hook情况，用单向链表连接
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
//为指定层分配实体的IN或者OUT IO，考虑一个层存在多个IN IO或OUT IO的情况，用单项链表连接
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
//功能API，用于上下层之间的双向连接，pre_layer.out_io.hook->curr_layer.in_io // curr_layer.in_io.hook->pre_layer.out_io
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

static microinfer_layer_t *model_active(microinfer_activation_t *act, microinfer_layer_t *target)
{
	target->actail = act;
	return target;
}

//模型初始化，分配模型描述符的空间，指定操作函数
microinfer_model_t* model_create(microinfer_model_t* model)
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
    //指定类操作函数
    m->hook = model_hook; 
    m->active = model_active;
    return m;
}
//从模型描述符中，分配已经实例化过的内存块描述符（并不是真正的内存）
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
    //将第一个没有所属者的内存块（不代表没有被用过，可能是用了又回收了），作为最优分配目标
	free = &list[idx];
	return free;
}
//回收内存块，只收回所属权限，但实际上并不修改其它属性或清空内存，下次再被用到的时候再改
static void release_block(microinfer_mem_block_t *block)
{
	if (block->owners > 0)
		block->owners -= 1;
	if (block->owners == 0)
		block->state = MICROINFER_BUF_EMPTY;
}
//回收指定层的所有IN IO内存块或者OUT IO内存块；仅回收内存块，并不回收描述符
static void release_input_mem(microinfer_layer_t *layer)
{
	microinfer_layer_io_t *in;
	in = layer->in;
	while (in != NULL)
	{
		release_block(in->mem);
		in = in->next;
	}
}
//回收指定层用于计算而分配的内存块
static void release_comp_mem(microinfer_layer_t *layer)
{
	if (layer->comp != NULL)
	{
		release_block(layer->comp->mem);
	}
}
//计算指定层的所有IN IO或OUT IO的tensor大小（理解为对应shape的数据buff，需要内存块总和的大小）
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
//遍历内存块描述符，获得所有内存块的大小
uint32_t mem_analysis_result(microinfer_model_t *m)
{
	uint32_t index;
	uint32_t total_mem = 0;
	MICROINFER_LOG("\n Memory cost by each block:\n ");
	for (index = 0; index < MICROINFER_BLOCK_NUM; index++)
	{
		total_mem += m->blocks[index].size;
		MICROINFER_LOG("blk_%d:%d  ", index, (uint32_t)(m->blocks[index].size));
	}
	MICROINFER_LOG("\n\n Memory will cost by all blocks: %d bytes\n", total_mem);
    
	return total_mem;
}
//将划分的一大块主内存，按照各个内存块事先计算的所需大小做划分（设置对应内存块的指针）
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
//将指定模型的每一层的IN IO和OUT IO的tensor（数据描述符）的数据指针和给IO对应实际分配的内存块连接
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
//输出每一层的名字，激活类型，输出shape，输入中间输出的buff大小，内存块生存周期
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
//输出各个内存块最终确认的所需内存大小
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
//遍历各层，计算各层输入输出中间的tensor，计算内存块所需数量和大小，目前仅支持顺序模型结构的单输入和单输出
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
        if(in->hook.io == NULL)//Input层
        {
            if(in->mem == NULL)//输入层未经过内存初始化
            {
                in_blk = allocate_block(block_pool);
                in_blk->owners += 1;
                mem_size = microinfer_alignto(tensor_size(in->tensor) , MICROINFER_ALIGN);//根据tensor的shape去计算所需内存块的大小
                in_blk->size = mem_size > in_blk->size ? mem_size : in_blk->size;//和内存块原来的size值相比较（可能之前被用过），取较大值
                in->mem = in_blk;
                in->mem->state = MICROINFER_BUF_FILLED;
            }
        }
        else//非Input层
        {
            while(in != NULL)
            {
                in->mem = in->hook.io->mem;//该层IN IO输入的内存块，一定等同于，上一层OUT IO输出的内存块
                in = in->next;
            }
        }
        
        layer->build(layer);//运行构造模型的函数，确定该层输出和计算的tensor大小

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
        
        if(layer->out->next == NULL && layer->out->hook.next == NULL) //单输出层 连接到 单输入层的情况
        {
            if(layer->in->type == MICROINFER_TENSOR_BUF_NULL || layer->out->type == MICROINFER_TENSOR_BUF_NULL)
            { //这个层的类型，是一个单buff，不需要其他计算（比如input）
                layer->out->mem = layer->in->mem; //
                print_memory_block_info(block_pool);
                release_comp_mem(layer);
            }
            else //这个层的类型，需要额外的中间运算，需要更多的buff（比如卷积）
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
                //除了输出层，输入和中间层的内存块全部回收
				release_input_mem(layer);
				release_comp_mem(layer);                
            }
        }
		if (layer->out->hook.io == NULL)
			return NN_SUCCESS;
        layer = layer->out->hook.io->owner; //通过钩子找到下一层
    }
    return NN_SUCCESS;
}
//执行模型计算图构建的API函数，根据层的属性，逐层根据数据流向去计算输入、输出、中间层的tensor，并按最少内存块使用的原则（近似贪心算法），分配实体内存
microinfer_status_t model_compile(microinfer_model_t* m , microinfer_layer_t* input , microinfer_layer_t* output)
{
    uint32_t buf_size;
    uint8_t* buf;
    uint32_t layer_num = 1;
    
    m->head = input;
    m->tail = output;

    MICROINFER_LOG("MicroInfer Version %d.%d.%d\n" , MICROINFER_MAJORVERSION, MICROINFER_SUBVERSION, MICROINFER_REVISION);
	#ifdef MICROINFER_USING_CHW
	    MICROINFER_LOG("Data format: Channel first (CHW)\n");
	#else
	    MICROINFER_LOG("Data format: Channel last (HWC)\n");
	#endif
	#ifdef MICROINFER_USING_CMSIS_NN
	   MICROINFER_LOG("Backend optimization: CMSIS-NN\n");
	#endif

    MICROINFER_LOG("Static memory size set to: %d bytes\n", (uint32_t)microinfer_buf_size);

	MICROINFER_LOG("Start compiling model...\n\n");
	MICROINFER_LOG("Layer(#)         Activation    output shape    mem(in, out, middle)   mem blk lifetime\n");
	MICROINFER_LOG("--------------------------------------------------------------------------------------\n");
    compile_layers(m->head , m->head , m->blocks , &layer_num); //计算模型总共所需要的内存块数量和大小
	MICROINFER_LOG("--------------------------------------------------------------------------------------\n");

    buf_size =  mem_analysis_result(m); //输出各个内存块的大小，返回需要的内存总和
	buf = microinfer_mem(buf_size); //先按照内存综合分配一大块内存（有助于减少碎片化）
	if (buf == NULL)
	{
		MICROINFER_LOG("ERROR: No enough memory for network buffer, required %d bytes\n", (uint32_t)buf_size);
		return NN_NO_MEMORY;
	}
    MICROINFER_LOG("\n Memory already cost by MircoInfer: %d bytes\n", microinfer_buf_curr);
    block_mem_set(m, buf); //切割大内存，分配给各个内存块描述符
    tensor_mem_set(m); //将该层的IO tensor指向内存块,相当于是数据形状的描述和存储数据的内存绑定
    return NN_SUCCESS;
}

microinfer_status_t model_run(microinfer_model_t *m)
{
	uint32_t layer_num = 1;
	microinfer_status_t result;
	microinfer_layer_t *layer;

	layer = m->head;
	
	// using shortcut run
	while (layer)
	{
		// run layer
		//result = layer_run(layer);
        result = layer->run(layer);
        if(layer->actail != NULL)
        {
            layer->actail->run(layer->actail);
        }
		if (result != NN_SUCCESS)
		{
			MICROINFER_LOG("Error: #%d %s layer return error code:%d\n", layer_num, default_layer_names[layer->type], result);
			return result;
		}
		// run callback
        /*
		if(m->layer_callback != NULL)
		{
			result = m->layer_callback(m, layer);
			if (result != NN_SUCCESS)
			{
				NNOM_LOG("Error: Callback return error code %d at #%d %s layer\n", result, layer_num, default_layer_names[layer->type]);
				return result;
			}
		}
        */		
		// check if finished
		if (layer->out->hook.io == NULL)
			break;
		layer = layer->out->hook.io->owner;
		layer_num++;
	}

	return NN_SUCCESS;   
}

microinfer_status_t microinfer_predict(microinfer_model_t *m, uint32_t *label, float *prob)
{
	if (!m)
		return NN_ARGUMENT_ERROR;
	model_run(m);
	microinfer_output(m, label);
	return NN_SUCCESS;
}

microinfer_status_t microinfer_output(microinfer_model_t *m, uint32_t *label)
{
	int32_t max_val, max_index, sum;
	int8_t *output;
	// get the output memory
	output = m->tail->out->tensor->p_data;

	// multiple neural output
	if (tensor_size(m->tail->out->tensor) > 0)
	{
		// Top 1
		max_val = output[0];
		max_index = 0;
		sum = max_val;
		for (uint32_t i = 1; i < tensor_size(m->tail->out->tensor); i++)
		{
			if (output[i] > max_val)
			{
				max_val = output[i];
				max_index = i;
			}
			sum += output[i];
		}
		// send results
		*label = max_index;
	}
	
	return NN_SUCCESS;
}