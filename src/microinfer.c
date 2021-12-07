/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */

#include "microinfer.h"

//microinfer_layer_t test;

struct test
{
    int a;
    int b;
};

typedef struct test test_t;


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
            MICROINFER_LOG("please set static buffer first");
        else //计算出仍需要分配的资源大小，提醒需要增加的内存大小
            //MICROINFER_LOG("memory (%d) not big enough, please increase buffer size: (%d)", \
                        (uint32_t)microinfer_buf_size , (uint32_t)(size+microinfer_buf_curr-microinfer_buf_size);
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
    curr_in_io->hook.io = last->out; //将本层输入io的hook，指向上一层输出io

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