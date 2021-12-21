# microinfer

## windows 调试方法
### win安装make
https://www.cnblogs.com/jixiaohua/p/11724218.html

### 编译运行
cmd中输入“make”，生成bin.exe，输入“bin”运行

### 代码分析
基本要素：
通用层描述符
专用层描述符（包含通用层描述符）
tensor（用于描述某个内存中，存放数据的特征，比如CHW）
通用层所拥有的IO（IN或OUT两种）
通用层所拥有的BUF（其中包含用于中间计算的block和block的size）
IO所拥有的hook（用来指向下层的输入IO或上层的输出IO）
IO所拥有的block（用来存放IO中tensor描述的真实内存数据的指针）

逻辑思路：
1.创建各个层，主要工作是根据层的输入属性（如卷积核数量、大小等），初始化对应层的描述符和已经可确定的tensor
2.执行model_compile，主要工作是从头到位逐层遍历模型，确认每一层的输入、输出、计算，包括这三者所需要的tensor、内存块以及tensor和内存块的对应关系。
3.进行推理model_run