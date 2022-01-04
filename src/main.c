#include "stdio.h"
#include "microinfer.h"
#include "microinfer_layers.h"
#include "microinfer_local.h"
#include "weight.h"
#include "image.h"

microinfer_model_t *model;
static int8_t microinfer_input_data[1960];
static int8_t microinfer_output_data[10];
static uint8_t static_buff[1024*20];

const char codeLib[] = "@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.   ";
void print_img(int8_t * buf)
{
    for(int y = 0; y < 28; y++) 
	{
        for (int x = 0; x < 28; x++) 
		{
            int index =  69 / 127.0 * (127 - buf[y*28+x]); 
			if(index > 69) index =69;
			if(index < 0) index = 0;
            printf("%c",codeLib[index]);
			printf("%c",codeLib[index]);
        }
        printf("\n");
    }
}

microinfer_model_t*model_init(void)
{
	printf("start\n");
	static microinfer_model_t model;
	microinfer_layer_t* layer[15];
	model_create(&model);
	
    layer[0] = Input(shape(28, 28, 1), microinfer_input_data);
	layer[1] = model.hook(Conv2D(12, kernel(3, 3), stride(1, 1), dilation(1,1) , PADDING_SAME, &conv2d_1_w, &conv2d_1_b), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(MaxPool(kernel(2, 2), stride(2, 2), PADDING_SAME), layer[2]);
	layer[4] = model.hook(Conv2D(24, kernel(3, 3), stride(1, 1), dilation(1,1) ,  PADDING_SAME, &conv2d_2_w, &conv2d_2_b), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(MaxPool(kernel(2, 2), stride(2, 2), PADDING_SAME), layer[5]);
	layer[7] = model.hook(Conv2D(48, kernel(3, 3), stride(1, 1), dilation(1,1) , PADDING_SAME, &conv2d_3_w, &conv2d_3_b), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(MaxPool(kernel(2, 2), stride(2, 2), PADDING_SAME), layer[8]);
	layer[10] = model.hook(Dense(96, &dense_1_w, &dense_1_b), layer[9]);
	layer[11] = model.active(act_relu(), layer[10]);
	layer[12] = model.hook(Dense(10, &dense_2_w, &dense_2_b), layer[11]);
	layer[13] = model.hook(Softmax(), layer[12]);
	layer[14] = model.hook(Output(shape(10,1,1), microinfer_output_data), layer[13]);
	model_compile(&model, layer[0], layer[14]);
    
    /*
    layer[0] = Input(shape(49, 40, 1), microinfer_input_data);
	layer[1] = model.hook(Conv2D(8, kernel(10, 8), stride(2, 2), dilation(1,1) , PADDING_SAME, &conv2d_1_w, &conv2d_1_b), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(Dense(4, &dense_1_w, &dense_1_b), layer[2]);
	layer[4] = model.hook(Softmax(), layer[3]);
	model_compile(&model, layer[0], layer[4]);
	*/
	return &model;
}

int main()
{
	uint32_t predic_label;
	float prob;
	int32_t index = 5;
    microinfer_set_buf(static_buff , sizeof(static_buff)/sizeof(uint8_t));
    model = model_init();
	
	print_img((int8_t*)&img[index][0]);
	memcpy(microinfer_input_data, (int8_t*)&img[index][0], 784);
	microinfer_predict(model, &predic_label, &prob);

	printf("Truth label: %d\n", label[index]);
	printf("Predicted label: %d\n", predic_label);
	printf("Probability: %d%%\n", (int)(prob*100));
	
    return 0;
}