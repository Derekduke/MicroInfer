#include <stdio.h>
#include <microinfer.h>
#include <microinfer_layers.h>

microinfer_model_t *model;
static int8_t microinfer_input_data[784];

microinfer_model_t* microinfer_model_create(void)
{
    printf("input layer\n");
	static microinfer_model_t model;
	microinfer_layer_t* layer[15];
	model_init(&model);    
    layer[0] = Input(shape(28, 28, 1), microinfer_input_data);
    /*
	layer[1] = model.hook(Conv2D(12, kernel(3, 3), stride(1, 1), PADDING_SAME, &conv2d_1_w, &conv2d_1_b), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(MaxPool(kernel(2, 2), stride(2, 2), PADDING_SAME), layer[2]);
	layer[4] = model.hook(Conv2D(24, kernel(3, 3), stride(1, 1), PADDING_SAME, &conv2d_2_w, &conv2d_2_b), layer[3]);
	layer[5] = model.active(act_relu(), layer[4]);
	layer[6] = model.hook(MaxPool(kernel(2, 2), stride(2, 2), PADDING_SAME), layer[5]);
	layer[7] = model.hook(Conv2D(48, kernel(3, 3), stride(1, 1), PADDING_SAME, &conv2d_3_w, &conv2d_3_b), layer[6]);
	layer[8] = model.active(act_relu(), layer[7]);
	layer[9] = model.hook(MaxPool(kernel(2, 2), stride(2, 2), PADDING_SAME), layer[8]);
	layer[10] = model.hook(Dense(96, &dense_1_w, &dense_1_b), layer[9]);
	layer[11] = model.active(act_relu(), layer[10]);
	layer[12] = model.hook(Dense(10, &dense_2_w, &dense_2_b), layer[11]);
	layer[13] = model.hook(Softmax(), layer[12]);
	layer[14] = model.hook(Output(shape(10,1,1), nnom_output_data), layer[13]);
	model_compile(&model, layer[0], layer[14]);
    */
	return &model;
}

int main()
{
    model = microinfer_model_create();
    //model_run(model);
    return 0;
}