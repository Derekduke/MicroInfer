/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */
#ifndef __MICROINFER_LOCAL_H_
#define __MICROINFER_LOCAL_H_

#include "microinfer.h"

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

#define NNOM_ROUND(out_shift) ((0x1 << out_shift) >> 1 )

#ifndef __NNOM_SSAT
static inline int __NNOM_SSAT(int32_t value, int32_t bit) {
    int32_t min = -(1<<(bit-1));
    int32_t max = (1<<(bit-1)) - 1;
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}
#endif

#ifndef __NNOM_USAT
static inline int __NNOM_USAT(int32_t value, int32_t bit) {
    int32_t max = (1<<(bit-1)) - 1;
    if (value < 0)
        return 0;
    else if (value > max)
        return max;
    else
        return value;
}
#endif

void local_relu_q7(q7_t *data, uint32_t size);

void local_convolve_HWC_q7_nonsquare(const q7_t *Im_in,                // input image
	const uint16_t dim_im_in_x,                                        // input image dimention x
	const uint16_t dim_im_in_y,                                        // input image dimention y
	const uint16_t ch_im_in,                                           // number of input image channels
	const q7_t *wt,                                                    // kernel weights
	const uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
	const uint16_t dim_kernel_x,                                       // filter kernel size x
	const uint16_t dim_kernel_y,                                       // filter kernel size y
	const uint16_t padding_x,                                          // padding sizes x
	const uint16_t padding_y,                                          // padding sizes y
	const uint16_t stride_x,                                           // stride x
	const uint16_t stride_y,                                           // stride y
    const uint16_t dilation_x,                                         // dilation x
	const uint16_t dilation_y,                                         // dilation y
	const q7_t *bias,                                                  // bias
	const microinfer_qformat_param_t *bias_shift,                                        // bias shifts
    const microinfer_qformat_param_t *out_shift,                                         // output shift
    const microinfer_qtype_t q_type,                                         // per channel or per tensor
    q7_t *Im_out,                                                      // output image
	const uint16_t dim_im_out_x,                                       // output image dimension x
	const uint16_t dim_im_out_y,                                       // output image dimension y
	q15_t *bufferA,                                                    //buffer space for input
	q7_t *bufferB                                                      //buffer space for output
);

void local_maxpool_q7_HWC(const q7_t *Im_in,           // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	const uint16_t dim_kernel_x, // window kernel size
	const uint16_t dim_kernel_y, // window kernel size
	const uint16_t padding_x,    // padding sizes
	const uint16_t padding_y,    // padding sizes
	const uint16_t stride_x,     // stride
	const uint16_t stride_y,     // stride
	const uint16_t dim_im_out_x, // output image dimension x or W
	const uint16_t dim_im_out_y, // output image dimension y or H
	q7_t *bufferA,               // a buffer for local storage, NULL by now
	q7_t *Im_out);

void local_fully_connected_q7_opt(const q7_t *pV,               // pointer to vector
	const q7_t *pM,               // pointer to matrix
	const uint16_t dim_vec,       // length of the vector
	const uint16_t num_of_rows,   // numCol of A
	const uint16_t bias_shift,    // amount of left-shift for bias
	const uint16_t out_shift,     // amount of right-shift for output
	const q7_t *bias, q7_t *pOut, // output operand
	q15_t *vec_buffer);

void local_softmax_q7(const q7_t *vec_in, const uint32_t dim_vec, q7_t *p_out);
#endif