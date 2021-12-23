/*
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-06     derekduke   The first version
 */
#include "microinfer.h"
#include "microinfer_local.h"

void local_relu_q7(q7_t *data, uint32_t size)
{
    uint32_t i;

    for (i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
    }
}

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
)
{
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;
    int in_pix_loc, wt_loc;
    int shift_idx, shift_steps;
    if(q_type == MICROINFER_QTYPE_PER_AXIS)
        shift_steps = 1;
    else
        shift_steps = 0;

    for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            int32_t base_idx_y = stride_y * j - padding_y;
            for (k = 0; k < dim_im_out_x; k++)
            {
				int32_t base_idx_x = stride_x * k - padding_x;
                int32_t ker_y_start = MAX(0, -(base_idx_y-(dilation_y-1))/dilation_y);
                int32_t ker_x_start = MAX(0, -(base_idx_x-(dilation_x-1))/dilation_x);
                int32_t ker_y_end = MIN(dim_kernel_y, (dim_im_in_y - base_idx_y + (dilation_y-1))/dilation_y);
                int32_t ker_x_end = MIN(dim_kernel_x, (dim_im_in_x - base_idx_x + (dilation_x-1))/dilation_x);

                if(bias)
                    conv_out = ((q31_t)(bias[i]) << bias_shift[shift_idx]) + NNOM_ROUND(out_shift[shift_idx]);
                else
                    conv_out = (q31_t) NNOM_ROUND(out_shift[shift_idx]);

                for (m = ker_y_start; m < ker_y_end; m++)
                {
                    for (n = ker_x_start; n < ker_x_end; n++)
                    {
                        in_row = stride_y * j + m * dilation_y - padding_y;
                        in_col = stride_x * k + n * dilation_x - padding_x;

                        // pre-calculate the pixel location and weight location to improve the performance.
                        in_pix_loc = (in_row * dim_im_in_x + in_col) * ch_im_in;
                        wt_loc = i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in;
                        
                        for (l = 0; l < ch_im_in; l++)
                        {    
                            conv_out += Im_in[in_pix_loc + l] * wt[wt_loc + l];
                        } 
                    }
                }
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
            }
        }
    }
}

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
	q7_t *Im_out)
{
    int16_t i_ch_in, i_x, i_y;
    int16_t k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int max = -129;
                for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
                {
                    for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = max;
            }
        }
    }
}

void local_fully_connected_q7_opt(const q7_t *pV,               // pointer to vector
	const q7_t *pM,               // pointer to matrix
	const uint16_t dim_vec,       // length of the vector
	const uint16_t num_of_rows,   // numCol of A
	const uint16_t bias_shift,    // amount of left-shift for bias
	const uint16_t out_shift,     // amount of right-shift for output
	const q7_t *bias, q7_t *pOut, // output operand
	q15_t *vec_buffer)
{
    uint16_t rowCnt = num_of_rows >> 2;
    const q7_t *pB = pM;
    const q7_t *pA;
    q7_t *pO = pOut;
    const q7_t *pBias = bias;

    while (rowCnt)
    {
        pA = pV;
        q31_t     sum;
        q31_t     sum2;
        q31_t     sum3;
        q31_t     sum4;
        uint16_t colCnt = dim_vec >> 2;

        if(bias)
        {
            sum =  ((q31_t)(*pBias++) << bias_shift) + NNOM_ROUND(out_shift);
            sum2 = ((q31_t)(*pBias++) << bias_shift) + NNOM_ROUND(out_shift);
            sum3 = ((q31_t)(*pBias++) << bias_shift) + NNOM_ROUND(out_shift);
            sum4 = ((q31_t)(*pBias++) << bias_shift) + NNOM_ROUND(out_shift);
        }
        else
        {
            sum =  (q31_t) NNOM_ROUND(out_shift);
            sum2 = (q31_t) NNOM_ROUND(out_shift);
            sum3 = (q31_t) NNOM_ROUND(out_shift);
            sum4 = (q31_t) NNOM_ROUND(out_shift);
        }

        while (colCnt)
        {
            q7_t inA1 = *pA++;
            q7_t inA3 = *pA++;
            q7_t inA2 = *pA++;
            q7_t inA4 = *pA++;

            q7_t inB1 = *pB++;
            q7_t inB3 = *pB++;
            q7_t inB2 = *pB++;
            q7_t inB4 = *pB++;

            sum += inA1 * inB1 + inA2 * inB2;
            sum2 += inA1 * inB3 + inA2 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum3 += inA1 * inB1 + inA2 * inB2;
            sum4 += inA1 * inB3 + inA2 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum += inA3 * inB1 + inA4 * inB2;
            sum2 += inA3 * inB3 + inA4 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum3 += inA3 * inB1 + inA4 * inB2;
            sum4 += inA3 * inB3 + inA4 * inB4;

            colCnt--;
        }
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q7_t inA = *pA++;
            q7_t inB = *pB++;
            sum += inA * inB;
            inB = *pB++;
            sum2 += inA * inB;
            inB = *pB++;
            sum3 += inA * inB;
            inB = *pB++;
            sum4 += inA * inB;

            colCnt--;
        }
        *pO++ = (q7_t)__NNOM_SSAT((sum >> out_shift), 8);
        *pO++ = (q7_t)__NNOM_SSAT((sum2 >> out_shift), 8);
        *pO++ = (q7_t)__NNOM_SSAT((sum3 >> out_shift), 8);
        *pO++ = (q7_t)__NNOM_SSAT((sum4 >> out_shift), 8);

        rowCnt--;
    }

    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
		int ip_out;
        if(bias)
            ip_out=((q31_t)(*bias++) << bias_shift) + NNOM_ROUND(out_shift);
        else
            ip_out=(q31_t)NNOM_ROUND(out_shift);
        
        pA = pV;
        for (int j = 0; j < dim_vec; j++)
        {
            q7_t inA = *pA++;
            q7_t inB = *pB++;
            ip_out += inA * inB;
        }
        *pO++ = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);

        rowCnt--;
    }
}

void local_softmax_q7(const q7_t *vec_in, const uint32_t dim_vec, q7_t *p_out)
{
    q31_t sum;
    int32_t i;
    uint8_t shift;
    q15_t base;
    base = -257;
    /* We first search for the maximum */
    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }

    /* 
     * So the base is set to max-8, meaning 
     * that we ignore really small values. 
     * anyway, they will be 0 after shrinking to q7_t.
     */
    base = base - 8;

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            shift = (uint8_t)__NNOM_USAT(vec_in[i] - base, 5);
            sum += 0x1 << shift;
        }
    }

    /* This is effectively (0x1 << 20) / sum */
    int output_base = 0x100000 / sum;

    /* 
     * Final confidence will be output_base >> ( 13 - (vec_in[i] - base) )
     * so 128 (0x1<<7) -> 100% confidence when sum = 0x1 << 8, output_base = 0x1 << 12 
     * and vec_in[i]-base = 8
     */
    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            /* Here minimum value of 13+base-vec_in[i] will be 5 */
            shift = (uint8_t)__NNOM_USAT(13 + base - vec_in[i], 5);
            p_out[i] = (q7_t)__NNOM_SSAT((output_base >> shift), 8);
        }
        else
        {
            p_out[i] = 0;
        }
    }
}

