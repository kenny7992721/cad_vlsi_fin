/*
 * read_data.h
 *
 *  Created on: 2020撟�9���25�
 *      Author: weiche
 */

#ifndef SRC_READ_DATA_H_
#define SRC_READ_DATA_H_

#define IFMAP_SIZE	3072
#define OFMAP_SIZE	10

#define BIAS_SIZE_0	32
#define BIAS_SIZE_1	32
#define BIAS_SIZE_2	64

#define WEIGHTS_SIZE_0 864
#define WEIGHTS_SIZE_1 9216
#define WEIGHTS_SIZE_2 18432

#define FC_BIAS_0 64
#define FC_BIAS_1 10

#define FC_WEIGHTS_0 65536
#define FC_WEIGHTS_1 640

#define ALL_SIZE 97962

u64 *ifmap;
u64 all0[ALL_SIZE];
u64 ifmap0[IFMAP_SIZE];

u64 *bias;
u64 bias_0[BIAS_SIZE_0];
u64 bias_1[BIAS_SIZE_1];
u64 bias_2[BIAS_SIZE_2];

u64 *weights;
u64 weights_0[WEIGHTS_SIZE_0];
u64 weights_1[WEIGHTS_SIZE_1];
u64 weights_2[WEIGHTS_SIZE_2];

u64 fc_bias_0[FC_BIAS_0];
u64 fc_bias_1[FC_BIAS_1];

u64 fc_weights_0[FC_WEIGHTS_0];
u64 fc_weights_1[FC_WEIGHTS_1];

u64 *ofmap;
u64 data_out[OFMAP_SIZE];

int data_init();

#endif /* SRC_READ_DATA_H_ */
