/*
 * conv_ip.h
 *
 *  Created on: 2020-10-19謑�
 *      Author: wei-che
 */


#include "xaxidma.h"
#include "xparameters.h"
#include "xil_exception.h"
#include "xdebug.h"
/***************************** Include Files *********************************/

#include "xaxidma.h"
#include "xparameters.h"
#include "xil_exception.h"
#include "xdebug.h"
#include "xstatus.h"
#include "xscugic.h"
#include "conv_ip.h"
#include "axi_dma.h"
#include "sleep.h"

#include "read_data.h"

#include "xtime_l.h"

void TX_DATA(int size, u64* fmap){
	AXI_DMA_TxDone=0;
	AXI_DMA_RxDone=0;
	Xil_DCacheFlushRange((UINTPTR)(fmap), size*8);

	AXI_DMA_Transfer((UINTPTR)(fmap), size*8, XAXIDMA_DMA_TO_DEVICE);
	while (!AXI_DMA_TxDone) { }
	xil_printf("\n--- tx ifmap done ---");
}

void RX_DATA(int size, u64* fmap){
	AXI_DMA_TxDone=0;
	AXI_DMA_RxDone=0;
	Xil_DCacheFlushRange((UINTPTR)(fmap), size*8);

	AXI_DMA_Transfer((UINTPTR)(fmap), size*8, XAXIDMA_DEVICE_TO_DMA);
	while (!AXI_DMA_RxDone) { }
	xil_printf("\n--- rx ofmap done ---");
}

/***************************** Main Function *********************************/

void CONV_IP_0(int layer){

	XTime start, end;
	u32 time_used;
	XTime_GetTime(&start);

	int bias_size, weights_size, ifmap_size, ofmap_size, err;

//	u8 *gold_map;
//	gold_map	 = gold_conv_0;

	xil_printf("\nStart convolution...");
	TX_DATA(ALL_SIZE,(u64) &all0);

/*reserve if something go wrong //data_in
	AXI_DMA_TxDone=0;
	AXI_DMA_RxDone=0;
	ifmap 		 = all0;
	ifmap_size   = ALL_SIZE*8;
	Xil_DCacheFlushRange((UINTPTR)(ifmap), ifmap_size);
	AXI_DMA_Transfer((UINTPTR)(ifmap), ifmap_size, XAXIDMA_DMA_TO_DEVICE);
	while (!AXI_DMA_TxDone) { }
	xil_printf("\n--- input ifmap Done ---");
*/
/*
	TX_DATA(IFMAP_SIZE,(u64) &ifmap0);
	//TX_DATA(COE_SIZE,(u64) &all_coe);
	TX_DATA(M0EXPONENT_SIZE,(u64) &m0exponent0);
	TX_DATA(BIAS_SIZE_0,(u64) &bias_0);
	TX_DATA(WEIGHTS_SIZE_0,(u64) &weights_0);
	TX_DATA(BIAS_SIZE_1,(u64) &bias_1);
	TX_DATA(WEIGHTS_SIZE_1,(u64) &weights_1);
	TX_DATA(BIAS_SIZE_2,(u64) &bias_2);
	TX_DATA(WEIGHTS_SIZE_2,(u64) &weights_2);
	TX_DATA(FC_BIAS_0,(u64) &fc_bias_0);
	TX_DATA(FC_WEIGHTS_0,(u64) &fc_weights_0);
	TX_DATA(FC_BIAS_1,(u64) &fc_bias_1);
	TX_DATA(FC_WEIGHTS_1,(u64) &fc_weights_1);
*/
/*reserve if something go wrong //data_out
	AXI_DMA_TxDone=0;
	AXI_DMA_RxDone=0;
	ifmap 		 = data_out;
	ifmap_size   = OFMAP_SIZE*8;
	Xil_DCacheFlushRange((UINTPTR)(ifmap), ifmap_size);
	AXI_DMA_Transfer((UINTPTR)(ifmap), ifmap_size*8, XAXIDMA_DEVICE_TO_DMA);
	// Wait TX done and RX done
	while (!AXI_DMA_RxDone) { }
*/
	RX_DATA(OFMAP_SIZE,(u64) &data_out);

    xil_printf("\nConvolution done.", layer);

	XTime_GetTime(&end);
	time_used = ((end-start)*1000000)/(COUNTS_PER_SECOND);
	xil_printf("\n%d us\r\n", time_used);

/*Verify data
	err = 0;
	xil_printf("\n");
	for(int i=0;i<ofmap_size;i++){
		if((ofmap[i]- gold_map[i])>3 || (gold_map[i]- ofmap[i])>3){
			err++;
			xil_printf("err at %d , out = %d , gold = %d\n", i, ofmap[i], gold_map[i]);
		}
	}
	xil_printf("\ntotal pixel = %d , error pixel = %d", ofmap_size, err);
*/
}
