/*
 * read_data.c
 *
 *  Created on: 2020��蕭9嚙踝蕭嚙�25嚙踐��
 *      Author: weiche
 */

#include "ff.h"
#include "xaxidma.h"
#include "xparameters.h"
#include "xil_exception.h"
#include "xdebug.h"
#include "read_data.h"
#include "math.h"

static FIL fil_bias;
static FATFS fatfs;

//file names
//const static char file_input[32]        = "all4.dat";
//const static char file_input[32]        = "all4.dat";
//const static char file_input[32]        = "all8.dat";
//const static char file_input[32]        = "all9.dat";
const static char file_input[32]        = "all10.dat";
/*
const static char file_input[32]        = "all_n1.dat";
const static char file_input[32]        = "all_n2.dat";
const static char file_input[32]        = "all_n3.dat";
const static char file_input[32]        = "all_n4.dat";
const static char file_input[32]        = "all_n5.dat";
const static char file_coe[32]        	= "all_coe.dat";
const static char file_ifmap[32]        = "ifmap.dat";
const static char file_ifmap[32]        = "layer0_INPUT_MAP_1.dat";
const static char file_ifmap[32]        = "layer0_INPUT_MAP_2.dat";
const static char file_ifmap[32]        = "layer0_INPUT_MAP_3.dat";
const static char file_ifmap[32]        = "layer0_INPUT_MAP_4.dat";
const static char file_ifmap[32]        = "layer0_INPUT_MAP_5.dat";
const static char file_m0exponent[32]   = "m0exponent.dat";
const static char file_conv_bias_0[32] 	= "bias1.dat";
const static char file_conv_bias_1[32] 	= "bias2.dat";
const static char file_conv_bias_2[32] 	= "bias3.dat";
const static char file_conv_0[32]		= "weight1.dat";
const static char file_conv_1[32]		= "weight2.dat";
const static char file_conv_2[32]		= "weight3.dat";
const static char file_fc_bias_0[32]	= "fully_b1.dat";
const static char file_fc_bias_1[32]	= "fully_b2.dat";
const static char file_fc_0[32]			= "fully1.dat";
const static char file_fc_1[32]			= "fully2.dat";
*/

//convert function
long str_to_long(char *temp2){
	long cal = 0, pow = 1, digit = 0;
		for(int j=0; j<8; j++){
			if(temp2[7-j] >= '0' && temp2[7-j] <= '9')			//0~9
				digit = temp2[7-j] - '0';
			else												//a~f
				digit = temp2[7-j] - 'a' + 10;
			cal += digit * pow;
			pow = pow *16;
		}
	return cal;
}

//read data function
int READ_u64(int bias_size,char** file_name, u64* out){
	FRESULT Res;
	UINT NumBytesRead;

	u8  temp[18];
	char  temp1[9];
	char  temp2[9];
	long temp3;
	long temp4;
	u64 temp5;
	u64 temp6;

	Res = f_open(&fil_bias,file_name, FA_READ);
	if(Res){
		xil_printf("-- Failed at stage 2 --\r\n");
		return XST_FAILURE;
	}

	// Set pointer to beginning of file.
	Res = f_lseek(&fil_bias, 0);
	if(Res){
		xil_printf("-- Failed at stage 3 --\r\n");
		return XST_FAILURE;
	}

	// Read data from file.
	for(int i=0; i<bias_size; i++){
		Res = f_read(&fil_bias, (void*)temp, 16,  &NumBytesRead);
		if(Res){
			xil_printf("-- Failed at stage 4 --\r\n");
			return XST_FAILURE;
		}
		temp1[8] = '\0';
		temp2[8] = '\0';

		for(int j=0; j<8; j++){
			temp1[j] = temp[j];
		}
		for(int j=0; j<8; j++){
			temp2[j] = temp[j+8];
		}

		temp3=str_to_long(temp1);
		temp4=str_to_long(temp2);
		temp5 = temp3 & 0x00000000FFFFFFFF;
		temp6 = temp4 & 0x00000000FFFFFFFF;
		out[i] = (temp5<<32) + temp6;
	}

	// Close file.
	Res = f_close(&fil_bias);
	if(Res){
		xil_printf("-- Failed at stage 5 --\r\n");
		return XST_FAILURE;
	}
	
	xil_printf("--- read data done --\r\n");
}

int data_read(){
	xil_printf("---------------- SD Card Reading ...... ----------------\r\n");

	FRESULT Res;
	UINT NumBytesRead;
	TCHAR *Path = "0:/";

	Res = f_mount(&fatfs, Path, 0);
	if(Res != FR_OK){
		xil_printf("-- Failed at stage 1 --\r\n");
		return XST_FAILURE;
	}

// Read input data
	Res = READ_u64(ALL_SIZE, &file_input, (u64) &all0);

    for (int i=0; i <OFMAP_SIZE; i++){
    	data_out[i]=0;
    }

	xil_printf("---------------- SD Card Reading Done.  ----------------\r\n");
	return XST_SUCCESS;
}
