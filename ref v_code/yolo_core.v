// ============================================================================
// Copyright (C) 2019 NARLabs TSRI. All rights reserved.
//
// Designer : Liu Yi-Jun
// Date     : 2019.10.31
// Ver      : 1.2
// Module   : yolo_core
// Func     : 
//            1.) Bypass  
//            2.) adder: incoming every two operand, output one result
//
//
// ============================================================================

`timescale 1 ns / 1 ps

module yolo_core #(
        parameter TBITS = 64 ,
        parameter TBYTE = 4
) (
        input  wire [TBITS-1:0] isif_data_dout ,  // {last,user,strb,data}
        input  wire [TBYTE-1:0] isif_strb_dout ,
        input  wire [1 - 1:0]   isif_last_dout ,  // eol
        input  wire [1 - 1:0]   isif_user_dout ,  // sof
        input  wire             isif_empty_n ,
        output wire             isif_read ,
        //
        output wire [TBITS-1:0] osif_data_din ,
        output wire [TBYTE-1:0] osif_strb_din ,
        output wire [1 - 1:0]   osif_last_din ,
        output wire [1 - 1:0]   osif_user_din ,
        input  wire             osif_full_n ,
        output wire             osif_write ,
        //
        input  wire             rst ,
        input  wire             clk
);  // filter_core

// ============================================================================
// Parameter
// ============================================================================
// local signal
//
// ISIF
// OSIF
// ============================================================================
// Body
`define data_size 32    //image data=32bits

reg read, write, last;
reg [63:0] op_data;
wire empty_n;

assign isif_read = read;
assign empty_n = isif_empty_n;
assign osif_data_din = op_data;
assign osif_strb_din = 'hff;
//assign osif_last_din = isif_last_dout ; //not sure why
assign osif_last_din = last ; //not sure why last year version
assign osif_user_din = 0 ; //no sure why
assign osif_write = write;

//state
parameter IDLE 					= 5'd0;
parameter OP_YC_READ_PICTURE	= 5'd1;
parameter OP_YC_READ_BIAS_1		= 5'd2;
parameter OP_YC_READ_WEIGHT_1	= 5'd3;
parameter OP_YC_CONV_1			= 5'd4;
parameter OP_YC_CM1				= 5'd22; //empty state
parameter OP_YC_MAX_1			= 5'd5;
//CONV2
parameter OP_YC_READ_BIAS_2		= 5'd6;
parameter OP_YC_READ_WEIGHT_2	= 5'd7;
parameter OP_YC_CONV_2			= 5'd8;
parameter OP_YC_CM2				= 5'd23; //empty state
parameter OP_YC_MAX_2			= 5'd9;
//CONV3
parameter OP_YC_READ_BIAS_3		= 5'd10;
parameter OP_YC_READ_WEIGHT_3	= 5'd11;
parameter OP_YC_CONV_3			= 5'd12;
parameter OP_YC_CM3				= 5'd24; //empty state 
parameter OP_YC_MAX_3			= 5'd13;
//DENSE1
parameter OP_YC_READ_DENSE_BIAS_1   = 5'd14;
parameter OP_YC_READ_DENSE_WEIGHT_1 = 5'd15;
parameter OP_YC_DENSE_1			    = 5'd16;
//DENSE2
parameter OP_YC_READ_DENSE_BIAS_2   = 5'd17;
parameter OP_YC_READ_DENSE_WEIGHT_2 = 5'd18;
parameter OP_YC_DENSE_2			    = 5'd19;

parameter OP_YC_WB      		= 5'd20;
parameter DONE 					= 5'd21;

//read_cnt	
reg [4:0] cs,ns;							//21states 2^5=32	
reg [11:0]	read_cnt1;	//1:picture 3070 2:bias 3:weight
reg [11:0]	read_cnt2;
reg [11:0]	read_cnt3;
reg	[10:0]	ker_cnt;						//count pixels of feature map 1:32*32=1024 2:16*16=256 3:8*8=64 
reg [6:0]	kernel_shift_z;					//count feature map nums 1:32 2:32 3:64 
reg [6:0] 	mp_kernel_shift_z;				//count feature map nums 1:32 2:32 3:64 
wire [`data_size-1:0] dout_a2;				//memory2 out
reg [6:0] dense_cnt;						//count dense outputs nums 1:64
reg [10:0] dense_pcnt,dense_pcnt_2;						//count dense pixels 1:1024
wire [`data_size-1:0] dout_a1,dout_a3,dout_a4;				//memory1 out 32bits
wire	[14:0]	mp_position_1,mp_position_2,mp_position_3;	//memory開32768 2^15=32768 cause CONV1 feature map32*32*32=32768pixels
reg [8:0] kernel_position_del_1;
reg [8:0] kernel_position_del_2;
reg [6:0] kernel_shift_z_del_1,kernel_shift_z_del_2;
reg [13:0]	max_cnt;								//?????count pixels of feature map
reg [10:0] padding_position_dim1_c1_del_1,padding_position_dim1_c1_del_2,padding_position_dim1_c1_del_3;
reg [8:0] padding_position_dim1_c2_del_1,padding_position_dim1_c2_del_2,padding_position_dim1_c2_del_3;
reg [8:0] padding_position_dim1_c3_del_1,padding_position_dim1_c3_del_2,padding_position_dim1_c3_del_3;
reg [6:0] kernel_shift_z_del_3,kernel_shift_z_del_4,kernel_shift_z_del_5;
reg [10:0] dense_pcnt_del_1, dense_pcnt_del_2, dense_pcnt_del_3;
reg [6:0] dense_cnt_del_1,dense_cnt_del_2,dense_cnt_del_3,dense_cnt_del_4;
reg [3:0] wb_cnt,wb_cnt_del_1,wb_cnt_del_2,wb_cnt_del_3;

//state
always @(posedge clk or posedge rst)begin
    if(rst)begin
        cs <= IDLE;
    end 
	else begin
        cs <= ns;
    end
end

//FSM
always @(posedge clk)begin
    case(cs)
		IDLE           		:  ns = (empty_n)			?	OP_YC_READ_PICTURE  : 	IDLE;
        OP_YC_READ_PICTURE	:  ns = (read_cnt1>=3070)	?   OP_YC_READ_BIAS_1 	:	OP_YC_READ_PICTURE;		//image=32*32*3=3072pixels
        OP_YC_READ_BIAS_1   :  ns = (read_cnt2>=30)		? 	OP_YC_READ_WEIGHT_1 :	OP_YC_READ_BIAS_1;		//bias 1:32 2:32 3:64
		OP_YC_READ_WEIGHT_1 :  ns = (read_cnt3>=25)		?	OP_YC_CONV_1		:	OP_YC_READ_WEIGHT_1;	//load weight 27+0=28nums
		OP_YC_CONV_1	   	:  begin	if(kernel_shift_z_del_5>31)begin			//finish CONV1(32 feature maps)
											ns <= OP_YC_CM1;
										end
										else if(ker_cnt>1024)begin					//finish one feature map(32*32=1024pixels)
											ns <= OP_YC_READ_WEIGHT_1;		
										end
										else begin 
											ns <= OP_YC_CONV_1;
										end
								end
		OP_YC_CM1			:  ns = OP_YC_MAX_1;
		OP_YC_MAX_1	   		:  ns = (max_cnt>8192)			?	OP_YC_READ_BIAS_2	:	OP_YC_MAX_1;		//finish MAX1(32feature maps)16*16*32=8192
		OP_YC_READ_BIAS_2	:  ns = (read_cnt2>=30)			?	OP_YC_READ_WEIGHT_2	:	OP_YC_READ_BIAS_2;	//bias 1:32 2:32 3:64
		OP_YC_READ_WEIGHT_2	:  ns = (read_cnt3>=286)		?	OP_YC_CONV_2		:	OP_YC_READ_WEIGHT_2;//3*3*32=288
		OP_YC_CONV_2	   	:  begin	if(kernel_shift_z_del_5>31)begin			//finish CONV2(32 feature maps)
											ns <= OP_YC_CM2;
										end
										else if(ker_cnt>256)begin					//finish one feature map(16*16=256pixels)
											ns <= OP_YC_READ_WEIGHT_2;		
										end
										else begin 
											ns <= OP_YC_CONV_2;
										end
								end
		OP_YC_CM2			:  ns = OP_YC_MAX_2;
		OP_YC_MAX_2	   		:  ns = (max_cnt>2048)			?	OP_YC_READ_BIAS_3	:	OP_YC_MAX_2;		//finish MAX2(32feature maps)8*8*32=2048
		OP_YC_READ_BIAS_3	:  ns = (read_cnt2>=62)			?	OP_YC_READ_WEIGHT_3	:	OP_YC_READ_BIAS_3;	//bias 1:32 2:32 3:64
		OP_YC_READ_WEIGHT_3	:  ns = (read_cnt3>=286)		?	OP_YC_CONV_3		:	OP_YC_READ_WEIGHT_3;//3*3*32=288
		OP_YC_CONV_3	   	:  begin	if(kernel_shift_z_del_5>63)begin			//finish CONV3(64 feature maps)
											ns <= OP_YC_CM3;
										end
										else if(ker_cnt>64)begin					//finish one feature map(8*8=64pixels)
											ns <= OP_YC_READ_WEIGHT_3;		
										end
										else begin 
											ns <= OP_YC_CONV_3;
										end
								end	
		OP_YC_CM3			:  ns = OP_YC_MAX_3;
		OP_YC_MAX_3	   		:  ns = (max_cnt>1024)			?	OP_YC_READ_DENSE_BIAS_1		:	OP_YC_MAX_3;	//finish MAX3(64feature maps)4*4*64=1024	
		OP_YC_READ_DENSE_BIAS_1	  : ns = (read_cnt2>=62) 	?	OP_YC_READ_DENSE_WEIGHT_1	:	OP_YC_READ_DENSE_BIAS_1;	//bias:32
		OP_YC_READ_DENSE_WEIGHT_1 : ns = (read_cnt3>=1022) 	?	OP_YC_DENSE_1				:	OP_YC_READ_DENSE_WEIGHT_1;
		OP_YC_DENSE_1			  : begin 	if(dense_cnt_del_4>63)begin		//finish DENSE1(64 outputs)
												ns <= OP_YC_READ_DENSE_BIAS_2;
											end
											else if(dense_pcnt_2>1028)begin				//finish one 
												ns <= OP_YC_READ_DENSE_WEIGHT_1;		
											end
											else begin 
												ns <= OP_YC_DENSE_1;
											end
									end
		OP_YC_READ_DENSE_BIAS_2	  : ns = (read_cnt2>=8)  ? OP_YC_READ_DENSE_WEIGHT_2	: OP_YC_READ_DENSE_BIAS_2;	//bias:10
		OP_YC_READ_DENSE_WEIGHT_2 : ns = (read_cnt3>=62) ? OP_YC_DENSE_2				: OP_YC_READ_DENSE_WEIGHT_2;
		OP_YC_DENSE_2			  : begin	if(dense_cnt_del_4>9)begin			//finish DENSE2(10 outputs)
												ns <= OP_YC_WB;
											end 
											else if(dense_pcnt_2>68)begin				//finish one
												ns <= OP_YC_READ_DENSE_WEIGHT_2;		
											end
											else begin 
												ns <= OP_YC_DENSE_2;
											end
									end
		OP_YC_WB       		:  ns = (wb_cnt_del_3>=9)		? 	DONE				:	OP_YC_WB;  //Write		
		DONE          		:  ns = DONE;
        default        		:  ns = IDLE;
    endcase
end

//read
always @(posedge clk or posedge rst)begin
    if(rst)begin
        read <= 0;
    end 
	else begin
		if((cs==OP_YC_READ_PICTURE) || ((cs==OP_YC_READ_BIAS_1)&&read_cnt2<=30) || ((cs==OP_YC_READ_BIAS_2)&&read_cnt2<=30) || ((cs==OP_YC_READ_BIAS_3)&&read_cnt2<=62) || 
		  ((cs==OP_YC_READ_WEIGHT_1)&&read_cnt3<=25) ||((cs==OP_YC_READ_WEIGHT_2)&&read_cnt3<=286) || ((cs==OP_YC_READ_WEIGHT_3)&&read_cnt3<=286) || 
		   (cs==OP_YC_READ_DENSE_BIAS_1) || (cs==OP_YC_READ_DENSE_BIAS_2) || ((cs==OP_YC_READ_DENSE_WEIGHT_1)&&read_cnt3<=1022) || ((cs==OP_YC_READ_DENSE_WEIGHT_2)&&read_cnt3<=62))begin				
			read <= 1;		//load data
        end 
		else begin
            read <= 0;
        end
    end
end

//read_cnt1 for read image 0~3070
always@(posedge clk or posedge rst) begin
    if(rst) begin
        read_cnt1 <= 0;
    end
	else begin
        if(cs==OP_YC_READ_PICTURE && read) begin
			read_cnt1 <= read_cnt1 + 1;		//+2cause one clk two pixels
        end
		else begin
            read_cnt1 <= 0;
        end
    end
end

//read_cnt2 for read conv_bias and dense_bias
always@(posedge clk or posedge rst) begin
    if(rst) begin
        read_cnt2 <= 0;
    end
	else begin
        if((cs==OP_YC_READ_BIAS_1 || cs==OP_YC_READ_BIAS_2 ||cs==OP_YC_READ_BIAS_3 || cs==OP_YC_READ_DENSE_BIAS_1 || cs==OP_YC_READ_DENSE_BIAS_2) && read) begin
			read_cnt2 <= read_cnt2 + 1;		//+2cause one clk two bias data
        end
		else begin
            read_cnt2 <= 0;
        end
    end
end

//read_cnt3 for read weight
always@(posedge clk or posedge rst) begin
    if(rst) begin
        read_cnt3 <= 0;
    end
	else begin
        if((cs==OP_YC_READ_WEIGHT_1 || cs==OP_YC_READ_WEIGHT_2 ||cs==OP_YC_READ_WEIGHT_3 || cs == OP_YC_READ_DENSE_WEIGHT_1 || cs == OP_YC_READ_DENSE_WEIGHT_2) && read) begin
			read_cnt3 <= read_cnt3 + 1;		//+2cause one clk two weight data
        end
		else begin
            read_cnt3 <= 0;
        end
    end
end

//reg bits 
reg	[1:0]	kernel_x,kernel_y;				//0~2 //12bits cause 2^12=4096>padding_position34*34*3=3468
reg [5:0]	kernel_z;						//1:0~2 2:0~31 3:0~31	
reg	[5:0]	kernel_shift_x,kernel_shift_y;	//1:0~31 2:0~15 3:0~7

//kernel_shift		
always@(posedge clk or posedge rst)begin
	if(rst)begin		
		kernel_shift_x <= 0;
		kernel_shift_y <= 0;
		kernel_shift_z <= 0;		//count feature map	nums	
		kernel_x <= 0;
		kernel_y <= 0;
		kernel_z <= 0;
		ker_cnt  <= 0;				//count pixels of feature map ker_cnt>32*32=1024 
	end
	else begin
		if(cs==OP_YC_CONV_1)begin	//CONV1
			if(kernel_shift_x==31&&kernel_shift_y==31&&kernel_x==2&&kernel_y==2&&kernel_z==2)begin
				kernel_shift_z <= kernel_shift_z + 1;	//finish one kernel weight and switch another bias							
				kernel_shift_y <= 0;
				kernel_shift_x <= 0;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map
			end
			if(kernel_shift_x==31&&kernel_x==2&&kernel_y==2&&kernel_z==2)begin
				kernel_shift_y <= kernel_shift_y + 1;
				kernel_shift_x <= 0;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map
			end
			else if(kernel_x==2&&kernel_y==2&&kernel_z==2)begin
				kernel_shift_x <= kernel_shift_x+1;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map
			end
			else if(kernel_x==2&&kernel_y==2)begin
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= kernel_z + 1;				
			end
			else if(kernel_x==2)begin
				kernel_y <= kernel_y + 1;
				kernel_x <= 0;
			end
			else begin
				kernel_x <= kernel_x + 1;
			end
		end
		else if(cs==OP_YC_CONV_2)begin	//CONV2
			if(kernel_shift_x==15&&kernel_shift_y==15&&kernel_x==2&&kernel_y==2&&kernel_z==31)begin
				kernel_shift_z <= kernel_shift_z + 1;	//finish one kernel weight and switch another bias								
				kernel_shift_y <= 0;
				kernel_shift_x <= 0;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map	
			end
			if(kernel_shift_x==15&&kernel_x==2&&kernel_y==2&&kernel_z==31)begin
				kernel_shift_y <= kernel_shift_y + 1;
				kernel_shift_x <= 0;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map
			end
			else if(kernel_x==2&&kernel_y==2&&kernel_z==31)begin
				kernel_shift_x <= kernel_shift_x+1;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map
			end
			else if(kernel_x==2&&kernel_y==2)begin
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= kernel_z + 1;				
			end
			else if(kernel_x==2)begin
				kernel_y <= kernel_y + 1;
				kernel_x <= 0;
			end
			else begin
				kernel_x <= kernel_x + 1;
			end
		end
		else if(cs==OP_YC_CONV_3)begin	//CONV3
			if(kernel_shift_x==7&&kernel_shift_y==7&&kernel_x==2&&kernel_y==2&&kernel_z==31)begin
				kernel_shift_z <= kernel_shift_z + 1;	//finish one kernel weight and switch another bias								
				kernel_shift_y <= 0;
				kernel_shift_x <= 0;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map	
			end
			if(kernel_shift_x==7&&kernel_x==2&&kernel_y==2&&kernel_z==31)begin
				kernel_shift_y <= kernel_shift_y + 1;
				kernel_shift_x <= 0;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map
			end
			else if(kernel_x==2&&kernel_y==2&&kernel_z==31)begin
				kernel_shift_x <= kernel_shift_x+1;
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= 0;
				ker_cnt	 <= ker_cnt +1;					//get one pixel of feature map
			end
			else if(kernel_x==2&&kernel_y==2)begin
				kernel_x <= 0;
				kernel_y <= 0;
				kernel_z <= kernel_z + 1;				
			end
			else if(kernel_x==2)begin
				kernel_y <= kernel_y + 1;
				kernel_x <= 0;
			end
			else begin
				kernel_x <= kernel_x + 1;
			end
		end
		else if(cs==OP_YC_READ_WEIGHT_1 || cs==OP_YC_READ_WEIGHT_2 || cs==OP_YC_READ_WEIGHT_3)begin
			kernel_shift_z <= kernel_shift_z;	//avoid become 0 for jumping back to forward state((load weight)->CONV->load weight->(CONV))
			kernel_shift_y <= 0;
			kernel_shift_x <= 0;
			kernel_x <= 0;
			kernel_y <= 0;
			kernel_z <= 0;
			ker_cnt	 <= 0;					
		end
		else begin
			kernel_shift_x <= 0;
			kernel_shift_y <= 0;
			kernel_shift_z <= 0;
			kernel_x <= 0;
			kernel_y <= 0;
			kernel_z <= 0;
			ker_cnt  <= 0;
		end
	end
end

//wire
`define pad_pos 14												//2^14=16384 cause > 2:18*18*32=10368
wire	[8:0]	kernel_position;								//1:0~26 2:0~287 3:0~287
assign 	kernel_position = kernel_x + 3*kernel_y	+ 9*kernel_z;	//1:0~26 2:0~287 3:0~287

//CONV_1
wire	[`pad_pos-1:0]	padding_position_c1;					//0~3467 34*34*3=3468
wire	[10:0]	padding_position_dim1_c1;						//0~1155 11bits cause 2^11=2048 > 34*34=1156
assign	padding_position_c1 = kernel_x + kernel_shift_x + 34*(kernel_y + kernel_shift_y) + 1156*kernel_z;
assign  padding_position_dim1_c1 = padding_position_c1%1156;	//convert to 1dimension0~1155 34*34=1156

//CONV_2
wire	[`pad_pos-1:0]	padding_position_c2;					//0~10367 18*18*32=10368
wire	[8:0]	padding_position_dim1_c2;						//0~323 18*18=324
assign	padding_position_c2 = kernel_x + kernel_shift_x + 18*(kernel_y + kernel_shift_y) + 324*kernel_z;
assign  padding_position_dim1_c2 = padding_position_c2%324;		//convert to 1dimension0~323 18*18=324

//CONV_3
wire	[`pad_pos-1:0]	padding_position_c3;					//0~3199 10*10*32=3200
wire	[6:0]	padding_position_dim1_c3;						//0~99 8*8=64
assign	padding_position_c3 = kernel_x + kernel_shift_x + 10*(kernel_y + kernel_shift_y) + 100*kernel_z;
assign  padding_position_dim1_c3 = padding_position_c3%100;		//convert to 1dimension0~99 10*10=100

//iaddr_conv1		
wire	[10:0]	iaddr_c1,iaddr_c2,iaddr_c3;
wire	[10:0]	input_position_1_c1;
wire	[9:0]	input_position_2_c1;
assign input_position_1_c1 = padding_position_dim1_c1 - 33 - 2*(kernel_shift_y+kernel_y);
assign input_position_2_c1 = {input_position_1_c1[9:0]};		
assign iaddr_c1 = ((padding_position_dim1_c1>=0&&padding_position_dim1_c1<=33) || (padding_position_dim1_c1>=1122&&padding_position_dim1_c1<=1155) || (padding_position_dim1_c1%34)==0 || (padding_position_dim1_c1&34)==33) ? iaddr_c1 : input_position_2_c1;

//iaddr_conv2
wire	[10:0]	input_position_1_c2;
wire	[9:0]	input_position_2_c2;
assign input_position_1_c2 = padding_position_dim1_c2 - 17 - 2*(kernel_shift_y+kernel_y);
assign input_position_2_c2 = {input_position_1_c2[7:0]};
assign iaddr_c2 = ((padding_position_dim1_c2>=0&&padding_position_dim1_c2<=17) || (padding_position_dim1_c2>=306&&padding_position_dim1_c2<=323) || (padding_position_dim1_c2%18)==0 || (padding_position_dim1_c2&18)==17) ? iaddr_c2 : input_position_2_c2;

//iaddr_conv3
wire	[10:0]	input_position_1_c3;
wire	[9:0]	input_position_2_c3;
assign input_position_1_c3 = padding_position_dim1_c3 - 9 - 2*(kernel_shift_y+kernel_y);
assign input_position_2_c3 = {input_position_1_c3[5:0]};
assign iaddr_c3 = ((padding_position_dim1_c3>=0&&padding_position_dim1_c3<=9) || (padding_position_dim1_c3>=90&&padding_position_dim1_c3<=99) || (padding_position_dim1_c3%10)==0 || (padding_position_dim1_c3&10)==9) ? iaddr_c3 : input_position_2_c3;

//zero_padding
reg [`data_size-1:0] padding_picture;			//image size=32bits

always@(posedge clk or posedge rst)begin
	if(rst)begin
		padding_picture <= 0;
	end
	else begin
		if(cs==OP_YC_CONV_1)begin
			if((padding_position_dim1_c1_del_3>=0&&padding_position_dim1_c1_del_3<=33) || (padding_position_dim1_c1_del_3>=1122&&padding_position_dim1_c1_del_3<=1155) || 
			   (padding_position_dim1_c1_del_3%34)==0 || (padding_position_dim1_c1_del_3%34)==33)begin
				padding_picture <= 0;
			end
			else begin
				padding_picture <= dout_a2;	
			end
		end
		else if(cs==OP_YC_CONV_2)begin
			if((padding_position_dim1_c2_del_3>=0&&padding_position_dim1_c2_del_3<=17) || (padding_position_dim1_c2_del_3>=306&&padding_position_dim1_c2_del_3<=323) || 
			   (padding_position_dim1_c2_del_3%18)==0 || (padding_position_dim1_c2_del_3%18)==17)begin
				padding_picture <= 0;
			end
			else begin
				padding_picture <= dout_a2;
			end
		end
		else if(cs==OP_YC_CONV_3)begin
			if((padding_position_dim1_c3_del_3>=0&&padding_position_dim1_c3_del_3<=9) || (padding_position_dim1_c3_del_3>=90&&padding_position_dim1_c3_del_3<=99) || 
			   (padding_position_dim1_c3_del_3%10)==0 || (padding_position_dim1_c3_del_3%10)==9)begin
				padding_picture <= 0;
			end
			else begin
				padding_picture <= dout_a2;
			end
		end
		else begin
			padding_picture <= 0;
		end
	end
end

//kernel_position_del 
//delay
reg [8:0] kernel_position_del_3, kernel_position_del_4;

always@(posedge clk or posedge rst)begin
	if(rst)begin
		kernel_position_del_1 <= 0;
		kernel_position_del_2 <= 0;
		kernel_position_del_3 <= 0;
		kernel_position_del_4 <= 0;
		
		kernel_shift_z_del_1 <= 0;
        kernel_shift_z_del_2 <= 0;
        kernel_shift_z_del_3 <= 0;
        kernel_shift_z_del_4 <= 0;
        kernel_shift_z_del_5 <= 0;
        
		padding_position_dim1_c1_del_1 <= 0;
        padding_position_dim1_c1_del_2 <= 0;
        padding_position_dim1_c1_del_3 <= 0;
		
        padding_position_dim1_c2_del_1 <= 0;
        padding_position_dim1_c2_del_2 <= 0;
        padding_position_dim1_c2_del_3 <= 0;
		
        padding_position_dim1_c3_del_1 <= 0;
        padding_position_dim1_c3_del_2 <= 0;
        padding_position_dim1_c3_del_3 <= 0;
        
        dense_pcnt_del_1 <= 0;
        dense_pcnt_del_2 <= 0;
        dense_pcnt_del_3 <= 0;
        
        dense_cnt_del_1<= 0;
        dense_cnt_del_2<= 0;
        dense_cnt_del_3<= 0;
        dense_cnt_del_4<= 0;
        
        wb_cnt_del_1 <= 0;
        wb_cnt_del_2 <= 0;
        wb_cnt_del_3 <= 0;
	end
	else begin
		kernel_position_del_1 <= kernel_position;
		kernel_position_del_2 <= kernel_position_del_1;
		kernel_position_del_3 <= kernel_position_del_2;
		kernel_position_del_4 <= kernel_position_del_3;
		
		kernel_shift_z_del_1 <= kernel_shift_z;
        kernel_shift_z_del_2 <= kernel_shift_z_del_1;
        kernel_shift_z_del_3 <= kernel_shift_z_del_2;
        kernel_shift_z_del_4 <= kernel_shift_z_del_3;
        kernel_shift_z_del_5 <= kernel_shift_z_del_4;
        
		padding_position_dim1_c1_del_1 <= padding_position_dim1_c1;
        padding_position_dim1_c1_del_2 <= padding_position_dim1_c1_del_1;
        padding_position_dim1_c1_del_3 <= padding_position_dim1_c1_del_2;
		
        padding_position_dim1_c2_del_1 <= padding_position_dim1_c2;
        padding_position_dim1_c2_del_2 <= padding_position_dim1_c2_del_1;
        padding_position_dim1_c2_del_3 <= padding_position_dim1_c2_del_2;
		
        padding_position_dim1_c3_del_1 <= padding_position_dim1_c3;
        padding_position_dim1_c3_del_2 <= padding_position_dim1_c3_del_1;
        padding_position_dim1_c3_del_3 <= padding_position_dim1_c3_del_2;
        
        dense_pcnt_del_1 <= dense_pcnt;
        dense_pcnt_del_2 <= dense_pcnt_del_1;
        dense_pcnt_del_3 <= dense_pcnt_del_2;
        
        dense_cnt_del_1<= dense_cnt;
        dense_cnt_del_2<= dense_cnt_del_1;
        dense_cnt_del_3<= dense_cnt_del_2;
        dense_cnt_del_4<= dense_cnt_del_3;
        
        wb_cnt_del_1 <= wb_cnt;
        wb_cnt_del_2 <= wb_cnt_del_1;
        wb_cnt_del_3 <= wb_cnt_del_2;
	end
end

//reg mac
reg	[`data_size-1:0] mac_a,mac_b;			//a:padimage b:weight
wire [`data_size-1:0] mac_result;			//a*b=result

always@(posedge clk or posedge rst)begin
	if(rst)begin
		mac_a <= 0;
		mac_b <= 0;
	end
	else begin
		if(cs==OP_YC_CONV_1 || cs==OP_YC_CONV_2 || cs==OP_YC_CONV_3) begin	
			mac_a <= padding_picture;
			mac_b <= dout_a3;
		end	
		else if(cs==OP_YC_DENSE_1) begin
			mac_a <= dout_a2;
			mac_b <= dout_a3;
		end
		else if(cs==OP_YC_DENSE_2)begin			
            mac_a <= dout_a1;
            mac_b <= dout_a3;
		end
		else begin
			mac_a <= 0;
			mac_b <= 0;
		end
	end
end

mul conv_mul_1(
.s_axis_a_tvalid(),
.s_axis_a_tdata(mac_a),				//padding_picture
.s_axis_b_tvalid(),
.s_axis_b_tdata(mac_b),				//weight
.m_axis_result_tdata(mac_result),
.m_axis_result_tvalid()
);

//reg add
reg	[`data_size-1:0] add_a, add_b, add_ma, add_mb;	//a:mac_result b:bias
wire [`data_size-1:0] add_result, add_result_m;	//32bits

//dense_cnt
always@(posedge clk or posedge rst)begin
	if(rst)begin
		dense_pcnt <= 0;			//count dense pixels 1:1024
		dense_cnt <= 0;				//count dense outputs nums 1:64
		dense_pcnt_2 <= 0;
	end
	else begin
		if(cs==OP_YC_DENSE_1)begin
			if(dense_pcnt>=1023)begin			//////////
				dense_cnt  <= dense_cnt + 1;	
				dense_pcnt <= 0;
				dense_pcnt_2 <= dense_pcnt_2 + 1;
			end
			else begin
				dense_cnt  <= dense_cnt;
				dense_pcnt <= dense_pcnt + 1;
				dense_pcnt_2 <= dense_pcnt_2 +1;
			end
		end
		else if(cs==OP_YC_DENSE_2)begin
			if(dense_pcnt>=63)begin				///////////
				dense_cnt  <= dense_cnt + 1;
				dense_pcnt <= 0;
				dense_pcnt_2 <= dense_pcnt_2 + 1;
			end
			else begin
				dense_cnt  <= dense_cnt;
				dense_pcnt <= dense_pcnt + 1;
				dense_pcnt_2 <= dense_pcnt_2 + 1;
			end
		end
		else if(cs==OP_YC_READ_DENSE_WEIGHT_1 || cs ==OP_YC_READ_DENSE_WEIGHT_2)begin
			dense_cnt  <= dense_cnt;
			dense_pcnt <= 0;
			dense_pcnt_2 <= 0;
		end
		else begin
			dense_cnt  <= 0;
			dense_pcnt <= 0;
			dense_pcnt_2 <= 0;
		end
	end
end

//add of mac
always@(posedge clk or posedge rst)begin
	if(rst)begin
		add_ma<=0;
		add_mb <= 0;
	end
	else begin
		if(cs==OP_YC_CONV_1 || cs==OP_YC_CONV_2 || cs==OP_YC_CONV_3)begin
		    if ( kernel_position_del_4==1 && ker_cnt>0 ) begin
                add_ma <= mac_result;
                add_mb <= 0;
			end
			else begin
			    add_ma <= mac_result;
                add_mb <= add_result_m;
			end
		end
		else if(cs==OP_YC_DENSE_1 || cs==OP_YC_DENSE_2)begin
			if (dense_pcnt_del_3==1 && dense_cnt>0) begin
                add_ma <= mac_result;
                add_mb <= 0;
            end
            else begin
                add_ma <= mac_result;
                add_mb <= add_result_m;
            end
		end
		else begin
			add_ma <= 0;
			add_mb <= 0;
		end
	end
end

//add of add bias
always@(posedge clk or posedge rst)begin
	if(rst)begin
		add_a <= 0;
		add_b <= 0;
	end
	else begin
		if(cs==OP_YC_CONV_1 || cs==OP_YC_CONV_2 || cs==OP_YC_CONV_3)begin
			if(kernel_position_del_4==1 && ker_cnt>0) begin
				add_a <= add_result_m;
				add_b <= dout_a4;
			end
			else begin
				add_a <= add_a;
				add_b <= add_b;
			end
		end
		else if(cs==OP_YC_DENSE_1 || cs==OP_YC_DENSE_2)begin
			if(dense_pcnt_del_3==1 && dense_cnt>0) begin
				add_a <= add_result_m;
				add_b <= dout_a4;
			end
			else begin
				add_a <= add_a;
				add_b <= add_b;
			end
		end
		else begin
			add_a <= 0;
			add_b <= 0;
		end
	end
end

add add_bias_1(
.s_axis_a_tdata(add_a),				//mac_result 
.s_axis_a_tvalid(),
.s_axis_b_tdata(add_b),				//bias
.s_axis_b_tvalid(),
.m_axis_result_tdata(add_result),	//add_a+bias=result
.m_axis_result_tvalid()
); 

add add_mul_1(
.s_axis_a_tdata(add_ma),				//mac_result 
.s_axis_a_tvalid(),
.s_axis_b_tdata(add_mb),				//bias
.s_axis_b_tvalid(),
.m_axis_result_tdata(add_result_m),	//add_a+bias=result
.m_axis_result_tvalid()
);

//ReLU
wire [`data_size-1:0] relu_result;	//data_size=32bits
assign relu_result = (add_result[31]==1) ? 32'd0 : add_result;

//mem1_reg
reg [14:0] addr_a1;					//memory開32768 2^15=32768 cause CONV1 feature map32*32*32=32768pixels
reg [`data_size-1:0] din_a1;		//relu_result
reg wenable_a1;						//write enable

//mem1 data signal
always@(posedge clk or posedge rst) begin
	if(rst) begin
		addr_a1	<= -15'd1;
		din_a1	<= 32'd0;
		wenable_a1	<= 0;
	end
	else begin
		if(cs==OP_YC_CONV_1 || cs==OP_YC_CONV_2 || cs==OP_YC_CONV_3) begin //conv write part
			if(kernel_position_del_4==2 && ker_cnt>0) begin //right data write to mem1
				addr_a1	<= addr_a1+1;
				din_a1	<= relu_result;
				wenable_a1	<= 1;
			end
			else begin //wrong data
				addr_a1	<= addr_a1;
				din_a1	<= din_a1;
				wenable_a1	<= 0;
			end
        end	
        else if(cs==OP_YC_READ_WEIGHT_1 || cs==OP_YC_READ_WEIGHT_2 || cs==OP_YC_READ_WEIGHT_3 || cs==OP_YC_READ_DENSE_WEIGHT_1 || cs==OP_YC_READ_DENSE_WEIGHT_2) begin //weight stay
            addr_a1	<= addr_a1;
            din_a1	<= din_a1;
            wenable_a1	<= 0;
        end
		else if(cs==OP_YC_MAX_1) begin //maxpool read
			addr_a1	<= mp_position_1;
			din_a1	<= 32'd0;
			wenable_a1	<= 0;
		end
		else if(cs==OP_YC_MAX_2) begin //maxpool read
			addr_a1	<= mp_position_2;
			din_a1	<= 32'd0;
			wenable_a1	<= 0;
		end
		else if(cs==OP_YC_MAX_3) begin //maxpool read
			addr_a1	<= mp_position_3;
			din_a1	<= 32'd0;
			wenable_a1	<= 0;
		end
		else if(cs==OP_YC_DENSE_1) begin //dense1 write
			if(dense_pcnt_del_3==2 && dense_pcnt_2>=1028) begin //right data write to mem1
				addr_a1	<= addr_a1+1;
				din_a1	<= relu_result;
				wenable_a1	<= 1;
			end
			else begin //wrong data
				addr_a1	<= addr_a1;
				din_a1	<= din_a1;
				wenable_a1	<= 0;
			end
        end
		else if(cs==OP_YC_DENSE_2) begin //dense1 read
			addr_a1	<= dense_pcnt;
			din_a1	<= 0;
			wenable_a1	<= 0;
        end
		else begin
			addr_a1	<= -1;
			din_a1	<= 0;
			wenable_a1	<= 0;
		end
    end
end

mem_1 memory_1(
.addra(addr_a1),
.clka(clk),
.dina(din_a1),			//relu_result
.douta(dout_a1),		//for maxp		
.wea(wenable_a1)
);

//reg_mp
reg [`data_size-1:0] mp_kernel [0:3];				//data_size=32bits
reg [10:0] 	mp_kernel_x,mp_kernel_y;				//0~1	
reg [10:0]	mp_kernel_shift_x,mp_kernel_shift_y;	//1:0~30 2:0~14 3:0~6		

//wire_mp
wire	[10:0]	mp_kernel_position;

assign mp_kernel_position = mp_kernel_x + 2*mp_kernel_y;	//0~3
assign mp_position_1 = mp_kernel_x + mp_kernel_shift_x + 32*(mp_kernel_y + mp_kernel_shift_y) + 1024*mp_kernel_shift_z;	
assign mp_position_2 = mp_kernel_x + mp_kernel_shift_x + 16*(mp_kernel_y + mp_kernel_shift_y) + 256*mp_kernel_shift_z;		
assign mp_position_3 = mp_kernel_x + mp_kernel_shift_x + 8*(mp_kernel_y + mp_kernel_shift_y) + 64*mp_kernel_shift_z;

//mp_kernel_shift 
always@(posedge clk or posedge rst)begin
	if(rst)begin
		mp_kernel_shift_x <= 0;
		mp_kernel_shift_y <= 0;
		mp_kernel_shift_z <= 0;
		mp_kernel_x <= 0;
		mp_kernel_y <= 0;
		max_cnt <= 0;											//count pixels of feature map
	end
	else begin
		if(cs==OP_YC_MAX_1)begin
			if(mp_kernel_shift_x==30&&mp_kernel_x==1&&mp_kernel_shift_y==30&&mp_kernel_y==1)begin
				mp_kernel_shift_z <= mp_kernel_shift_z + 1;		//count feature map	nums 1:0~32 2:0~32 3:0~64
				mp_kernel_shift_x <= 0;
				mp_kernel_shift_y <= 0;
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_shift_x==30&&mp_kernel_x==1&&mp_kernel_y==1)begin
				mp_kernel_shift_y <= mp_kernel_shift_y + 2;		//stride=2
				mp_kernel_shift_x <= 0;
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_x==1&&mp_kernel_y==1)begin
				mp_kernel_shift_x <= mp_kernel_shift_x + 2;		//stride=2
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_x==1)begin
				mp_kernel_y <= mp_kernel_y + 1;
				mp_kernel_x <= 0;
			end
			else begin
				mp_kernel_x <= mp_kernel_x + 1;
			end			
		end
		else if(cs==OP_YC_MAX_2)begin
			if(mp_kernel_shift_x==14&&mp_kernel_x==1&&mp_kernel_shift_y==14&&mp_kernel_y==1)begin
				mp_kernel_shift_z <= mp_kernel_shift_z + 1;		//count feature map	nums
				mp_kernel_shift_x <= 0;
				mp_kernel_shift_y <= 0;
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_shift_x==14&&mp_kernel_x==1&&mp_kernel_y==1)begin
				mp_kernel_shift_y <= mp_kernel_shift_y + 2;		//stride=2
				mp_kernel_shift_x <= 0;
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_x==1&&mp_kernel_y==1)begin
				mp_kernel_shift_x <= mp_kernel_shift_x + 2;		//stride=2
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_x==1)begin
				mp_kernel_y <= mp_kernel_y + 1;
				mp_kernel_x <= 0;
			end
			else begin
				mp_kernel_x <= mp_kernel_x + 1;
			end			
		end
		else if(cs==OP_YC_MAX_3)begin
			if(mp_kernel_shift_x==6&&mp_kernel_x==1&&mp_kernel_shift_y==6&&mp_kernel_y==1)begin
				mp_kernel_shift_z <= mp_kernel_shift_z + 1;		//count feature map	nums
				mp_kernel_shift_x <= 0;
				mp_kernel_shift_y <= 0;
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_shift_x==6&&mp_kernel_x==1&&mp_kernel_y==1)begin
				mp_kernel_shift_y <= mp_kernel_shift_y + 2;		//stride=2
				mp_kernel_shift_x <= 0;
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_x==1&&mp_kernel_y==1)begin
				mp_kernel_shift_x <= mp_kernel_shift_x + 2;		//stride=2
				mp_kernel_x <= 0;
				mp_kernel_y <= 0;
				max_cnt <= max_cnt + 1;							//get one pixel of feature map
			end
			else if(mp_kernel_x==1)begin
				mp_kernel_y <= mp_kernel_y + 1;
				mp_kernel_x <= 0;
			end
			else begin
				mp_kernel_x <= mp_kernel_x + 1;
			end			
		end
		else begin
			mp_kernel_shift_x <= 0;
			mp_kernel_shift_y <= 0;
			mp_kernel_shift_z <= 0;
			mp_kernel_x <= 0;
			mp_kernel_y <= 0;
			max_cnt <= 0;
		end
	end
end

//mp_kernel
always@(posedge clk or posedge rst)begin		
	if(rst)begin
		mp_kernel[0] <= 0;
		mp_kernel[1] <= 0;
		mp_kernel[2] <= 0;
		mp_kernel[3] <= 0;
	end
	else begin
		if(cs==OP_YC_MAX_1 || cs==OP_YC_MAX_2 || cs==OP_YC_MAX_3)begin
			mp_kernel[0] <= dout_a1;				//接mem1output
			mp_kernel[1] <= mp_kernel[0];
			mp_kernel[2] <= mp_kernel[1];
			mp_kernel[3] <= mp_kernel[2];
		end
		else begin
			mp_kernel[0] <= 0;				
			mp_kernel[1] <= 0;
			mp_kernel[2] <= 0;
			mp_kernel[3] <= 0;
		end
	end
end

//compare
wire	[`data_size-1:0]	mp_result_temp_0,mp_result_temp_1;	//data_size=32bits
wire	[`data_size-1:0]	mp_result;							//data_size=32bits

assign mp_result_temp_0 = (mp_kernel[0] >= mp_kernel[1]) ? mp_kernel[0] : mp_kernel[1];
assign mp_result_temp_1 = (mp_kernel[2] >= mp_kernel[3]) ? mp_kernel[2] : mp_kernel[3];
assign mp_result = (mp_result_temp_0 >= mp_result_temp_1) ? mp_result_temp_0 :mp_result_temp_1;

//mem2_reg
reg [14:0] addr_a2;					//memory開 feature map 32*32*32=32768 pixels
reg [`data_size-1:0] din_a2;		//mp_result=32bits
reg wenable_a2;				//write enable

//mem2
always@(posedge clk or posedge rst) begin
    if(rst) begin
		wenable_a2 <= 0;
		addr_a2    <= -15'd1;
		din_a2     <= 0;	
    end
	else begin	
		if(cs==OP_YC_READ_PICTURE && read)begin
			wenable_a2 <= 1;
			addr_a2    <= addr_a2 + 1;
			din_a2     <= isif_data_dout[31:0];
		end		
		else if(cs==OP_YC_CONV_1)begin
			wenable_a2 <= 0;
			addr_a2	   <= (iaddr_c1+(kernel_z*1024));
			din_a2     <= 0;
		end			
        else if(cs==OP_YC_MAX_1 || cs==OP_YC_MAX_2 || cs==OP_YC_MAX_3) begin
			if(mp_kernel_position >= 3 && max_cnt>0) begin //write right data to mem2
				wenable_a2 <= 1;
				addr_a2    <= addr_a2 + 1;
				din_a2     <= mp_result;			
			end
			else begin //wrong data
				wenable_a2 <= 0;
				addr_a2	   <= addr_a2;
				din_a2	   <= din_a2;			
			end
        end			
		else if(cs==OP_YC_CONV_2)begin //conv read part
			wenable_a2 <= 0;
			addr_a2	   <= (iaddr_c2+(kernel_z*256));
			din_a2     <= 0;
        end
		else if(cs==OP_YC_CONV_3)begin //conv read part
			wenable_a2 <= 0;
			addr_a2    <= (iaddr_c3+(kernel_z*64));
			din_a2     <= 0;		
        end
		else if(cs==OP_YC_READ_WEIGHT_2 || cs==OP_YC_READ_WEIGHT_3 || cs==OP_YC_READ_DENSE_WEIGHT_1 || cs==OP_YC_READ_DENSE_WEIGHT_2)begin //weight stay
            wenable_a2 <= 0;
			addr_a2    <= addr_a2;
            din_a2     <= din_a2;                   
        end
		else if(cs==OP_YC_DENSE_2)begin
			if(dense_pcnt_del_3==2 && dense_pcnt_2>=68) begin //write right data to mem2
				wenable_a2 <= 1;
				addr_a2    <= addr_a2 + 1;
				din_a2     <= add_result;			
			end
			else begin //wrong data
				wenable_a2 <= 0;
				addr_a2	   <= addr_a2;
				din_a2	   <= din_a2;			
			end
        end
		else if(cs==OP_YC_DENSE_1)begin // read part
			wenable_a2 <= 0;
			addr_a2	   <= dense_pcnt;
			din_a2     <= 0;
        end	
		else if(cs==OP_YC_WB)begin				////////
			wenable_a2 <= 0;
			addr_a2	   <= wb_cnt;
			din_a2     <= 0;
		end	
		
		else begin
			wenable_a2 <= 0;
			addr_a2    <= -15'd1;
			din_a2     <= 0;			
		end
    end
end

mem_1 memory_2(
.addra(addr_a2),
.clka(clk),
.dina(din_a2),
.douta(dout_a2),
.wea(wenable_a2)
);

//mem3_reg
reg [9:0] addr_a3;					//memory開 feature map 32*32*32=32768 pixels
reg [`data_size-1:0] din_a3;		//mp_result=32bits
reg wenable_a3;				//write enable

//mem3 weight
always@(posedge clk or posedge rst) begin
    if(rst) begin
		wenable_a3 <= 0;
		addr_a3    <= -1;
		din_a3     <= 0;	
    end
	else begin
		if((cs==OP_YC_READ_WEIGHT_1 || cs==OP_YC_READ_WEIGHT_2 || cs==OP_YC_READ_WEIGHT_3 || cs==OP_YC_READ_DENSE_WEIGHT_1 || cs==OP_YC_READ_DENSE_WEIGHT_2) && read)begin
			wenable_a3 <= 1;
			addr_a3    <= addr_a3 + 1;
			din_a3     <= isif_data_dout[31:0];
		end
		else if(cs==OP_YC_CONV_1 || cs==OP_YC_CONV_2 || cs==OP_YC_CONV_3)begin
			wenable_a3 <= 0;
			addr_a3    <= kernel_position_del_1;
			din_a3     <= 0;
		end
		else if(cs==OP_YC_DENSE_1 || cs==OP_YC_DENSE_2)begin
			wenable_a3 <= 0;
			addr_a3    <= dense_pcnt;
			din_a3     <= 0;
		end		
		else begin
			wenable_a3 <= 0;
			addr_a3    <= -1;
			din_a3     <= 0;			
		end
    end
end

mem_weight memory_3(
.addra(addr_a3),
.clka(clk),
.dina(din_a3),
.douta(dout_a3),
.wea(wenable_a3)
);

//mem4_reg
reg [5:0] addr_a4;					
reg [`data_size-1:0] din_a4;
reg wenable_a4;

//mem4 bias
always@(posedge clk or posedge rst) begin
    if(rst) begin
		wenable_a4 <= 0;
		addr_a4    <= -1;
		din_a4     <= 0;	
    end
	else begin	
		if((cs==OP_YC_READ_BIAS_1 || cs==OP_YC_READ_BIAS_2 || cs==OP_YC_READ_BIAS_3 || cs==OP_YC_READ_DENSE_BIAS_1 || cs==OP_YC_READ_DENSE_BIAS_2) && read)begin
			wenable_a4 <= 1;
			addr_a4    <= addr_a4 + 1;
			din_a4     <= isif_data_dout[31:0];
		end
		else if(cs==OP_YC_CONV_1 || cs==OP_YC_CONV_2 || cs==OP_YC_CONV_3)begin
			wenable_a4 <= 0;
			addr_a4    <= kernel_shift_z_del_3;
			din_a4     <= isif_data_dout[31:0];
		end
		else if(cs==OP_YC_DENSE_1 || cs==OP_YC_DENSE_2)begin
			wenable_a4 <= 0;
			addr_a4    <= dense_cnt_del_3;
			din_a4     <= isif_data_dout[31:0];
		end
		else begin
			wenable_a4 <= 0;
			addr_a4    <= -1;
			din_a4     <= 0;			
		end
    end
end

mem_bias memory_4(
.addra(addr_a4),
.clka(clk),
.dina(din_a4),
.douta(dout_a4),
.wea(wenable_a4)
);

always @(posedge clk or posedge rst)begin
    if(rst)begin             
        op_data <= 0;
        write <= 0;
        last <= 0;
    end 
	else begin
        if(cs == OP_YC_WB)begin
			if(wb_cnt_del_2 >0)begin
				write <= 1;
				last <=(wb_cnt_del_3>=9)? 1:0;
				op_data <= {32'h00000000,dout_a2};
			end
			else begin
				write <= 0;
				last <= 0;
				op_data <=0;
			end
		end
        else begin
            write <= 0;
            last <= 0;
        end    
    end
end

always@(posedge clk or posedge rst)begin
	if(rst)begin
		wb_cnt <= 0;
	end
	else begin
		if(cs==OP_YC_WB)begin
			wb_cnt <= wb_cnt + 1;
		end
		else begin
			wb_cnt <= 0;
		end
	end
end

endmodule 