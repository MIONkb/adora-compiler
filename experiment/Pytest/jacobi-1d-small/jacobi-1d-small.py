
"""
Copyright (c) 2025 ADORA
All rights reserved.

Automatically generated file for pytest/cocotb based CGRA call function from ADORA.
"""
from test_runif import DeviceData, DeviceConfig, DeviceStream, DeviceRuntime
from typing import List

async def aux_stream(
    stream: DeviceStream, config: List[DeviceConfig], 
    iptrs: List[DeviceData], idata: List, 
    optrs: List[DeviceData], odata: List, olen: List):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------
    await stream.apply(config)
    await stream.config(config_id=0)
    # ------------------------------
    # 2. Host -> Device transfer
    # ------------------------------
    for i in range(len(iptrs)):
        await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]), dtype='i')
    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    await stream.execution_finish()
    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i], dtype='i')

    ## await stream.release()
    return

## ===----------------------------------------------------------------------===//
## Configuration Data 
## ===----------------------------------------------------------------------===//
""" kernel: jacobi_1d_kernel_0,  cfgNum: 30"""
cfgbit_jacobi_1d_kernel_0 = [
		0x2000, 0xd800, 0x0008,
		0x0001, 0x0000, 0x0009,
		0x0000, 0x0000, 0x000a,
		0x0000, 0x0002, 0x000b,
		0x2800, 0xd800, 0x0010,
		0x0001, 0x0000, 0x0011,
		0x0000, 0x0000, 0x0012,
		0x0000, 0x0902, 0x0013,
		0x0000, 0x0000, 0x0014,
		0x0030, 0x0000, 0x0058,
		0x0000, 0x0000, 0x00e8,
		0x0060, 0x0000, 0x00e9,
		0xaa3b, 0x3eaa, 0x0130,
		0x000a, 0x2000, 0x0131,
		0x0000, 0x0002, 0x0178,
		0x0000, 0x0000, 0x0180,
		0x000c, 0x0c00, 0x01c1,
		0x0000, 0x0000, 0x0210,
		0x000c, 0x2300, 0x0259,
		0x0010, 0x0000, 0x02a0,
		0x0002, 0x0000, 0x02a8,
		0x0000, 0x0001, 0x02b0,
		0x2000, 0xd800, 0x02e0,
		0x0001, 0x0000, 0x02e1,
		0x0000, 0x0000, 0x02e2,
		0x0000, 0x0002, 0x02e3,
		0x2000, 0xd800, 0x02f8,
		0x0001, 0x0000, 0x02f9,
		0x0000, 0x0000, 0x02fa,
		0x0000, 0x0002, 0x02fb,
	]


async def jacobi_1d_kernel_0(runtime: DeviceRuntime, arg_0: List, arg_1: List):
    # axibus.log.info("[ADORA] Starting CGRA call (jacobi_1d_kernel_0)")
    iptrs, idata = [],[]
    optrs, odata, olen = [],[],[]
    configs, data_ptr = [],[]    
    ## %0 = ADORA.BlockLoad %arg0 [0] : memref<120xf32> -> memref<120xf32>  {Id = "0", KernelName = "jacobi_1d_kernel_0"}
    idata.append(arg_0[0:0+120])
    iptrs.append(DeviceData(0x10000, 480))    
    
    ## %1 = ADORA.BlockLoad %arg0 [1] : memref<120xf32> -> memref<120xf32>  {Id = "1", KernelName = "jacobi_1d_kernel_0"}
    idata.append(arg_0[1:1+120])
    iptrs.append(DeviceData(0x18000, 480))    
    
    ## %2 = ADORA.BlockLoad %arg0 [2] : memref<120xf32> -> memref<120xf32>  {Id = "2", KernelName = "jacobi_1d_kernel_0"}
    idata.append(arg_0[2:2+120])
    iptrs.append(DeviceData(0x0, 480))    
    
    ## %3 = ADORA.LocalMemAlloc memref<120xf32>  {Id = "3", KernelName = "jacobi_1d_kernel_0"}
    data_ptr.append(DeviceData(0x2000, 480))
    
    ### jacobi_1d_kernel_0
    data_ptr.append(iptrs)
    config_jacobi_1d_kernel_0= DeviceConfig(
    	config_values=cfgbit_jacobi_1d_kernel_0,
    	iob_en=[0x03,0x12],
    	data_ptr=data_ptr
    )
    configs.append(config_jacobi_1d_kernel_0)
    

    
    ## ADORA.BlockStore %3, %arg1 [1] : memref<120xf32> -> memref<120xf32>  {Id = "3", KernelName = "jacobi_1d_kernel_0"}
    odata.append(arg_1[1:1+120])
    optrs.append(DeviceData(0x2000, 480))
    olen.append(480)
    
    stream = runtime.create_stream()

    await aux_stream(	stream=stream, config=configs,	iptrs=iptrs, idata=idata,	optrs=optrs, odata=odata, olen =olen)

    await stream.synchronize()
