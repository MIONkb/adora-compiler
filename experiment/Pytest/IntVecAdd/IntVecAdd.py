
"""
Copyright (c) 2025 ADORA
All rights reserved.

Automatically generated file for pytest/cocotb based CGRA call function from ADORA.
"""
from test_runif import DeviceData, DeviceConfig, DeviceStream, DeviceRuntime, aux_stream
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

    finally:
        # Always release the stream, even if an error occurs
        await stream.release()
        return";

//===----------------------------------------------------------------------===//
// Configuration Data 
//===----------------------------------------------------------------------===//
void IntVecAdd(void* arg_0 ,void* arg_1 ,void* arg_2){
    {
    /// %0 = ADORA.BlockLoad %arg0 [0] : memref<?xi32> -> memref<20xi32>  {Id = "0", KernelName = "IntVecAdd"}
    uint64_t dramoffset_0 = 0;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    spadoffset_0 = spadoffset_0 + 80;
    
    }
    {
    /// %1 = ADORA.BlockLoad %arg1 [0] : memref<?xi32> -> memref<20xi32>  {Id = "1", KernelName = "IntVecAdd"}
    uint64_t dramoffset_1 = 0;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    spadoffset_1 = spadoffset_1 + 80;
    
    }
    {
    /// ADORA.BlockStore %2, %arg2 [0] : memref<20xi32> -> memref<?xi32>  {Id = "2", KernelName = "IntVecAdd"}
    uint64_t dramoffset_2 = 0;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    store(arg_2 + dramoffset_2 + roffset_2, 0x0 + spadoffset_2, 80, _task_id, 0);
    spadoffset_2 = spadoffset_2 + 80;
    
    }
    _task_id++;
  fence(1);
}
