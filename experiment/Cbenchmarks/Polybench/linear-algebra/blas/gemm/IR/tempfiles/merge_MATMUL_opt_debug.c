
"""
Copyright (c) 2025 ADORA
All rights reserved.
Automatically generated file for pytest/cocotb based CGRA call function from ADORA.
Generated on: 2025-09-11 21:55:48

"""
from test_runif import DeviceData, DeviceConfig, DeviceStream, DeviceRuntime
from typing import List
from numpy import ndarray

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
        await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]))
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

def DeviceData_Pong(ptr : DeviceData) -> DeviceData:
    new_ptr = DeviceData(ptr.address+ptr.size, ptr.size)
    return new_ptr
  
async def aux_stream_pingpong(
    stream: DeviceStream, 
    # config: List[DeviceConfig], 
    config_id:int,
    iptrs: List[DeviceData], idata: List[ndarray], 
    optrs: List[DeviceData], odata: List, olen: List, 
    pingpong: bool):
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
    pingpong : bool
        Indicates the pingpong phase(ping-phase or pong-phase)
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------     
    await stream.config(config_id=config_id)
        # ------------------------------
    # 2. Host -> Device transfer
    # ------------------------------
    for i in range(len(iptrs)):
        if(pingpong == 0):
            await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]), depend_type=2)
        else:
            await stream.memcpyHostToDevice(DeviceData_Pong(iptrs[i]), h_data=idata[i], size=len(idata[i]), depend_type=2)

    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    await stream.execution_finish()
    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        if(pingpong == 0):
            await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i])
        else :
            await stream.memcpyDeviceToHost(DeviceData_Pong(optrs[i]), h_data=odata[i], size=olen[i])
    
    # await stream.synchronize()

    # await stream.release()
    return

async def aux_stream_pingpong_init(
    stream: DeviceStream, config: List[DeviceConfig]
    ):
    """
    Apply stream configuration
    """
    cfg_copy = list(config)
    await stream.apply(cfg_copy)  
    await stream.config(config_id=0)
    
    # await stream.release()
    return

## ===----------------------------------------------------------------------===//
## Configuration Data 
## ===----------------------------------------------------------------------===//
""" kernel: GEMMOS,  cfgNum: 620"""
cfgbit_GEMMOS = [
		0xc000, 0x4000, 0x0008,
		0xfe00, 0x041f, 0x0009,
		0x0000, 0x0000, 0x000a,
		0x0000, 0x2988, 0x000b,
		0x0000, 0x0000, 0x000c,
		0x9000, 0x4000, 0x0010,
		0xfe00, 0x041f, 0x0011,
		0x0000, 0x0000, 0x0012,
		0x0000, 0x0488, 0x0013,
		0xa000, 0x4000, 0x0020,
		0xfe00, 0x041f, 0x0021,
		0x0000, 0x0000, 0x0022,
		0x0000, 0x0308, 0x0023,
		0xd000, 0x4000, 0x0028,
		0xfe00, 0x041f, 0x0029,
		0x0000, 0x0000, 0x002a,
		0x0000, 0x2a08, 0x002b,
		0x0000, 0x0000, 0x002c,
		0x8000, 0x0000, 0x0030,
		0xc104, 0x041f, 0x0031,
		0x0000, 0x0000, 0x0032,
		0x0000, 0x0008, 0x0033,
		0xb000, 0x4000, 0x0038,
		0xfe00, 0x041f, 0x0039,
		0x0000, 0x0000, 0x003a,
		0x0000, 0x2988, 0x003b,
		0x0000, 0x0000, 0x003c,
		0xe000, 0x4000, 0x0040,
		0xfe00, 0x041f, 0x0041,
		0x0000, 0x0000, 0x0042,
		0x0000, 0x2888, 0x0043,
		0x0000, 0x0000, 0x0044,
		0xa000, 0x4000, 0x0048,
		0xfe00, 0x041f, 0x0049,
		0x0000, 0x0000, 0x004a,
		0x0000, 0x0408, 0x004b,
		0x9000, 0x0000, 0x0058,
		0xc104, 0x041f, 0x0059,
		0x0000, 0x0000, 0x005a,
		0x0000, 0x0008, 0x005b,
		0xd000, 0x4000, 0x0060,
		0xfe00, 0x041f, 0x0061,
		0x0000, 0x0000, 0x0062,
		0x0000, 0x2b88, 0x0063,
		0x0080, 0x0000, 0x0064,
		0xb000, 0x4000, 0x0070,
		0xfe00, 0x041f, 0x0071,
		0x0000, 0x0000, 0x0072,
		0x0000, 0x0408, 0x0073,
		0xc000, 0x4000, 0x0078,
		0xfe00, 0x041f, 0x0079,
		0x0000, 0x0000, 0x007a,
		0x0000, 0x2a08, 0x007b,
		0x0000, 0x0000, 0x007c,
		0x8000, 0x0000, 0x0080,
		0xc104, 0x041f, 0x0081,
		0x0000, 0x0000, 0x0082,
		0x0000, 0x0008, 0x0083,
		0x0000, 0x0000, 0x0090,
		0x0000, 0x0020, 0x0098,
		0x0000, 0x0400, 0x00a0,
		0x0000, 0x0000, 0x00a1,
		0x0000, 0xc000, 0x00a8,
		0x0100, 0x8000, 0x00b0,
		0x000c, 0x0000, 0x00b1,
		0x0000, 0x8000, 0x00b8,
		0x000c, 0x0000, 0x00b9,
		0x0100, 0x8800, 0x00c0,
		0x0001, 0x0000, 0x00c1,
		0x0140, 0x0820, 0x00c8,
		0x0028, 0x0000, 0x00c9,
		0x0000, 0x0800, 0x00d0,
		0x0004, 0x0000, 0x00d1,
		0x0000, 0x0820, 0x00d8,
		0x0000, 0x0800, 0x00e0,
		0x0004, 0x0000, 0x00e1,
		0x0000, 0x0800, 0x00e8,
		0x0000, 0x0000, 0x00e9,
		0x0000, 0x0800, 0x00f0,
		0x0000, 0x0800, 0x00f8,
		0x0100, 0x0800, 0x0100,
		0x0000, 0x0000, 0x0101,
		0x0000, 0x0000, 0x0108,
		0x0004, 0x0000, 0x0109,
		0x0000, 0x002c, 0x0118,
		0x0200, 0x0000, 0x0119,
		0x0010, 0x100d, 0x011a,
		0x0080, 0x0000, 0x011b,
		0x0000, 0x300a, 0x0128,
		0x1c00, 0x0000, 0x0129,
		0x0000, 0x400a, 0x0140,
		0x0b00, 0x0000, 0x0141,
		0x0000, 0x010a, 0x0148,
		0x1a00, 0x0000, 0x0149,
		0x0000, 0x020a, 0x0158,
		0x1a00, 0x0000, 0x0159,
		0x0000, 0xa00c, 0x0178,
		0x2300, 0x0000, 0x0179,
		0x0800, 0x0000, 0x01a0,
		0x0000, 0x0000, 0x01a9,
		0x0000, 0x0000, 0x01aa,
		0x8000, 0x0001, 0x01b0,
		0x0000, 0x0300, 0x01b1,
		0x0001, 0x0000, 0x01b8,
		0x0004, 0x0210, 0x01b9,
		0x0000, 0x0200, 0x01c0,
		0x3008, 0x2400, 0x01c1,
		0x0006, 0x0000, 0x01c2,
		0x3000, 0x0000, 0x01c8,
		0x2000, 0x1200, 0x01c9,
		0x0000, 0x0000, 0x01d0,
		0x0010, 0x1010, 0x01d1,
		0x0006, 0x0000, 0x01d2,
		0x0000, 0x0000, 0x01d8,
		0x2000, 0x1080, 0x01d9,
		0x0007, 0x0000, 0x01da,
		0x8000, 0x0000, 0x01e0,
		0x2000, 0x1000, 0x01e1,
		0x0001, 0x0000, 0x01e2,
		0x0000, 0x0000, 0x01e8,
		0x3000, 0xa000, 0x01e9,
		0x0000, 0x0000, 0x01ea,
		0x2000, 0x1000, 0x01f1,
		0x0000, 0x1300, 0x01f9,
		0x3000, 0x0040, 0x0200,
		0x0008, 0x0000, 0x0208,
		0x2000, 0x0010, 0x0211,
		0x0000, 0xc000, 0x0219,
		0x0000, 0x0000, 0x021a,
		0x0000, 0x012c, 0x0238,
		0x0200, 0x0000, 0x0239,
		0x0010, 0x100b, 0x023a,
		0x0080, 0x0000, 0x023b,
		0x0000, 0x030a, 0x0240,
		0x1a00, 0x0000, 0x0241,
		0x0000, 0x500c, 0x0248,
		0x0c00, 0x0000, 0x0249,
		0x0000, 0x002c, 0x0250,
		0x0200, 0x0000, 0x0251,
		0x0010, 0x100b, 0x0252,
		0x0080, 0x0000, 0x0253,
		0x0000, 0x002c, 0x0260,
		0x0100, 0x0000, 0x0261,
		0x0010, 0x100b, 0x0262,
		0x0080, 0x0000, 0x0263,
		0x0000, 0x002c, 0x0270,
		0x0100, 0x0000, 0x0271,
		0x0010, 0x100b, 0x0272,
		0x0080, 0x0000, 0x0273,
		0x0000, 0x020a, 0x0288,
		0x2100, 0x0000, 0x0289,
		0x0000, 0x022c, 0x0290,
		0x0300, 0x0000, 0x0291,
		0x0010, 0x100b, 0x0292,
		0x0080, 0x0000, 0x0293,
		0x0800, 0x0000, 0x02b0,
		0x0000, 0xe400, 0x02b9,
		0x0000, 0x0000, 0x02ba,
		0x0000, 0x0040, 0x02c0,
		0x0000, 0x5040, 0x02c1,
		0x0000, 0x0000, 0x02c2,
		0x8000, 0x0000, 0x02c8,
		0x0000, 0x0004, 0x02c9,
		0x0001, 0x0000, 0x02ca,
		0x4000, 0x0390, 0x02d1,
		0x0004, 0x0200, 0x02d8,
		0x0000, 0x0000, 0x02d9,
		0x0001, 0x0000, 0x02da,
		0x3000, 0x0800, 0x02e1,
		0x0006, 0x0000, 0x02e2,
		0x0000, 0x0340, 0x02e9,
		0x0005, 0x0000, 0x02ea,
		0x0000, 0x00c0, 0x02f0,
		0x2004, 0x2100, 0x02f1,
		0x0004, 0x0000, 0x02f2,
		0x0000, 0x4040, 0x02f9,
		0x0004, 0x0000, 0x02fa,
		0x0020, 0x0000, 0x0318,
		0x0000, 0x0010, 0x0321,
		0x0000, 0x0000, 0x0322,
		0x0000, 0xc000, 0x0329,
		0x0000, 0x0000, 0x032a,
		0x0000, 0x002c, 0x0348,
		0x0100, 0x0000, 0x0349,
		0x0010, 0x100c, 0x034a,
		0x0080, 0x0000, 0x034b,
		0x0000, 0x002c, 0x0358,
		0x0400, 0x0000, 0x0359,
		0x0010, 0x100a, 0x035a,
		0x0080, 0x0000, 0x035b,
		0x0000, 0x500c, 0x0360,
		0x0c00, 0x0000, 0x0361,
		0x0000, 0x500c, 0x0368,
		0x1300, 0x0000, 0x0369,
		0x0000, 0x022c, 0x0370,
		0x0200, 0x0000, 0x0371,
		0x0010, 0x100e, 0x0372,
		0x0080, 0x0000, 0x0373,
		0x0000, 0x040a, 0x0378,
		0x1100, 0x0000, 0x0379,
		0x0800, 0x0000, 0x03c0,
		0x0000, 0x0600, 0x03c8,
		0x0000, 0x018c, 0x03c9,
		0x8008, 0x0200, 0x03d1,
		0x0000, 0x2200, 0x03d9,
		0x0001, 0x0000, 0x03da,
		0x0000, 0x1490, 0x03e1,
		0x0004, 0x0000, 0x03e2,
		0x0000, 0x0200, 0x03e8,
		0x0000, 0xd480, 0x03e9,
		0x0000, 0x0000, 0x03ea,
		0x2002, 0x0000, 0x03f0,
		0x0010, 0x1400, 0x03f1,
		0x0007, 0x0000, 0x03f2,
		0x0000, 0x8000, 0x03f8,
		0xa000, 0xa301, 0x03f9,
		0x0000, 0x0000, 0x03fa,
		0x0000, 0x0040, 0x0400,
		0x8195, 0x0020, 0x0401,
		0x0000, 0x0001, 0x0409,
		0x0000, 0x0080, 0x0429,
		0x0000, 0xc000, 0x0431,
		0x0000, 0x0000, 0x0432,
		0x0000, 0xc000, 0x0439,
		0x0000, 0x0000, 0x043a,
		0x0000, 0x022c, 0x0448,
		0x0400, 0x0000, 0x0449,
		0x0010, 0x100b, 0x044a,
		0x0080, 0x0000, 0x044b,
		0x0000, 0x13f2, 0x0450,
		0xca33, 0x0008, 0x0451,
		0x0010, 0x0052, 0x0452,
		0x0004, 0x0000, 0x0453,
		0x0000, 0x000a, 0x0458,
		0x1300, 0x0000, 0x0459,
		0x0000, 0x042c, 0x0460,
		0x0400, 0x0000, 0x0461,
		0x0010, 0x100c, 0x0462,
		0x0080, 0x0000, 0x0463,
		0x0000, 0x010a, 0x0470,
		0x0a00, 0x0000, 0x0471,
		0x0000, 0x22f2, 0x0480,
		0xa242, 0x0002, 0x0481,
		0x0010, 0x0055, 0x0482,
		0x0004, 0x0000, 0x0483,
		0x0000, 0x10f2, 0x0488,
		0xd143, 0x0002, 0x0489,
		0x0010, 0x0050, 0x048a,
		0x0004, 0x0000, 0x048b,
		0x0000, 0x500c, 0x04b8,
		0x2300, 0x0000, 0x04b9,
		0x4000, 0x0000, 0x04d0,
		0x0000, 0x0008, 0x04d8,
		0x0000, 0x0280, 0x04d9,
		0x2600, 0x0000, 0x04e0,
		0x0808, 0x0480, 0x04e9,
		0x0000, 0x0000, 0x04f0,
		0x3000, 0x0390, 0x04f1,
		0x0004, 0x0000, 0x04f2,
		0x3014, 0xc090, 0x04f9,
		0x0004, 0x0000, 0x04fa,
		0x3000, 0xe401, 0x0501,
		0x0006, 0x0000, 0x0502,
		0x3000, 0x9800, 0x0509,
		0x0004, 0x0000, 0x050a,
		0x0008, 0x0008, 0x0510,
		0x1000, 0x8000, 0x0511,
		0x0000, 0x0000, 0x0512,
		0x3000, 0x0000, 0x0519,
		0x3000, 0x0000, 0x0521,
		0x4000, 0x0000, 0x0529,
		0x0040, 0x0080, 0x0539,
		0x4000, 0x0000, 0x0540,
		0x4000, 0xc000, 0x0541,
		0x0000, 0x0000, 0x0542,
		0x0018, 0x0000, 0x0548,
		0x0008, 0x0000, 0x0549,
		0x0000, 0x500c, 0x0558,
		0x2200, 0x0000, 0x0559,
		0x0000, 0x100a, 0x0560,
		0x1b00, 0x0000, 0x0561,
		0x0000, 0x21f2, 0x0568,
		0xda33, 0x0004, 0x0569,
		0x0010, 0x0050, 0x056a,
		0x0004, 0x0000, 0x056b,
		0x0000, 0x000a, 0x0578,
		0x1a00, 0x0000, 0x0579,
		0x0000, 0x022c, 0x0588,
		0x0400, 0x0000, 0x0589,
		0x0010, 0x100f, 0x058a,
		0x0080, 0x0000, 0x058b,
		0x0000, 0x002c, 0x0590,
		0x0300, 0x0000, 0x0591,
		0x0010, 0x100a, 0x0592,
		0x0080, 0x0000, 0x0593,
		0x0000, 0x002c, 0x0598,
		0x0300, 0x0000, 0x0599,
		0x0010, 0x100a, 0x059a,
		0x0080, 0x0000, 0x059b,
		0x0000, 0x042c, 0x05b0,
		0x0400, 0x0000, 0x05b1,
		0x0010, 0x100c, 0x05b2,
		0x0080, 0x0000, 0x05b3,
		0x0000, 0x20f2, 0x05b8,
		0xd343, 0x0008, 0x05b9,
		0x0010, 0x0051, 0x05ba,
		0x0004, 0x0000, 0x05bb,
		0x0000, 0x100a, 0x05c8,
		0x1a00, 0x0000, 0x05c9,
		0x4800, 0x0000, 0x05e0,
		0xa008, 0x0001, 0x05e8,
		0x0060, 0x2090, 0x05e9,
		0x0000, 0x000b, 0x05f0,
		0xb000, 0x1005, 0x05f1,
		0x4010, 0x1422, 0x05f9,
		0x0000, 0x0000, 0x05fa,
		0x0000, 0x00c0, 0x0600,
		0x4002, 0x9288, 0x0601,
		0x0000, 0x0000, 0x0602,
		0x0000, 0x4000, 0x0608,
		0x0000, 0x1204, 0x0609,
		0x0001, 0x0000, 0x060a,
		0x2c40, 0x9280, 0x0611,
		0x0006, 0x0000, 0x0612,
		0x4000, 0x0000, 0x0618,
		0x3008, 0x1001, 0x0619,
		0x0004, 0x0000, 0x061a,
		0x4000, 0x0000, 0x0620,
		0x3000, 0x1000, 0x0621,
		0x0000, 0x0400, 0x0628,
		0x3000, 0x0000, 0x0629,
		0x1000, 0x0000, 0x0631,
		0x0001, 0x0000, 0x0632,
		0x3000, 0x0000, 0x0639,
		0x4000, 0x000a, 0x0640,
		0x0000, 0x0000, 0x0641,
		0x0600, 0x0000, 0x0648,
		0x0040, 0x0260, 0x0649,
		0x8000, 0x02c1, 0x0650,
		0x4000, 0x0020, 0x0651,
		0x0800, 0x0010, 0x0659,
		0x0400, 0x0000, 0x0660,
		0x0000, 0x020a, 0x0668,
		0x1400, 0x0000, 0x0669,
		0x0000, 0x002c, 0x0670,
		0x0300, 0x0000, 0x0671,
		0x0010, 0x100d, 0x0672,
		0x0080, 0x0000, 0x0673,
		0x0000, 0x0000, 0x0678,
		0x0200, 0x0000, 0x0679,
		0x0000, 0x002c, 0x0680,
		0x0300, 0x0000, 0x0681,
		0x0010, 0x100c, 0x0682,
		0x0080, 0x0000, 0x0683,
		0x0000, 0x12f2, 0x0688,
		0xd143, 0x0002, 0x0689,
		0x0010, 0x0052, 0x068a,
		0x0004, 0x0000, 0x068b,
		0x0000, 0x22f2, 0x0690,
		0x5433, 0x0004, 0x0691,
		0x0010, 0x0053, 0x0692,
		0x0004, 0x0000, 0x0693,
		0x0000, 0x002c, 0x0698,
		0x0200, 0x0000, 0x0699,
		0x0010, 0x100d, 0x069a,
		0x0080, 0x0000, 0x069b,
		0x0000, 0x050a, 0x06a0,
		0x1400, 0x0000, 0x06a1,
		0x0000, 0x012c, 0x06a8,
		0x0300, 0x0000, 0x06a9,
		0x0010, 0x100a, 0x06aa,
		0x0080, 0x0000, 0x06ab,
		0x0000, 0x040a, 0x06b0,
		0x0b00, 0x0000, 0x06b1,
		0x0000, 0x002c, 0x06b8,
		0x0300, 0x0000, 0x06b9,
		0x0010, 0x100c, 0x06ba,
		0x0080, 0x0000, 0x06bb,
		0x0000, 0x052c, 0x06c0,
		0x0300, 0x0000, 0x06c1,
		0x0010, 0x100e, 0x06c2,
		0x0080, 0x0000, 0x06c3,
		0x0000, 0x010a, 0x06c8,
		0x1300, 0x0000, 0x06c9,
		0x0000, 0x032c, 0x06d0,
		0x0200, 0x0000, 0x06d1,
		0x0010, 0x100b, 0x06d2,
		0x0080, 0x0000, 0x06d3,
		0x0000, 0x22f2, 0x06d8,
		0xc942, 0x0004, 0x06d9,
		0x0010, 0x0052, 0x06da,
		0x0004, 0x0000, 0x06db,
		0x4400, 0x0000, 0x06f0,
		0x2003, 0x0000, 0x06f8,
		0x0010, 0x0086, 0x06f9,
		0x4000, 0x0030, 0x0701,
		0x1000, 0x0000, 0x0708,
		0x0000, 0x1890, 0x0709,
		0x0000, 0x0602, 0x0710,
		0x0000, 0x1080, 0x0711,
		0x0008, 0x1000, 0x0719,
		0x0001, 0x0600, 0x0720,
		0x0000, 0x1400, 0x0721,
		0x0006, 0x0000, 0x0722,
		0x0000, 0x0800, 0x0728,
		0x4010, 0x1210, 0x0729,
		0x4004, 0x0000, 0x0730,
		0x4000, 0x1204, 0x0731,
		0x0001, 0x0000, 0x0732,
		0x4000, 0x0080, 0x0738,
		0x1000, 0x0200, 0x0739,
		0x0000, 0x0000, 0x0740,
		0x0040, 0xc200, 0x0741,
		0x0000, 0x0000, 0x0742,
		0x1000, 0x0000, 0x0748,
		0x3000, 0x0200, 0x0749,
		0x5000, 0x0000, 0x0750,
		0x3000, 0x0230, 0x0751,
		0x3000, 0x0200, 0x0759,
		0x0000, 0x0004, 0x0760,
		0x3000, 0x0480, 0x0761,
		0x0000, 0x0040, 0x0768,
		0x4000, 0x0010, 0x0769,
		0x0000, 0x0000, 0x0770,
		0x0000, 0x022c, 0x0778,
		0x0200, 0x0000, 0x0779,
		0x0010, 0x100c, 0x077a,
		0x0080, 0x0000, 0x077b,
		0x0000, 0x022c, 0x0780,
		0x0100, 0x0000, 0x0781,
		0x0010, 0x100b, 0x0782,
		0x0080, 0x0000, 0x0783,
		0x0000, 0x500a, 0x0788,
		0x2400, 0x0000, 0x0789,
		0x0000, 0x062c, 0x0790,
		0x0400, 0x0000, 0x0791,
		0x0010, 0x100d, 0x0792,
		0x0080, 0x0000, 0x0793,
		0x0000, 0x100a, 0x0798,
		0x0a00, 0x0000, 0x0799,
		0x0000, 0x002c, 0x07a0,
		0x0300, 0x0000, 0x07a1,
		0x0010, 0x100b, 0x07a2,
		0x0080, 0x0000, 0x07a3,
		0x0000, 0x030a, 0x07a8,
		0x0a00, 0x0000, 0x07a9,
		0x0000, 0x030a, 0x07b0,
		0x0c00, 0x0000, 0x07b1,
		0x0000, 0x500c, 0x07c0,
		0x1100, 0x0000, 0x07c1,
		0x0000, 0x020a, 0x07c8,
		0x1c00, 0x0000, 0x07c9,
		0x0000, 0x042c, 0x07d0,
		0x0400, 0x0000, 0x07d1,
		0x0010, 0x100c, 0x07d2,
		0x0080, 0x0000, 0x07d3,
		0x0000, 0x032c, 0x07e0,
		0x0300, 0x0000, 0x07e1,
		0x0010, 0x100b, 0x07e2,
		0x0080, 0x0000, 0x07e3,
		0x0000, 0x032c, 0x07f0,
		0x0100, 0x0000, 0x07f1,
		0x0010, 0x100c, 0x07f2,
		0x0080, 0x0000, 0x07f3,
		0x4400, 0x0000, 0x0800,
		0x0000, 0x0200, 0x0808,
		0x8000, 0x0890, 0x0809,
		0x0010, 0x0440, 0x0811,
		0x0019, 0x0200, 0x0818,
		0x3004, 0x0490, 0x0819,
		0x0000, 0x0200, 0x0820,
		0x3014, 0x0080, 0x0821,
		0x0000, 0x0000, 0x0828,
		0xb000, 0x0001, 0x0829,
		0xb004, 0x0305, 0x0831,
		0xb000, 0x0241, 0x0839,
		0x0004, 0x0000, 0x0840,
		0xb010, 0xc281, 0x0841,
		0x0000, 0x0000, 0x0842,
		0xb000, 0x220d, 0x0849,
		0x8000, 0x0000, 0x0850,
		0xb008, 0x0201, 0x0851,
		0x0003, 0x0000, 0x0858,
		0x4000, 0x0a02, 0x0859,
		0x0001, 0x0040, 0x0860,
		0x0004, 0x1084, 0x0861,
		0x3000, 0x0000, 0x0868,
		0x1040, 0x0400, 0x0869,
		0x4000, 0x228c, 0x0871,
		0x0000, 0x0400, 0x0878,
		0x8000, 0x0088, 0x0879,
		0x0000, 0x300a, 0x0888,
		0x2400, 0x0000, 0x0889,
		0x0000, 0x500a, 0x0890,
		0x0a00, 0x0000, 0x0891,
		0x0000, 0x002c, 0x0898,
		0x0200, 0x0000, 0x0899,
		0x0010, 0x100a, 0x089a,
		0x0080, 0x0000, 0x089b,
		0x0000, 0x300a, 0x08a0,
		0x0a00, 0x0000, 0x08a1,
		0x0000, 0x000a, 0x08a8,
		0x0c00, 0x0000, 0x08a9,
		0x0000, 0x002c, 0x08b0,
		0x0200, 0x0000, 0x08b1,
		0x0010, 0x1009, 0x08b2,
		0x0080, 0x0000, 0x08b3,
		0x0000, 0x010a, 0x08b8,
		0x2300, 0x0000, 0x08b9,
		0x0000, 0x200a, 0x08c0,
		0x1a00, 0x0000, 0x08c1,
		0x0000, 0x200a, 0x08c8,
		0x2300, 0x0000, 0x08c9,
		0x0000, 0x500c, 0x08d0,
		0x1a00, 0x0000, 0x08d1,
		0x0000, 0x000a, 0x08d8,
		0x2400, 0x0000, 0x08d9,
		0x0000, 0x002c, 0x08e0,
		0x0200, 0x0000, 0x08e1,
		0x0010, 0x100c, 0x08e2,
		0x0080, 0x0000, 0x08e3,
		0x0000, 0x500a, 0x08e8,
		0x1100, 0x0000, 0x08e9,
		0x0000, 0x200a, 0x08f0,
		0x2300, 0x0000, 0x08f1,
		0x0000, 0x100a, 0x0900,
		0x0b00, 0x0000, 0x0901,
		0x0000, 0x0000, 0x0910,
		0x0001, 0x0000, 0x0918,
		0x0001, 0x0000, 0x0919,
		0x0000, 0x0c00, 0x0920,
		0x0004, 0x0000, 0x0921,
		0x0000, 0x4c00, 0x0928,
		0x0008, 0x0000, 0x0929,
		0x0000, 0x0c00, 0x0930,
		0x0008, 0x0000, 0x0931,
		0x0001, 0x0c00, 0x0938,
		0x0018, 0x0000, 0x0939,
		0x0200, 0x0c00, 0x0940,
		0x0024, 0x0000, 0x0941,
		0x0408, 0x0c00, 0x0948,
		0x0028, 0x0000, 0x0949,
		0x0300, 0x3420, 0x0950,
		0x002b, 0x0000, 0x0951,
		0x4000, 0x3c00, 0x0958,
		0x0006, 0x0000, 0x0959,
		0x0000, 0x3400, 0x0960,
		0x0008, 0x0000, 0x0961,
		0x0009, 0x3000, 0x0968,
		0x001b, 0x0000, 0x0969,
		0x0000, 0x3c00, 0x0970,
		0x0029, 0x0000, 0x0971,
		0x0200, 0x5c00, 0x0978,
		0x0020, 0x0000, 0x0979,
		0x0000, 0x0400, 0x0980,
		0x0021, 0x0000, 0x0981,
		0x0300, 0x0000, 0x0988,
		0x0000, 0x0000, 0x0989,
		0x2000, 0x0010, 0x0998,
		0x2104, 0x0418, 0x0999,
		0x0000, 0x0000, 0x099a,
		0x0000, 0x0008, 0x099b,
		0x8000, 0x0000, 0x09a0,
		0xc104, 0x041f, 0x09a1,
		0x0000, 0x0000, 0x09a2,
		0x0000, 0x0008, 0x09a3,
		0x1000, 0x0010, 0x09a8,
		0x2104, 0x0418, 0x09a9,
		0x0000, 0x0000, 0x09aa,
		0x0000, 0x0008, 0x09ab,
		0xd000, 0x4000, 0x09b0,
		0xfe00, 0x041f, 0x09b1,
		0x0000, 0x0000, 0x09b2,
		0x0000, 0x0288, 0x09b3,
		0x3000, 0x0010, 0x09b8,
		0x2104, 0x0418, 0x09b9,
		0x0000, 0x0000, 0x09ba,
		0x0000, 0x0008, 0x09bb,
		0xc000, 0x4000, 0x09c0,
		0xfe00, 0x041f, 0x09c1,
		0x0000, 0x0000, 0x09c2,
		0x0000, 0x0408, 0x09c3,
		0xe000, 0x4000, 0x09d0,
		0xfe00, 0x041f, 0x09d1,
		0x0000, 0x0000, 0x09d2,
		0x0000, 0x2a08, 0x09d3,
		0x0080, 0x0000, 0x09d4,
		0x8000, 0x0000, 0x09d8,
		0xc104, 0x041f, 0x09d9,
		0x0000, 0x0000, 0x09da,
		0x0000, 0x0008, 0x09db,
		0xf000, 0x4000, 0x09e0,
		0xfe00, 0x041f, 0x09e1,
		0x0000, 0x0000, 0x09e2,
		0x0000, 0x2908, 0x09e3,
		0x0000, 0x0000, 0x09e4,
		0xb000, 0x0000, 0x09e8,
		0xc104, 0x041f, 0x09e9,
		0x0000, 0x0000, 0x09ea,
		0x0000, 0x0008, 0x09eb,
		0xa000, 0x0000, 0x09f0,
		0xc104, 0x041f, 0x09f1,
		0x0000, 0x0000, 0x09f2,
		0x0000, 0x0008, 0x09f3,
		0xd000, 0x4000, 0x09f8,
		0xfe00, 0x041f, 0x09f9,
		0x0000, 0x0000, 0x09fa,
		0x0000, 0x0308, 0x09fb,
		0x4000, 0x0010, 0x0a00,
		0x2104, 0x0418, 0x0a01,
		0x0000, 0x0000, 0x0a02,
		0x0000, 0x0008, 0x0a03,
		0x9000, 0x0000, 0x0a08,
		0xc104, 0x041f, 0x0a09,
		0x0000, 0x0000, 0x0a0a,
		0x0000, 0x0008, 0x0a0b,
		0xe000, 0x4000, 0x0a10,
		0xfe00, 0x041f, 0x0a11,
		0x0000, 0x0000, 0x0a12,
		0x0000, 0x0288, 0x0a13,
	]


""" kernel: GEMMOS, ping-phase"""
cfgbit_GEMMOS_ping = [
		0xc000, 0x4000, 0x0008,
		0x9000, 0x4000, 0x0010,
		0xa000, 0x4000, 0x0020,
		0xd000, 0x4000, 0x0028,
		0x8000, 0x0000, 0x0030,
		0xb000, 0x4000, 0x0038,
		0xe000, 0x4000, 0x0040,
		0xa000, 0x4000, 0x0048,
		0x9000, 0x0000, 0x0058,
		0xd000, 0x4000, 0x0060,
		0xb000, 0x4000, 0x0070,
		0xc000, 0x4000, 0x0078,
		0x8000, 0x0000, 0x0080,
		0x2000, 0x0010, 0x0998,
		0x8000, 0x0000, 0x09a0,
		0x1000, 0x0010, 0x09a8,
		0xd000, 0x4000, 0x09b0,
		0x3000, 0x0010, 0x09b8,
		0xc000, 0x4000, 0x09c0,
		0xe000, 0x4000, 0x09d0,
		0x8000, 0x0000, 0x09d8,
		0xf000, 0x4000, 0x09e0,
		0xb000, 0x0000, 0x09e8,
		0xa000, 0x0000, 0x09f0,
		0xd000, 0x4000, 0x09f8,
		0x4000, 0x0010, 0x0a00,
		0x9000, 0x0000, 0x0a08,
		0xe000, 0x4000, 0x0a10,
	]

""" kernel: GEMMOS, pong-phase"""

cfgbit_GEMMOS_pong = [
		0xc100, 0x4000, 0x0008,
		0x9100, 0x4000, 0x0010,
		0xa100, 0x4000, 0x0020,
		0xd100, 0x4000, 0x0028,
		0x8080, 0x0000, 0x0030,
		0xb100, 0x4000, 0x0038,
		0xe100, 0x4000, 0x0040,
		0xa100, 0x4000, 0x0048,
		0x9080, 0x0000, 0x0058,
		0xd100, 0x4000, 0x0060,
		0xb100, 0x4000, 0x0070,
		0xc100, 0x4000, 0x0078,
		0x8080, 0x0000, 0x0080,
		0x3000, 0x0010, 0x0998,
		0x8080, 0x0000, 0x09a0,
		0x2000, 0x0010, 0x09a8,
		0xd100, 0x4000, 0x09b0,
		0x4000, 0x0010, 0x09b8,
		0xc100, 0x4000, 0x09c0,
		0xe100, 0x4000, 0x09d0,
		0x8080, 0x0000, 0x09d8,
		0xf100, 0x4000, 0x09e0,
		0xb080, 0x0000, 0x09e8,
		0xa080, 0x0000, 0x09f0,
		0xd100, 0x4000, 0x09f8,
		0x5000, 0x0010, 0x0a00,
		0x9080, 0x0000, 0x0a08,
		0xe100, 0x4000, 0x0a10,
	]


async def matmul_0(runtime: DeviceRuntime, arg_0: ndarray, arg_1: ndarray, arg_2: ndarray):
    # runtime.log.info("[ADORA] Starting CGRA call (matmul_0)")
    iptrs, idata = [],[]
    optrs, odata, olen = [],[],[]
    configs, data_ptr = [],[]
    #######################################
    ### Emit GemmOp: %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "OutputStationary", tile_size = array<i64: 32, 64, 8, 4>} : (memref<32x64xbf16>, memref<64x128xbf16>, memref<32x128xbf16>) -> memref<32x128xbf16>
    #######################################
    ptrs_ping, ptrs_pong = [], []
    ### Pingpong DataBlockLoadOp: %7 = ADORA.BlockLoad %arg0 [%6, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "3", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x32000, 128))
    ptrs_pong.append(DeviceData(0x32000+128, 128))
    ### Pingpong DataBlockLoadOp: %15 = ADORA.BlockLoad %arg0 [%14, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "7", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x36000, 128))
    ptrs_pong.append(DeviceData(0x36000+128, 128))
    ### Pingpong DataBlockLoadOp: %13 = ADORA.BlockLoad %arg0 [%12, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "6", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x20000, 128))
    ptrs_pong.append(DeviceData(0x20000+128, 128))
    ### Pingpong DataBlockLoadOp: %33 = ADORA.BlockLoad %arg2 [%31, %32] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "18", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x3c000, 256))
    ptrs_pong.append(DeviceData(0x3c000+256, 256))
    ### Pingpong DataBlockLoadOp: %23 = ADORA.BlockLoad %arg2 [%arg3, %arg4] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "12", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x28000, 256))
    ptrs_pong.append(DeviceData(0x28000+256, 256))
    ### Pingpong DataBlockLoadOp: %9 = ADORA.BlockLoad %arg0 [%8, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "4", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x12000, 128))
    ptrs_pong.append(DeviceData(0x12000+128, 128))
    ### Pingpong DataBlockLoadOp: %1 = ADORA.BlockLoad %arg0 [%arg3, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "0", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x0, 128))
    ptrs_pong.append(DeviceData(0x0+128, 128))
    ### Pingpong DataBlockLoadOp: %18 = ADORA.BlockLoad %arg1 [%arg5, %17] : memref<64x128xbf16> -> memref<64x32xbf16> , stride [1, 4] {ADORAGemm, Id = "9", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x22000, 4096))
    ptrs_pong.append(DeviceData(0x22000+4096, 4096))
    ### Pingpong DataBlockLoadOp: %26 = ADORA.BlockLoad %arg2 [%25, %arg4] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "14", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x3a000, 256))
    ptrs_pong.append(DeviceData(0x3a000+256, 256))
    ### Pingpong DataBlockLoadOp: %43 = ADORA.BlockLoad %arg2 [%arg3, %42] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "24", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x4000, 256))
    ptrs_pong.append(DeviceData(0x4000+256, 256))
    ### Pingpong DataBlockLoadOp: %3 = ADORA.BlockLoad %arg0 [%2, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "1", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x10000, 128))
    ptrs_pong.append(DeviceData(0x10000+128, 128))
    ### Pingpong DataBlockLoadOp: %29 = ADORA.BlockLoad %arg2 [%arg3, %28] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "16", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x14000, 256))
    ptrs_pong.append(DeviceData(0x14000+256, 256))
    ### Pingpong DataBlockLoadOp: %36 = ADORA.BlockLoad %arg2 [%arg3, %35] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "20", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x2a000, 256))
    ptrs_pong.append(DeviceData(0x2a000+256, 256))
    ### Pingpong DataBlockLoadOp: %16 = ADORA.BlockLoad %arg1 [%arg5, %arg4] : memref<64x128xbf16> -> memref<64x32xbf16> , stride [1, 4] {ADORAGemm, Id = "8", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x38000, 4096))
    ptrs_pong.append(DeviceData(0x38000+4096, 4096))
    ### Pingpong DataBlockLoadOp: %5 = ADORA.BlockLoad %arg0 [%4, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "2", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x30000, 128))
    ptrs_pong.append(DeviceData(0x30000+128, 128))
    ### Pingpong DataBlockLoadOp: %11 = ADORA.BlockLoad %arg0 [%10, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "5", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x34000, 128))
    ptrs_pong.append(DeviceData(0x34000+128, 128))
    ### Pingpong DataBlockLoadOp: %40 = ADORA.BlockLoad %arg2 [%38, %39] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "22", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x2000, 256))
    ptrs_pong.append(DeviceData(0x2000+256, 256))
    ### Pingpong DataBlockLoadOp: %22 = ADORA.BlockLoad %arg1 [%arg5, %21] : memref<64x128xbf16> -> memref<64x32xbf16> , stride [1, 4] {ADORAGemm, Id = "11", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x26000, 4096))
    ptrs_pong.append(DeviceData(0x26000+4096, 4096))
    ### Pingpong DataBlockLoadOp: %20 = ADORA.BlockLoad %arg1 [%arg5, %19] : memref<64x128xbf16> -> memref<64x32xbf16> , stride [1, 4] {ADORAGemm, Id = "10", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x24000, 4096))
    ptrs_pong.append(DeviceData(0x24000+4096, 4096))
    ### Pingpong DataBlockLoadOp: %47 = ADORA.BlockLoad %arg2 [%45, %46] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "26", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x16000, 256))
    ptrs_pong.append(DeviceData(0x16000+256, 256))
    ### Pingpong LocalMemAllocOp: %34 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "19", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x2c000, 256))
    ptrs_pong.append(DeviceData(0x2c000+256, 256))
    ### Pingpong LocalMemAllocOp: %30 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "17", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x6000, 256))
    ptrs_pong.append(DeviceData(0x6000+256, 256))
    ### Pingpong LocalMemAllocOp: %24 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x3e000, 256))
    ptrs_pong.append(DeviceData(0x3e000+256, 256))
    ### Pingpong LocalMemAllocOp: %27 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x18000, 256))
    ptrs_pong.append(DeviceData(0x18000+256, 256))
    ### Pingpong LocalMemAllocOp: %44 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "25", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0xc000, 256))
    ptrs_pong.append(DeviceData(0xc000+256, 256))
    ### Pingpong LocalMemAllocOp: %37 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "21", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x8000, 256))
    ptrs_pong.append(DeviceData(0x8000+256, 256))
    ### Pingpong LocalMemAllocOp: %41 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "23", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0xa000, 256))
    ptrs_pong.append(DeviceData(0xa000+256, 256))
    ### Pingpong LocalMemAllocOp: %48 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "27", KernelName = "GEMMOS", Pingpong}
    ptrs_ping.append(DeviceData(0x1a000, 256))
    ptrs_pong.append(DeviceData(0x1a000+256, 256))
    pingpong = False
    stream = runtime.create_stream()
    config_GEMMOS = DeviceConfig(config_values=cfgbit_GEMMOS, iob_en=[0xfb,0xed,0xbf,0xff], data_ptr=data_ptr)
    config_GEMMOS_ping = DeviceConfig(config_values=cfgbit_GEMMOS_ping, iob_en=[0xfb,0xed,0xbf,0xff], data_ptr=ptrs_ping)
    config_GEMMOS_pong = DeviceConfig(config_values=cfgbit_GEMMOS_pong, iob_en=[0xfb,0xed,0xbf,0xff], data_ptr=ptrs_pong)
    await aux_stream_pingpong_init(stream, [config_GEMMOS, config_GEMMOS_ping, config_GEMMOS_pong])

    for int_3 in range(0, 32, 8):
        for int_4 in range(0, 128, 128):
            for int_5 in range(0, 64, 64):
                
                ## %1 = ADORA.BlockLoad %arg0 [%arg3, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "0", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_0[int_3:int_3+1,int_5:int_5+64])
                iptrs.append(DeviceData(0x0+128 if pingpong else 0x0, 128))

                int_6 = int_3 + 1
                
                ## %3 = ADORA.BlockLoad %arg0 [%2, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "1", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_0[int_6:int_6+1,int_5:int_5+64])
                iptrs.append(DeviceData(0x10000+128 if pingpong else 0x10000, 128))

                int_7 = int_3 + 2
                
                ## %5 = ADORA.BlockLoad %arg0 [%4, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "2", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_0[int_7:int_7+1,int_5:int_5+64])
                iptrs.append(DeviceData(0x30000+128 if pingpong else 0x30000, 128))

                int_8 = int_3 + 3
                
                ## %7 = ADORA.BlockLoad %arg0 [%6, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "3", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_0[int_8:int_8+1,int_5:int_5+64])
                iptrs.append(DeviceData(0x32000+128 if pingpong else 0x32000, 128))

                int_9 = int_3 + 4
                
                ## %9 = ADORA.BlockLoad %arg0 [%8, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "4", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_0[int_9:int_9+1,int_5:int_5+64])
                iptrs.append(DeviceData(0x12000+128 if pingpong else 0x12000, 128))

                int_10 = int_3 + 5
                
                ## %11 = ADORA.BlockLoad %arg0 [%10, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "5", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_0[int_10:int_10+1,int_5:int_5+64])
                iptrs.append(DeviceData(0x34000+128 if pingpong else 0x34000, 128))

                int_11 = int_3 + 6
                
                ## %13 = ADORA.BlockLoad %arg0 [%12, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "6", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_0[int_11:int_11+1,int_5:int_5+64])
                iptrs.append(DeviceData(0x20000+128 if pingpong else 0x20000, 128))

                int_12 = int_3 + 7
                
                ## %15 = ADORA.BlockLoad %arg0 [%14, %arg5] : memref<32x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "7", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_0[int_12:int_12+1,int_5:int_5+64])
                iptrs.append(DeviceData(0x36000+128 if pingpong else 0x36000, 128))

                
                ## %16 = ADORA.BlockLoad %arg1 [%arg5, %arg4] : memref<64x128xbf16> -> memref<64x32xbf16> , stride [1, 4] {ADORAGemm, Id = "8", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_1[int_5:int_5+64:1,int_4:int_4+128:4])
                iptrs.append(DeviceData(0x38000+4096 if pingpong else 0x38000, 4096))

                int_13 = int_4 + 1
                
                ## %18 = ADORA.BlockLoad %arg1 [%arg5, %17] : memref<64x128xbf16> -> memref<64x32xbf16> , stride [1, 4] {ADORAGemm, Id = "9", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_1[int_5:int_5+64:1,int_13:int_13+128:4])
                iptrs.append(DeviceData(0x22000+4096 if pingpong else 0x22000, 4096))

                int_14 = int_4 + 2
                
                ## %20 = ADORA.BlockLoad %arg1 [%arg5, %19] : memref<64x128xbf16> -> memref<64x32xbf16> , stride [1, 4] {ADORAGemm, Id = "10", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_1[int_5:int_5+64:1,int_14:int_14+128:4])
                iptrs.append(DeviceData(0x24000+4096 if pingpong else 0x24000, 4096))

                int_15 = int_4 + 3
                
                ## %22 = ADORA.BlockLoad %arg1 [%arg5, %21] : memref<64x128xbf16> -> memref<64x32xbf16> , stride [1, 4] {ADORAGemm, Id = "11", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_1[int_5:int_5+64:1,int_15:int_15+128:4])
                iptrs.append(DeviceData(0x26000+4096 if pingpong else 0x26000, 4096))

                
                ## %23 = ADORA.BlockLoad %arg2 [%arg3, %arg4] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "12", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_2[int_3:int_3+4:1,int_4:int_4+128:4])
                iptrs.append(DeviceData(0x28000+256 if pingpong else 0x28000, 256))

                
                ## %24 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMOS", Pingpong}
                data_ptr.append(DeviceData(0x3e000+256 if pingpong else 0x3e000, 256))
                int_16 = int_3 + 4
                
                ## %26 = ADORA.BlockLoad %arg2 [%25, %arg4] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "14", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_2[int_16:int_16+4:1,int_4:int_4+128:4])
                iptrs.append(DeviceData(0x3a000+256 if pingpong else 0x3a000, 256))

                
                ## %27 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMOS", Pingpong}
                data_ptr.append(DeviceData(0x18000+256 if pingpong else 0x18000, 256))
                int_17 = int_4 + 1
                
                ## %29 = ADORA.BlockLoad %arg2 [%arg3, %28] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "16", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_2[int_3:int_3+4:1,int_17:int_17+128:4])
                iptrs.append(DeviceData(0x14000+256 if pingpong else 0x14000, 256))

                
                ## %30 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "17", KernelName = "GEMMOS", Pingpong}
                data_ptr.append(DeviceData(0x6000+256 if pingpong else 0x6000, 256))
                int_18 = int_3 + 4
                int_19 = int_4 + 1
                
                ## %33 = ADORA.BlockLoad %arg2 [%31, %32] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "18", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_2[int_18:int_18+4:1,int_19:int_19+128:4])
                iptrs.append(DeviceData(0x3c000+256 if pingpong else 0x3c000, 256))

                
                ## %34 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "19", KernelName = "GEMMOS", Pingpong}
                data_ptr.append(DeviceData(0x2c000+256 if pingpong else 0x2c000, 256))
                int_20 = int_4 + 2
                
                ## %36 = ADORA.BlockLoad %arg2 [%arg3, %35] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "20", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_2[int_3:int_3+4:1,int_20:int_20+128:4])
                iptrs.append(DeviceData(0x2a000+256 if pingpong else 0x2a000, 256))

                
                ## %37 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "21", KernelName = "GEMMOS", Pingpong}
                data_ptr.append(DeviceData(0x8000+256 if pingpong else 0x8000, 256))
                int_21 = int_3 + 4
                int_22 = int_4 + 2
                
                ## %40 = ADORA.BlockLoad %arg2 [%38, %39] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "22", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_2[int_21:int_21+4:1,int_22:int_22+128:4])
                iptrs.append(DeviceData(0x2000+256 if pingpong else 0x2000, 256))

                
                ## %41 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "23", KernelName = "GEMMOS", Pingpong}
                data_ptr.append(DeviceData(0xa000+256 if pingpong else 0xa000, 256))
                int_23 = int_4 + 3
                
                ## %43 = ADORA.BlockLoad %arg2 [%arg3, %42] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "24", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_2[int_3:int_3+4:1,int_23:int_23+128:4])
                iptrs.append(DeviceData(0x4000+256 if pingpong else 0x4000, 256))

                
                ## %44 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "25", KernelName = "GEMMOS", Pingpong}
                data_ptr.append(DeviceData(0xc000+256 if pingpong else 0xc000, 256))
                int_24 = int_3 + 4
                int_25 = int_4 + 3
                
                ## %47 = ADORA.BlockLoad %arg2 [%45, %46] : memref<32x128xbf16> -> memref<4x32xbf16> , stride [1, 4] {ADORAGemm, Id = "26", KernelName = "GEMMOS", Pingpong}
                idata.append(arg_2[int_24:int_24+4:1,int_25:int_25+128:4])
                iptrs.append(DeviceData(0x16000+256 if pingpong else 0x16000, 256))

                
                ## %48 = ADORA.LocalMemAlloc memref<4x32xbf16>  {ADORAGemm, Id = "27", KernelName = "GEMMOS", Pingpong}
                data_ptr.append(DeviceData(0x1a000+256 if pingpong else 0x1a000, 256))
                
                ### GEMMOS
                data_ptr.append(iptrs)
                config_GEMMOS= DeviceConfig(
                	config_values=cfgbit_GEMMOS,
                	iob_en=[0xfb,0xed,0xbf,0xff],
                	data_ptr=data_ptr
                )
                configs.append(config_GEMMOS)
                

                
                ## ADORA.BlockStore %24, %arg2 [%arg3, %arg4] : memref<4x32xbf16> -> memref<32x128xbf16> , stride [1, 4] {ADORAGemm, Id = "13", KernelName = "GEMMOS", Pingpong}
                odata.append(arg_2[int_3:int_3+4:1,int_4:int_4+128:4])
                optrs.append(DeviceData(0x3e000+256 if pingpong else 0x3e000, 256))
                olen.append(256)

                int_26 = int_3 + 4
                
                ## ADORA.BlockStore %27, %arg2 [%49, %arg4] : memref<4x32xbf16> -> memref<32x128xbf16> , stride [1, 4] {ADORAGemm, Id = "15", KernelName = "GEMMOS", Pingpong}
                odata.append(arg_2[int_26:int_26+4:1,int_4:int_4+128:4])
                optrs.append(DeviceData(0x18000+256 if pingpong else 0x18000, 256))
                olen.append(256)

                int_27 = int_4 + 1
                
                ## ADORA.BlockStore %30, %arg2 [%arg3, %50] : memref<4x32xbf16> -> memref<32x128xbf16> , stride [1, 4] {ADORAGemm, Id = "17", KernelName = "GEMMOS", Pingpong}
                odata.append(arg_2[int_3:int_3+4:1,int_27:int_27+128:4])
                optrs.append(DeviceData(0x6000+256 if pingpong else 0x6000, 256))
                olen.append(256)

                int_28 = int_3 + 4
                int_29 = int_4 + 1
                
                ## ADORA.BlockStore %34, %arg2 [%51, %52] : memref<4x32xbf16> -> memref<32x128xbf16> , stride [1, 4] {ADORAGemm, Id = "19", KernelName = "GEMMOS", Pingpong}
                odata.append(arg_2[int_28:int_28+4:1,int_29:int_29+128:4])
                optrs.append(DeviceData(0x2c000+256 if pingpong else 0x2c000, 256))
                olen.append(256)

                int_30 = int_4 + 2
                
                ## ADORA.BlockStore %37, %arg2 [%arg3, %53] : memref<4x32xbf16> -> memref<32x128xbf16> , stride [1, 4] {ADORAGemm, Id = "21", KernelName = "GEMMOS", Pingpong}
                odata.append(arg_2[int_3:int_3+4:1,int_30:int_30+128:4])
                optrs.append(DeviceData(0x8000+256 if pingpong else 0x8000, 256))
                olen.append(256)

                int_31 = int_3 + 4
                int_32 = int_4 + 2
                
                ## ADORA.BlockStore %41, %arg2 [%54, %55] : memref<4x32xbf16> -> memref<32x128xbf16> , stride [1, 4] {ADORAGemm, Id = "23", KernelName = "GEMMOS", Pingpong}
                odata.append(arg_2[int_31:int_31+4:1,int_32:int_32+128:4])
                optrs.append(DeviceData(0xa000+256 if pingpong else 0xa000, 256))
                olen.append(256)

                int_33 = int_4 + 3
                
                ## ADORA.BlockStore %44, %arg2 [%arg3, %56] : memref<4x32xbf16> -> memref<32x128xbf16> , stride [1, 4] {ADORAGemm, Id = "25", KernelName = "GEMMOS", Pingpong}
                odata.append(arg_2[int_3:int_3+4:1,int_33:int_33+128:4])
                optrs.append(DeviceData(0xc000+256 if pingpong else 0xc000, 256))
                olen.append(256)

                int_34 = int_3 + 4
                int_35 = int_4 + 3
                
                ## ADORA.BlockStore %48, %arg2 [%57, %58] : memref<4x32xbf16> -> memref<32x128xbf16> , stride [1, 4] {ADORAGemm, Id = "27", KernelName = "GEMMOS", Pingpong}
                odata.append(arg_2[int_34:int_34+4:1,int_35:int_35+128:4])
                optrs.append(DeviceData(0x1a000+256 if pingpong else 0x1a000, 256))
                olen.append(256)

                await aux_stream_pingpong(stream=stream,
                	config_id = 1 if pingpong==False else 2,
                	iptrs=iptrs, idata=idata,
                	optrs=optrs, odata=odata, olen =olen,
                	pingpong=pingpong
                )

                iptrs.clear(), idata.clear()
                optrs.clear(), odata.clear(), olen.clear()

                pingpong = not pingpong



    #######################################
    ### End of GemmOp:%0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "OutputStationary", tile_size = array<i64: 32, 64, 8, 4>} : (memref<32x64xbf16>, memref<64x128xbf16>, memref<32x128xbf16>) -> memref<32x128xbf16>
    #######################################

    await stream.synchronize()
