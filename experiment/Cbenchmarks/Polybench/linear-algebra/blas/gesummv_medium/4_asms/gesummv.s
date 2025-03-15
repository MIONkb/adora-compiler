	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.file	"LLVMDialectModule"
	.globl	gesummv                         # -- Begin function gesummv
	.p2align	1
	.type	gesummv,@function
gesummv:                                # @gesummv
	.cfi_startproc
# %bb.0:
	mv	a6, a4
	mv	a5, a1
	mv	a4, a0
	mv	a0, a2
	mv	a1, a6
	mv	a2, a4
	mv	a4, a5
	tail	gesummv_kernel_0@plt
.Lfunc_end0:
	.size	gesummv, .Lfunc_end0-gesummv
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
