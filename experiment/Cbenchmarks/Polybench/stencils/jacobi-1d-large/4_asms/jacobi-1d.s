	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.file	"LLVMDialectModule"
	.globl	jacobi_1d                       # -- Begin function jacobi_1d
	.p2align	1
	.type	jacobi_1d,@function
jacobi_1d:                              # @jacobi_1d
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -32
	.cfi_def_cfa_offset 32
	sd	ra, 24(sp)                      # 8-byte Folded Spill
	sd	s0, 16(sp)                      # 8-byte Folded Spill
	sd	s1, 8(sp)                       # 8-byte Folded Spill
	sd	s2, 0(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	mv	s2, a1
	mv	s1, a0
	li	s0, 500
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	mv	a0, s1
	mv	a1, s2
	call	jacobi_1d_kernel_0@plt
	mv	a0, s2
	mv	a1, s1
	call	jacobi_1d_kernel_1@plt
	addi	s0, s0, -1
	bnez	s0, .LBB0_1
# %bb.2:
	ld	ra, 24(sp)                      # 8-byte Folded Reload
	ld	s0, 16(sp)                      # 8-byte Folded Reload
	ld	s1, 8(sp)                       # 8-byte Folded Reload
	ld	s2, 0(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 32
	ret
.Lfunc_end0:
	.size	jacobi_1d, .Lfunc_end0-jacobi_1d
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
