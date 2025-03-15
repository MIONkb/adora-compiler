	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.file	"LLVMDialectModule"
	.globl	atax                            # -- Begin function atax
	.p2align	1
	.type	atax,@function
atax:                                   # @atax
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -64
	.cfi_def_cfa_offset 64
	sd	ra, 56(sp)                      # 8-byte Folded Spill
	sd	s0, 48(sp)                      # 8-byte Folded Spill
	sd	s1, 40(sp)                      # 8-byte Folded Spill
	sd	s2, 32(sp)                      # 8-byte Folded Spill
	sd	s3, 24(sp)                      # 8-byte Folded Spill
	sd	s4, 16(sp)                      # 8-byte Folded Spill
	sd	s5, 8(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	mv	s4, a3
	mv	s2, a2
	mv	s3, a1
	mv	s1, a0
	li	s0, 0
	li	s5, 390
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	mv	a0, s1
	mv	a1, s0
	mv	a2, s3
	mv	a3, s4
	call	atax_kernel_0@plt
	mv	a0, s4
	mv	a1, s0
	mv	a2, s2
	mv	a3, s1
	call	atax_kernel_1@plt
	addi	s0, s0, 1
	bne	s0, s5, .LBB0_1
# %bb.2:
	ld	ra, 56(sp)                      # 8-byte Folded Reload
	ld	s0, 48(sp)                      # 8-byte Folded Reload
	ld	s1, 40(sp)                      # 8-byte Folded Reload
	ld	s2, 32(sp)                      # 8-byte Folded Reload
	ld	s3, 24(sp)                      # 8-byte Folded Reload
	ld	s4, 16(sp)                      # 8-byte Folded Reload
	ld	s5, 8(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 64
	ret
.Lfunc_end0:
	.size	atax, .Lfunc_end0-atax
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
