	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.file	"LLVMDialectModule"
	.globl	deriche                         # -- Begin function deriche
	.p2align	1
	.type	deriche,@function
deriche:                                # @deriche
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -80
	.cfi_def_cfa_offset 80
	sd	ra, 72(sp)                      # 8-byte Folded Spill
	sd	s0, 64(sp)                      # 8-byte Folded Spill
	sd	s1, 56(sp)                      # 8-byte Folded Spill
	sd	s2, 48(sp)                      # 8-byte Folded Spill
	sd	s3, 40(sp)                      # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	mv	s2, a3
	mv	s1, a2
	mv	s3, a1
	mv	s0, a0
	addi	a0, sp, 8
	addi	a1, sp, 12
	mv	a2, sp
	mv	a3, s0
	mv	a4, s1
	call	deriche_kernel_0@plt
	addi	a0, sp, 32
	addi	a1, sp, 36
	addi	a2, sp, 16
	addi	a3, sp, 20
	mv	a4, s2
	mv	a5, s0
	call	deriche_kernel_1@plt
	mv	a0, s1
	mv	a1, s2
	mv	a2, s3
	call	deriche_kernel_2@plt
	addi	a0, sp, 4
	addi	a1, sp, 8
	addi	a2, sp, 12
	mv	a3, s3
	mv	a4, s1
	call	deriche_kernel_3@plt
	addi	a0, sp, 24
	addi	a1, sp, 28
	addi	a2, sp, 32
	addi	a3, sp, 36
	mv	a4, s2
	mv	a5, s3
	call	deriche_kernel_4@plt
	mv	a0, s1
	mv	a1, s2
	mv	a2, s3
	call	deriche_kernel_5@plt
	ld	ra, 72(sp)                      # 8-byte Folded Reload
	ld	s0, 64(sp)                      # 8-byte Folded Reload
	ld	s1, 56(sp)                      # 8-byte Folded Reload
	ld	s2, 48(sp)                      # 8-byte Folded Reload
	ld	s3, 40(sp)                      # 8-byte Folded Reload
	addi	sp, sp, 80
	ret
.Lfunc_end0:
	.size	deriche, .Lfunc_end0-deriche
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
