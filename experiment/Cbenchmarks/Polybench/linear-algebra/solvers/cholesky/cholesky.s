	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.file	"LLVMDialectModule"
	.globl	cholesky                        # -- Begin function cholesky
	.p2align	1
	.type	cholesky,@function
cholesky:                               # @cholesky
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -80
	.cfi_def_cfa_offset 80
	sd	ra, 72(sp)                      # 8-byte Folded Spill
	sd	s0, 64(sp)                      # 8-byte Folded Spill
	sd	s1, 56(sp)                      # 8-byte Folded Spill
	sd	s2, 48(sp)                      # 8-byte Folded Spill
	sd	s3, 40(sp)                      # 8-byte Folded Spill
	sd	s4, 32(sp)                      # 8-byte Folded Spill
	sd	s5, 24(sp)                      # 8-byte Folded Spill
	sd	s6, 16(sp)                      # 8-byte Folded Spill
	sd	s7, 8(sp)                       # 8-byte Folded Spill
	sd	s8, 0(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	mv	s2, a0
	li	s7, 0
	lui	a0, 2
	addiw	s3, a0, -188
	addiw	s4, a0, -192
	li	s5, 2000
	mv	s6, s2
	j	.LBB0_2
.LBB0_1:                                # %._crit_edge
                                        #   in Loop: Header=BB0_2 Depth=1
	mv	a0, s2
	mv	a1, s7
	call	cholesky_kernel_1@plt
	mul	a0, s7, s3
	add	a0, a0, s2
	flw	fa5, 0(a0)
	fsqrt.s	fa5, fa5
	fsw	fa5, 0(a0)
	addi	s7, s7, 1
	add	s6, s6, s4
	beq	s7, s5, .LBB0_5
.LBB0_2:                                # %.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	beqz	s7, .LBB0_1
# %bb.3:                                # %.lr.ph
                                        #   in Loop: Header=BB0_2 Depth=1
	li	s0, 0
	mv	s8, s2
	mv	s1, s6
.LBB0_4:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mv	a0, s2
	mv	a1, s7
	mv	a2, s0
	call	cholesky_kernel_0@plt
	flw	fa5, 0(s8)
	flw	fa4, 0(s1)
	fdiv.s	fa5, fa4, fa5
	fsw	fa5, 0(s1)
	addi	s0, s0, 1
	addi	s1, s1, 4
	add	s8, s8, s3
	bne	s7, s0, .LBB0_4
	j	.LBB0_1
.LBB0_5:
	ld	ra, 72(sp)                      # 8-byte Folded Reload
	ld	s0, 64(sp)                      # 8-byte Folded Reload
	ld	s1, 56(sp)                      # 8-byte Folded Reload
	ld	s2, 48(sp)                      # 8-byte Folded Reload
	ld	s3, 40(sp)                      # 8-byte Folded Reload
	ld	s4, 32(sp)                      # 8-byte Folded Reload
	ld	s5, 24(sp)                      # 8-byte Folded Reload
	ld	s6, 16(sp)                      # 8-byte Folded Reload
	ld	s7, 8(sp)                       # 8-byte Folded Reload
	ld	s8, 0(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 80
	ret
.Lfunc_end0:
	.size	cholesky, .Lfunc_end0-cholesky
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
