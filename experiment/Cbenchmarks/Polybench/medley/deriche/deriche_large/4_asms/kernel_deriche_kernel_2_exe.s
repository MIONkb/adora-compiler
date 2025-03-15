	.file	"kernel_deriche_kernel_2_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	deriche_kernel_2
	.type	deriche_kernel_2, @function
deriche_kernel_2:
	addi	sp,sp,-240
	li	a5,1132462080
	li	t3,4096
	li	a4,566231040
	addi	t6,a5,3
	sd	s0,232(sp)
	sd	s2,216(sp)
	sd	s3,208(sp)
	addi	t3,t3,224
	addi	a5,a5,1
	li	s0,35389440
	addi	a4,a4,1
	li	s3,-4096
	li	s2,135
	li	t0,15
	li	t1,8192
	sd	s1,224(sp)
	sd	s4,200(sp)
	sd	s5,192(sp)
	add	a6,a1,t3
	mv	a3,a2
	add	s0,a0,s0
	lla	t4,.LANCHOR0+160
	slli	t2,a4,15
	addi	s3,s3,-224
	ld	s5,.LC1
	slli	s2,s2,37
	ld	s4,.LC2
	ld	s1,.LC3
	slli	t0,t0,33
	slli	t6,t6,14
	slli	t5,a5,14
	addi	t1,t1,448
.L3:
	mv	a1,a0
	mv	a2,t2
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a6,s3
	mv	a2,s5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a0,t3
	mv	a2,s2
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a6
	mv	a2,s4
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	addi	a4,sp,8
.L2:
	ld	a2,0(a5)
	ld	a7,8(a5)
	ld	a1,16(a5)
	sd	a2,0(a4)
	ld	a2,24(a5)
	sd	a7,8(a4)
	sd	a1,16(a4)
	sd	a2,24(a4)
	addi	a5,a5,32
	addi	a4,a4,32
	bne	a5,t4,.L2
	ld	a1,0(a5)
	ld	a2,8(a5)
	lw	a5,16(a5)
	sd	a1,0(a4)
	sd	a2,8(a4)
	sw	a5,16(a4)
	addi	a1,sp,8
	mv	a2,s1
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,t0
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,119
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a3
	mv	a2,t6
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a3,a3,t3
	mv	a1,a3
	mv	a2,t5
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a0,a0,t1
	add	a6,a6,t1
	bne	s0,a0,.L3
	ld	s0,232(sp)
	ld	s1,224(sp)
	ld	s2,216(sp)
	ld	s3,208(sp)
	ld	s4,200(sp)
	ld	s5,192(sp)
	addi	sp,sp,240
	jr	ra
	.size	deriche_kernel_2, .-deriche_kernel_2
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	18554258759680
	.align	3
.LC2:
	.dword	18554258726912
	.align	3
.LC3:
	.dword	36029570113208320
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	10240
	.half	-8192
	.half	8
	.half	16
	.half	0
	.half	9
	.half	0
	.half	256
	.half	10
	.half	0
	.half	0
	.half	11
	.half	8192
	.half	-8192
	.half	16
	.half	16
	.half	0
	.half	17
	.half	0
	.half	256
	.half	18
	.half	0
	.half	0
	.half	19
	.half	12288
	.half	-8192
	.half	24
	.half	16
	.half	0
	.half	25
	.half	0
	.half	-30464
	.half	26
	.half	0
	.half	0
	.half	27
	.half	8192
	.half	-8192
	.half	40
	.half	16
	.half	0
	.half	41
	.half	0
	.half	256
	.half	42
	.half	0
	.half	0
	.half	43
	.half	10240
	.half	-8192
	.half	48
	.half	16
	.half	0
	.half	49
	.half	0
	.half	256
	.half	50
	.half	0
	.half	0
	.half	51
	.half	12288
	.half	-8192
	.half	56
	.half	16
	.half	0
	.half	57
	.half	0
	.half	-30464
	.half	58
	.half	0
	.half	0
	.half	59
	.half	512
	.half	0
	.half	88
	.half	16
	.half	0
	.half	96
	.half	512
	.half	0
	.half	120
	.half	16
	.half	0
	.half	128
	.half	14
	.half	18
	.half	161
	.half	14
	.half	18
	.half	193
	.ident	"GCC: (g2ee5e430018) 12.2.0"
