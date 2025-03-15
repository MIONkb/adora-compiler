	.file	"kernel_deriche_kernel_5_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	deriche_kernel_5
	.type	deriche_kernel_5, @function
deriche_kernel_5:
	addi	sp,sp,-288
	li	t5,4096
	li	a5,566231040
	li	a4,283115520
	addi	t3,t5,224
	sd	s0,280(sp)
	sd	s4,248(sp)
	sd	s5,240(sp)
	sd	s6,232(sp)
	li	s0,35389440
	addi	s6,a5,1
	li	s5,-4096
	addi	a5,a5,3
	addi	a4,a4,1
	li	s4,135
	li	t0,37
	li	t1,8192
	sd	s1,272(sp)
	sd	s2,264(sp)
	sd	s3,256(sp)
	add	a6,a1,t3
	mv	a3,a2
	add	s0,a0,s0
	lla	t4,.LANCHOR0+200
	slli	s6,s6,15
	addi	s5,s5,-224
	slli	t2,a4,16
	slli	s4,s4,37
	slli	t6,a5,15
	ld	s3,.LC1
	slli	t0,t0,32
	addi	t5,t5,1084
	ld	s2,.LC2
	ld	s1,.LC3
	addi	t1,t1,448
.L3:
	mv	a1,a0
	mv	a2,s6
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a6,s5
	mv	a2,t2
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a0,t3
	mv	a2,s4
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a6
	mv	a2,t6
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	mv	a4,sp
.L2:
	ld	a1,0(a5)
	ld	a2,8(a5)
	ld	a7,16(a5)
	sd	a1,0(a4)
	ld	a1,24(a5)
	sd	a2,8(a4)
	ld	a2,32(a5)
	sd	a7,16(a4)
	sd	a1,24(a4)
	sd	a2,32(a4)
	addi	a5,a5,40
	addi	a4,a4,40
	bne	a5,t4,.L2
	ld	a7,0(a5)
	ld	a1,8(a5)
	lw	a2,16(a5)
	sd	a7,0(a4)
	lhu	a5,20(a5)
	sd	a1,8(a4)
	sw	a2,16(a4)
	sh	a5,20(a4)
	mv	a1,sp
	mv	a2,s3
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
	li	a2,0
	mv	a1,t5
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a3
	mv	a2,s2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a3,a3,t3
	mv	a1,a3
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a0,a0,t1
	add	a6,a6,t1
	bne	s0,a0,.L3
	ld	s0,280(sp)
	ld	s1,272(sp)
	ld	s2,264(sp)
	ld	s3,256(sp)
	ld	s4,248(sp)
	ld	s5,240(sp)
	ld	s6,232(sp)
	addi	sp,sp,288
	jr	ra
	.size	deriche_kernel_5, .-deriche_kernel_5
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36029750501834752
	.align	3
.LC2:
	.dword	18554258726912
	.align	3
.LC3:
	.dword	18554258759680
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	10240
	.half	-8192
	.half	24
	.half	16
	.half	0
	.half	25
	.half	0
	.half	-29952
	.half	26
	.half	512
	.half	0
	.half	27
	.half	8192
	.half	-8192
	.half	32
	.half	16
	.half	0
	.half	33
	.half	0
	.half	256
	.half	34
	.half	0
	.half	0
	.half	35
	.half	10240
	.half	-8192
	.half	40
	.half	16
	.half	0
	.half	41
	.half	0
	.half	-29952
	.half	42
	.half	0
	.half	0
	.half	43
	.half	8192
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
	.half	515
	.half	0
	.half	104
	.half	16
	.half	48
	.half	112
	.half	0
	.half	0
	.half	120
	.half	526
	.half	24
	.half	177
	.half	0
	.half	0
	.half	248
	.half	2
	.half	8
	.half	256
	.half	14
	.half	52
	.half	321
	.half	512
	.half	0
	.half	392
	.half	0
	.half	4096
	.half	400
	.half	0
	.half	4096
	.half	536
	.half	0
	.half	4096
	.half	544
	.half	0
	.half	4
	.half	680
	.half	0
	.half	0
	.half	688
	.half	8192
	.half	-8192
	.half	744
	.half	16
	.half	0
	.half	745
	.half	0
	.half	256
	.half	746
	.half	0
	.half	0
	.half	747
	.half	8192
	.half	-8192
	.half	760
	.half	16
	.half	0
	.half	761
	.half	0
	.half	256
	.half	762
	.half	0
	.half	0
	.half	763
	.ident	"GCC: (g2ee5e430018) 12.2.0"
