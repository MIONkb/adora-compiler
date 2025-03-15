	.file	"kernel_deriche_kernel_0_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	deriche_kernel_0
	.type	deriche_kernel_0, @function
deriche_kernel_0:
	addi	sp,sp,-480
	li	t1,4194304
	li	a7,524288
	li	a6,2097152
	li	a5,566231040
	sd	s0,472(sp)
	sd	s1,464(sp)
	addi	s0,t1,1
	sd	s2,456(sp)
	sd	s3,448(sp)
	sd	s9,400(sp)
	sd	s10,392(sp)
	addi	s9,t1,9
	sd	s11,384(sp)
	addi	t1,t1,13
	li	s11,35389440
	addi	a7,a7,1
	addi	a6,a6,5
	addi	a5,a5,3
	li	s3,31
	li	s2,40960
	li	s1,1
	li	t0,4096
	li	s10,8192
	sd	s4,440(sp)
	sd	s5,432(sp)
	sd	s6,424(sp)
	sd	s7,416(sp)
	sd	s8,408(sp)
	mv	t5,a1
	mv	t4,a2
	add	s11,a3,s11
	lla	t6,.LANCHOR0+360
	slli	s8,a7,16
	slli	s9,s9,13
	slli	s7,a6,14
	slli	s6,a5,15
	ld	s5,.LC1
	slli	s3,s3,33
	addi	s2,s2,-253
	ld	s4,.LC2
	slli	s1,s1,35
	slli	s0,s0,13
	slli	t1,t1,13
	addi	t0,t0,224
	addi	s10,s10,448
.L2:
	li	a7,0
.L4:
	mv	a1,t4
	mv	a2,s8
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t5
	mv	a2,s9
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a0
	mv	a2,s7
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a3,a7
	mv	a2,s6
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	addi	a6,sp,8
.L3:
	ld	t3,0(a5)
	ld	a1,8(a5)
	ld	a2,16(a5)
	sd	t3,0(a6)
	ld	t3,24(a5)
	sd	a1,8(a6)
	ld	a1,32(a5)
	sd	a2,16(a6)
	sd	t3,24(a6)
	sd	a1,32(a6)
	addi	a5,a5,40
	addi	a6,a6,40
	bne	a5,t6,.L3
	ld	a2,0(a5)
	lw	a5,8(a5)
	addi	a1,sp,8
	sd	a2,0(a6)
	sw	a5,8(a6)
	mv	a2,s5
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s3
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s2
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a4,a7
	mv	a2,s4
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a0
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t5
	mv	a2,s0
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t4
	mv	a2,t1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	beq	a7,t0,.L10
	mv	a7,t0
	j	.L4
.L10:
	add	a3,a3,s10
	add	a4,a4,a7
	bne	a3,s11,.L2
	ld	s0,472(sp)
	ld	s1,464(sp)
	ld	s2,456(sp)
	ld	s3,448(sp)
	ld	s4,440(sp)
	ld	s5,432(sp)
	ld	s6,424(sp)
	ld	s7,416(sp)
	ld	s8,408(sp)
	ld	s9,400(sp)
	ld	s10,392(sp)
	ld	s11,384(sp)
	addi	sp,sp,480
	jr	ra
	.size	deriche_kernel_0, .-deriche_kernel_0
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36030394746929152
	.align	3
.LC2:
	.dword	18554258808832
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	2048
	.half	-8192
	.half	8
	.half	16
	.half	0
	.half	9
	.half	0
	.half	-29952
	.half	10
	.half	512
	.half	0
	.half	11
	.half	0
	.half	-8192
	.half	16
	.half	16
	.half	0
	.half	17
	.half	0
	.half	-27392
	.half	18
	.half	0
	.half	0
	.half	19
	.half	50
	.half	0
	.half	88
	.half	0
	.half	3
	.half	96
	.half	0
	.half	4096
	.half	232
	.half	0
	.half	4096
	.half	240
	.half	0
	.half	3072
	.half	376
	.half	0
	.half	4224
	.half	384
	.half	1550
	.half	56
	.half	457
	.half	0
	.half	8192
	.half	520
	.half	256
	.half	12288
	.half	528
	.half	-32768
	.half	8192
	.half	536
	.half	0
	.half	0
	.half	537
	.half	-32768
	.half	4
	.half	544
	.half	0
	.half	12
	.half	552
	.half	0
	.half	128
	.half	560
	.half	17816
	.half	-16613
	.half	584
	.half	13
	.half	64
	.half	585
	.half	17661
	.half	16215
	.half	592
	.half	13
	.half	8
	.half	593
	.half	-19124
	.half	15841
	.half	600
	.half	13
	.half	6
	.half	601
	.half	526
	.half	20
	.half	609
	.half	526
	.half	20
	.half	617
	.half	13764
	.half	-16831
	.half	632
	.half	13
	.half	8
	.half	633
	.half	2
	.half	0
	.half	664
	.half	16
	.half	0
	.half	672
	.half	768
	.half	0
	.half	680
	.half	12288
	.half	0
	.half	696
	.half	0
	.half	3
	.half	704
	.half	0
	.half	1
	.half	712
	.half	2048
	.half	-8192
	.half	728
	.half	16
	.half	0
	.half	729
	.half	0
	.half	256
	.half	730
	.half	0
	.half	0
	.half	731
	.half	0
	.half	-8192
	.half	736
	.half	16
	.half	0
	.half	737
	.half	0
	.half	256
	.half	738
	.half	0
	.half	0
	.half	739
	.half	4096
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
	.half	14336
	.half	-8192
	.half	752
	.half	16
	.half	0
	.half	753
	.half	0
	.half	-27904
	.half	754
	.half	0
	.half	0
	.half	755
	.half	2048
	.half	-8192
	.half	760
	.half	16
	.half	0
	.half	761
	.half	0
	.half	-30464
	.half	762
	.half	512
	.half	0
	.half	763
	.half	8192
	.half	-8192
	.half	784
	.half	16
	.half	0
	.half	785
	.half	0
	.half	256
	.half	786
	.half	0
	.half	0
	.half	787
	.ident	"GCC: (g2ee5e430018) 12.2.0"
