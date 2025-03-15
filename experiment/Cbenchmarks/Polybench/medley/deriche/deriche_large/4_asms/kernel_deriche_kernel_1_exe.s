	.file	"kernel_deriche_kernel_1_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	deriche_kernel_1
	.type	deriche_kernel_1, @function
deriche_kernel_1:
	addi	sp,sp,-544
	li	a6,524288
	mv	t1,a5
	sd	s7,480(sp)
	mv	t4,a2
	li	s7,566231040
	mv	a2,a4
	addi	a6,a6,1
	li	a4,35389440
	add	t3,t1,a4
	addi	s7,s7,1
	slli	a4,a6,16
	sd	a4,8(sp)
	slli	a4,s7,15
	li	t2,4194304
	li	a7,2097152
	sd	s0,536(sp)
	sd	a4,16(sp)
	li	s0,1048576
	li	a4,8192
	addi	a5,a7,5
	sd	s1,528(sp)
	sd	s2,520(sp)
	li	s1,4096
	sd	s3,512(sp)
	sd	s4,504(sp)
	addi	s3,t2,5
	sd	s9,464(sp)
	sd	s10,456(sp)
	addi	s9,a7,1
	addi	s10,t2,1
	sd	s11,448(sp)
	addi	t2,t2,3
	li	s11,1
	li	s4,69
	addi	s0,s0,3
	li	s2,-8192
	addi	a4,a4,448
	sd	s5,496(sp)
	sd	s6,488(sp)
	sd	s8,472(sp)
	addi	t6,s1,224
	mv	t5,a1
	lla	a7,.LANCHOR0+400
	slli	s11,s11,35
	slli	s10,s10,13
	slli	s9,s9,14
	ld	s6,.LC1
	slli	s4,s4,32
	addi	s1,s1,1855
	ld	s5,.LC2
	slli	s3,s3,13
	slli	s0,s0,15
	slli	a5,a5,14
	slli	t2,t2,13
	addi	s2,s2,-448
	sd	a4,24(sp)
	mv	s8,a2
.L2:
	li	a6,0
	add	t0,t1,t6
	add	s8,s8,t6
.L4:
	mv	a1,t4
	mv	a2,s11
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a0
	mv	a2,s10
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a3
	mv	a2,s9
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t5
	ld	a2,8(sp)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,t0,a6
	ld	a2,16(sp)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LANCHOR0
	addi	a4,sp,32
.L3:
	ld	a1,0(a2)
	ld	s7,8(a2)
	addi	a2,a2,40
	sd	a1,0(a4)
	sd	s7,8(a4)
	ld	s7,-24(a2)
	ld	a1,-16(a2)
	addi	a4,a4,40
	sd	s7,-24(a4)
	sd	a1,-16(a4)
	ld	a1,-8(a2)
	sd	a1,-8(a4)
	bne	a2,a7,.L3
	ld	s7,0(a7)
	lw	a1,8(a7)
	lhu	a2,12(a7)
	sd	s7,0(a4)
	sw	a1,8(a4)
	sh	a2,12(a4)
	addi	a1,sp,32
	mv	a2,s6
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s4
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s1
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,s8,a6
	mv	a2,s5
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t5
	mv	a2,s3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a3
	mv	a2,s0
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a0
	mv	a2,a5
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t4
	mv	a2,t2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	sub	a6,a6,t6
	bne	a6,s2,.L4
	ld	a4,24(sp)
	add	t1,t1,a4
	bne	t3,t1,.L2
	ld	s0,536(sp)
	ld	s1,528(sp)
	ld	s2,520(sp)
	ld	s3,512(sp)
	ld	s4,504(sp)
	ld	s5,496(sp)
	ld	s6,488(sp)
	ld	s7,480(sp)
	ld	s8,472(sp)
	ld	s9,464(sp)
	ld	s10,456(sp)
	ld	s11,448(sp)
	addi	sp,sp,544
	jr	ra
	.size	deriche_kernel_1, .-deriche_kernel_1
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36030575135555584
	.align	3
.LC2:
	.dword	18554258792448
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	6144
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
	.half	256
	.half	18
	.half	0
	.half	0
	.half	19
	.half	4096
	.half	-8192
	.half	24
	.half	16
	.half	0
	.half	25
	.half	0
	.half	256
	.half	26
	.half	0
	.half	0
	.half	27
	.half	2048
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
	.half	2048
	.half	-8192
	.half	40
	.half	16
	.half	0
	.half	41
	.half	0
	.half	-30976
	.half	42
	.half	0
	.half	0
	.half	43
	.half	-8192
	.half	-7169
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
	.half	2
	.half	0
	.half	88
	.half	512
	.half	2
	.half	96
	.half	512
	.half	2
	.half	104
	.half	0
	.half	2
	.half	112
	.half	0
	.half	0
	.half	120
	.half	5908
	.half	-16836
	.half	152
	.half	13
	.half	4
	.half	153
	.half	24616
	.half	15850
	.half	168
	.half	13
	.half	16
	.half	169
	.half	17661
	.half	16215
	.half	176
	.half	13
	.half	16
	.half	177
	.half	0
	.half	0
	.half	232
	.half	0
	.half	0
	.half	240
	.half	3
	.half	0
	.half	241
	.half	0
	.half	0
	.half	248
	.half	14
	.half	34
	.half	305
	.half	526
	.half	38
	.half	313
	.half	0
	.half	0
	.half	384
	.half	3
	.half	0
	.half	385
	.half	1038
	.half	68
	.half	449
	.half	1
	.half	0
	.half	521
	.half	0
	.half	0
	.half	528
	.half	3
	.half	0
	.half	529
	.half	17816
	.half	-16613
	.half	600
	.half	13
	.half	64
	.half	601
	.half	8960
	.half	0
	.half	664
	.half	0
	.half	48
	.half	672
	.half	2
	.half	32
	.half	680
	.half	512
	.half	0
	.half	688
	.half	-6144
	.half	-7169
	.half	728
	.half	16
	.half	0
	.half	729
	.half	0
	.half	-28416
	.half	730
	.half	512
	.half	0
	.half	731
	.half	4096
	.half	-8192
	.half	736
	.half	16
	.half	0
	.half	737
	.half	0
	.half	-28416
	.half	738
	.half	0
	.half	0
	.half	739
	.half	0
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
	.half	0
	.half	-8192
	.half	760
	.half	16
	.half	0
	.half	761
	.half	0
	.half	-29440
	.half	762
	.half	0
	.half	0
	.half	763
	.ident	"GCC: (g2ee5e430018) 12.2.0"
