	.file	"kernel_deriche_kernel_3_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	deriche_kernel_3
	.type	deriche_kernel_3, @function
deriche_kernel_3:
	li	a5,1132462080
	addi	sp,sp,-576
	addi	a5,a5,1
	slli	a5,a5,14
	sd	s0,568(sp)
	li	a6,566231040
	li	s0,524288
	li	a7,4194304
	li	t2,2097152
	addi	s0,s0,1
	addi	a6,a6,1
	sd	a5,8(sp)
	li	a5,8192
	li	t1,4096
	li	t3,0
	addi	a5,a5,448
	sd	s4,536(sp)
	sd	s6,520(sp)
	mv	s4,a4
	sd	s7,512(sp)
	addi	a4,a7,9
	addi	s7,a7,1
	sd	s8,504(sp)
	addi	a7,t2,3
	sd	s9,496(sp)
	addi	t2,t2,5
	mv	s9,a3
	slli	s6,s0,16
	slli	a3,a6,15
	li	s8,1
	li	s0,37
	sd	s1,560(sp)
	sd	s2,552(sp)
	sd	s3,544(sp)
	sd	s10,488(sp)
	sd	s11,480(sp)
	sd	a3,0(sp)
	addi	s2,t1,-529
	sd	a5,24(sp)
	sd	s5,528(sp)
	mv	a5,t3
	mv	t5,a1
	mv	t4,a2
	mv	t6,s4
	li	s11,0
	lla	t0,.LANCHOR0+440
	slli	s8,s8,35
	slli	s7,s7,13
	ld	s10,.LC1
	ld	s3,.LC2
	slli	s0,s0,33
	ld	s1,.LC3
	slli	a4,a4,13
	slli	a3,a7,14
	slli	t2,t2,14
	addi	t1,t1,224
	mv	t3,a0
.L2:
	mv	a7,a5
	li	a0,0
	sd	s11,16(sp)
.L4:
	mv	a1,t3
	mv	a2,s6
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t4
	mv	a2,s8
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t5
	mv	a2,s7
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,s9,a7
	ld	a2,0(sp)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,8(sp)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,s4,a7
	mv	a2,s10
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LANCHOR0
	addi	a6,sp,32
.L3:
	ld	s11,0(a2)
	ld	a1,8(a2)
	ld	s5,16(a2)
	sd	s11,0(a6)
	ld	s11,24(a2)
	sd	a1,8(a6)
	ld	a1,32(a2)
	sd	s5,16(a6)
	sd	s11,24(a6)
	sd	a1,32(a6)
	addi	a2,a2,40
	addi	a6,a6,40
	bne	a2,t0,.L3
	lw	s11,0(a2)
	addi	a1,sp,32
	mv	a2,s3
	sw	s11,0(a6)
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s0
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
	add	a1,t6,a0
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t5
	mv	a2,a4
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t4
	mv	a2,a3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t3
	mv	a2,t2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a7,a7,t1
	beq	a0,t1,.L10
	mv	a0,t1
	j	.L4
.L10:
	ld	a2,24(sp)
	ld	s11,16(sp)
	add	t6,t6,a0
	add	a5,a5,a2
	addiw	s11,s11,1080
	li	a2,4423680
	bne	s11,a2,.L2
	ld	s0,568(sp)
	ld	s1,560(sp)
	ld	s2,552(sp)
	ld	s3,544(sp)
	ld	s4,536(sp)
	ld	s5,528(sp)
	ld	s6,520(sp)
	ld	s7,512(sp)
	ld	s8,504(sp)
	ld	s9,496(sp)
	ld	s10,488(sp)
	ld	s11,480(sp)
	addi	sp,sp,576
	jr	ra
	.size	deriche_kernel_3, .-deriche_kernel_3
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	18554258759680
	.align	3
.LC2:
	.dword	36030703984574464
	.align	3
.LC3:
	.dword	18554258743296
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	12288
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
	.half	14336
	.half	-8192
	.half	16
	.half	16
	.half	0
	.half	17
	.half	0
	.half	-26880
	.half	18
	.half	512
	.half	0
	.half	19
	.half	0
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
	.half	4096
	.half	-8192
	.half	48
	.half	16
	.half	0
	.half	49
	.half	0
	.half	-30976
	.half	50
	.half	0
	.half	0
	.half	51
	.half	10240
	.half	-8192
	.half	56
	.half	16
	.half	0
	.half	57
	.half	0
	.half	256
	.half	58
	.half	0
	.half	0
	.half	59
	.half	8192
	.half	-8192
	.half	64
	.half	16
	.half	0
	.half	65
	.half	0
	.half	256
	.half	66
	.half	0
	.half	0
	.half	67
	.half	0
	.half	0
	.half	88
	.half	3
	.half	0
	.half	96
	.half	-32768
	.half	0
	.half	104
	.half	0
	.half	49
	.half	112
	.half	32
	.half	2
	.half	120
	.half	0
	.half	0
	.half	128
	.half	0
	.half	0
	.half	136
	.half	17816
	.half	-16613
	.half	160
	.half	13
	.half	4
	.half	161
	.half	1550
	.half	38
	.half	169
	.half	17661
	.half	16215
	.half	184
	.half	13
	.half	2
	.half	185
	.half	13764
	.half	-16831
	.half	200
	.half	13
	.half	4
	.half	201
	.half	3
	.half	0
	.half	233
	.half	16448
	.half	12
	.half	240
	.half	0
	.half	384
	.half	248
	.half	0
	.half	384
	.half	256
	.half	3
	.half	0
	.half	257
	.half	0
	.half	384
	.half	264
	.half	0
	.half	0
	.half	272
	.half	1038
	.half	52
	.half	305
	.half	142
	.half	18
	.half	313
	.half	512
	.half	0
	.half	376
	.half	3
	.half	0
	.half	377
	.half	3
	.half	0
	.half	401
	.half	0
	.half	1024
	.half	520
	.half	3
	.half	0
	.half	521
	.half	3
	.half	0
	.half	545
	.half	-19124
	.half	15841
	.half	584
	.half	13
	.half	64
	.half	585
	.half	2
	.half	48
	.half	664
	.half	512
	.half	0
	.half	672
	.half	8192
	.half	0
	.half	688
	.half	0
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
	.half	4096
	.half	-8192
	.half	744
	.half	16
	.half	0
	.half	745
	.half	0
	.half	-29440
	.half	746
	.half	0
	.half	0
	.half	747
	.half	2048
	.half	-8192
	.half	752
	.half	16
	.half	0
	.half	753
	.half	0
	.half	-29440
	.half	754
	.half	512
	.half	0
	.half	755
	.ident	"GCC: (g2ee5e430018) 12.2.0"
