	.file	"kernel_deriche_kernel_4_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	deriche_kernel_4
	.type	deriche_kernel_4, @function
deriche_kernel_4:
	li	a7,524288
	li	a6,1048576
	mv	t2,a2
	addi	a2,a7,1
	addi	sp,sp,-608
	addi	t1,a6,1
	slli	a2,a2,16
	sd	a2,8(sp)
	slli	a2,t1,15
	sd	s0,600(sp)
	sd	s1,592(sp)
	li	s0,2097152
	li	s1,4194304
	sd	s2,584(sp)
	li	t5,17690624
	li	s2,1132462080
	sd	a2,16(sp)
	li	a2,-8192
	sd	s3,576(sp)
	sd	s4,568(sp)
	addi	s3,s0,3
	addi	s4,s1,15
	sd	s5,560(sp)
	sd	s6,552(sp)
	sd	s8,536(sp)
	sd	s9,528(sp)
	addi	s8,a6,3
	sd	s11,512(sp)
	addi	s1,s1,1
	mv	s11,a4
	addi	s0,s0,1
	addi	t5,t5,-224
	li	t6,4423680
	li	a6,35381248
	li	s9,1
	li	s6,39
	li	s5,61440
	addi	s2,s2,7
	li	t4,4096
	addi	a2,a2,-448
	sd	s7,544(sp)
	sd	s10,520(sp)
	mv	a4,a1
	mv	s10,a5
	add	t5,s11,t5
	addi	t6,t6,-1080
	addi	a6,a6,-448
	lla	a5,.LANCHOR0+448
	slli	s9,s9,35
	slli	s8,s8,15
	ld	s7,.LC3
	slli	s6,s6,33
	addi	s5,s5,702
	slli	s2,s2,14
	slli	s4,s4,13
	slli	s1,s1,13
	slli	s3,s3,14
	slli	s0,s0,14
	addi	t4,t4,224
	sd	a2,24(sp)
.L2:
	mv	a7,a6
	li	t3,0
.L4:
	mv	a1,a0
	mv	a2,s9
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t2
	mv	a2,s8
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a4
	ld	a2,8(sp)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a3
	ld	a2,16(sp)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LC1
	add	a1,s10,a7
	ld	a2,0(a2)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LC2
	add	a1,s11,a7
	ld	a2,0(a2)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LANCHOR0
	addi	a1,sp,40
.L3:
	ld	t0,0(a2)
	ld	t1,8(a2)
	addi	a2,a2,32
	sd	t0,0(a1)
	sd	t1,8(a1)
	ld	t1,-16(a2)
	ld	t0,-8(a2)
	addi	a1,a1,32
	sd	t1,-16(a1)
	sd	t0,-8(a1)
	bne	a2,a5,.L3
	ld	t0,0(a2)
	ld	t1,8(a2)
	lw	a2,16(a2)
	sd	t0,0(a1)
	sd	t1,8(a1)
	sw	a2,16(a1)
	addi	a1,sp,40
	mv	a2,s7
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s6
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s5
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,t5,t3
	mv	a2,s2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a3
	mv	a2,s4
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a4
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t2
	mv	a2,s3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,a0
	mv	a2,s0
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a7,a7,t4
	beq	t3,t4,.L10
	mv	t3,t4
	j	.L4
.L10:
	ld	a2,24(sp)
	addiw	t6,t6,-1080
	sub	t5,t5,t3
	add	a6,a6,a2
	li	a2,-1080
	bne	t6,a2,.L2
	ld	s0,600(sp)
	ld	s1,592(sp)
	ld	s2,584(sp)
	ld	s3,576(sp)
	ld	s4,568(sp)
	ld	s5,560(sp)
	ld	s6,552(sp)
	ld	s7,544(sp)
	ld	s8,536(sp)
	ld	s9,528(sp)
	ld	s10,520(sp)
	ld	s11,512(sp)
	addi	sp,sp,608
	jr	ra
	.size	deriche_kernel_4, .-deriche_kernel_4
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	18554258825216
	.align	3
.LC2:
	.dword	18554258759680
	.align	3
.LC3:
	.dword	36030807063789568
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
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
	.half	-29440
	.half	26
	.half	512
	.half	0
	.half	27
	.half	2048
	.half	-8192
	.half	32
	.half	16
	.half	0
	.half	33
	.half	0
	.half	-30976
	.half	34
	.half	0
	.half	0
	.half	35
	.half	0
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
	.half	4096
	.half	-8192
	.half	48
	.half	16
	.half	0
	.half	49
	.half	0
	.half	-30464
	.half	50
	.half	512
	.half	0
	.half	51
	.half	10240
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
	.half	512
	.half	0
	.half	88
	.half	0
	.half	0
	.half	96
	.half	34
	.half	0
	.half	104
	.half	512
	.half	2
	.half	112
	.half	0
	.half	2
	.half	120
	.half	2
	.half	3
	.half	128
	.half	0
	.half	0
	.half	136
	.half	5908
	.half	-16836
	.half	160
	.half	13
	.half	16
	.half	161
	.half	17816
	.half	-16613
	.half	184
	.half	13
	.half	16
	.half	185
	.half	1
	.half	0
	.half	233
	.half	1
	.half	0
	.half	257
	.half	0
	.half	4096
	.half	272
	.half	8192
	.half	0
	.half	376
	.half	8192
	.half	0
	.half	400
	.half	0
	.half	4096
	.half	416
	.half	14
	.half	66
	.half	449
	.half	78
	.half	66
	.half	473
	.half	0
	.half	0
	.half	528
	.half	0
	.half	16384
	.half	536
	.half	4096
	.half	0
	.half	544
	.half	4
	.half	48
	.half	552
	.half	0
	.half	0
	.half	553
	.half	0
	.half	4224
	.half	560
	.half	24616
	.half	15850
	.half	600
	.half	13
	.half	6
	.half	601
	.half	526
	.half	34
	.half	617
	.half	17661
	.half	16215
	.half	632
	.half	13
	.half	8
	.half	633
	.half	16
	.half	0
	.half	672
	.half	13056
	.half	0
	.half	696
	.half	0
	.half	3
	.half	704
	.half	0
	.half	1
	.half	712
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
	.half	6144
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
	.half	12288
	.half	-8192
	.half	768
	.half	16
	.half	0
	.half	769
	.half	0
	.half	-27904
	.half	770
	.half	0
	.half	0
	.half	771
	.half	10240
	.half	-8192
	.half	776
	.half	16
	.half	0
	.half	777
	.half	0
	.half	256
	.half	778
	.half	0
	.half	0
	.half	779
	.half	0
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
