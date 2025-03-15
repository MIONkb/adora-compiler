	.file	"conv2d_kernel_0_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	conv2d_kernel_0
	.type	conv2d_kernel_0, @function
conv2d_kernel_0:
	mv	t3,a2
	li	a2,12288
	addi	sp,sp,-864
	addi	a2,a2,1168
	li	a4,15728640
	addi	a4,a4,1
	li	a6,308281344
	li	a3,62914560
	sd	a2,48(sp)
	li	a2,29
	sd	s3,832(sp)
	sd	s8,792(sp)
	sd	s11,768(sp)
	li	t6,9
	slli	s11,a4,15
	li	s3,4096
	li	a4,268472320
	addi	a6,a6,1
	slli	s8,a2,34
	addi	a3,a3,3
	li	a2,65536
	sd	s4,824(sp)
	sd	s5,816(sp)
	sd	s6,808(sp)
	sd	s7,800(sp)
	sd	s9,784(sp)
	sd	s10,776(sp)
	addi	s9,a2,-1
	sd	s0,856(sp)
	sd	s1,848(sp)
	sd	s2,840(sp)
	li	t1,0
	lla	s4,.LANCHOR0+680
	slli	a4,a4,24
	slli	t6,t6,36
	addi	s3,s3,-64
	slli	s5,a6,14
	ld	s7,.LC7
	slli	s10,a3,13
	mv	a2,a1
	mv	s6,a0
.L2:
	ld	a5,48(sp)
	sd	a2,16(sp)
	sd	t1,56(sp)
	mul	a3,t1,a5
	slli	a5,t1,4
	sub	a5,a5,t1
	slli	a5,a5,3
	add	a0,a5,t3
	sd	a0,40(sp)
	li	a5,0
	add	a6,a3,t3
	sd	a6,32(sp)
.L10:
	li	a3,232
	mul	a2,a5,a3
	ld	a1,40(sp)
	slli	a3,a5,4
	sub	a3,a3,a5
	slli	a3,a3,3
	add	s2,a3,a1
	slli	s0,a5,8
	sd	a5,24(sp)
	li	a0,0
	li	t0,65536
	mv	a3,a2
	ld	a2,32(sp)
	mv	a5,t3
	add	a3,a3,a2
.L7:
	add	a1,a3,a0
	mv	a2,s11
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	t2,1008
	sd	s2,0(sp)
	add	s1,s6,s0
	li	t4,40960
	li	t3,98304
	li	t1,73728
	li	a7,81920
	li	a6,8192
	li	t5,16384
	sd	a3,8(sp)
	mv	s2,a0
.L3:
	addi	a0,t2,-1008
	mv	a1,s1
.L4:
	addw	a2,t0,a0
	sext.w	a3,a0
	or	a2,a2,a4
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	or	a2,a3,a4
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addw	a2,t4,a3
	or	a2,a2,a4
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addw	a2,t3,a3
	or	a2,a2,a4
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addw	a2,t1,a3
	or	a2,a2,a4
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addw	a2,a7,a3
	or	a2,a2,a4
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addw	a3,a6,a3
	or	a2,a3,t6
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a0,a0,144
	addi	a1,a1,256
	bne	a0,t2,.L4
	addi	t2,a0,1008
	add	s1,s1,t5
	bne	t2,s3,.L3
	lla	a2,.LC1
	mv	a0,s2
	ld	a3,8(sp)
	ld	s2,0(sp)
	ld	a1,16(sp)
	ld	a2,0(a2)
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LC2
	ld	a2,0(a2)
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LC3
	ld	a2,0(a2)
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LC4
	ld	a2,0(a2)
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LC5
	ld	a2,0(a2)
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LC6
	ld	a2,0(a2)
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a2,s5
 #APP
# 41 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LANCHOR0
	addi	a6,sp,72
.L6:
	ld	a7,0(a2)
	ld	a1,8(a2)
	ld	t1,16(a2)
	sd	a7,0(a6)
	ld	a7,24(a2)
	sd	a1,8(a6)
	ld	a1,32(a2)
	sd	t1,16(a6)
	sd	a7,24(a6)
	sd	a1,32(a6)
	addi	a2,a2,40
	addi	a6,a6,40
	bne	a2,s4,.L6
	ld	a7,0(a2)
	ld	a2,8(a2)
	addi	a1,sp,72
	sd	a7,0(a6)
	sd	a2,8(a6)
	mv	a2,s7
 #APP
# 33 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s8
 #APP
# 57 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s9
 #APP
# 65 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,s2,a0
	mv	a2,s10
 #APP
# 49 "/home/jhlou/chipyard/generators/ADORA/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,116
	addi	s0,s0,116
	beq	a0,a2,.L16
	li	a0,116
	j	.L7
.L16:
	mv	t3,a5
	ld	a5,24(sp)
	li	a3,58
	addi	a5,a5,1
	bne	a5,a3,.L10
	ld	t1,56(sp)
	ld	a2,16(sp)
	li	a5,6
	addi	t1,t1,1
	addi	a2,a2,588
	bne	t1,a5,.L2
	ld	s0,856(sp)
	ld	s1,848(sp)
	ld	s2,840(sp)
	ld	s3,832(sp)
	ld	s4,824(sp)
	ld	s5,816(sp)
	ld	s6,808(sp)
	ld	s7,800(sp)
	ld	s8,792(sp)
	ld	s9,784(sp)
	ld	s10,776(sp)
	ld	s11,768(sp)
	addi	sp,sp,864
	jr	ra
	.size	conv2d_kernel_0, .-conv2d_kernel_0
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	4508650509017088
	.align	3
.LC2:
	.dword	4508650508959744
	.align	3
.LC3:
	.dword	4508650509025280
	.align	3
.LC4:
	.dword	4508650509033472
	.align	3
.LC5:
	.dword	4508650509000704
	.align	3
.LC6:
	.dword	4508650508967936
	.align	3
.LC7:
	.dword	36031786316333056
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	-4096
	.half	7168
	.half	8
	.half	448
	.half	24
	.half	9
	.half	-8262
	.half	257
	.half	10
	.half	0
	.half	0
	.half	11
	.half	-32768
	.half	7172
	.half	16
	.half	2304
	.half	-32744
	.half	17
	.half	-8552
	.half	257
	.half	18
	.half	0
	.half	0
	.half	19
	.half	6144
	.half	7168
	.half	24
	.half	0
	.half	-32744
	.half	25
	.half	-12288
	.half	-3839
	.half	26
	.half	4100
	.half	0
	.half	27
	.half	-30720
	.half	7172
	.half	32
	.half	2304
	.half	-32744
	.half	33
	.half	-8552
	.half	257
	.half	34
	.half	0
	.half	0
	.half	35
	.half	-4096
	.half	7168
	.half	40
	.half	448
	.half	24
	.half	41
	.half	-8262
	.half	257
	.half	42
	.half	0
	.half	0
	.half	43
	.half	-30720
	.half	7172
	.half	48
	.half	2304
	.half	-32744
	.half	49
	.half	-8552
	.half	257
	.half	50
	.half	0
	.half	0
	.half	51
	.half	-2048
	.half	7168
	.half	56
	.half	448
	.half	24
	.half	57
	.half	-8262
	.half	257
	.half	58
	.half	0
	.half	0
	.half	59
	.half	0
	.half	7168
	.half	64
	.half	0
	.half	-32744
	.half	65
	.half	-12288
	.half	257
	.half	66
	.half	0
	.half	0
	.half	67
	.half	0
	.half	0
	.half	88
	.half	1024
	.half	0
	.half	96
	.half	258
	.half	0
	.half	104
	.half	0
	.half	49
	.half	112
	.half	0
	.half	2
	.half	120
	.half	0
	.half	0
	.half	128
	.half	0
	.half	16
	.half	136
	.half	136
	.half	80
	.half	169
	.half	8
	.half	136
	.half	177
	.half	18442
	.half	280
	.half	185
	.half	4
	.half	0
	.half	241
	.half	0
	.half	128
	.half	248
	.half	192
	.half	0
	.half	256
	.half	3
	.half	0
	.half	257
	.half	8
	.half	0
	.half	264
	.half	0
	.half	384
	.half	272
	.half	3
	.half	0
	.half	273
	.half	0
	.half	256
	.half	280
	.half	10250
	.half	96
	.half	321
	.half	2058
	.half	88
	.half	329
	.half	0
	.half	24576
	.half	384
	.half	4096
	.half	0
	.half	392
	.half	192
	.half	1024
	.half	400
	.half	3
	.half	0
	.half	401
	.half	0
	.half	0
	.half	408
	.half	3
	.half	0
	.half	417
	.half	26
	.half	8
	.half	465
	.half	0
	.half	4096
	.half	466
	.half	19712
	.half	1029
	.half	467
	.half	0
	.half	0
	.half	468
	.half	6154
	.half	224
	.half	473
	.half	2058
	.half	264
	.half	481
	.half	6154
	.half	280
	.half	489
	.half	0
	.half	8192
	.half	528
	.half	0
	.half	16384
	.half	536
	.half	0
	.half	16416
	.half	544
	.half	-32768
	.half	0
	.half	552
	.half	8192
	.half	0
	.half	560
	.half	0
	.half	0
	.half	568
	.half	8
	.half	280
	.half	593
	.half	264
	.half	152
	.half	609
	.half	8
	.half	280
	.half	617
	.half	4106
	.half	80
	.half	625
	.half	2056
	.half	200
	.half	633
	.half	8
	.half	280
	.half	641
	.half	16
	.half	0
	.half	664
	.half	0
	.half	16
	.half	672
	.half	32
	.half	0
	.half	680
	.half	16
	.half	0
	.half	688
	.half	2
	.half	0
	.half	696
	.half	0
	.half	0
	.half	704
	.half	16
	.half	0
	.half	712
	.half	-30720
	.half	7172
	.half	728
	.half	2304
	.half	-32744
	.half	729
	.half	-8552
	.half	257
	.half	730
	.half	0
	.half	0
	.half	731
	.half	-28672
	.half	7172
	.half	736
	.half	2304
	.half	-32744
	.half	737
	.half	-8552
	.half	257
	.half	738
	.half	0
	.half	0
	.half	739
	.half	-2048
	.half	7168
	.half	744
	.half	448
	.half	24
	.half	745
	.half	-8262
	.half	257
	.half	746
	.half	0
	.half	0
	.half	747
	.half	-32768
	.half	7172
	.half	752
	.half	2304
	.half	-32744
	.half	753
	.half	-8552
	.half	257
	.half	754
	.half	0
	.half	0
	.half	755
	.half	-6144
	.half	7168
	.half	760
	.half	448
	.half	24
	.half	761
	.half	-8262
	.half	257
	.half	762
	.half	0
	.half	0
	.half	763
	.half	-4096
	.half	7168
	.half	768
	.half	448
	.half	24
	.half	769
	.half	-8262
	.half	257
	.half	770
	.half	0
	.half	0
	.half	771
	.half	-32768
	.half	7172
	.half	776
	.half	2304
	.half	-32744
	.half	777
	.half	-8552
	.half	257
	.half	778
	.half	0
	.half	0
	.half	779
	.half	-2048
	.half	7168
	.half	784
	.half	448
	.half	24
	.half	785
	.half	-8262
	.half	257
	.half	786
	.half	0
	.half	0
	.half	787
	.ident	"GCC: (g2ee5e430018) 12.2.0"
