	.file	"forward_kernel_0_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	forward_kernel_0
	.type	forward_kernel_0, @function
forward_kernel_0:
	addi	sp,sp,-848
	sd	s10,760(sp)
	sd	s11,752(sp)
	mv	a4,a2
	sd	s0,840(sp)
	sd	s1,832(sp)
	sd	s2,824(sp)
	sd	s3,816(sp)
	sd	s4,808(sp)
	sd	s5,800(sp)
	sd	s6,792(sp)
	sd	s7,784(sp)
	sd	s8,776(sp)
	sd	s9,768(sp)
	mv	s11,a0
	mv	s10,a1
	lla	a5,.LANCHOR0
	addi	a3,sp,32
	lla	a2,.LANCHOR0+720
.L2:
	ld	a0,0(a5)
	ld	a1,8(a5)
	ld	a6,16(a5)
	sd	a0,0(a3)
	ld	a0,24(a5)
	sd	a1,8(a3)
	ld	a1,32(a5)
	sd	a6,16(a3)
	sd	a0,24(a3)
	sd	a1,32(a3)
	addi	a5,a5,40
	addi	a3,a3,40
	bne	a5,a2,.L2
	addi	a1,sp,32
	ld	a2,.LC1
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,268435456
	addi	a1,a5,1
	li	a2,134217728
	addi	a5,a5,3
	li	a3,16777216
	li	t0,8388608
	slli	a5,a5,13
	li	s9,67108864
	li	s1,2097152
	addi	a2,a2,7
	addi	s7,a3,9
	addi	s6,t0,1
	addi	s3,a3,7
	addi	s2,a3,15
	addi	t0,t0,5
	addi	a3,a3,11
	addi	s10,s10,96
	addi	s9,s9,3
	slli	a1,a1,13
	slli	a2,a2,14
	sd	a5,24(sp)
	addi	s1,s1,1
	li	a5,1
	li	s5,15
	li	s4,65536
	slli	t6,a3,13
	mv	s0,a4
	li	t2,0
	slli	s9,s9,15
	sd	a1,8(sp)
	sd	a2,16(sp)
	slli	a5,a5,37
	li	a7,110592
	slli	s1,s1,16
	li	a6,36864
	slli	s7,s7,13
	slli	s6,s6,14
	slli	s5,s5,35
	addi	s4,s4,-1
	slli	s3,s3,13
	slli	t0,t0,14
	slli	s2,s2,13
	li	s8,224
	mv	a3,s10
.L8:
	mv	a1,s11
	mv	a2,s9
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
	ld	a2,16(sp)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,24(sp)
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	s10,96
	mv	t5,a3
	mv	t4,s0
	mv	t3,a4
	li	t1,45056
	li	a0,53248
	sd	s0,0(sp)
.L7:
	mv	a1,t3
	mv	a2,a5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	s0,106496
	addi	a1,t5,-96
.L3:
	or	a2,s0,a5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	s0,s0,32
	addi	a1,a1,256
	bne	s0,a7,.L3
	addi	a1,t3,32
	mv	a2,s1
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	s0,32768
	addi	a1,t5,-64
.L4:
	or	a2,s0,a5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	s0,s0,32
	addi	a1,a1,256
	bne	s0,a6,.L4
	addi	a1,t3,64
	mv	a2,s7
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	s0,40960
	addi	a1,t5,-32
.L5:
	or	a2,s0,a5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	s0,s0,32
	addi	a1,a1,256
	bne	s0,t1,.L5
	addi	a1,t3,96
	mv	a2,s6
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	s0,49152
	mv	a1,t5
.L6:
	or	a2,s0,a5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	s0,s0,32
	addi	a1,a1,256
	bne	s0,a0,.L6
	li	a1,0
	mv	a2,s5
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s4
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,t4
	mv	a2,s3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,t4,32
	mv	a2,t0
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,t4,64
	mv	a2,s2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,t4,96
	mv	a2,t6
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	t3,t3,128
	addi	t4,t4,128
	addi	t5,t5,128
	beq	s10,s8,.L18
	li	s10,224
	j	.L7
.L18:
	ld	s0,0(sp)
	addiw	t2,t2,8
	li	a2,512
	addi	s11,s11,512
	addi	a4,a4,256
	addi	s0,s0,32
	bne	t2,a2,.L8
	ld	s0,840(sp)
	ld	s1,832(sp)
	ld	s2,824(sp)
	ld	s3,816(sp)
	ld	s4,808(sp)
	ld	s5,800(sp)
	ld	s6,792(sp)
	ld	s7,784(sp)
	ld	s8,776(sp)
	ld	s9,768(sp)
	ld	s10,760(sp)
	ld	s11,752(sp)
	addi	sp,sp,848
	jr	ra
	.size	forward_kernel_0, .-forward_kernel_0
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36031889395548160
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	10240
	.half	0
	.half	8
	.half	-8126
	.half	71
	.half	9
	.half	0
	.half	256
	.half	10
	.half	0
	.half	0
	.half	11
	.half	0
	.half	0
	.half	16
	.half	66
	.half	64
	.half	17
	.half	0
	.half	768
	.half	18
	.half	0
	.half	0
	.half	19
	.half	14336
	.half	0
	.half	24
	.half	-8126
	.half	71
	.half	25
	.half	0
	.half	256
	.half	26
	.half	0
	.half	0
	.half	27
	.half	4096
	.half	0
	.half	32
	.half	66
	.half	64
	.half	33
	.half	0
	.half	256
	.half	34
	.half	0
	.half	0
	.half	35
	.half	6144
	.half	0
	.half	40
	.half	66
	.half	64
	.half	41
	.half	0
	.half	-27392
	.half	42
	.half	0
	.half	0
	.half	43
	.half	2048
	.half	1
	.half	48
	.half	578
	.half	71
	.half	49
	.half	0
	.half	256
	.half	50
	.half	0
	.half	0
	.half	51
	.half	4096
	.half	1
	.half	56
	.half	578
	.half	71
	.half	57
	.half	0
	.half	256
	.half	58
	.half	0
	.half	0
	.half	59
	.half	0
	.half	1
	.half	64
	.half	578
	.half	71
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
	.half	0
	.half	0
	.half	96
	.half	1024
	.half	0
	.half	104
	.half	272
	.half	0
	.half	112
	.half	-32768
	.half	16
	.half	120
	.half	-32768
	.half	0
	.half	128
	.half	0
	.half	0
	.half	136
	.half	2062
	.half	40
	.half	169
	.half	2574
	.half	24
	.half	177
	.half	525
	.half	34
	.half	185
	.half	1037
	.half	38
	.half	193
	.half	0
	.half	24576
	.half	232
	.half	4
	.half	0
	.half	233
	.half	0
	.half	16384
	.half	240
	.half	0
	.half	16384
	.half	248
	.half	1
	.half	16384
	.half	256
	.half	128
	.half	384
	.half	264
	.half	3
	.half	0
	.half	265
	.half	0
	.half	512
	.half	272
	.half	16
	.half	4
	.half	321
	.half	0
	.half	1024
	.half	322
	.half	384
	.half	264
	.half	323
	.half	0
	.half	0
	.half	324
	.half	16
	.half	4
	.half	329
	.half	0
	.half	1024
	.half	330
	.half	512
	.half	264
	.half	331
	.half	0
	.half	0
	.half	332
	.half	3
	.half	0
	.half	377
	.half	1
	.half	0
	.half	401
	.half	0
	.half	-32768
	.half	408
	.half	3
	.half	0
	.half	409
	.half	4096
	.half	0
	.half	416
	.half	2062
	.half	56
	.half	465
	.half	16
	.half	2
	.half	489
	.half	0
	.half	1024
	.half	490
	.half	448
	.half	264
	.half	491
	.half	0
	.half	0
	.half	492
	.half	3
	.half	0
	.half	521
	.half	512
	.half	0
	.half	536
	.half	0
	.half	8
	.half	544
	.half	0
	.half	0
	.half	545
	.half	-16384
	.half	3076
	.half	552
	.half	0
	.half	128
	.half	560
	.half	2574
	.half	52
	.half	609
	.half	16
	.half	4
	.half	617
	.half	0
	.half	1024
	.half	618
	.half	448
	.half	264
	.half	619
	.half	0
	.half	0
	.half	620
	.half	77
	.half	24
	.half	625
	.half	13
	.half	56
	.half	633
	.half	8192
	.half	0
	.half	664
	.half	0
	.half	16
	.half	672
	.half	64
	.half	0
	.half	680
	.half	768
	.half	0
	.half	688
	.half	0
	.half	0
	.half	704
	.half	0
	.half	0
	.half	712
	.half	6144
	.half	0
	.half	728
	.half	66
	.half	64
	.half	729
	.half	0
	.half	-27392
	.half	730
	.half	512
	.half	0
	.half	731
	.half	0
	.half	0
	.half	736
	.half	66
	.half	64
	.half	737
	.half	0
	.half	768
	.half	738
	.half	0
	.half	0
	.half	739
	.half	4096
	.half	0
	.half	744
	.half	66
	.half	64
	.half	745
	.half	0
	.half	-27392
	.half	746
	.half	512
	.half	0
	.half	747
	.half	2048
	.half	0
	.half	752
	.half	66
	.half	64
	.half	753
	.half	0
	.half	256
	.half	754
	.half	0
	.half	0
	.half	755
	.half	6144
	.half	0
	.half	760
	.half	66
	.half	64
	.half	761
	.half	0
	.half	-27392
	.half	762
	.half	0
	.half	0
	.half	763
	.half	2048
	.half	1
	.half	768
	.half	578
	.half	71
	.half	769
	.half	0
	.half	256
	.half	770
	.half	0
	.half	0
	.half	771
	.half	12288
	.half	0
	.half	776
	.half	-8126
	.half	71
	.half	777
	.half	0
	.half	256
	.half	778
	.half	0
	.half	0
	.half	779
	.half	8192
	.half	0
	.half	784
	.half	-8126
	.half	71
	.half	785
	.half	0
	.half	256
	.half	786
	.half	0
	.half	0
	.half	787
	.ident	"GCC: (g2ee5e430018) 12.2.0"
