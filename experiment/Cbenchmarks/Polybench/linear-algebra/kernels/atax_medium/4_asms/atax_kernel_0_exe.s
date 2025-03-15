	.file	"atax_kernel_0_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	atax_kernel_0
	.type	atax_kernel_0, @function
atax_kernel_0:
	li	a5,1640
	mulw	a5,a5,a1
	mv	a6,a1
	mv	a4,a2
	addi	sp,sp,-480
	ld	a2,.LC1
	add	a1,a0,a5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC2
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC3
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC4
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,859832320
	addi	a2,a5,13
	slli	a2,a2,13
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC5
	mv	a1,a4
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC6
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC7
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC8
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a2,a5,11
	slli	a2,a2,13
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	mv	a4,sp
	lla	a2,.LANCHOR0+448
.L2:
	ld	a1,0(a5)
	ld	a7,8(a5)
	ld	a0,16(a5)
	sd	a1,0(a4)
	ld	a1,24(a5)
	sd	a7,8(a4)
	sd	a0,16(a4)
	sd	a1,24(a4)
	addi	a5,a5,32
	addi	a4,a4,32
	bne	a5,a2,.L2
	ld	a0,0(a5)
	ld	a1,8(a5)
	ld	a2,16(a5)
	sd	a0,0(a4)
	lhu	a5,24(a5)
	sd	a1,8(a4)
	sd	a2,16(a4)
	sh	a5,24(a4)
	mv	a1,sp
	ld	a2,.LC9
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,79
	li	a1,0
	slli	a2,a2,32
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,65536
	addi	a1,a1,-80
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,4194304
	slliw	a1,a6,2
	addi	a2,a2,15
	add	a1,a3,a1
	slli	a2,a2,13
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,480
	jr	ra
	.size	atax_kernel_0, .-atax_kernel_0
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	4510643373834240
	.align	3
.LC2:
	.dword	4510643373801472
	.align	3
.LC3:
	.dword	4510643373768704
	.align	3
.LC4:
	.dword	4510643373809664
	.align	3
.LC5:
	.dword	4510643373817856
	.align	3
.LC6:
	.dword	4510643373850624
	.align	3
.LC7:
	.dword	4510643373776896
	.align	3
.LC8:
	.dword	4510643373785088
	.align	3
.LC9:
	.dword	36030832833593344
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	-24576
	.half	18432
	.half	40
	.half	1
	.half	0
	.half	41
	.half	0
	.half	256
	.half	42
	.half	0
	.half	0
	.half	43
	.half	-22528
	.half	18432
	.half	48
	.half	1
	.half	0
	.half	49
	.half	0
	.half	256
	.half	50
	.half	0
	.half	0
	.half	51
	.half	-20480
	.half	18432
	.half	64
	.half	1
	.half	0
	.half	65
	.half	0
	.half	256
	.half	66
	.half	0
	.half	0
	.half	67
	.half	0
	.half	48
	.half	104
	.half	0
	.half	1
	.half	112
	.half	-32256
	.half	0
	.half	120
	.half	0
	.half	2
	.half	128
	.half	0
	.half	0
	.half	136
	.half	13
	.half	38
	.half	185
	.half	13
	.half	18
	.half	193
	.half	0
	.half	24576
	.half	248
	.half	4288
	.half	0
	.half	256
	.half	1
	.half	0
	.half	265
	.half	46
	.half	66
	.half	329
	.half	0
	.half	4096
	.half	400
	.half	4
	.half	32
	.half	408
	.half	1038
	.half	38
	.half	473
	.half	1550
	.half	50
	.half	481
	.half	0
	.half	-32768
	.half	528
	.half	0
	.half	16384
	.half	536
	.half	-32704
	.half	4144
	.half	544
	.half	0
	.half	128
	.half	552
	.half	0
	.half	0
	.half	560
	.half	45
	.half	56
	.half	601
	.half	14
	.half	38
	.half	609
	.half	45
	.half	24
	.half	617
	.half	525
	.half	70
	.half	625
	.half	16
	.half	2
	.half	633
	.half	0
	.half	1024
	.half	634
	.half	8832
	.half	261
	.half	635
	.half	0
	.half	0
	.half	636
	.half	0
	.half	16
	.half	664
	.half	64
	.half	4
	.half	672
	.half	1
	.half	16
	.half	680
	.half	0
	.half	37
	.half	688
	.half	33
	.half	0
	.half	696
	.half	2
	.half	1
	.half	704
	.half	0
	.half	0
	.half	712
	.half	-20480
	.half	18432
	.half	728
	.half	1
	.half	0
	.half	729
	.half	0
	.half	256
	.half	730
	.half	0
	.half	0
	.half	731
	.half	-18432
	.half	18432
	.half	736
	.half	1
	.half	0
	.half	737
	.half	0
	.half	256
	.half	738
	.half	0
	.half	0
	.half	739
	.half	-24576
	.half	18432
	.half	744
	.half	1
	.half	0
	.half	745
	.half	0
	.half	256
	.half	746
	.half	0
	.half	0
	.half	747
	.half	-22528
	.half	18432
	.half	752
	.half	1
	.half	0
	.half	753
	.half	0
	.half	256
	.half	754
	.half	0
	.half	0
	.half	755
	.half	-24576
	.half	18432
	.half	760
	.half	1
	.half	0
	.half	761
	.half	0
	.half	256
	.half	762
	.half	0
	.half	0
	.half	763
	.half	-20480
	.half	18432
	.half	768
	.half	1
	.half	0
	.half	769
	.half	0
	.half	256
	.half	770
	.half	0
	.half	0
	.half	771
	.half	-22528
	.half	18432
	.half	776
	.half	1
	.half	0
	.half	777
	.half	0
	.half	256
	.half	778
	.half	0
	.half	0
	.half	779
	.half	6144
	.half	18432
	.half	784
	.half	1
	.half	0
	.half	785
	.half	0
	.half	-26880
	.half	786
	.half	0
	.half	0
	.half	787
	.ident	"GCC: (g2ee5e430018) 12.2.0"
