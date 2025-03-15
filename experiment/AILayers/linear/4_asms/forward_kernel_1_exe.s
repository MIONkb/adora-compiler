	.file	"forward_kernel_1_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	forward_kernel_1
	.type	forward_kernel_1, @function
forward_kernel_1:
	li	a6,536870912
	addi	a4,a6,1
	mv	a5,a1
	mv	a3,a2
	addi	sp,sp,-368
	mv	a1,a0
	slli	a2,a4,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a4,134217728
	addi	a2,a4,5
	mv	a1,a5
	slli	a2,a2,13
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,4096
	addi	a2,a6,3
	add	a1,a0,a1
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a2,a4,13
	mv	a1,a5
	slli	a2,a2,13
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1073741824
	li	a1,8192
	addi	a2,a2,3
	add	a1,a0,a1
	slli	a2,a2,14
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1
	mv	a1,a5
	slli	a2,a2,40
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,268435456
	li	a1,12288
	addi	a2,a2,1
	add	a1,a0,a1
	slli	a2,a2,16
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a2,a4,9
	mv	a1,a5
	slli	a2,a2,13
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	addi	a4,sp,8
	lla	a2,.LANCHOR0+360
.L2:
	ld	a0,0(a5)
	ld	a1,8(a5)
	ld	a6,16(a5)
	sd	a0,0(a4)
	ld	a0,24(a5)
	sd	a1,8(a4)
	ld	a1,32(a5)
	sd	a6,16(a4)
	sd	a0,24(a4)
	sd	a1,32(a4)
	addi	a5,a5,40
	addi	a4,a4,40
	bne	a5,a2,.L2
	addi	a1,sp,8
	ld	a2,.LC1
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,15
	li	a1,0
	slli	a2,a2,34
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,28672
	addi	a1,a1,2044
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC2
	mv	a1,a3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,1073741824
	addi	a2,a5,7
	li	a1,4096
	add	a1,a3,a1
	slli	a2,a2,14
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,8192
	add	a1,a3,a1
	ld	a2,.LC3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,12288
	addi	a2,a5,5
	add	a1,a3,a1
	slli	a2,a2,14
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,368
	jr	ra
	.size	forward_kernel_1, .-forward_kernel_1
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36030343207321600
	.align	3
.LC2:
	.dword	17592186101760
	.align	3
.LC3:
	.dword	17592186052608
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	8192
	.half	0
	.half	24
	.half	-4031
	.half	135
	.half	25
	.half	0
	.half	256
	.half	26
	.half	0
	.half	0
	.half	27
	.half	10240
	.half	0
	.half	32
	.half	65
	.half	128
	.half	33
	.half	0
	.half	-30464
	.half	34
	.half	512
	.half	0
	.half	35
	.half	12288
	.half	0
	.half	40
	.half	65
	.half	128
	.half	41
	.half	0
	.half	256
	.half	42
	.half	0
	.half	0
	.half	43
	.half	8192
	.half	0
	.half	48
	.half	65
	.half	128
	.half	49
	.half	0
	.half	256
	.half	50
	.half	0
	.half	0
	.half	51
	.half	14336
	.half	0
	.half	56
	.half	65
	.half	128
	.half	57
	.half	0
	.half	-30464
	.half	58
	.half	512
	.half	0
	.half	59
	.half	10240
	.half	0
	.half	64
	.half	-4031
	.half	135
	.half	65
	.half	0
	.half	256
	.half	66
	.half	0
	.half	0
	.half	67
	.half	0
	.half	0
	.half	104
	.half	1
	.half	0
	.half	112
	.half	0
	.half	0
	.half	128
	.half	1
	.half	0
	.half	136
	.half	14
	.half	20
	.half	177
	.half	14
	.half	34
	.half	201
	.half	14
	.half	70
	.half	593
	.half	14
	.half	70
	.half	625
	.half	16
	.half	0
	.half	664
	.half	2
	.half	0
	.half	672
	.half	272
	.half	0
	.half	696
	.half	0
	.half	0
	.half	704
	.half	8192
	.half	0
	.half	728
	.half	65
	.half	128
	.half	729
	.half	0
	.half	256
	.half	730
	.half	0
	.half	0
	.half	731
	.half	10240
	.half	0
	.half	736
	.half	-4031
	.half	135
	.half	737
	.half	0
	.half	256
	.half	738
	.half	0
	.half	0
	.half	739
	.half	12288
	.half	0
	.half	744
	.half	65
	.half	128
	.half	745
	.half	0
	.half	-30464
	.half	746
	.half	0
	.half	0
	.half	747
	.half	8192
	.half	0
	.half	760
	.half	65
	.half	128
	.half	761
	.half	0
	.half	256
	.half	762
	.half	0
	.half	0
	.half	763
	.half	12288
	.half	0
	.half	768
	.half	65
	.half	128
	.half	769
	.half	0
	.half	-30464
	.half	770
	.half	0
	.half	0
	.half	771
	.half	10240
	.half	0
	.half	776
	.half	-4031
	.half	135
	.half	777
	.half	0
	.half	256
	.half	778
	.half	0
	.half	0
	.half	779
	.ident	"GCC: (g2ee5e430018) 12.2.0"
