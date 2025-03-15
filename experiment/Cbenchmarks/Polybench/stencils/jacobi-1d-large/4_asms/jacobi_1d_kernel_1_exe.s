	.file	"jacobi_1d_kernel_1_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	jacobi_1d_kernel_1
	.type	jacobi_1d_kernel_1, @function
jacobi_1d_kernel_1:
	li	a6,524288000
	addi	a2,a6,3
	mv	a3,a1
	addi	sp,sp,-320
	mv	a1,a0
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,262144000
	addi	a2,a2,1
	addi	a1,a0,4
	slli	a2,a2,16
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a4,2097152000
	addi	a2,a4,9
	addi	a1,a0,8
	slli	a2,a2,13
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,4096
	addi	a1,a5,-100
	addi	a2,a6,1
	add	a1,a0,a1
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,a5,-96
	addi	a2,a4,5
	add	a1,a0,a1
	slli	a2,a2,13
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,125
	addi	a5,a5,-92
	add	a1,a0,a5
	slli	a2,a2,37
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	addi	a4,sp,8
	lla	a2,.LANCHOR0+288
.L2:
	ld	a1,0(a5)
	ld	a6,8(a5)
	ld	a0,16(a5)
	sd	a1,0(a4)
	ld	a1,24(a5)
	sd	a6,8(a4)
	sd	a0,16(a4)
	sd	a1,24(a4)
	addi	a5,a5,32
	addi	a4,a4,32
	bne	a5,a2,.L2
	ld	a1,0(a5)
	ld	a2,8(a5)
	ld	a5,16(a5)
	sd	a1,0(a4)
	sd	a2,8(a4)
	sd	a5,16(a4)
	addi	a1,sp,8
	ld	a2,.LC1
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,13
	li	a1,0
	slli	a2,a2,34
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,12288
	addi	a1,a1,-358
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1048576000
	addi	a2,a2,5
	addi	a1,a3,4
	slli	a2,a2,14
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,2097152000
	li	a1,4096
	addi	a1,a1,-96
	addi	a2,a2,1
	add	a1,a3,a1
	slli	a2,a2,13
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
# 82 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (10 << (7)) | (0 << (7+5)) | (0 << (7+5+1)) | (1 << (7+5+2)) | (0 << (7+5+3)) | (0 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 0) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,320
	jr	ra
	.size	jacobi_1d_kernel_1, .-jacobi_1d_kernel_1
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36030137048891392
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	8192
	.half	-25600
	.half	16
	.half	15
	.half	0
	.half	17
	.half	0
	.half	256
	.half	18
	.half	0
	.half	0
	.half	19
	.half	10240
	.half	-25600
	.half	32
	.half	15
	.half	0
	.half	33
	.half	0
	.half	-28416
	.half	34
	.half	512
	.half	0
	.half	35
	.half	10240
	.half	-25600
	.half	40
	.half	15
	.half	0
	.half	41
	.half	0
	.half	256
	.half	42
	.half	0
	.half	0
	.half	43
	.half	8192
	.half	-25600
	.half	64
	.half	15
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
	.half	96
	.half	0
	.half	8
	.half	104
	.half	5121
	.half	0
	.half	112
	.half	8192
	.half	0
	.half	120
	.half	1024
	.half	1
	.half	128
	.half	0
	.half	0
	.half	136
	.half	3
	.half	0
	.half	176
	.half	13
	.half	4
	.half	177
	.half	526
	.half	20
	.half	185
	.half	46
	.half	20
	.half	201
	.half	3
	.half	0
	.half	456
	.half	13
	.half	64
	.half	457
	.half	1
	.half	0
	.half	529
	.half	4
	.half	4
	.half	536
	.half	1038
	.half	68
	.half	601
	.half	526
	.half	56
	.half	609
	.half	768
	.half	16
	.half	672
	.half	64
	.half	0
	.half	680
	.half	1
	.half	0
	.half	688
	.half	0
	.half	1
	.half	696
	.half	8192
	.half	-25600
	.half	736
	.half	15
	.half	0
	.half	737
	.half	0
	.half	256
	.half	738
	.half	0
	.half	0
	.half	739
	.half	12288
	.half	-25600
	.half	744
	.half	15
	.half	0
	.half	745
	.half	0
	.half	-27904
	.half	746
	.half	0
	.half	0
	.half	747
	.half	10240
	.half	-25600
	.half	752
	.half	15
	.half	0
	.half	753
	.half	0
	.half	256
	.half	754
	.half	0
	.half	0
	.half	755
	.half	8192
	.half	-25600
	.half	768
	.half	15
	.half	0
	.half	769
	.half	0
	.half	256
	.half	770
	.half	0
	.half	0
	.half	771
	.ident	"GCC: (g2ee5e430018) 12.2.0"
