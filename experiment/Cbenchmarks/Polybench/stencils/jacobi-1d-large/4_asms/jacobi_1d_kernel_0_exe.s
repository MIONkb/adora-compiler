	.file	"jacobi_1d_kernel_0_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	jacobi_1d_kernel_0
	.type	jacobi_1d_kernel_0, @function
jacobi_1d_kernel_0:
	li	a2,125
	mv	a3,a1
	addi	sp,sp,-352
	mv	a1,a0
	slli	a2,a2,37
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,2097152000
	addi	a2,a2,1
	addi	a1,a0,4
	slli	a2,a2,13
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a4,524288000
	addi	a2,a4,3
	addi	a1,a0,8
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,4096
	li	a2,262144000
	addi	a1,a5,-100
	addi	a2,a2,1
	add	a1,a0,a1
	slli	a2,a2,16
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1048576000
	addi	a1,a5,-96
	addi	a2,a2,1
	add	a1,a0,a1
	slli	a2,a2,14
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a5,a5,-92
	addi	a2,a4,1
	add	a1,a0,a5
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	mv	a4,sp
	lla	a2,.LANCHOR0+336
.L2:
	ld	a6,0(a5)
	ld	a0,8(a5)
	ld	a1,16(a5)
	sd	a6,0(a4)
	ld	a6,24(a5)
	sd	a0,8(a4)
	ld	a0,32(a5)
	sd	a1,16(a4)
	ld	a1,40(a5)
	sd	a6,24(a4)
	sd	a0,32(a4)
	sd	a1,40(a4)
	addi	a5,a5,48
	addi	a4,a4,48
	bne	a5,a2,.L2
	ld	a2,0(a5)
	lw	a5,8(a5)
	mv	a1,sp
	sd	a2,0(a4)
	sw	a5,8(a4)
	ld	a2,.LC1
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,29
	li	a1,0
	slli	a2,a2,33
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,32768
	addi	a1,a1,1147
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,2097152000
	addi	a2,a2,5
	addi	a1,a3,4
	slli	a2,a2,13
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1048576000
	li	a1,4096
	addi	a1,a1,-96
	addi	a2,a2,3
	add	a1,a3,a1
	slli	a2,a2,14
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
# 82 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (10 << (7)) | (0 << (7+5)) | (0 << (7+5+1)) | (1 << (7+5+2)) | (0 << (7+5+3)) | (0 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 0) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,352
	jr	ra
	.size	jacobi_1d_kernel_0, .-jacobi_1d_kernel_0
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36030291667714048
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	12288
	.half	-25600
	.half	8
	.half	15
	.half	0
	.half	9
	.half	0
	.half	256
	.half	10
	.half	0
	.half	0
	.half	11
	.half	10240
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
	.half	8192
	.half	-25600
	.half	32
	.half	15
	.half	0
	.half	33
	.half	0
	.half	256
	.half	34
	.half	0
	.half	0
	.half	35
	.half	12288
	.half	-25600
	.half	40
	.half	15
	.half	0
	.half	41
	.half	0
	.half	-28416
	.half	42
	.half	512
	.half	0
	.half	43
	.half	10240
	.half	-25600
	.half	48
	.half	15
	.half	0
	.half	49
	.half	0
	.half	-28416
	.half	50
	.half	512
	.half	0
	.half	51
	.half	8192
	.half	-25600
	.half	56
	.half	15
	.half	0
	.half	57
	.half	0
	.half	256
	.half	58
	.half	0
	.half	0
	.half	59
	.half	0
	.half	0
	.half	88
	.half	0
	.half	4
	.half	96
	.half	256
	.half	0
	.half	104
	.half	-32768
	.half	0
	.half	112
	.half	1
	.half	2
	.half	120
	.half	1
	.half	0
	.half	128
	.half	526
	.half	24
	.half	161
	.half	14
	.half	20
	.half	169
	.half	526
	.half	34
	.half	177
	.half	3
	.half	0
	.half	184
	.half	13
	.half	6
	.half	185
	.half	3
	.half	0
	.half	192
	.half	13
	.half	6
	.half	193
	.half	2
	.half	0
	.half	240
	.half	0
	.half	0
	.half	248
	.half	4096
	.half	0
	.half	256
	.half	64
	.half	0
	.half	264
	.half	46
	.half	66
	.half	329
	.half	0
	.half	4096
	.half	384
	.half	8
	.half	0
	.half	408
	.half	0
	.half	384
	.half	416
	.half	0
	.half	512
	.half	424
	.half	0
	.half	4096
	.half	528
	.half	0
	.half	4096
	.half	568
	.half	0
	.half	0
	.half	672
	.half	0
	.half	0
	.half	712
	.half	8192
	.half	-25600
	.half	744
	.half	15
	.half	0
	.half	745
	.half	0
	.half	256
	.half	746
	.half	0
	.half	0
	.half	747
	.half	8192
	.half	-25600
	.half	784
	.half	15
	.half	0
	.half	785
	.half	0
	.half	256
	.half	786
	.half	0
	.half	0
	.half	787
	.ident	"GCC: (g2ee5e430018) 12.2.0"
