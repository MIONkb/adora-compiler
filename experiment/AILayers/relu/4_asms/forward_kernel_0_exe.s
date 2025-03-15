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
	li	a5,1073741824
	addi	a2,a5,3
	mv	a3,a1
	addi	sp,sp,-384
	mv	a1,a0
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,8192
	addi	a2,a5,1
	add	a1,a0,a1
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,16384
	li	a2,1
	add	a1,a0,a1
	slli	a2,a2,45
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,536870912
	li	a1,24576
	addi	a2,a2,1
	add	a1,a0,a1
	slli	a2,a2,16
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	mv	a4,sp
	lla	a6,.LANCHOR0+360
.L2:
	ld	a1,0(a5)
	ld	a2,8(a5)
	ld	a0,16(a5)
	sd	a1,0(a4)
	ld	a1,24(a5)
	sd	a2,8(a4)
	ld	a2,32(a5)
	sd	a0,16(a4)
	sd	a1,24(a4)
	sd	a2,32(a4)
	addi	a5,a5,40
	addi	a4,a4,40
	bne	a5,a6,.L2
	ld	a1,0(a5)
	ld	a2,8(a5)
	lhu	a5,16(a5)
	sd	a1,0(a4)
	sd	a2,8(a4)
	sh	a5,16(a4)
	mv	a1,sp
	ld	a2,.LC1
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,63
	li	a1,0
	slli	a2,a2,32
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,61440
	addi	a1,a1,-1885
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
	li	a5,8192
	add	a1,a3,a5
	ld	a2,.LC3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,16384
	slli	a2,a5,32
	add	a1,a3,a1
	add	a2,a2,a5
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,24576
	add	a1,a3,a1
	ld	a2,.LC4
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
# 82 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (10 << (7)) | (0 << (7+5)) | (0 << (7+5+1)) | (1 << (7+5+2)) | (0 << (7+5+3)) | (0 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 0) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,384
	jr	ra
	.size	forward_kernel_0, .-forward_kernel_0
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36030420516732928
	.align	3
.LC2:
	.dword	35184372195328
	.align	3
.LC3:
	.dword	35184372129792
	.align	3
.LC4:
	.dword	35184372203520
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	8192
	.half	0
	.half	8
	.half	65
	.half	256
	.half	9
	.half	0
	.half	256
	.half	10
	.half	0
	.half	0
	.half	11
	.half	10240
	.half	0
	.half	16
	.half	65
	.half	256
	.half	17
	.half	0
	.half	-29952
	.half	18
	.half	512
	.half	0
	.half	19
	.half	10240
	.half	0
	.half	48
	.half	65
	.half	256
	.half	49
	.half	0
	.half	-29440
	.half	50
	.half	512
	.half	0
	.half	51
	.half	8192
	.half	0
	.half	64
	.half	65
	.half	256
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
	.half	1
	.half	0
	.half	96
	.half	0
	.half	0
	.half	128
	.half	512
	.half	0
	.half	136
	.half	0
	.half	0
	.half	152
	.half	18
	.half	16
	.half	153
	.half	0
	.half	0
	.half	160
	.half	524
	.half	400
	.half	161
	.half	0
	.half	0
	.half	200
	.half	1036
	.half	416
	.half	201
	.half	0
	.half	0
	.half	208
	.half	18
	.half	16
	.half	209
	.half	0
	.half	0
	.half	232
	.half	2048
	.half	0
	.half	272
	.half	1
	.half	0
	.half	281
	.half	0
	.half	3072
	.half	416
	.half	0
	.half	256
	.half	424
	.half	0
	.half	0
	.half	496
	.half	12
	.half	560
	.half	497
	.half	512
	.half	0
	.half	568
	.half	1
	.half	0
	.half	569
	.half	0
	.half	0
	.half	576
	.half	0
	.half	0
	.half	608
	.half	18
	.half	64
	.half	609
	.half	0
	.half	0
	.half	616
	.half	524
	.half	432
	.half	617
	.half	0
	.half	0
	.half	640
	.half	18
	.half	64
	.half	641
	.half	2
	.half	0
	.half	688
	.half	0
	.half	0
	.half	696
	.half	8192
	.half	0
	.half	712
	.half	8192
	.half	0
	.half	752
	.half	65
	.half	256
	.half	753
	.half	0
	.half	256
	.half	754
	.half	0
	.half	0
	.half	755
	.half	12288
	.half	0
	.half	768
	.half	65
	.half	256
	.half	769
	.half	0
	.half	-29952
	.half	770
	.half	0
	.half	0
	.half	771
	.half	10240
	.half	0
	.half	776
	.half	65
	.half	256
	.half	777
	.half	0
	.half	-29952
	.half	778
	.half	512
	.half	0
	.half	779
	.half	8192
	.half	0
	.half	784
	.half	65
	.half	256
	.half	785
	.half	0
	.half	256
	.half	786
	.half	0
	.half	0
	.half	787
	.ident	"GCC: (g2ee5e430018) 12.2.0"
