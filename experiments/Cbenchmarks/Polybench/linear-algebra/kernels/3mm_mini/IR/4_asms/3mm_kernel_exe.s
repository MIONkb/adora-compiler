	.file	"3mm_kernel_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	kernel_3mm
	.type	kernel_3mm, @function
kernel_3mm:
	li	a7,167772160
	mv	t1,a2
	addi	a2,a7,3
	addi	sp,sp,-384
	slli	a2,a2,15
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a7,754974720
	addi	a2,a7,13
	mv	a1,t1
	slli	a2,a2,13
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a2,.LANCHOR0
	addi	a7,sp,248
	lla	a1,.LANCHOR0+128
.L2:
	ld	t1,0(a2)
	ld	t4,8(a2)
	ld	t3,16(a2)
	sd	t1,0(a7)
	ld	t1,24(a2)
	sd	t4,8(a7)
	sd	t3,16(a7)
	sd	t1,24(a7)
	addi	a2,a2,32
	addi	a7,a7,32
	bne	a2,a1,.L2
	lw	t1,0(a2)
	addi	a1,sp,248
	ld	a2,.LC3
	sw	t1,0(a7)
 #APP
# 33 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,11
	li	a1,0
	slli	a2,a2,33
 #APP
# 57 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,28672
	addi	a1,a1,11
	li	a2,0
 #APP
# 65 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,301989888
	addi	a2,a2,7
	mv	a1,a0
	slli	a2,a2,14
 #APP
# 49 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1073852416
	mv	a1,a4
	slli	a2,a2,26
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC4
	mv	a1,a5
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0+136
	addi	a4,sp,8
	lla	a2,.LANCHOR0+256
.L3:
	ld	a7,0(a5)
	ld	a1,8(a5)
	ld	t1,16(a5)
	sd	a7,0(a4)
	ld	a7,24(a5)
	sd	a1,8(a4)
	ld	a1,32(a5)
	sd	t1,16(a4)
	sd	a7,24(a4)
	sd	a1,32(a4)
	addi	a5,a5,40
	addi	a4,a4,40
	bne	a5,a2,.L3
	addi	a1,sp,8
	ld	a2,.LC5
 #APP
# 33 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,4194304
	addi	a2,a2,5
	li	a1,0
	slli	a2,a2,34
 #APP
# 57 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,28672
	li	a2,1
	addi	a1,a1,11
	slli	a2,a2,56
 #APP
# 65 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC6
	mv	a1,a3
 #APP
# 49 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,12845056
	addi	a2,a2,9
	mv	a1,a0
	slli	a2,a2,39
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC7
	mv	a1,a3
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0+256
	addi	a4,sp,128
	lla	a3,.LANCHOR0+376
.L4:
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
	bne	a5,a3,.L4
	addi	a1,sp,128
	ld	a2,.LC8
 #APP
# 33 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,8388608
	addi	a2,a2,5
	li	a1,0
	slli	a2,a2,34
 #APP
# 57 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,28672
	li	a2,1
	addi	a1,a1,11
	slli	a2,a2,57
 #APP
# 65 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC9
	mv	a1,a6
 #APP
# 49 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,384
	jr	ra
	.size	kernel_3mm, .-kernel_3mm
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC3:
	.dword	36029363954778112
	.align	3
.LC4:
	.dword	72066665008865280
	.align	3
.LC5:
	.dword	108086906453098496
	.align	3
.LC6:
	.dword	72064397266141184
	.align	3
.LC7:
	.dword	2449965000517754880
	.align	3
.LC8:
	.dword	180144500491026432
	.align	3
.LC9:
	.dword	144121235389825024
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	26
	.half	32
	.half	481
	.half	0
	.half	4096
	.half	482
	.half	1280
	.half	1029
	.half	483
	.half	0
	.half	0
	.half	484
	.half	1
	.half	0
	.half	553
	.half	0
	.half	0
	.half	560
	.half	8
	.half	280
	.half	633
	.half	8192
	.half	0
	.half	696
	.half	16
	.half	0
	.half	704
	.half	2
	.half	0
	.half	712
	.half	4096
	.half	20480
	.half	760
	.half	64
	.half	-32624
	.half	761
	.half	0
	.half	24833
	.half	762
	.half	4100
	.half	0
	.half	763
	.half	18432
	.half	20482
	.half	768
	.half	-21824
	.half	-32617
	.half	769
	.half	3916
	.half	257
	.half	770
	.half	0
	.half	0
	.half	771
	.half	8192
	.half	20480
	.half	776
	.half	-1216
	.half	-32617
	.half	777
	.half	0
	.half	257
	.half	778
	.half	0
	.half	0
	.half	779
	.zero	4
.LC1:
	.half	8192
	.half	24576
	.half	8
	.half	-1472
	.half	-32585
	.half	9
	.half	8192
	.half	257
	.half	10
	.half	0
	.half	0
	.half	11
	.half	4096
	.half	24576
	.half	16
	.half	64
	.half	-32592
	.half	17
	.half	8192
	.half	28929
	.half	18
	.half	4100
	.half	0
	.half	19
	.half	-14336
	.half	24578
	.half	32
	.half	-32320
	.half	-32585
	.half	33
	.half	12024
	.half	257
	.half	34
	.half	0
	.half	0
	.half	35
	.half	0
	.half	0
	.half	88
	.half	4353
	.half	0
	.half	96
	.half	0
	.half	0
	.half	104
	.half	26
	.half	16
	.half	161
	.half	0
	.half	4096
	.half	162
	.half	1536
	.half	1030
	.half	163
	.half	0
	.half	0
	.half	164
	.half	2056
	.half	136
	.half	169
.LC2:
	.half	8192
	.half	18432
	.half	8
	.half	-1088
	.half	-32585
	.half	9
	.half	0
	.half	257
	.half	10
	.half	0
	.half	0
	.half	11
	.half	4096
	.half	18432
	.half	16
	.half	64
	.half	-32592
	.half	17
	.half	0
	.half	28929
	.half	18
	.half	4
	.half	0
	.half	19
	.half	-14336
	.half	18434
	.half	32
	.half	-23872
	.half	-32585
	.half	33
	.half	3898
	.half	257
	.half	34
	.half	0
	.half	0
	.half	35
	.half	4112
	.half	0
	.half	88
	.half	-32768
	.half	0
	.half	96
	.half	0
	.half	0
	.half	104
	.half	26
	.half	16
	.half	153
	.half	0
	.half	4096
	.half	154
	.half	-31232
	.half	1028
	.half	155
	.half	0
	.half	0
	.half	156
	.half	136
	.half	136
	.half	161
	.ident	"GCC: (g2ee5e430018) 12.2.0"
