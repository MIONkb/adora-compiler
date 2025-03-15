	.file	"atax_kernel_1_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	atax_kernel_1
	.type	atax_kernel_1, @function
atax_kernel_1:
	mv	a3,a2
	li	a2,107479040
	addi	a2,a2,1
	addi	sp,sp,-160
	mv	a1,a3
	slli	a2,a2,16
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	mv	a4,sp
	lla	a2,.LANCHOR0+144
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
	li	a2,13
	li	a1,0
	slli	a2,a2,33
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,4096
	addi	a1,a1,-752
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,429916160
	addi	a2,a2,5
	mv	a1,a3
	slli	a2,a2,14
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,160
	jr	ra
	.size	atax_kernel_1, .-atax_kernel_1
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36029467033993216
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	8192
	.half	6144
	.half	40
	.half	6
	.half	0
	.half	41
	.half	0
	.half	256
	.half	42
	.half	0
	.half	0
	.half	43
	.half	0
	.half	16
	.half	112
	.half	3
	.half	0
	.half	257
	.half	0
	.half	8
	.half	400
	.half	13
	.half	52
	.half	465
	.half	0
	.half	48
	.half	528
	.half	512
	.half	0
	.half	536
	.half	110
	.half	38
	.half	593
	.half	16
	.half	0
	.half	664
	.half	0
	.half	0
	.half	672
	.half	0
	.half	0
	.half	680
	.half	8192
	.half	6144
	.half	728
	.half	6
	.half	0
	.half	729
	.half	0
	.half	256
	.half	730
	.half	0
	.half	0
	.half	731
	.half	12288
	.half	6144
	.half	744
	.half	6
	.half	0
	.half	745
	.half	0
	.half	-28928
	.half	746
	.half	0
	.half	0
	.half	747
	.half	2048
	.half	6144
	.half	752
	.half	6
	.half	0
	.half	753
	.half	0
	.half	256
	.half	754
	.half	0
	.half	0
	.half	755
	.ident	"GCC: (g2ee5e430018) 12.2.0"
