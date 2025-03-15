	.file	"cholesky_kernel_0_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	cholesky_kernel_0
	.type	cholesky_kernel_0, @function
cholesky_kernel_0:
	slliw	a5,a1,4
	subw	a5,a5,a1
	mv	a3,a1
	mv	a6,a2
	slliw	a1,a5,5
	li	a2,15
	addi	sp,sp,-192
	mv	t1,a0
	add	a1,a0,a1
	slli	a2,a2,37
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	slliw	a1,a6,4
	subw	a1,a1,a6
	li	a2,62914560
	slliw	a1,a1,5
	addi	a2,a2,3
	add	a1,a0,a1
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	slliw	a1,a5,3
	addw	a1,a1,a6
	li	a2,524288
	slliw	a1,a1,2
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
	lla	a7,.LANCHOR0+160
.L2:
	ld	a2,0(a5)
	ld	a0,8(a5)
	ld	a1,16(a5)
	sd	a2,0(a4)
	ld	a2,24(a5)
	sd	a0,8(a4)
	sd	a1,16(a4)
	sd	a2,24(a4)
	addi	a5,a5,32
	addi	a4,a4,32
	bne	a5,a7,.L2
	ld	a0,0(a5)
	ld	a1,8(a5)
	ld	a2,16(a5)
	sd	a0,0(a4)
	lhu	a5,24(a5)
	sd	a1,8(a4)
	sd	a2,16(a4)
	sh	a5,24(a4)
	lhu	a5,2(sp)
	slliw	a4,a6,10
	slli	a4,a4,48
	srli	a4,a4,48
	andi	a5,a5,1023
	or	a5,a4,a5
	sh	a5,2(sp)
	lhu	a2,6(sp)
	sraiw	a5,a6,6
	slli	a5,a5,48
	slli	a2,a2,48
	srli	a2,a2,48
	srli	a5,a5,48
	andi	a2,a2,-1024
	or	a2,a5,a2
	sh	a2,6(sp)
	lhu	a2,164(sp)
	li	a0,4096
	addi	t5,a0,-1
	andi	a2,a2,1023
	or	a2,a4,a2
	sh	a2,164(sp)
	lhu	a2,168(sp)
	slliw	t4,a6,12
	sraiw	t3,a6,8
	slli	a2,a2,48
	srli	a2,a2,48
	andi	a2,a2,-1024
	or	a2,a5,a2
	sh	a2,168(sp)
	lhu	a7,116(sp)
	mv	a1,sp
	ld	a2,.LC1
	andi	a7,a7,1023
	or	a7,a4,a7
	sh	a7,116(sp)
	lhu	a7,120(sp)
	slli	a7,a7,48
	srli	a7,a7,48
	andi	a7,a7,-1024
	or	a7,a5,a7
	sh	a7,120(sp)
	lhu	a7,140(sp)
	andi	a7,a7,1023
	or	a4,a4,a7
	sh	a4,140(sp)
	lhu	a4,144(sp)
	slli	a4,a4,48
	srli	a4,a4,48
	andi	a4,a4,-1024
	or	a5,a5,a4
	sh	a5,144(sp)
	lhu	a5,54(sp)
	and	a5,a5,t5
	or	a5,a5,t4
	slli	a5,a5,48
	srli	a5,a5,48
	sh	a5,54(sp)
	lhu	a5,56(sp)
	slli	a5,a5,48
	srli	a5,a5,48
	andi	a5,a5,-256
	or	a5,a5,t3
	slli	a5,a5,48
	srli	a5,a5,48
	sh	a5,56(sp)
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,31
	li	a1,0
	slli	a2,a2,32
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	addi	a1,a0,264
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
# 82 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (10 << (7)) | (0 << (7+5)) | (0 << (7+5+1)) | (1 << (7+5+2)) | (0 << (7+5+3)) | (0 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 0) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	slliw	a1,a3,1
	addw	a1,a1,a6
	slliw	a1,a1,2
	li	a2,1
	add	a1,t1,a1
	slli	a2,a2,35
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
# 82 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (10 << (7)) | (0 << (7+5)) | (0 << (7+5+1)) | (1 << (7+5+2)) | (0 << (7+5+3)) | (0 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 0) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,192
	jr	ra
	.size	cholesky_kernel_0, .-cholesky_kernel_0
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36029595883012096
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	8192
	.half	-2048
	.half	32
	.half	19
	.half	0
	.half	33
	.half	0
	.half	256
	.half	34
	.half	0
	.half	0
	.half	35
	.half	0
	.half	0
	.half	112
	.half	3
	.half	0
	.half	257
	.half	3
	.half	0
	.half	401
	.half	16
	.half	8
	.half	465
	.half	0
	.half	1024
	.half	466
	.half	-7808
	.half	335
	.half	467
	.half	0
	.half	0
	.half	468
	.half	0
	.half	12
	.half	528
	.half	0
	.half	0
	.half	536
	.half	8192
	.half	0
	.half	544
	.half	2062
	.half	52
	.half	593
	.half	525
	.half	66
	.half	617
	.half	0
	.half	0
	.half	664
	.half	0
	.half	0
	.half	672
	.half	2
	.half	0
	.half	696
	.half	0
	.half	-2048
	.half	728
	.half	19
	.half	0
	.half	729
	.half	0
	.half	256
	.half	730
	.half	0
	.half	0
	.half	731
	.half	0
	.half	-2048
	.half	744
	.half	19
	.half	0
	.half	745
	.half	0
	.half	-28416
	.half	746
	.half	0
	.half	0
	.half	747
	.half	8192
	.half	-2048
	.half	760
	.half	19
	.half	0
	.half	761
	.half	0
	.half	256
	.half	762
	.half	0
	.half	0
	.half	763
	.ident	"GCC: (g2ee5e430018) 12.2.0"
