	.file	"cholesky_kernel_1_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	cholesky_kernel_1
	.type	cholesky_kernel_1, @function
cholesky_kernel_1:
	li	a5,484
	mulw	a5,a5,a1
	li	a2,1
	mv	a3,a1
	addi	sp,sp,-176
	slli	a2,a2,35
	add	a1,a0,a5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,62914560
	slli	a6,a3,2
	sub	a6,a1,a6
	addi	a2,a2,3
	mv	a1,a6
	slli	a2,a2,15
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	addi	a4,sp,8
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
	lhu	a5,0(a5)
	slliw	a0,a3,10
	slli	a0,a0,48
	sh	a5,0(a4)
	lhu	a4,148(sp)
	srli	a0,a0,48
	sraiw	a5,a3,6
	andi	a4,a4,1023
	or	a4,a0,a4
	sh	a4,148(sp)
	lhu	a4,152(sp)
	slli	a5,a5,48
	srli	a5,a5,48
	slli	a4,a4,48
	srli	a4,a4,48
	andi	a4,a4,-1024
	or	a4,a5,a4
	sh	a4,152(sp)
	lhu	a4,10(sp)
	slliw	t1,a3,12
	sraiw	a7,a3,8
	andi	a4,a4,1023
	or	a4,a0,a4
	sh	a4,10(sp)
	lhu	a4,14(sp)
	addi	a1,sp,8
	ld	a2,.LC1
	slli	a4,a4,48
	srli	a4,a4,48
	andi	a4,a4,-1024
	or	a4,a5,a4
	sh	a4,14(sp)
	lhu	a4,34(sp)
	andi	a4,a4,1023
	or	a0,a0,a4
	sh	a0,34(sp)
	lhu	a4,38(sp)
	slli	a4,a4,48
	srli	a4,a4,48
	andi	a4,a4,-1024
	or	a5,a5,a4
	sh	a5,38(sp)
	lhu	a5,110(sp)
	slli	a5,a5,52
	srli	a5,a5,52
	or	a5,a5,t1
	slli	a5,a5,48
	srli	a5,a5,48
	sh	a5,110(sp)
	lhu	a5,112(sp)
	slli	a5,a5,48
	srli	a5,a5,48
	andi	a5,a5,-256
	or	a5,a5,a7
	slli	a5,a5,48
	srli	a5,a5,48
	sh	a5,112(sp)
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,27
	li	a1,0
	slli	a2,a2,32
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,16384
	addi	a1,a1,24
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,-468
	mul	a1,a3,a5
	li	a2,1048576
	addi	a2,a2,1
	slli	a2,a2,15
	add	a1,a6,a1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
# 82 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (10 << (7)) | (0 << (7+5)) | (0 << (7+5+1)) | (1 << (7+5+2)) | (0 << (7+5+3)) | (0 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 0) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,176
	jr	ra
	.size	cholesky_kernel_1, .-cholesky_kernel_1
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36029492803796992
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	0
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
	.half	-2048
	.half	40
	.half	19
	.half	0
	.half	41
	.half	0
	.half	-27904
	.half	42
	.half	512
	.half	0
	.half	43
	.half	0
	.half	0
	.half	112
	.half	1024
	.half	0
	.half	120
	.half	2574
	.half	24
	.half	193
	.half	1
	.half	0
	.half	272
	.half	0
	.half	512
	.half	280
	.half	0
	.half	-32768
	.half	416
	.half	4096
	.half	0
	.half	424
	.half	16
	.half	2
	.half	497
	.half	0
	.half	1024
	.half	498
	.half	-7808
	.half	335
	.half	499
	.half	0
	.half	0
	.half	500
	.half	0
	.half	3072
	.half	560
	.half	0
	.half	128
	.half	568
	.half	13
	.half	6
	.half	641
	.half	16
	.half	0
	.half	712
	.half	8192
	.half	-2048
	.half	776
	.half	19
	.half	0
	.half	777
	.half	0
	.half	256
	.half	778
	.half	0
	.half	0
	.half	779
	.ident	"GCC: (g2ee5e430018) 12.2.0"
