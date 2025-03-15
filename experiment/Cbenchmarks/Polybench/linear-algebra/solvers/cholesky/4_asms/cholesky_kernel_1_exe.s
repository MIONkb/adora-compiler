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
	li	a5,8192
	addiw	a5,a5,-188
	mulw	a5,a5,a1
	li	a2,1048576
	addi	a2,a2,1
	mv	a3,a1
	addi	sp,sp,-160
	slli	a2,a2,15
	add	a1,a0,a5
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1048576000
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
	lla	a7,.LANCHOR0+144
.L2:
	ld	a0,0(a5)
	ld	a1,8(a5)
	ld	a2,16(a5)
	sd	a0,0(a4)
	ld	a0,24(a5)
	sd	a1,8(a4)
	ld	a1,32(a5)
	sd	a2,16(a4)
	ld	a2,40(a5)
	sd	a0,24(a4)
	sd	a1,32(a4)
	sd	a2,40(a4)
	addi	a5,a5,48
	addi	a4,a4,48
	bne	a5,a7,.L2
	lw	a1,0(a5)
	lhu	a2,4(a5)
	slliw	a5,a3,10
	sw	a1,0(a4)
	sh	a2,4(a4)
	lhu	a2,136(sp)
	slli	a4,a5,48
	srli	a4,a4,48
	andi	a2,a2,1023
	or	a2,a4,a2
	sh	a2,136(sp)
	lhu	a2,140(sp)
	sraiw	a5,a3,6
	slli	a5,a5,48
	slli	a2,a2,48
	srli	a2,a2,48
	srli	a5,a5,48
	andi	a2,a2,-1024
	or	a2,a5,a2
	sh	a2,140(sp)
	lhu	a2,10(sp)
	slliw	t1,a3,12
	sraiw	a7,a3,8
	andi	a2,a2,1023
	or	a2,a4,a2
	sh	a2,10(sp)
	lhu	a0,14(sp)
	addi	a1,sp,8
	ld	a2,.LC1
	slli	a0,a0,48
	srli	a0,a0,48
	andi	a0,a0,-1024
	or	a0,a5,a0
	sh	a0,14(sp)
	lhu	a0,34(sp)
	andi	a0,a0,1023
	or	a4,a4,a0
	sh	a4,34(sp)
	lhu	a4,38(sp)
	slli	a4,a4,48
	srli	a4,a4,48
	andi	a4,a4,-1024
	or	a5,a5,a4
	sh	a5,38(sp)
	lhu	a5,104(sp)
	slli	a5,a5,52
	srli	a5,a5,52
	or	a5,a5,t1
	slli	a5,a5,48
	srli	a5,a5,48
	sh	a5,104(sp)
	lhu	a5,106(sp)
	slli	a5,a5,48
	srli	a5,a5,48
	andi	a5,a5,-256
	or	a5,a5,a7
	slli	a5,a5,48
	srli	a5,a5,48
	sh	a5,106(sp)
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,25
	li	a1,0
	slli	a2,a2,32
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,32768
	addi	a1,a1,48
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,-8192
	addi	a5,a5,204
	mul	a1,a3,a5
	li	a2,4194304
	addi	a2,a2,5
	slli	a2,a2,13
	add	a1,a6,a1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,160
	jr	ra
	.size	cholesky_kernel_1, .-cholesky_kernel_1
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36029441264189440
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	0
	.half	-2048
	.half	40
	.half	19
	.half	0
	.half	41
	.half	0
	.half	256
	.half	42
	.half	0
	.half	0
	.half	43
	.half	2048
	.half	-2048
	.half	48
	.half	19
	.half	0
	.half	49
	.half	0
	.half	-28928
	.half	50
	.half	0
	.half	0
	.half	51
	.half	0
	.half	16
	.half	112
	.half	48
	.half	0
	.half	120
	.half	-16384
	.half	0
	.half	256
	.half	0
	.half	1024
	.half	264
	.half	1550
	.half	24
	.half	329
	.half	0
	.half	0
	.half	408
	.half	16
	.half	8
	.half	481
	.half	0
	.half	1024
	.half	482
	.half	-7872
	.half	335
	.half	483
	.half	0
	.half	0
	.half	484
	.half	0
	.half	0
	.half	560
	.half	13
	.half	8
	.half	633
	.half	0
	.half	0
	.half	712
	.half	8192
	.half	-2048
	.half	784
	.half	19
	.half	0
	.half	785
	.half	0
	.half	256
	.half	786
	.half	0
	.half	0
	.half	787
	.ident	"GCC: (g2ee5e430018) 12.2.0"
