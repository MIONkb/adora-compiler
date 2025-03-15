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
	li	a5,8192
	addiw	a5,a5,-192
	mv	a3,a1
	mulw	a1,a5,a1
	li	a4,524288000
	mv	a6,a2
	addi	a2,a4,1
	addi	sp,sp,-192
	slli	a2,a2,16
	add	a1,a0,a1
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mulw	a1,a5,a6
	li	a2,125
	slli	a2,a2,38
	add	a1,a0,a1
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,2000
	mulw	a1,a1,a3
	li	a2,4194304
	addi	a2,a2,9
	slli	a2,a2,13
	addw	a1,a1,a6
	slliw	a1,a1,2
	add	a1,a0,a1
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	mv	a4,sp
	lla	t1,.LANCHOR0+192
.L2:
	ld	a2,0(a5)
	ld	a7,8(a5)
	ld	a1,16(a5)
	sd	a2,0(a4)
	ld	a2,24(a5)
	sd	a7,8(a4)
	sd	a1,16(a4)
	sd	a2,24(a4)
	addi	a5,a5,32
	addi	a4,a4,32
	bne	a5,t1,.L2
	lhu	a5,146(sp)
	slliw	a4,a6,10
	slli	a4,a4,48
	srli	a4,a4,48
	andi	a5,a5,1023
	or	a5,a4,a5
	sh	a5,146(sp)
	lhu	a2,150(sp)
	sraiw	a5,a6,6
	slli	a5,a5,48
	slli	a2,a2,48
	srli	a2,a2,48
	srli	a5,a5,48
	andi	a2,a2,-1024
	or	a2,a5,a2
	sh	a2,150(sp)
	lhu	a2,2(sp)
	slliw	t3,a6,12
	sraiw	t1,a6,8
	andi	a2,a2,1023
	or	a2,a4,a2
	sh	a2,2(sp)
	lhu	a7,6(sp)
	mv	a1,sp
	ld	a2,.LC1
	slli	a7,a7,48
	srli	a7,a7,48
	andi	a7,a7,-1024
	or	a7,a5,a7
	sh	a7,6(sp)
	lhu	a7,122(sp)
	andi	a7,a7,1023
	or	a7,a4,a7
	sh	a7,122(sp)
	lhu	a7,126(sp)
	slli	a7,a7,48
	srli	a7,a7,48
	andi	a7,a7,-1024
	or	a7,a5,a7
	sh	a7,126(sp)
	lhu	a7,170(sp)
	andi	a7,a7,1023
	or	a4,a4,a7
	sh	a4,170(sp)
	lhu	a4,174(sp)
	slli	a4,a4,48
	srli	a4,a4,48
	andi	a4,a4,-1024
	or	a5,a5,a4
	sh	a5,174(sp)
	lhu	a5,42(sp)
	slli	a5,a5,52
	srli	a5,a5,52
	or	a5,a5,t3
	slli	a5,a5,48
	srli	a5,a5,48
	sh	a5,42(sp)
	lhu	a5,44(sp)
	slli	a5,a5,48
	srli	a5,a5,48
	andi	a5,a5,-256
	or	a5,a5,t1
	slli	a5,a5,48
	srli	a5,a5,48
	sh	a5,44(sp)
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1
	li	a1,0
	slli	a2,a2,37
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,770
	li	a2,0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	slliw	a1,a3,1
	addw	a1,a1,a6
	li	a2,2097152
	slliw	a1,a1,2
	addi	a2,a2,5
	add	a1,a0,a1
	slli	a2,a2,14
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,192
	jr	ra
	.size	cholesky_kernel_0, .-cholesky_kernel_0
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC1:
	.dword	36029621652815872
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	8192
	.half	-2048
	.half	16
	.half	19
	.half	0
	.half	17
	.half	0
	.half	256
	.half	18
	.half	0
	.half	0
	.half	19
	.half	0
	.half	16
	.half	88
	.half	16
	.half	6
	.half	169
	.half	0
	.half	1024
	.half	170
	.half	-7808
	.half	335
	.half	171
	.half	0
	.half	0
	.half	172
	.half	-16384
	.half	0
	.half	232
	.half	64
	.half	0
	.half	240
	.half	1
	.half	0
	.half	241
	.half	13
	.half	22
	.half	305
	.half	192
	.half	0
	.half	376
	.half	3
	.half	0
	.half	385
	.half	0
	.half	4096
	.half	520
	.half	0
	.half	8
	.half	528
	.half	2574
	.half	52
	.half	593
	.half	0
	.half	0
	.half	664
	.half	0
	.half	0
	.half	672
	.half	4096
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
	.half	8192
	.half	-2048
	.half	736
	.half	19
	.half	0
	.half	737
	.half	0
	.half	256
	.half	738
	.half	0
	.half	0
	.half	739
	.half	0
	.half	-2048
	.half	744
	.half	19
	.half	0
	.half	745
	.half	0
	.half	-27904
	.half	746
	.half	0
	.half	0
	.half	747
	.ident	"GCC: (g2ee5e430018) 12.2.0"
