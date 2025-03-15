	.file	"deriche_kernel_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	kernel_deriche
	.type	kernel_deriche, @function
kernel_deriche:
	addi	sp,sp,-1664
	li	a5,1048576
	li	a6,536870912
	li	a4,4194304
	addi	t2,a5,1
	sd	s0,1656(sp)
	sd	s1,1648(sp)
	addi	a5,a5,3
	addi	a6,a6,1
	li	s1,27
	li	s0,65536
	addi	a4,a4,13
	sd	s2,1640(sp)
	sd	s3,1632(sp)
	sd	s4,1624(sp)
	sd	s5,1616(sp)
	sd	s6,1608(sp)
	sd	s7,1600(sp)
	sd	s8,1592(sp)
	sd	s9,1584(sp)
	mv	t3,a1
	mv	t4,a2
	li	a7,0
	lla	t5,.LANCHOR0+320
	slli	a6,a6,16
	ld	s3,.LC6
	slli	s1,s1,33
	addi	s0,s0,-1025
	slli	t2,t2,15
	slli	t1,a5,15
	slli	t6,a4,13
	ld	s4,.LC7
	li	s2,8192
.L3:
	add	a1,a7,a0
	mv	a2,a6
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	addi	a4,sp,544
.L2:
	ld	a2,0(a5)
	ld	t0,8(a5)
	ld	a1,16(a5)
	sd	a2,0(a4)
	ld	a2,24(a5)
	sd	t0,8(a4)
	sd	a1,16(a4)
	sd	a2,24(a4)
	addi	a5,a5,32
	addi	a4,a4,32
	bne	a5,t5,.L2
	lw	a5,0(a5)
	addi	a1,sp,544
	mv	a2,s3
	sw	a5,0(a4)
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s1
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s0
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,36
	mv	a2,t2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,24
	mv	a2,t1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,28
	mv	a2,t6
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,t4,a7
	mv	a2,s4
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	beq	a7,s2,.L36
	li	a7,8192
	j	.L3
.L36:
	li	a5,4194304
	li	s0,1073741824
	li	t2,1048576
	li	t0,2097152
	addi	s1,a5,5
	addi	s0,s0,1
	addi	a5,a5,1
	li	s4,29
	li	s3,65536
	li	s2,1
	addi	t2,t2,3
	addi	t0,t0,3
	li	t1,0
	lla	t5,.LANCHOR0+664
	slli	s0,s0,15
	ld	s6,.LC8
	slli	s4,s4,33
	addi	s3,s3,-1025
	slli	s2,s2,45
	slli	s1,s1,13
	slli	t2,t2,15
	slli	t0,t0,14
	slli	t6,a5,13
	li	s5,8192
.L5:
	add	a1,a0,t1
	mv	a2,s0
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0+328
	addi	a4,sp,872
.L4:
	ld	a6,0(a5)
	ld	a1,8(a5)
	ld	a2,16(a5)
	sd	a6,0(a4)
	ld	a6,24(a5)
	sd	a1,8(a4)
	ld	a1,32(a5)
	sd	a2,16(a4)
	ld	a2,40(a5)
	sd	a6,24(a4)
	sd	a1,32(a4)
	sd	a2,40(a4)
	addi	a5,a5,48
	addi	a4,a4,48
	bne	a5,t5,.L4
	ld	a2,0(a5)
	lw	a5,8(a5)
	addi	a1,sp,872
	sd	a2,0(a4)
	sw	a5,8(a4)
	mv	a2,s6
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s4
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s3
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a3,t1
	mv	a2,s2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,16
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,20
	mv	a2,t2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,4
	mv	a2,t0
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,sp
	mv	a2,t6
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	beq	t1,s5,.L37
	mv	t1,a7
	j	.L5
.L37:
	li	a7,536870912
	li	t2,1
	addi	a7,a7,1
	li	t0,19
	li	t6,65536
	li	t5,0
	lla	a6,.LANCHOR0+776
	slli	t2,t2,45
	slli	a7,a7,16
	ld	s2,.LC9
	slli	t0,t0,32
	addi	t6,t6,-1025
	ld	s1,.LC10
	li	s0,8192
.L7:
	add	a1,t4,t5
	mv	a2,t2
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a3,t5
	mv	a2,a7
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0+680
	addi	a4,sp,152
.L6:
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
	bne	a5,a6,.L6
	ld	a1,0(a5)
	ld	a2,8(a5)
	lhu	a5,16(a5)
	sd	a1,0(a4)
	sd	a2,8(a4)
	sh	a5,16(a4)
	addi	a1,sp,152
	mv	a2,s2
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,t0
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,t6
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,t3,t5
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	beq	t5,s0,.L38
	mv	t5,t1
	j	.L7
.L38:
	li	a5,4194304
	li	s0,1048576
	li	a4,1
	addi	s1,a5,13
	li	s3,45
	addi	a5,a5,5
	li	s2,65536
	addi	s0,s0,1
	slli	t1,a4,39
	li	t6,0
	lla	t0,.LANCHOR0+1056
	li	a7,106496
	ld	s5,.LC11
	slli	s3,s3,32
	addi	s2,s2,-1025
	slli	s0,s0,15
	slli	s1,s1,13
	slli	t2,a5,13
	slli	a4,a4,45
	li	a6,8192
	li	a0,573440
	li	s4,128
.L11:
	add	a1,t3,t6
	li	a5,98304
.L8:
	or	a2,a5,t1
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a5,a5,128
	addi	a1,a1,256
	bne	a5,a7,.L8
	lla	a5,.LANCHOR0+800
	addi	a2,sp,272
.L9:
	ld	a1,0(a5)
	ld	s7,8(a5)
	ld	s6,16(a5)
	sd	a1,0(a2)
	ld	a1,24(a5)
	sd	s7,8(a2)
	sd	s6,16(a2)
	sd	a1,24(a2)
	addi	a5,a5,32
	addi	a2,a2,32
	bne	a5,t0,.L9
	ld	s6,0(t0)
	lw	a1,8(t0)
	lhu	a5,12(t0)
	sd	s6,0(a2)
	sw	a1,8(a2)
	sh	a5,12(a2)
	addi	a1,sp,272
	mv	a2,s5
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s3
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s2
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,24
	mv	a2,s0
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,32
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,28
	mv	a2,t2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,49152
	add	a1,t4,t6
.L10:
	or	a2,a5,a4
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a5,a5,a6
	addi	a1,a1,256
	bne	a5,a0,.L10
	beq	t6,s4,.L39
	li	t6,128
	j	.L11
.L39:
	li	a2,4194304
	li	a5,2097152
	li	a4,1
	addi	s3,a2,13
	addi	s2,a5,7
	addi	a2,a2,1
	addi	a5,a5,1
	li	s5,59
	li	s4,65536
	slli	t1,a4,39
	li	t2,0
	lla	t0,.LANCHOR0+1424
	li	a7,106496
	ld	s7,.LC12
	slli	s5,s5,32
	addi	s4,s4,-1025
	slli	s3,s3,13
	slli	s2,s2,14
	slli	a4,a4,45
	li	a6,8192
	li	a0,524288
	slli	s1,a2,13
	slli	s0,a5,14
	li	s6,128
.L15:
	add	a1,t3,t2
	li	a5,98304
.L12:
	or	a2,a5,t1
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a5,a5,128
	addi	a1,a1,256
	bne	a5,a7,.L12
	lla	a5,.LANCHOR0+1072
	addi	s8,sp,1224
.L13:
	ld	a2,0(a5)
	ld	s9,8(a5)
	ld	a1,16(a5)
	sd	a2,0(s8)
	ld	a2,24(a5)
	sd	s9,8(s8)
	sd	a1,16(s8)
	sd	a2,24(s8)
	addi	a5,a5,32
	addi	s8,s8,32
	bne	a5,t0,.L13
	lhu	a5,0(t0)
	addi	a1,sp,1224
	mv	a2,s7
	sh	a5,0(s8)
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,s5
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,s4
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,12
	mv	a2,s3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	a1,sp,8
	mv	a2,s2
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a5,0
	add	a1,a3,t2
.L14:
	or	a2,a5,a4
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a5,a5,a6
	addi	a1,a1,256
	bne	a5,a0,.L14
	addi	a1,sp,4
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	mv	a1,sp
	mv	a2,s0
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	beq	t2,s6,.L40
	mv	t2,t6
	j	.L15
.L40:
	li	a5,1073741824
	addi	t2,a5,1
	li	t0,9
	addi	a5,a5,3
	li	t6,65536
	li	a6,0
	lla	a7,.LANCHOR0+1528
	slli	t2,t2,15
	slli	t1,a5,15
	ld	s2,.LC13
	slli	t0,t0,33
	addi	t6,t6,-1025
	ld	s1,.LC14
	li	s0,8192
.L17:
	add	a1,t4,a6
	mv	a2,t2
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,a3,a6
	mv	a2,t1
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0+1432
	addi	a4,sp,40
.L16:
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
	bne	a5,a7,.L16
	ld	a2,0(a5)
	lw	a5,8(a5)
	addi	a1,sp,40
	sd	a2,0(a4)
	sw	a5,8(a4)
	mv	a2,s2
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,0
	mv	a2,t0
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,0
	mv	a1,t6
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	add	a1,t3,a6
	mv	a2,s1
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	beq	a6,s0,.L41
	mv	a6,t5
	j	.L17
.L41:
	ld	s0,1656(sp)
	ld	s1,1648(sp)
	ld	s2,1640(sp)
	ld	s3,1632(sp)
	ld	s4,1624(sp)
	ld	s5,1616(sp)
	ld	s6,1608(sp)
	ld	s7,1600(sp)
	ld	s8,1592(sp)
	ld	s9,1584(sp)
	addi	sp,sp,1664
	jr	ra
	.size	kernel_deriche, .-kernel_deriche
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC6:
	.dword	36030188588498944
	.align	3
.LC7:
	.dword	35184372203520
	.align	3
.LC8:
	.dword	36030291667714048
	.align	3
.LC9:
	.dword	36029286645366784
	.align	3
.LC10:
	.dword	35184372162560
	.align	3
.LC11:
	.dword	36029956660264960
	.align	3
.LC12:
	.dword	36030317437517824
	.align	3
.LC13:
	.dword	36029260875563008
	.align	3
.LC14:
	.dword	35184372195328
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	2048
	.half	0
	.half	56
	.half	1
	.half	256
	.half	57
	.half	0
	.half	-32512
	.half	58
	.half	4
	.half	0
	.half	59
	.half	0
	.half	12
	.half	104
	.half	0
	.half	8
	.half	112
	.half	0
	.half	8
	.half	120
	.half	32
	.half	0
	.half	128
	.half	0
	.half	4096
	.half	248
	.half	0
	.half	8192
	.half	384
	.half	4096
	.half	4096
	.half	392
	.half	0
	.half	-32768
	.half	400
	.half	4096
	.half	0
	.half	408
	.half	-19124
	.half	15841
	.half	440
	.half	8
	.half	256
	.half	441
	.half	266
	.half	280
	.half	449
	.half	10
	.half	264
	.half	465
	.half	6154
	.half	280
	.half	473
	.half	80
	.half	8
	.half	481
	.half	4
	.half	12
	.half	520
	.half	4
	.half	-32256
	.half	528
	.half	0
	.half	2048
	.half	536
	.half	0
	.half	3072
	.half	544
	.half	8
	.half	132
	.half	552
	.half	0
	.half	0
	.half	553
	.half	0
	.half	128
	.half	560
	.half	80
	.half	16
	.half	585
	.half	13764
	.half	-16831
	.half	592
	.half	8
	.half	24
	.half	593
	.half	17661
	.half	16215
	.half	616
	.half	8
	.half	16
	.half	617
	.half	80
	.half	8
	.half	625
	.half	17816
	.half	-16613
	.half	632
	.half	8
	.half	8
	.half	633
	.half	16
	.half	16
	.half	664
	.half	0
	.half	8
	.half	672
	.half	8960
	.half	0
	.half	696
	.half	0
	.half	0
	.half	704
	.half	10240
	.half	0
	.half	728
	.half	65
	.half	256
	.half	729
	.half	0
	.half	256
	.half	730
	.half	0
	.half	0
	.half	731
	.half	4096
	.half	0
	.half	760
	.half	1
	.half	256
	.half	761
	.half	0
	.half	-32512
	.half	762
	.half	4100
	.half	0
	.half	763
	.half	8192
	.half	0
	.half	768
	.half	65
	.half	256
	.half	769
	.half	0
	.half	-32512
	.half	770
	.half	4
	.half	0
	.half	771
	.half	0
	.half	0
	.half	776
	.half	1
	.half	256
	.half	777
	.half	0
	.half	16640
	.half	778
	.half	4
	.half	0
	.half	779
	.zero	4
.LC1:
	.half	-8192
	.half	1023
	.half	24
	.half	8129
	.half	256
	.half	25
	.half	0
	.half	28928
	.half	26
	.half	4100
	.half	0
	.half	27
	.half	4096
	.half	0
	.half	32
	.half	1
	.half	256
	.half	33
	.half	0
	.half	12544
	.half	34
	.half	4
	.half	0
	.half	35
	.half	2048
	.half	0
	.half	40
	.half	1
	.half	256
	.half	41
	.half	0
	.half	28928
	.half	42
	.half	4
	.half	0
	.half	43
	.half	4096
	.half	0
	.half	56
	.half	1
	.half	256
	.half	57
	.half	0
	.half	8448
	.half	58
	.half	4
	.half	0
	.half	59
	.half	-8192
	.half	1023
	.half	64
	.half	8129
	.half	256
	.half	65
	.half	0
	.half	256
	.half	66
	.half	0
	.half	0
	.half	67
	.half	0
	.half	12
	.half	88
	.half	4352
	.half	4
	.half	96
	.half	1072
	.half	0
	.half	104
	.half	4112
	.half	0
	.half	112
	.half	4144
	.half	60
	.half	128
	.half	0
	.half	32
	.half	136
	.half	17816
	.half	-16613
	.half	160
	.half	8
	.half	16
	.half	161
	.half	80
	.half	8
	.half	169
	.half	6154
	.half	80
	.half	177
	.half	138
	.half	224
	.half	185
	.half	24616
	.half	15850
	.half	192
	.half	8
	.half	16
	.half	193
	.half	80
	.half	32
	.half	201
	.half	80
	.half	24
	.half	209
	.half	0
	.half	3072
	.half	232
	.half	0
	.half	128
	.half	240
	.half	0
	.half	1024
	.half	248
	.half	0
	.half	0
	.half	256
	.half	16384
	.half	0
	.half	264
	.half	0
	.half	27652
	.half	272
	.half	3
	.half	0
	.half	273
	.half	8320
	.half	0
	.half	280
	.half	80
	.half	16
	.half	313
	.half	17661
	.half	16215
	.half	320
	.half	8
	.half	24
	.half	321
	.half	10
	.half	80
	.half	337
	.half	5908
	.half	-16836
	.half	344
	.half	8
	.half	16
	.half	345
	.half	128
	.half	8
	.half	353
	.half	0
	.half	0
	.half	392
	.half	3
	.half	0
	.half	417
	.half	3
	.half	0
	.half	561
	.half	8192
	.half	0
	.half	704
	.half	2048
	.half	0
	.half	768
	.half	1
	.half	256
	.half	769
	.half	0
	.half	24832
	.half	770
	.half	4100
	.half	0
	.half	771
	.zero	4
.LC2:
	.half	8192
	.half	0
	.half	16
	.half	65
	.half	256
	.half	17
	.half	0
	.half	256
	.half	18
	.half	0
	.half	0
	.half	19
	.half	0
	.half	0
	.half	96
	.half	3
	.half	0
	.half	241
	.half	3
	.half	0
	.half	385
	.half	8192
	.half	0
	.half	528
	.half	2058
	.half	264
	.half	601
	.half	0
	.half	0
	.half	672
	.half	0
	.half	0
	.half	680
	.half	10240
	.half	0
	.half	736
	.half	65
	.half	256
	.half	737
	.half	0
	.half	20736
	.half	738
	.half	4100
	.half	0
	.half	739
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
	.zero	6
.LC3:
	.half	4096
	.half	0
	.half	48
	.half	1
	.half	256
	.half	49
	.half	0
	.half	-32512
	.half	50
	.half	4100
	.half	0
	.half	51
	.half	0
	.half	0
	.half	56
	.half	1
	.half	256
	.half	57
	.half	0
	.half	16640
	.half	58
	.half	4100
	.half	0
	.half	59
	.half	0
	.half	4
	.half	64
	.half	2113
	.half	262
	.half	65
	.half	0
	.half	-32512
	.half	66
	.half	4
	.half	0
	.half	67
	.half	8192
	.half	0
	.half	128
	.half	16
	.half	1
	.half	136
	.half	80
	.half	16
	.half	193
	.half	522
	.half	280
	.half	201
	.half	80
	.half	24
	.half	209
	.half	16384
	.half	0
	.half	264
	.half	64
	.half	0
	.half	272
	.half	16388
	.half	4
	.half	280
	.half	17816
	.half	-16613
	.half	336
	.half	8
	.half	64
	.half	337
	.half	10
	.half	272
	.half	345
	.half	17661
	.half	16215
	.half	352
	.half	8
	.half	64
	.half	353
	.half	4
	.half	16
	.half	424
	.half	4106
	.half	152
	.half	489
	.half	-19124
	.half	15841
	.half	496
	.half	8
	.half	24
	.half	497
	.half	0
	.half	-32768
	.half	552
	.half	4160
	.half	0
	.half	560
	.half	64
	.half	0
	.half	568
	.half	13764
	.half	-16831
	.half	624
	.half	8
	.half	32
	.half	625
	.half	208
	.half	8
	.half	633
	.half	0
	.half	12
	.half	696
	.half	4096
	.half	1
	.half	704
	.half	2048
	.half	0
	.half	768
	.half	1
	.half	256
	.half	769
	.half	0
	.half	12544
	.half	770
	.half	4100
	.half	0
	.half	771
	.half	2048
	.half	4
	.half	776
	.half	2113
	.half	262
	.half	777
	.half	0
	.half	256
	.half	778
	.half	0
	.half	0
	.half	779
	.zero	2
.LC4:
	.half	2048
	.half	0
	.half	8
	.half	1
	.half	256
	.half	9
	.half	0
	.half	28928
	.half	10
	.half	4100
	.half	0
	.half	11
	.half	0
	.half	1020
	.half	16
	.half	-1983
	.half	257
	.half	17
	.half	0
	.half	28928
	.half	18
	.half	4
	.half	0
	.half	19
	.half	4096
	.half	0
	.half	32
	.half	1
	.half	256
	.half	33
	.half	0
	.half	12544
	.half	34
	.half	4
	.half	0
	.half	35
	.half	4147
	.half	0
	.half	88
	.half	48
	.half	0
	.half	104
	.half	17816
	.half	-16613
	.half	152
	.half	8
	.half	16
	.half	153
	.half	336
	.half	32
	.half	161
	.half	138
	.half	288
	.half	177
	.half	0
	.half	0
	.half	232
	.half	0
	.half	12
	.half	240
	.half	0
	.half	1024
	.half	248
	.half	6
	.half	0
	.half	256
	.half	6154
	.half	80
	.half	305
	.half	80
	.half	24
	.half	313
	.half	17661
	.half	16215
	.half	320
	.half	8
	.half	24
	.half	321
	.half	0
	.half	0
	.half	384
	.half	0
	.half	4
	.half	392
	.half	0
	.half	0
	.half	400
	.half	0
	.half	12
	.half	408
	.half	0
	.half	128
	.half	416
	.half	80
	.half	16
	.half	457
	.half	256
	.half	32
	.half	465
	.half	10
	.half	208
	.half	473
	.half	5908
	.half	-16836
	.half	488
	.half	8
	.half	32
	.half	489
	.half	16384
	.half	0
	.half	528
	.half	0
	.half	8192
	.half	536
	.half	257
	.half	0
	.half	544
	.half	0
	.half	384
	.half	552
	.half	0
	.half	-32384
	.half	560
	.half	4096
	.half	128
	.half	568
	.half	24616
	.half	15850
	.half	600
	.half	8
	.half	64
	.half	601
	.half	80
	.half	8
	.half	641
	.half	12288
	.half	12
	.half	704
	.half	0
	.half	1
	.half	712
	.half	2048
	.half	0
	.half	768
	.half	1
	.half	256
	.half	769
	.half	0
	.half	16640
	.half	770
	.half	4100
	.half	0
	.half	771
	.half	4096
	.half	0
	.half	776
	.half	1
	.half	256
	.half	777
	.half	0
	.half	4352
	.half	778
	.half	4100
	.half	0
	.half	779
	.half	0
	.half	1020
	.half	784
	.half	-1983
	.half	257
	.half	785
	.half	0
	.half	256
	.half	786
	.half	0
	.half	0
	.half	787
	.zero	6
.LC5:
	.half	8192
	.half	0
	.half	56
	.half	65
	.half	256
	.half	57
	.half	0
	.half	256
	.half	58
	.half	0
	.half	0
	.half	59
	.half	0
	.half	16
	.half	128
	.half	3
	.half	0
	.half	273
	.half	3
	.half	0
	.half	417
	.half	0
	.half	8
	.half	560
	.half	2058
	.half	272
	.half	625
	.half	2
	.half	0
	.half	704
	.half	8192
	.half	0
	.half	768
	.half	65
	.half	256
	.half	769
	.half	0
	.half	256
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
	.half	20736
	.half	778
	.half	4
	.half	0
	.half	779
	.ident	"GCC: (g2ee5e430018) 12.2.0"
