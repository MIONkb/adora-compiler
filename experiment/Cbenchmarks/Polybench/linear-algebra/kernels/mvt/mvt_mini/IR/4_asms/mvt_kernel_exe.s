	.file	"mvt_kernel_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	kernel_mvt
	.type	kernel_mvt, @function
kernel_mvt:
	li	a7,10485760
	mv	a5,a2
	addi	a2,a7,1
	mv	a6,a1
	addi	sp,sp,-368
	mv	a1,a0
	slli	a2,a2,16
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,838860800
	addi	a2,a2,3
	mv	a1,a4
	slli	a2,a2,15
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,5
	mv	a1,a5
	slli	a2,a2,37
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0
	addi	a2,sp,176
	lla	a1,.LANCHOR0+192
.L2:
	ld	a7,0(a5)
	ld	t3,8(a5)
	ld	t1,16(a5)
	sd	a7,0(a2)
	ld	a7,24(a5)
	sd	t3,8(a2)
	sd	t1,16(a2)
	sd	a7,24(a2)
	addi	a5,a5,32
	addi	a2,a2,32
	bne	a5,a1,.L2
	addi	a1,sp,176
	ld	a2,.LC2
 #APP
# 33 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,1
	li	a1,0
	slli	a2,a2,37
 #APP
# 57 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,32768
	addi	a1,a1,-1020
	li	a2,0
 #APP
# 65 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC3
	mv	a1,a0
 #APP
# 49 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC4
	mv	a1,a6
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC5
	mv	a1,a4
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC3
	mv	a1,a3
 #APP
# 41 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	lla	a5,.LANCHOR0+192
	addi	a4,sp,8
	lla	a3,.LANCHOR0+352
.L3:
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
	bne	a5,a3,.L3
	ld	a5,0(a5)
	addi	a1,sp,8
	ld	a2,.LC6
	sd	a5,0(a4)
 #APP
# 33 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a2,4194304
	addi	a2,a2,7
	li	a1,0
	slli	a2,a2,34
 #APP
# 57 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	li	a1,32768
	li	a2,1
	addi	a1,a1,-1020
	slli	a2,a2,56
 #APP
# 65 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	ld	a2,.LC7
	mv	a1,a6
 #APP
# 49 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
	addi	sp,sp,368
	jr	ra
	.size	kernel_mvt, .-kernel_mvt
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC2:
	.dword	36029621652815872
	.align	3
.LC3:
	.dword	72058281232801792
	.align	3
.LC4:
	.dword	72058281232793600
	.align	3
.LC5:
	.dword	72085081828687872
	.align	3
.LC6:
	.dword	108087112611528704
	.align	3
.LC7:
	.dword	72058281232809984
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	10240
	.half	-24576
	.half	24
	.half	-2496
	.half	327
	.half	25
	.half	0
	.half	256
	.half	26
	.half	0
	.half	0
	.half	27
	.half	0
	.half	0
	.half	104
	.half	0
	.half	24576
	.half	248
	.half	-32768
	.half	0
	.half	256
	.half	8
	.half	96
	.half	329
	.half	1
	.half	0
	.half	408
	.half	0
	.half	512
	.half	416
	.half	26
	.half	8
	.half	481
	.half	0
	.half	4096
	.half	482
	.half	1536
	.half	1034
	.half	483
	.half	0
	.half	0
	.half	484
	.half	0
	.half	0
	.half	552
	.half	0
	.half	4096
	.half	560
	.half	8202
	.half	208
	.half	617
	.half	0
	.half	0
	.half	688
	.half	0
	.half	0
	.half	696
	.half	0
	.half	0
	.half	704
	.half	0
	.half	-24576
	.half	752
	.half	64
	.half	320
	.half	753
	.half	0
	.half	256
	.half	754
	.half	0
	.half	0
	.half	755
	.half	4096
	.half	-24576
	.half	768
	.half	64
	.half	320
	.half	769
	.half	0
	.half	-32512
	.half	770
	.half	4
	.half	0
	.half	771
	.half	8192
	.half	-24576
	.half	776
	.half	64
	.half	320
	.half	777
	.half	0
	.half	256
	.half	778
	.half	0
	.half	0
	.half	779
.LC1:
	.half	26
	.half	24
	.half	473
	.half	0
	.half	4096
	.half	474
	.half	1280
	.half	1034
	.half	475
	.half	0
	.half	0
	.half	476
	.half	64
	.half	0
	.half	544
	.half	0
	.half	0
	.half	552
	.half	8
	.half	280
	.half	609
	.half	6154
	.half	264
	.half	625
	.half	16
	.half	0
	.half	680
	.half	0
	.half	0
	.half	688
	.half	256
	.half	0
	.half	696
	.half	0
	.half	0
	.half	704
	.half	0
	.half	-24571
	.half	744
	.half	31296
	.half	326
	.half	745
	.half	0
	.half	256
	.half	746
	.half	0
	.half	0
	.half	747
	.half	10240
	.half	-24576
	.half	760
	.half	-2496
	.half	327
	.half	761
	.half	0
	.half	256
	.half	762
	.half	0
	.half	0
	.half	763
	.half	4096
	.half	-24576
	.half	768
	.half	64
	.half	320
	.half	769
	.half	0
	.half	28928
	.half	770
	.half	4
	.half	0
	.half	771
	.half	0
	.half	-24576
	.half	776
	.half	64
	.half	320
	.half	777
	.half	0
	.half	256
	.half	778
	.half	0
	.half	0
	.half	779
	.ident	"GCC: (g2ee5e430018) 12.2.0"
