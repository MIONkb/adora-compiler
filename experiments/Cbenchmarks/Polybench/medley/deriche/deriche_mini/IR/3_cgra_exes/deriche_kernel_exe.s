	.file	"deriche_kernel_exe.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
.Ltext0:
	.cfi_sections	.debug_frame
	.file 0 "/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/medley/deriche/deriche_mini/IR/3_cgra_exes" "deriche_kernel_exe.c"
	.align	1
	.globl	kernel_deriche
	.type	kernel_deriche, @function
kernel_deriche:
.LFB10:
	.file 1 "deriche_kernel_exe.c"
	.loc 1 18 72
	.cfi_startproc
.LVL0:
	.loc 1 19 3
	.loc 1 20 3
	.loc 1 21 3
	.loc 1 22 3
	.loc 1 23 3
	.loc 1 24 3
	.loc 1 25 3
	.loc 1 26 3
	.loc 1 27 3
	.loc 1 28 3
	.loc 1 29 3
	.loc 1 30 3
	.loc 1 31 3
	.loc 1 32 3
	.loc 1 33 3
	.loc 1 34 3
	.loc 1 35 3
	.loc 1 36 3
	.loc 1 37 3
	.loc 1 38 3
	.loc 1 39 3
.LBB61:
	.loc 1 39 8
	.loc 1 39 31
.LBE61:
	.loc 1 18 72 is_stmt 0
	addi	sp,sp,-976
	.cfi_def_cfa_offset 976
	sd	s0,968(sp)
	sd	s1,960(sp)
	sd	s2,952(sp)
	sd	s3,944(sp)
	sd	s4,936(sp)
	sd	s5,928(sp)
	sd	s6,920(sp)
	sd	s7,912(sp)
	sd	s8,904(sp)
	sd	s9,896(sp)
	sd	s10,888(sp)
	.cfi_offset 8, -8
	.cfi_offset 9, -16
	.cfi_offset 18, -24
	.cfi_offset 19, -32
	.cfi_offset 20, -40
	.cfi_offset 21, -48
	.cfi_offset 22, -56
	.cfi_offset 23, -64
	.cfi_offset 24, -72
	.cfi_offset 25, -80
	.cfi_offset 26, -88
.LBB122:
.LBB62:
	.loc 1 45 5
	lla	s0,_task_id
	lbu	a7,0(s0)
.LBE62:
.LBB66:
.LBB67:
	.file 2 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h"
	.loc 2 41 11
	li	t4,1048576
.LBE67:
.LBE66:
.LBB72:
.LBB73:
	li	t5,4194304
.LBE73:
.LBE72:
.LBB78:
.LBB69:
	addi	s1,t4,1
.LBE69:
.LBE78:
.LBB79:
.LBB80:
.LBB81:
	.loc 2 49 74
	li	s4,63
.LBE81:
.LBE80:
.LBE79:
.LBB102:
.LBB103:
	.loc 2 41 11
	addi	t4,t4,3
.LBE103:
.LBE102:
.LBB107:
.LBB85:
.LBB86:
	.loc 2 57 11
	li	s3,1
.LBB87:
	.loc 2 58 2
	li	s2,61440
.LBE87:
.LBE86:
.LBE85:
.LBE107:
.LBB108:
.LBB75:
	.loc 2 41 11
	addi	t5,t5,9
.LBE75:
.LBE108:
.LBE122:
	.loc 1 18 72
	mv	s5,a2
.LBB123:
.LBB109:
.LBB92:
.LBB93:
.LBB94:
	.loc 2 26 2
	mv	t1,a7
	li	t3,0
	ld	s10,.LC2
	lla	s6,.LANCHOR0+360
	ld	s9,.LC3
	ld	s8,.LC4
.LBE94:
.LBE93:
.LBE92:
.LBB97:
.LBB83:
	.loc 2 49 74
	slli	s4,s4,32
.LBE83:
.LBE97:
.LBB98:
.LBB90:
	.loc 2 57 11
	slli	s3,s3,61
.LBB88:
	.loc 2 58 2
	addi	s2,s2,-1487
.LBE88:
.LBE90:
.LBE98:
.LBE109:
.LBB110:
.LBB70:
	.loc 2 41 11
	slli	s1,s1,15
.LBE70:
.LBE110:
.LBB111:
.LBB105:
	slli	t4,t4,15
.LBE105:
.LBE111:
.LBB112:
.LBB76:
	slli	t5,t5,13
	li	s7,8192
.LVL1:
.L3:
.LBE76:
.LBE112:
.LBB113:
	.loc 1 42 5 is_stmt 1 discriminator 3
	.loc 1 43 5 discriminator 3
	.loc 1 44 5 discriminator 3
	.loc 1 45 5 discriminator 3
.LBB63:
.LBB64:
	.loc 2 33 2 discriminator 3
	.loc 2 33 67 is_stmt 0 discriminator 3
	slli	a6,t1,56
.LVL2:
.LBB65:
	.loc 2 34 2 is_stmt 1 discriminator 3
	add	a1,a0,t3
	.loc 2 34 2 discriminator 3
	or	a2,a6,s10
.LVL3:
	.loc 2 34 2 discriminator 3
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE65:
	.loc 2 34 61 discriminator 3
	.loc 2 35 2 discriminator 3
.LVL4:
.LBE64:
.LBE63:
	.loc 1 46 5 discriminator 3
.LBE113:
.LBB114:
	.loc 1 51 5 discriminator 3
	.loc 1 51 29 is_stmt 0 discriminator 3
	lla	a5,.LANCHOR0
	addi	a4,sp,40
.L2:
	ld	a2,0(a5)
	ld	t6,8(a5)
	ld	t2,16(a5)
	sd	a2,0(a4)
	ld	t0,24(a5)
	sd	t6,8(a4)
	ld	t6,32(a5)
	sd	t2,16(a4)
	sd	t0,24(a4)
	sd	t6,32(a4)
	addi	a5,a5,40
	addi	a4,a4,40
	bne	a5,s6,.L2
	ld	a1,0(a5)
	ld	a2,8(a5)
	lhu	a5,16(a5)
	sd	a1,0(a4)
	sd	a2,8(a4)
	sh	a5,16(a4)
	.loc 1 117 5 is_stmt 1 discriminator 3
.LVL5:
.LBB99:
.LBB96:
	.loc 2 25 2 discriminator 3
.LBB95:
	.loc 2 26 2 discriminator 3
	addi	a1,sp,40
.LVL6:
	.loc 2 26 2 discriminator 3
	or	a2,a6,s9
.LVL7:
	.loc 2 26 2 discriminator 3
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE95:
	.loc 2 26 61 discriminator 3
	.loc 2 27 2 discriminator 3
.LVL8:
.LBE96:
.LBE99:
	.loc 1 118 5 discriminator 3
.LBB100:
.LBB84:
	.loc 2 48 2 discriminator 3
	.loc 2 49 2 discriminator 3
.LBB82:
	.loc 2 50 2 discriminator 3
	li	a1,0
	.loc 2 50 2 discriminator 3
	or	a2,a6,s4
.LVL9:
	.loc 2 50 2 discriminator 3
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE82:
	.loc 2 50 43 discriminator 3
	.loc 2 51 2 discriminator 3
.LVL10:
.LBE84:
.LBE100:
	.loc 1 119 5 discriminator 3
.LBB101:
.LBB91:
	.loc 2 56 2 discriminator 3
	.loc 2 57 2 discriminator 3
.LBB89:
	.loc 2 58 2 discriminator 3
	mv	a1,s2
	.loc 2 58 2 discriminator 3
	or	a2,a6,s3
.LVL11:
	.loc 2 58 2 discriminator 3
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE89:
	.loc 2 58 43 discriminator 3
	.loc 2 59 2 discriminator 3
.LVL12:
.LBE91:
.LBE101:
.LBE114:
	.loc 1 123 5 discriminator 3
.LBB115:
.LBB71:
	.loc 2 41 2 discriminator 3
.LBB68:
	.loc 2 42 2 discriminator 3
	addi	a1,sp,32
.LVL13:
	.loc 2 42 2 discriminator 3
	or	a2,a6,s1
.LVL14:
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE68:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL15:
.LBE71:
.LBE115:
	.loc 1 128 5 discriminator 3
.LBB116:
.LBB106:
	.loc 2 41 2 discriminator 3
.LBB104:
	.loc 2 42 2 discriminator 3
	addi	a1,sp,28
.LVL16:
	.loc 2 42 2 discriminator 3
	or	a2,a6,t4
.LVL17:
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE104:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL18:
.LBE106:
.LBE116:
	.loc 1 133 5 discriminator 3
.LBB117:
.LBB77:
	.loc 2 41 2 discriminator 3
.LBB74:
	.loc 2 42 2 discriminator 3
	addi	a1,sp,36
.LVL19:
	.loc 2 42 2 discriminator 3
	or	a2,a6,t5
.LVL20:
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE74:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL21:
.LBE77:
.LBE117:
.LBB118:
	.loc 1 138 5 discriminator 3
	.loc 1 139 5 discriminator 3
	.loc 1 140 5 discriminator 3
	.loc 1 141 5 discriminator 3
.LBB119:
.LBB120:
	.loc 2 41 2 discriminator 3
.LBB121:
	.loc 2 42 2 discriminator 3
	add	a1,s5,t3
.LVL22:
	.loc 2 42 2 discriminator 3
	or	a2,a6,s8
.LVL23:
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE121:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL24:
.LBE120:
.LBE119:
	.loc 1 142 5 discriminator 3
.LBE118:
	.loc 1 145 5 discriminator 3
	.loc 1 145 13 is_stmt 0 discriminator 3
	addiw	t1,t1,1
	andi	t1,t1,0xff
	.loc 1 39 44 is_stmt 1 discriminator 3
.LVL25:
	.loc 1 39 31 discriminator 3
	beq	t3,s7,.L12
	li	t3,8192
.LVL26:
	j	.L3
.LVL27:
.L12:
.LBE123:
.LBB124:
.LBB125:
.LBB126:
	.loc 2 41 11 is_stmt 0
	li	t4,4194304
.LBE126:
.LBE125:
.LBB131:
.LBB132:
	li	t0,1048576
.LBE132:
.LBE131:
.LBB138:
.LBB139:
.LBB140:
	.loc 2 57 11
	li	t2,1
.LBE140:
.LBE139:
.LBE138:
.LBB168:
.LBB128:
	.loc 2 41 11
	addi	s1,t4,13
.LBE128:
.LBE168:
.LBE124:
.LBB196:
	.loc 1 145 13
	addiw	a7,a7,2
.LBE196:
.LBB197:
.LBB169:
.LBB170:
	.loc 2 41 11
	addi	t4,t4,5
.LBE170:
.LBE169:
.LBB174:
.LBB148:
.LBB149:
	.loc 2 49 74
	li	s3,19
.LBE149:
.LBE148:
.LBB153:
.LBB144:
.LBB141:
	.loc 2 58 2
	li	s2,61440
.LBE141:
.LBE144:
.LBE153:
.LBE174:
.LBB175:
.LBB135:
	.loc 2 41 11
	addi	t0,t0,3
.LBE135:
.LBE175:
.LBB176:
.LBB154:
.LBB145:
	.loc 2 57 11
	slli	s6,t2,61
.LBE145:
.LBE154:
.LBE176:
.LBE197:
.LBB198:
	.loc 1 145 13
	andi	a7,a7,0xff
.LBE198:
.LBB199:
.LBB177:
.LBB155:
.LBB156:
.LBB157:
	.loc 2 26 2
	li	t1,0
	ld	s9,.LC5
	lla	s4,.LANCHOR0+832
	ld	s8,.LC6
	ld	s7,.LC7
.LBE157:
.LBE156:
.LBE155:
.LBB162:
.LBB151:
	.loc 2 49 74
	slli	s3,s3,34
.LBE151:
.LBE162:
.LBB163:
.LBB146:
.LBB142:
	.loc 2 58 2
	addi	s2,s2,-1487
.LBE142:
.LBE146:
.LBE163:
.LBE177:
.LBB178:
.LBB136:
	.loc 2 41 74
	slli	t2,t2,35
	.loc 2 41 11
	slli	t0,t0,15
.LBE136:
.LBE178:
.LBB179:
.LBB129:
	slli	s1,s1,13
.LBE129:
.LBE179:
.LBB180:
.LBB172:
	slli	t4,t4,13
.LBE172:
.LBE180:
	.loc 1 150 31
	li	s5,8192
.LVL28:
.L5:
.LBB181:
	.loc 1 153 5 is_stmt 1 discriminator 3
	.loc 1 154 5 discriminator 3
	.loc 1 155 5 discriminator 3
	.loc 1 156 5 discriminator 3
.LBB182:
.LBB183:
	.loc 2 33 2 discriminator 3
	.loc 2 33 67 is_stmt 0 discriminator 3
	slli	a6,a7,56
.LVL29:
.LBB184:
	.loc 2 34 2 is_stmt 1 discriminator 3
	add	a1,a0,t1
	.loc 2 34 2 discriminator 3
	or	a2,a6,s9
.LVL30:
	.loc 2 34 2 discriminator 3
 #APP
# 34 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE184:
	.loc 2 34 61 discriminator 3
	.loc 2 35 2 discriminator 3
.LVL31:
.LBE183:
.LBE182:
	.loc 1 157 5 discriminator 3
.LBE181:
.LBB185:
	.loc 1 162 5 discriminator 3
	.loc 1 162 29 is_stmt 0 discriminator 3
	lla	a5,.LANCHOR0+384
	addi	a4,sp,424
.L4:
	ld	a2,0(a5)
	ld	t6,8(a5)
	ld	t5,16(a5)
	sd	a2,0(a4)
	ld	a2,24(a5)
	sd	t6,8(a4)
	sd	t5,16(a4)
	sd	a2,24(a4)
	addi	a5,a5,32
	addi	a4,a4,32
	bne	a5,s4,.L4
	ld	a5,0(a5)
.LBB164:
.LBB160:
.LBB158:
	.loc 2 26 2 discriminator 3
	addi	a1,sp,424
	or	a2,a6,s8
.LBE158:
.LBE160:
.LBE164:
	.loc 1 162 29 discriminator 3
	sd	a5,0(a4)
	.loc 1 246 5 is_stmt 1 discriminator 3
.LVL32:
.LBB165:
.LBB161:
	.loc 2 25 2 discriminator 3
.LBB159:
	.loc 2 26 2 discriminator 3
	.loc 2 26 2 discriminator 3
	.loc 2 26 2 discriminator 3
 #APP
# 26 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 1) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE159:
	.loc 2 26 61 discriminator 3
	.loc 2 27 2 discriminator 3
.LVL33:
.LBE161:
.LBE165:
	.loc 1 247 5 discriminator 3
.LBB166:
.LBB152:
	.loc 2 48 2 discriminator 3
	.loc 2 49 2 discriminator 3
.LBB150:
	.loc 2 50 2 discriminator 3
	li	a1,0
	.loc 2 50 2 discriminator 3
	or	a2,a6,s3
.LVL34:
	.loc 2 50 2 discriminator 3
 #APP
# 50 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 3) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE150:
	.loc 2 50 43 discriminator 3
	.loc 2 51 2 discriminator 3
.LVL35:
.LBE152:
.LBE166:
	.loc 1 248 5 discriminator 3
.LBB167:
.LBB147:
	.loc 2 56 2 discriminator 3
	.loc 2 57 2 discriminator 3
.LBB143:
	.loc 2 58 2 discriminator 3
	mv	a1,s2
	.loc 2 58 2 discriminator 3
	or	a2,a6,s6
.LVL36:
	.loc 2 58 2 discriminator 3
 #APP
# 58 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 4) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE143:
	.loc 2 58 43 discriminator 3
	.loc 2 59 2 discriminator 3
.LVL37:
.LBE147:
.LBE167:
.LBE185:
	.loc 1 252 5 discriminator 3
.LBB186:
.LBB137:
	.loc 2 41 2 discriminator 3
.LBB133:
	.loc 2 42 2 discriminator 3
.LBE133:
	.loc 2 41 74 is_stmt 0 discriminator 3
	or	a5,a6,t2
.LBB134:
	.loc 2 42 2 discriminator 3
	addi	a1,sp,16
.LVL38:
	.loc 2 42 2 is_stmt 1 discriminator 3
	or	a2,a6,t0
.LVL39:
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE134:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL40:
.LBE137:
.LBE186:
	.loc 1 257 5 discriminator 3
.LBB187:
.LBB130:
	.loc 2 41 2 discriminator 3
.LBB127:
	.loc 2 42 2 discriminator 3
	addi	a1,sp,12
.LVL41:
	.loc 2 42 2 discriminator 3
	or	a2,a6,s1
.LVL42:
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE127:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL43:
.LBE130:
.LBE187:
	.loc 1 262 5 discriminator 3
.LBB188:
.LBB173:
	.loc 2 41 2 discriminator 3
.LBB171:
	.loc 2 42 2 discriminator 3
	addi	a1,sp,20
.LVL44:
	.loc 2 42 2 discriminator 3
	or	a2,a6,t4
.LVL45:
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE171:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL46:
.LBE173:
.LBE188:
.LBB189:
	.loc 1 267 5 discriminator 3
	.loc 1 268 5 discriminator 3
	.loc 1 269 5 discriminator 3
	.loc 1 270 5 discriminator 3
.LBB190:
.LBB191:
	.loc 2 41 2 discriminator 3
.LBB192:
	.loc 2 42 2 discriminator 3
	add	a1,a3,t1
.LVL47:
	.loc 2 42 2 discriminator 3
	or	a2,a6,s7
.LVL48:
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE192:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL49:
.LBE191:
.LBE190:
	.loc 1 271 5 discriminator 3
.LBE189:
	.loc 1 276 5 discriminator 3
.LBB193:
.LBB194:
	.loc 2 41 2 discriminator 3
.LBB195:
	.loc 2 42 2 discriminator 3
	addi	a1,sp,24
.LVL50:
	.loc 2 42 2 discriminator 3
	mv	a2,a5
	.loc 2 42 2 discriminator 3
 #APP
# 42 "/home/jhlou/chipyard/generators/fdra/software/tests/include/ISA.h" 1
	.word 0b0001011 | (0 << (7)) | (1 << (7+5)) | (1 << (7+5+1)) | (0 << (7+5+2)) | (11 << (7+5+3)) | (12 << (7+5+3+5)) | ((((~(~0 << 7) << 0) & 2) >> 0) << (7+5+3+5+5))
	
# 0 "" 2
 #NO_APP
.LBE195:
	.loc 2 42 62 discriminator 3
	.loc 2 43 2 discriminator 3
.LVL51:
.LBE194:
.LBE193:
	.loc 1 279 5 discriminator 3
	.loc 1 279 13 is_stmt 0 discriminator 3
	addiw	a7,a7,1
	andi	a7,a7,0xff
	sb	a7,0(s0)
	.loc 1 150 44 is_stmt 1 discriminator 3
.LVL52:
	.loc 1 150 31 discriminator 3
	beq	t1,s5,.L13
	mv	t1,t3
.LVL53:
	j	.L5
.LVL54:
.L13:
.LBE199:
	.loc 1 284 1 is_stmt 0
	ld	s0,968(sp)
	.cfi_restore 8
	ld	s1,960(sp)
	.cfi_restore 9
	ld	s2,952(sp)
	.cfi_restore 18
	ld	s3,944(sp)
	.cfi_restore 19
	ld	s4,936(sp)
	.cfi_restore 20
	ld	s5,928(sp)
	.cfi_restore 21
	ld	s6,920(sp)
	.cfi_restore 22
	ld	s7,912(sp)
	.cfi_restore 23
	ld	s8,904(sp)
	.cfi_restore 24
	ld	s9,896(sp)
	.cfi_restore 25
	ld	s10,888(sp)
	.cfi_restore 26
	addi	sp,sp,976
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE10:
	.size	kernel_deriche, .-kernel_deriche
	.globl	_task_id
	.section	.srodata.cst8,"aM",@progbits,8
	.align	3
.LC2:
	.dword	4611721202799542272
	.align	3
.LC3:
	.dword	4647716438944120832
	.align	3
.LC4:
	.dword	35184372129792
	.align	3
.LC5:
	.dword	4611721202799509504
	.align	3
.LC6:
	.dword	4647716773951569920
	.align	3
.LC7:
	.dword	35184372203520
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC0:
	.half	0
	.half	0
	.half	40
	.half	1
	.half	256
	.half	41
	.half	0
	.half	0
	.half	42
	.half	0
	.half	2314
	.half	43
	.half	32
	.half	0
	.half	44
	.half	10240
	.half	0
	.half	48
	.half	65
	.half	256
	.half	49
	.half	0
	.half	0
	.half	50
	.half	0
	.half	2314
	.half	51
	.half	0
	.half	0
	.half	52
	.half	0
	.half	48
	.half	112
	.half	51
	.half	1
	.half	120
	.half	17816
	.half	-16613
	.half	192
	.half	8
	.half	32
	.half	193
	.half	0
	.half	24576
	.half	256
	.half	4096
	.half	-32768
	.half	264
	.half	4096
	.half	0
	.half	272
	.half	266
	.half	264
	.half	337
	.half	80
	.half	8
	.half	345
	.half	0
	.half	20480
	.half	346
	.half	0
	.half	-32752
	.half	347
	.half	64
	.half	0
	.half	348
	.half	0
	.half	3072
	.half	408
	.half	4
	.half	128
	.half	416
	.half	4106
	.half	280
	.half	473
	.half	10
	.half	280
	.half	481
	.half	80
	.half	8
	.half	489
	.half	0
	.half	20480
	.half	490
	.half	768
	.half	-32752
	.half	491
	.half	64
	.half	0
	.half	492
	.half	64
	.half	0
	.half	544
	.half	0
	.half	0
	.half	552
	.half	16384
	.half	0
	.half	560
	.half	0
	.half	0
	.half	569
	.half	13764
	.half	-16831
	.half	608
	.half	8
	.half	256
	.half	609
	.half	336
	.half	24
	.half	617
	.half	0
	.half	20480
	.half	618
	.half	0
	.half	-32752
	.half	619
	.half	64
	.half	0
	.half	620
	.half	-19124
	.half	15841
	.half	624
	.half	8
	.half	24
	.half	625
	.half	17661
	.half	16215
	.half	632
	.half	8
	.half	64
	.half	633
	.half	12288
	.half	0
	.half	672
	.half	0
	.half	1
	.half	680
	.half	18
	.half	0
	.half	688
	.half	0
	.half	0
	.half	696
	.half	768
	.half	0
	.half	712
	.half	2048
	.half	0
	.half	736
	.half	1
	.half	256
	.half	737
	.half	0
	.half	0
	.half	738
	.half	0
	.half	2186
	.half	739
	.half	32
	.half	0
	.half	740
	.half	8192
	.half	0
	.half	752
	.half	65
	.half	256
	.half	753
	.half	0
	.half	0
	.half	754
	.half	0
	.half	10
	.half	755
	.half	0
	.half	0
	.half	784
	.half	1
	.half	256
	.half	785
	.half	0
	.half	0
	.half	786
	.half	0
	.half	2186
	.half	787
	.half	0
	.half	0
	.half	788
	.zero	6
.LC1:
	.half	0
	.half	0
	.half	8
	.half	1
	.half	256
	.half	9
	.half	0
	.half	0
	.half	10
	.half	0
	.half	2186
	.half	11 
	/////// 000b

	.half	32
	.half	0
	.half	12
	.half	-8192
	.half	1023
	.half	40
	.half	8129
	.half	256
	.half	41
	.half	0
	.half	0
	.half	42
	/////// 002a

	.half	0
	.half	10
	.half	43
	.half	2048
	.half	0
	.half	48
	.half	1
	.half	256
	.half	49
	.half	0
	.half	0
	.half	50
	/////// 0032

	.half	0
	.half	2090
	.half	51
	.half	0
	.half	0
	.half	52
	.half	2
	.half	0
	.half	88
	.half	0
	.half	2
	.half	96
	/////// 0060
	
	.half	0
	.half	50
	.half	104
	.half	0
	.half	0
	.half	112
	.half	16
	.half	0
	.half	120
	.half	208
	.half	24
	.half	185
	/////// 00B9

	.half	0
	.half	20480
	.half	186
	.half	0
	.half	-32752
	.half	187
	.half	64
	.half	0
	.half	188
	.half	5908
	.half	-16836
	.half	192
	.half	8
	.half	24
	.half	193
	.half	0
	.half	24576
	.half	248
	.half	128
	.half	0
	.half	256
	.half	16384
	.half	0
	.half	264
	.half	384
	.half	16
	.half	321
	.half	10
	.half	88
	.half	337
	.half	17816
	.half	-16613
	.half	344
	.half	8
	.half	32
	.half	345
	.half	0
	.half	0
	.half	400
	.half	16576
	.half	0
	.half	408
	.half	16384
	.half	-32768
	.half	416
	.half	4096
	.half	0
	.half	424
	.half	80
	.half	8
	.half	473
	.half	0
	.half	20480
	.half	474
	.half	0
	.half	-32752
	.half	475
	.half	64
	.half	0
	.half	476
	.half	2058
	.half	96
	.half	481
	.half	6154
	.half	88
	.half	489
	.half	208
	.half	8
	.half	497
	.half	0
	.half	20480
	.half	498
	.half	0
	.half	-32752
	.half	499
	.half	64
	.half	0
	.half	500
	.half	0
	.half	0
	.half	552
	.half	0
	.half	3072
	.half	560
	.half	4
	.half	0
	.half	561
	.half	0
	.half	132
	.half	568
	.half	0
	.half	0
	.half	569
	.half	24616
	.half	15850
	.half	624
	.half	8
	.half	8
	.half	625
	.half	17661
	.half	16215
	.half	632
	.half	8
	.half	16
	.half	633
	.half	80
	.half	8
	.half	641
	.half	0
	.half	20480
	.half	642
	.half	512
	.half	-32752
	.half	643
	.half	64
	.half	0
	.half	644
	.half	8192
	.half	0
	.half	704
	.half	8960
	.half	0
	.half	712
	.half	2048
	.half	0
	.half	768
	.half	1
	.half	256
	.half	769
	.half	0
	.half	0
	.half	770
	.half	0
	.half	2186
	.half	771
	.half	32
	.half	0
	.half	772
	.half	0
	.half	0
	.half	776
	.half	1
	.half	256
	.half	777
	.half	0
	.half	0
	.half	778
	.half	0
	.half	2282
	.half	779
	.half	32
	.half	0
	.half	780
	.half	-4096
	.half	1023
	.half	784
	.half	8129
	.half	256
	.half	785
	.half	0
	.half	0
	.half	786
	.half	0
	.half	2282
	.half	787
	.half	0
	.half	0
	.half	788
	.section	.sbss,"aw",@nobits
	.type	_task_id, @object
	.size	_task_id, 1
_task_id:
	.zero	1
	.text
.Letext0:
	.file 3 "/home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include/machine/_default_types.h"
	.file 4 "/home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include/sys/_stdint.h"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.4byte	0xc85
	.2byte	0x5
	.byte	0x1
	.byte	0x8
	.4byte	.Ldebug_abbrev0
	.byte	0x24
	.4byte	.LASF60
	.byte	0x1d
	.4byte	.LASF0
	.4byte	.LASF1
	.8byte	.Ltext0
	.8byte	.Letext0-.Ltext0
	.4byte	.Ldebug_line0
	.byte	0x7
	.byte	0x1
	.byte	0x6
	.4byte	.LASF2
	.byte	0xe
	.4byte	.LASF8
	.byte	0x3
	.byte	0x2b
	.byte	0x18
	.4byte	0x41
	.byte	0x7
	.byte	0x1
	.byte	0x8
	.4byte	.LASF3
	.byte	0x7
	.byte	0x2
	.byte	0x5
	.4byte	.LASF4
	.byte	0x7
	.byte	0x2
	.byte	0x7
	.4byte	.LASF5
	.byte	0x14
	.4byte	0x4f
	.byte	0x25
	.byte	0x4
	.byte	0x5
	.string	"int"
	.byte	0x7
	.byte	0x4
	.byte	0x7
	.4byte	.LASF6
	.byte	0x7
	.byte	0x8
	.byte	0x5
	.4byte	.LASF7
	.byte	0xe
	.4byte	.LASF9
	.byte	0x3
	.byte	0x69
	.byte	0x19
	.4byte	0x7c
	.byte	0x7
	.byte	0x8
	.byte	0x7
	.4byte	.LASF10
	.byte	0xe
	.4byte	.LASF11
	.byte	0x3
	.byte	0xe8
	.byte	0x1a
	.4byte	0x7c
	.byte	0xe
	.4byte	.LASF12
	.byte	0x4
	.byte	0x18
	.byte	0x13
	.4byte	0x35
	.byte	0xe
	.4byte	.LASF13
	.byte	0x4
	.byte	0x3c
	.byte	0x14
	.4byte	0x70
	.byte	0xe
	.4byte	.LASF14
	.byte	0x4
	.byte	0x52
	.byte	0x15
	.4byte	0x83
	.byte	0x7
	.byte	0x8
	.byte	0x5
	.4byte	.LASF15
	.byte	0x7
	.byte	0x10
	.byte	0x4
	.4byte	.LASF16
	.byte	0x26
	.byte	0x8
	.byte	0x7
	.byte	0x1
	.byte	0x8
	.4byte	.LASF17
	.byte	0x7
	.byte	0x8
	.byte	0x7
	.4byte	.LASF18
	.byte	0x27
	.4byte	.LASF37
	.byte	0x1
	.byte	0xa
	.byte	0x9
	.4byte	0x8f
	.byte	0x9
	.byte	0x3
	.8byte	_task_id
	.byte	0x28
	.4byte	.LASF61
	.byte	0x1
	.byte	0x12
	.byte	0x6
	.8byte	.LFB10
	.8byte	.LFE10-.LFB10
	.byte	0x1
	.byte	0x9c
	.4byte	0xa50
	.byte	0x1b
	.4byte	.LASF19
	.byte	0x1b
	.4byte	0xc1
	.byte	0x1
	.byte	0x5a
	.byte	0x1c
	.4byte	.LASF20
	.byte	0x28
	.4byte	0xc1
	.4byte	.LLST0
	.byte	0x1c
	.4byte	.LASF21
	.byte	0x35
	.4byte	0xc1
	.4byte	.LLST1
	.byte	0x1b
	.4byte	.LASF22
	.byte	0x42
	.4byte	0xc1
	.byte	0x1
	.byte	0x5d
	.byte	0xb
	.4byte	.LASF23
	.byte	0x13
	.4byte	0xa50
	.byte	0x3
	.byte	0x91
	.byte	0xbc,0x78
	.byte	0xb
	.4byte	.LASF24
	.byte	0x15
	.4byte	0xa50
	.byte	0x3
	.byte	0x91
	.byte	0xc0,0x78
	.byte	0x15
	.4byte	.LASF25
	.byte	0x17
	.4byte	0xa50
	.byte	0x4
	.4byte	0
	.byte	0x15
	.4byte	.LASF26
	.byte	0x19
	.4byte	0xa50
	.byte	0x4
	.4byte	0
	.byte	0xb
	.4byte	.LASF27
	.byte	0x1b
	.4byte	0xa50
	.byte	0x3
	.byte	0x91
	.byte	0xc4,0x78
	.byte	0xb
	.4byte	.LASF28
	.byte	0x1d
	.4byte	0xa50
	.byte	0x3
	.byte	0x91
	.byte	0xc8,0x78
	.byte	0xb
	.4byte	.LASF29
	.byte	0x1f
	.4byte	0xa50
	.byte	0x3
	.byte	0x91
	.byte	0xcc,0x78
	.byte	0xb
	.4byte	.LASF30
	.byte	0x21
	.4byte	0xa50
	.byte	0x3
	.byte	0x91
	.byte	0xd0,0x78
	.byte	0x15
	.4byte	.LASF31
	.byte	0x23
	.4byte	0xa50
	.byte	0x4
	.4byte	0
	.byte	0xb
	.4byte	.LASF32
	.byte	0x25
	.4byte	0xa50
	.byte	0x3
	.byte	0x91
	.byte	0xd4,0x78
	.byte	0x11
	.4byte	.LLRL2
	.4byte	0x5d1
	.byte	0xa
	.4byte	.LASF33
	.byte	0x27
	.byte	0xc
	.4byte	0x5b
	.4byte	.LLST3
	.byte	0x11
	.4byte	.LLRL4
	.4byte	0x28c
	.byte	0xa
	.4byte	.LASF34
	.byte	0x2a
	.byte	0xe
	.4byte	0x9b
	.4byte	.LLST5
	.byte	0xa
	.4byte	.LASF35
	.byte	0x2b
	.byte	0xe
	.4byte	0x9b
	.4byte	.LLST6
	.byte	0x16
	.4byte	.LASF36
	.byte	0x2c
	.4byte	0x9b
	.byte	0x17
	.4byte	0xbb2
	.8byte	.LBB63
	.8byte	.LBE63-.LBB63
	.byte	0x2d
	.byte	0x1
	.4byte	0xbf7
	.4byte	.LLST7
	.byte	0x1
	.4byte	0xbec
	.4byte	.LLST8
	.byte	0x1
	.4byte	0xbe1
	.4byte	.LLST9
	.byte	0x1
	.4byte	0xbd6
	.4byte	.LLST10
	.byte	0x1
	.4byte	0xbcb
	.4byte	.LLST11
	.byte	0x1
	.4byte	0xbc0
	.4byte	.LLST12
	.byte	0x4
	.4byte	0xc02
	.4byte	.LLST13
	.byte	0x8
	.4byte	0xc0c
	.8byte	.LBB65
	.8byte	.LBE65-.LBB65
	.byte	0x2
	.4byte	0xc0d
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xc17
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0x11
	.4byte	.LLRL26
	.4byte	0x3d0
	.byte	0x1d
	.string	"cin"
	.byte	0x33
	.4byte	0xa6d
	.byte	0x3
	.byte	0x91
	.byte	0xd8,0x78
	.byte	0xc
	.4byte	0xae7
	.8byte	.LBB80
	.4byte	.LLRL27
	.byte	0x76
	.4byte	0x312
	.byte	0x1
	.4byte	0xb16
	.4byte	.LLST28
	.byte	0x5
	.4byte	0xb0b
	.byte	0x1
	.4byte	0xb00
	.4byte	.LLST29
	.byte	0x1
	.4byte	0xaf5
	.4byte	.LLST28
	.byte	0x6
	.4byte	.LLRL27
	.byte	0x1e
	.4byte	0xb21
	.byte	0x4
	.4byte	0xb2b
	.4byte	.LLST31
	.byte	0x8
	.4byte	0xb35
	.8byte	.LBB82
	.8byte	.LBE82-.LBB82
	.byte	0x2
	.4byte	0xb36
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xb40
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0xc
	.4byte	0xa8d
	.8byte	.LBB85
	.4byte	.LLRL32
	.byte	0x77
	.4byte	0x36c
	.byte	0x1
	.4byte	0xab1
	.4byte	.LLST33
	.byte	0x5
	.4byte	0xaa6
	.byte	0x1
	.4byte	0xa9b
	.4byte	.LLST34
	.byte	0x6
	.4byte	.LLRL32
	.byte	0x1f
	.4byte	0xabc
	.byte	0x4
	.4byte	0xac6
	.4byte	.LLST35
	.byte	0xf
	.4byte	0xad0
	.4byte	.LLRL36
	.byte	0x2
	.4byte	0xad1
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xadb
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0x18
	.4byte	0xc23
	.8byte	.LBB92
	.4byte	.LLRL37
	.byte	0x75
	.byte	0x1
	.4byte	0xc5c
	.4byte	.LLST38
	.byte	0x5
	.4byte	0xc51
	.byte	0x1
	.4byte	0xc46
	.4byte	.LLST39
	.byte	0x1
	.4byte	0xc3b
	.4byte	.LLST40
	.byte	0x1
	.4byte	0xc30
	.4byte	.LLST41
	.byte	0x6
	.4byte	.LLRL37
	.byte	0x4
	.4byte	0xc67
	.4byte	.LLST42
	.byte	0xf
	.4byte	0xc71
	.4byte	.LLRL37
	.byte	0x2
	.4byte	0xc72
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xc7c
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0x19
	.8byte	.LBB118
	.8byte	.LBE118-.LBB118
	.4byte	0x47b
	.byte	0xa
	.4byte	.LASF38
	.byte	0x8a
	.byte	0xe
	.4byte	0x9b
	.4byte	.LLST49
	.byte	0xa
	.4byte	.LASF39
	.byte	0x8b
	.byte	0xe
	.4byte	0x9b
	.4byte	.LLST50
	.byte	0x16
	.4byte	.LASF40
	.byte	0x8c
	.4byte	0x9b
	.byte	0x17
	.4byte	0xb4c
	.8byte	.LBB119
	.8byte	.LBE119-.LBB119
	.byte	0x8d
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST51
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST52
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST53
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST54
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST55
	.byte	0x8
	.4byte	0xb9b
	.8byte	.LBB121
	.8byte	.LBE121-.LBB121
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0xc
	.4byte	0xb4c
	.8byte	.LBB66
	.4byte	.LLRL14
	.byte	0x7b
	.4byte	0x4ee
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST15
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST16
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST17
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST18
	.byte	0x6
	.4byte	.LLRL14
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST19
	.byte	0x8
	.4byte	0xb9b
	.8byte	.LBB68
	.8byte	.LBE68-.LBB68
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0xc
	.4byte	0xb4c
	.8byte	.LBB72
	.4byte	.LLRL20
	.byte	0x85
	.4byte	0x561
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST21
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST22
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST23
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST24
	.byte	0x6
	.4byte	.LLRL20
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST25
	.byte	0x8
	.4byte	0xb9b
	.8byte	.LBB74
	.8byte	.LBE74-.LBB74
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0x18
	.4byte	0xb4c
	.8byte	.LBB102
	.4byte	.LLRL43
	.byte	0x80
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST44
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST45
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST46
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST47
	.byte	0x6
	.4byte	.LLRL43
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST48
	.byte	0x8
	.4byte	0xb9b
	.8byte	.LBB104
	.8byte	.LBE104-.LBB104
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0x6
	.4byte	.LLRL56
	.byte	0xa
	.4byte	.LASF41
	.byte	0x96
	.byte	0xc
	.4byte	0x5b
	.4byte	.LLST57
	.byte	0x19
	.8byte	.LBB181
	.8byte	.LBE181-.LBB181
	.4byte	0x69d
	.byte	0xa
	.4byte	.LASF34
	.byte	0x99
	.byte	0xe
	.4byte	0x9b
	.4byte	.LLST94
	.byte	0xa
	.4byte	.LASF35
	.byte	0x9a
	.byte	0xe
	.4byte	0x9b
	.4byte	.LLST95
	.byte	0x16
	.4byte	.LASF36
	.byte	0x9b
	.4byte	0x9b
	.byte	0x17
	.4byte	0xbb2
	.8byte	.LBB182
	.8byte	.LBE182-.LBB182
	.byte	0x9c
	.byte	0x1
	.4byte	0xbf7
	.4byte	.LLST96
	.byte	0x1
	.4byte	0xbec
	.4byte	.LLST97
	.byte	0x1
	.4byte	0xbe1
	.4byte	.LLST98
	.byte	0x1
	.4byte	0xbd6
	.4byte	.LLST99
	.byte	0x1
	.4byte	0xbcb
	.4byte	.LLST100
	.byte	0x1
	.4byte	0xbc0
	.4byte	.LLST101
	.byte	0x4
	.4byte	0xc02
	.4byte	.LLST102
	.byte	0x8
	.4byte	0xc0c
	.8byte	.LBB184
	.8byte	.LBE184-.LBB184
	.byte	0x2
	.4byte	0xc0d
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xc17
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0x11
	.4byte	.LLRL71
	.4byte	0x7e1
	.byte	0x1d
	.string	"cin"
	.byte	0xa2
	.4byte	0xa88
	.byte	0x3
	.byte	0x91
	.byte	0xd8,0x7b
	.byte	0xc
	.4byte	0xa8d
	.8byte	.LBB139
	.4byte	.LLRL72
	.byte	0xf8
	.4byte	0x70e
	.byte	0x1
	.4byte	0xab1
	.4byte	.LLST73
	.byte	0x5
	.4byte	0xaa6
	.byte	0x1
	.4byte	0xa9b
	.4byte	.LLST74
	.byte	0x6
	.4byte	.LLRL72
	.byte	0x1f
	.4byte	0xabc
	.byte	0x4
	.4byte	0xac6
	.4byte	.LLST75
	.byte	0xf
	.4byte	0xad0
	.4byte	.LLRL76
	.byte	0x2
	.4byte	0xad1
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xadb
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0xc
	.4byte	0xae7
	.8byte	.LBB148
	.4byte	.LLRL77
	.byte	0xf7
	.4byte	0x77d
	.byte	0x1
	.4byte	0xb16
	.4byte	.LLST78
	.byte	0x5
	.4byte	0xb0b
	.byte	0x1
	.4byte	0xb00
	.4byte	.LLST79
	.byte	0x1
	.4byte	0xaf5
	.4byte	.LLST78
	.byte	0x6
	.4byte	.LLRL77
	.byte	0x1e
	.4byte	0xb21
	.byte	0x4
	.4byte	0xb2b
	.4byte	.LLST81
	.byte	0x8
	.4byte	0xb35
	.8byte	.LBB150
	.8byte	.LBE150-.LBB150
	.byte	0x2
	.4byte	0xb36
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xb40
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0x18
	.4byte	0xc23
	.8byte	.LBB155
	.4byte	.LLRL82
	.byte	0xf6
	.byte	0x1
	.4byte	0xc5c
	.4byte	.LLST83
	.byte	0x5
	.4byte	0xc51
	.byte	0x1
	.4byte	0xc46
	.4byte	.LLST84
	.byte	0x1
	.4byte	0xc3b
	.4byte	.LLST85
	.byte	0x1
	.4byte	0xc30
	.4byte	.LLST86
	.byte	0x6
	.4byte	.LLRL82
	.byte	0x4
	.4byte	0xc67
	.4byte	.LLST87
	.byte	0xf
	.4byte	0xc71
	.4byte	.LLRL82
	.byte	0x2
	.4byte	0xc72
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xc7c
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0x19
	.8byte	.LBB189
	.8byte	.LBE189-.LBB189
	.4byte	0x891
	.byte	0x20
	.4byte	.LASF42
	.2byte	0x10b
	.4byte	0x9b
	.4byte	.LLST103
	.byte	0x20
	.4byte	.LASF43
	.2byte	0x10c
	.4byte	0x9b
	.4byte	.LLST104
	.byte	0x29
	.4byte	.LASF44
	.byte	0x1
	.2byte	0x10d
	.byte	0xe
	.4byte	0x9b
	.byte	0
	.byte	0x21
	.4byte	0xb4c
	.8byte	.LBB190
	.8byte	.LBE190-.LBB190
	.2byte	0x10e
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST105
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST106
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST107
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST108
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST109
	.byte	0x8
	.4byte	0xb9b
	.8byte	.LBB192
	.8byte	.LBE192-.LBB192
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0x22
	.4byte	0xb4c
	.8byte	.LBB125
	.4byte	.LLRL58
	.2byte	0x101
	.4byte	0x905
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST59
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST60
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST61
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST62
	.byte	0x6
	.4byte	.LLRL58
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST63
	.byte	0x8
	.4byte	0xb9b
	.8byte	.LBB127
	.8byte	.LBE127-.LBB127
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0xc
	.4byte	0xb4c
	.8byte	.LBB131
	.4byte	.LLRL64
	.byte	0xfc
	.4byte	0x96c
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST65
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST66
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST67
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST68
	.byte	0x6
	.4byte	.LLRL64
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST69
	.byte	0xf
	.4byte	0xb9b
	.4byte	.LLRL70
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0x22
	.4byte	0xb4c
	.8byte	.LBB169
	.4byte	.LLRL88
	.2byte	0x106
	.4byte	0x9e0
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST89
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST90
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST91
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST92
	.byte	0x6
	.4byte	.LLRL88
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST93
	.byte	0x8
	.4byte	0xb9b
	.8byte	.LBB171
	.8byte	.LBE171-.LBB171
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0x21
	.4byte	0xb4c
	.8byte	.LBB193
	.8byte	.LBE193-.LBB193
	.2byte	0x114
	.byte	0x1
	.4byte	0xb86
	.4byte	.LLST110
	.byte	0x5
	.4byte	0xb7b
	.byte	0x1
	.4byte	0xb70
	.4byte	.LLST111
	.byte	0x1
	.4byte	0xb65
	.4byte	.LLST110
	.byte	0x1
	.4byte	0xb5a
	.4byte	.LLST113
	.byte	0x4
	.4byte	0xb91
	.4byte	.LLST114
	.byte	0x8
	.4byte	0xb9b
	.8byte	.LBB195
	.8byte	.LBE195-.LBB195
	.byte	0x2
	.4byte	0xb9c
	.byte	0x1
	.byte	0x5b
	.byte	0x2
	.4byte	0xba6
	.byte	0x1
	.byte	0x5c
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0x7
	.byte	0x4
	.byte	0x4
	.4byte	.LASF45
	.byte	0x23
	.4byte	0x56
	.4byte	0xa6d
	.byte	0x12
	.4byte	0x7c
	.byte	0x3e
	.byte	0x12
	.4byte	0x7c
	.byte	0x2
	.byte	0
	.byte	0x14
	.4byte	0xa57
	.byte	0x23
	.4byte	0x56
	.4byte	0xa88
	.byte	0x12
	.4byte	0x7c
	.byte	0x4b
	.byte	0x12
	.4byte	0x7c
	.byte	0x2
	.byte	0
	.byte	0x14
	.4byte	0xa72
	.byte	0x13
	.4byte	.LASF51
	.byte	0x36
	.4byte	0x5b
	.4byte	0xae7
	.byte	0x3
	.4byte	.LASF46
	.byte	0x36
	.byte	0x24
	.4byte	0x9b
	.byte	0x3
	.4byte	.LASF47
	.byte	0x36
	.byte	0x31
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF48
	.byte	0x36
	.byte	0x3e
	.4byte	0x5b
	.byte	0xd
	.string	"rs1"
	.byte	0x38
	.4byte	0x9b
	.byte	0xd
	.string	"rs2"
	.byte	0x39
	.4byte	0x9b
	.byte	0x10
	.byte	0x9
	.4byte	.LASF49
	.byte	0x3a
	.4byte	0x9b
	.byte	0x9
	.4byte	.LASF50
	.byte	0x3a
	.4byte	0x9b
	.byte	0
	.byte	0
	.byte	0x13
	.4byte	.LASF52
	.byte	0x2e
	.4byte	0x5b
	.4byte	0xb4c
	.byte	0x3
	.4byte	.LASF53
	.byte	0x2e
	.byte	0x1e
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF54
	.byte	0x2e
	.byte	0x31
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF47
	.byte	0x2e
	.byte	0x3e
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF48
	.byte	0x2e
	.byte	0x4b
	.4byte	0x5b
	.byte	0xd
	.string	"rs1"
	.byte	0x30
	.4byte	0x9b
	.byte	0xd
	.string	"rs2"
	.byte	0x31
	.4byte	0x9b
	.byte	0x10
	.byte	0x9
	.4byte	.LASF49
	.byte	0x32
	.4byte	0x9b
	.byte	0x9
	.4byte	.LASF50
	.byte	0x32
	.4byte	0x9b
	.byte	0
	.byte	0
	.byte	0x13
	.4byte	.LASF55
	.byte	0x26
	.4byte	0x5b
	.4byte	0xbb2
	.byte	0x3
	.4byte	.LASF56
	.byte	0x26
	.byte	0x1f
	.4byte	0xc1
	.byte	0x3
	.4byte	.LASF57
	.byte	0x26
	.byte	0x2d
	.4byte	0x5b
	.byte	0x1a
	.string	"len"
	.byte	0x26
	.byte	0x3b
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF47
	.byte	0x26
	.byte	0x44
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF48
	.byte	0x26
	.byte	0x51
	.4byte	0x5b
	.byte	0xd
	.string	"rs2"
	.byte	0x29
	.4byte	0x9b
	.byte	0x10
	.byte	0x9
	.4byte	.LASF49
	.byte	0x2a
	.4byte	0x9b
	.byte	0x9
	.4byte	.LASF50
	.byte	0x2a
	.4byte	0x9b
	.byte	0
	.byte	0
	.byte	0x13
	.4byte	.LASF58
	.byte	0x1e
	.4byte	0x5b
	.4byte	0xc23
	.byte	0x3
	.4byte	.LASF56
	.byte	0x1e
	.byte	0x23
	.4byte	0xc1
	.byte	0x3
	.4byte	.LASF57
	.byte	0x1e
	.byte	0x31
	.4byte	0x5b
	.byte	0x1a
	.string	"len"
	.byte	0x1e
	.byte	0x3f
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF59
	.byte	0x1e
	.byte	0x48
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF47
	.byte	0x1e
	.byte	0x53
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF48
	.byte	0x1e
	.byte	0x60
	.4byte	0x5b
	.byte	0xd
	.string	"rs2"
	.byte	0x21
	.4byte	0x9b
	.byte	0x10
	.byte	0x9
	.4byte	.LASF49
	.byte	0x22
	.4byte	0x9b
	.byte	0x9
	.4byte	.LASF50
	.byte	0x22
	.4byte	0x9b
	.byte	0
	.byte	0
	.byte	0x2a
	.4byte	.LASF62
	.byte	0x2
	.byte	0x16
	.byte	0x13
	.4byte	0x5b
	.byte	0x3
	.byte	0x3
	.4byte	.LASF56
	.byte	0x16
	.byte	0x22
	.4byte	0xc1
	.byte	0x3
	.4byte	.LASF57
	.byte	0x16
	.byte	0x30
	.4byte	0x5b
	.byte	0x1a
	.string	"len"
	.byte	0x16
	.byte	0x3e
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF47
	.byte	0x16
	.byte	0x47
	.4byte	0x5b
	.byte	0x3
	.4byte	.LASF48
	.byte	0x16
	.byte	0x54
	.4byte	0x5b
	.byte	0xd
	.string	"rs2"
	.byte	0x19
	.4byte	0x9b
	.byte	0x10
	.byte	0x9
	.4byte	.LASF49
	.byte	0x1a
	.4byte	0x9b
	.byte	0x9
	.4byte	.LASF50
	.byte	0x1a
	.4byte	0x9b
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.byte	0x1
	.byte	0x5
	.byte	0
	.byte	0x31
	.byte	0x13
	.byte	0x2
	.byte	0x17
	.byte	0
	.byte	0
	.byte	0x2
	.byte	0x34
	.byte	0
	.byte	0x31
	.byte	0x13
	.byte	0x2
	.byte	0x18
	.byte	0
	.byte	0
	.byte	0x3
	.byte	0x5
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x2
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x4
	.byte	0x34
	.byte	0
	.byte	0x31
	.byte	0x13
	.byte	0x2
	.byte	0x17
	.byte	0
	.byte	0
	.byte	0x5
	.byte	0x5
	.byte	0
	.byte	0x31
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x6
	.byte	0xb
	.byte	0x1
	.byte	0x55
	.byte	0x17
	.byte	0
	.byte	0
	.byte	0x7
	.byte	0x24
	.byte	0
	.byte	0xb
	.byte	0xb
	.byte	0x3e
	.byte	0xb
	.byte	0x3
	.byte	0xe
	.byte	0
	.byte	0
	.byte	0x8
	.byte	0xb
	.byte	0x1
	.byte	0x31
	.byte	0x13
	.byte	0x11
	.byte	0x1
	.byte	0x12
	.byte	0x7
	.byte	0
	.byte	0
	.byte	0x9
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x2
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0x21
	.byte	0x2
	.byte	0x49
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0xa
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x1
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0x2
	.byte	0x17
	.byte	0
	.byte	0
	.byte	0xb
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x1
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0x21
	.byte	0x9
	.byte	0x49
	.byte	0x13
	.byte	0x2
	.byte	0x18
	.byte	0
	.byte	0
	.byte	0xc
	.byte	0x1d
	.byte	0x1
	.byte	0x31
	.byte	0x13
	.byte	0x52
	.byte	0x1
	.byte	0x55
	.byte	0x17
	.byte	0x58
	.byte	0x21
	.byte	0x1
	.byte	0x59
	.byte	0xb
	.byte	0x57
	.byte	0x21
	.byte	0x5
	.byte	0x1
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0xd
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0x8
	.byte	0x3a
	.byte	0x21
	.byte	0x2
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0x21
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0xe
	.byte	0x16
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0xb
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0xf
	.byte	0xb
	.byte	0x1
	.byte	0x31
	.byte	0x13
	.byte	0x55
	.byte	0x17
	.byte	0
	.byte	0
	.byte	0x10
	.byte	0xb
	.byte	0x1
	.byte	0
	.byte	0
	.byte	0x11
	.byte	0xb
	.byte	0x1
	.byte	0x55
	.byte	0x17
	.byte	0x1
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x12
	.byte	0x21
	.byte	0
	.byte	0x49
	.byte	0x13
	.byte	0x2f
	.byte	0xb
	.byte	0
	.byte	0
	.byte	0x13
	.byte	0x2e
	.byte	0x1
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x2
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0x21
	.byte	0x13
	.byte	0x27
	.byte	0x19
	.byte	0x49
	.byte	0x13
	.byte	0x20
	.byte	0x21
	.byte	0x3
	.byte	0x1
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x14
	.byte	0x35
	.byte	0
	.byte	0x49
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x15
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x1
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0x21
	.byte	0x9
	.byte	0x49
	.byte	0x13
	.byte	0x1c
	.byte	0xa
	.byte	0
	.byte	0
	.byte	0x16
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x1
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0x21
	.byte	0xe
	.byte	0x49
	.byte	0x13
	.byte	0x1c
	.byte	0x21
	.byte	0
	.byte	0
	.byte	0
	.byte	0x17
	.byte	0x1d
	.byte	0x1
	.byte	0x31
	.byte	0x13
	.byte	0x11
	.byte	0x1
	.byte	0x12
	.byte	0x7
	.byte	0x58
	.byte	0x21
	.byte	0x1
	.byte	0x59
	.byte	0xb
	.byte	0x57
	.byte	0x21
	.byte	0x5
	.byte	0
	.byte	0
	.byte	0x18
	.byte	0x1d
	.byte	0x1
	.byte	0x31
	.byte	0x13
	.byte	0x52
	.byte	0x1
	.byte	0x55
	.byte	0x17
	.byte	0x58
	.byte	0x21
	.byte	0x1
	.byte	0x59
	.byte	0xb
	.byte	0x57
	.byte	0x21
	.byte	0x5
	.byte	0
	.byte	0
	.byte	0x19
	.byte	0xb
	.byte	0x1
	.byte	0x11
	.byte	0x1
	.byte	0x12
	.byte	0x7
	.byte	0x1
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x1a
	.byte	0x5
	.byte	0
	.byte	0x3
	.byte	0x8
	.byte	0x3a
	.byte	0x21
	.byte	0x2
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x1b
	.byte	0x5
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x1
	.byte	0x3b
	.byte	0x21
	.byte	0x12
	.byte	0x39
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0x2
	.byte	0x18
	.byte	0
	.byte	0
	.byte	0x1c
	.byte	0x5
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x1
	.byte	0x3b
	.byte	0x21
	.byte	0x12
	.byte	0x39
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0x2
	.byte	0x17
	.byte	0
	.byte	0
	.byte	0x1d
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0x8
	.byte	0x3a
	.byte	0x21
	.byte	0x1
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0x21
	.byte	0x1d
	.byte	0x49
	.byte	0x13
	.byte	0x88,0x1
	.byte	0x21
	.byte	0x8
	.byte	0x2
	.byte	0x18
	.byte	0
	.byte	0
	.byte	0x1e
	.byte	0x34
	.byte	0
	.byte	0x31
	.byte	0x13
	.byte	0x1c
	.byte	0x21
	.byte	0
	.byte	0
	.byte	0
	.byte	0x1f
	.byte	0x34
	.byte	0
	.byte	0x31
	.byte	0x13
	.byte	0x1c
	.byte	0x21
	.byte	0xb1,0xd4,0x3
	.byte	0
	.byte	0
	.byte	0x20
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0x21
	.byte	0x1
	.byte	0x3b
	.byte	0x5
	.byte	0x39
	.byte	0x21
	.byte	0xe
	.byte	0x49
	.byte	0x13
	.byte	0x2
	.byte	0x17
	.byte	0
	.byte	0
	.byte	0x21
	.byte	0x1d
	.byte	0x1
	.byte	0x31
	.byte	0x13
	.byte	0x11
	.byte	0x1
	.byte	0x12
	.byte	0x7
	.byte	0x58
	.byte	0x21
	.byte	0x1
	.byte	0x59
	.byte	0x5
	.byte	0x57
	.byte	0x21
	.byte	0x5
	.byte	0
	.byte	0
	.byte	0x22
	.byte	0x1d
	.byte	0x1
	.byte	0x31
	.byte	0x13
	.byte	0x52
	.byte	0x1
	.byte	0x55
	.byte	0x17
	.byte	0x58
	.byte	0x21
	.byte	0x1
	.byte	0x59
	.byte	0x5
	.byte	0x57
	.byte	0x21
	.byte	0x5
	.byte	0x1
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x23
	.byte	0x1
	.byte	0x1
	.byte	0x49
	.byte	0x13
	.byte	0x1
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x24
	.byte	0x11
	.byte	0x1
	.byte	0x25
	.byte	0xe
	.byte	0x13
	.byte	0xb
	.byte	0x3
	.byte	0x1f
	.byte	0x1b
	.byte	0x1f
	.byte	0x11
	.byte	0x1
	.byte	0x12
	.byte	0x7
	.byte	0x10
	.byte	0x17
	.byte	0
	.byte	0
	.byte	0x25
	.byte	0x24
	.byte	0
	.byte	0xb
	.byte	0xb
	.byte	0x3e
	.byte	0xb
	.byte	0x3
	.byte	0x8
	.byte	0
	.byte	0
	.byte	0x26
	.byte	0xf
	.byte	0
	.byte	0xb
	.byte	0xb
	.byte	0
	.byte	0
	.byte	0x27
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0xb
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0x3f
	.byte	0x19
	.byte	0x2
	.byte	0x18
	.byte	0
	.byte	0
	.byte	0x28
	.byte	0x2e
	.byte	0x1
	.byte	0x3f
	.byte	0x19
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0xb
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0xb
	.byte	0x27
	.byte	0x19
	.byte	0x11
	.byte	0x1
	.byte	0x12
	.byte	0x7
	.byte	0x40
	.byte	0x18
	.byte	0x7a
	.byte	0x19
	.byte	0x1
	.byte	0x13
	.byte	0
	.byte	0
	.byte	0x29
	.byte	0x34
	.byte	0
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0xb
	.byte	0x3b
	.byte	0x5
	.byte	0x39
	.byte	0xb
	.byte	0x49
	.byte	0x13
	.byte	0x1c
	.byte	0xb
	.byte	0
	.byte	0
	.byte	0x2a
	.byte	0x2e
	.byte	0x1
	.byte	0x3
	.byte	0xe
	.byte	0x3a
	.byte	0xb
	.byte	0x3b
	.byte	0xb
	.byte	0x39
	.byte	0xb
	.byte	0x27
	.byte	0x19
	.byte	0x49
	.byte	0x13
	.byte	0x20
	.byte	0xb
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_loclists,"",@progbits
	.4byte	.Ldebug_loc3-.Ldebug_loc2
.Ldebug_loc2:
	.2byte	0x5
	.byte	0x8
	.byte	0
	.4byte	0
.Ldebug_loc0:
.LLST0:
	.byte	0x7
	.8byte	.LVL0
	.8byte	.LVL1
	.byte	0x1
	.byte	0x5b
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LFE10
	.byte	0x4
	.byte	0xa3
	.byte	0x1
	.byte	0x5b
	.byte	0x9f
	.byte	0
.LLST1:
	.byte	0x7
	.8byte	.LVL0
	.8byte	.LVL1
	.byte	0x1
	.byte	0x5c
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL28
	.byte	0x1
	.byte	0x65
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LFE10
	.byte	0x4
	.byte	0xa3
	.byte	0x1
	.byte	0x5c
	.byte	0x9f
	.byte	0
.LLST3:
	.byte	0x7
	.8byte	.LVL0
	.8byte	.LVL1
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL25
	.byte	0x5
	.byte	0x8c
	.byte	0
	.byte	0x38
	.byte	0x25
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL25
	.8byte	.LVL26
	.byte	0x7
	.byte	0x8c
	.byte	0
	.byte	0x38
	.byte	0x25
	.byte	0x23
	.byte	0x20
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL27
	.8byte	.LFE10
	.byte	0x7
	.byte	0x8c
	.byte	0
	.byte	0x38
	.byte	0x25
	.byte	0x23
	.byte	0x20
	.byte	0x9f
	.byte	0
.LLST5:
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL26
	.byte	0x1
	.byte	0x6c
	.byte	0x7
	.8byte	.LVL27
	.8byte	.LFE10
	.byte	0x1
	.byte	0x6c
	.byte	0
.LLST6:
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL4
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL4
	.8byte	.LFE10
	.byte	0x4
	.byte	0xa
	.2byte	0x2000
	.byte	0x9f
	.byte	0
.LLST7:
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL4
	.byte	0x2
	.byte	0x32
	.byte	0x9f
	.byte	0
.LLST8:
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL4
	.byte	0xf
	.byte	0x3
	.8byte	_task_id
	.byte	0x94
	.byte	0x1
	.byte	0x8
	.byte	0xff
	.byte	0x1a
	.byte	0x9f
	.byte	0
.LLST9:
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL4
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST10:
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL4
	.byte	0x4
	.byte	0xa
	.2byte	0x2000
	.byte	0x9f
	.byte	0
.LLST11:
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL4
	.byte	0x4
	.byte	0x40
	.byte	0x3c
	.byte	0x24
	.byte	0x9f
	.byte	0
.LLST12:
	.byte	0x7
	.8byte	.LVL1
	.8byte	.LVL4
	.byte	0x6
	.byte	0x7a
	.byte	0
	.byte	0x8c
	.byte	0
	.byte	0x22
	.byte	0x9f
	.byte	0
.LLST13:
	.byte	0x7
	.8byte	.LVL2
	.8byte	.LVL3
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x8a
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL3
	.8byte	.LVL4
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST28:
	.byte	0x7
	.8byte	.LVL8
	.8byte	.LVL10
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST29:
	.byte	0x7
	.8byte	.LVL8
	.8byte	.LVL10
	.byte	0x3
	.byte	0x8
	.byte	0x3f
	.byte	0x9f
	.byte	0
.LLST31:
	.byte	0x7
	.8byte	.LVL8
	.8byte	.LVL9
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x84
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL9
	.8byte	.LVL10
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST33:
	.byte	0x7
	.8byte	.LVL10
	.8byte	.LVL12
	.byte	0x2
	.byte	0x31
	.byte	0x9f
	.byte	0
.LLST34:
	.byte	0x7
	.8byte	.LVL10
	.8byte	.LVL12
	.byte	0x4
	.byte	0xa
	.2byte	0xea31
	.byte	0x9f
	.byte	0
.LLST35:
	.byte	0x7
	.8byte	.LVL10
	.8byte	.LVL11
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x83
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL11
	.8byte	.LVL12
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST38:
	.byte	0x7
	.8byte	.LVL5
	.8byte	.LVL8
	.byte	0x2
	.byte	0x32
	.byte	0x9f
	.byte	0
.LLST39:
	.byte	0x7
	.8byte	.LVL5
	.8byte	.LVL8
	.byte	0x4
	.byte	0xa
	.2byte	0x17a
	.byte	0x9f
	.byte	0
.LLST40:
	.byte	0x7
	.8byte	.LVL5
	.8byte	.LVL8
	.byte	0x4
	.byte	0x40
	.byte	0x3d
	.byte	0x24
	.byte	0x9f
	.byte	0
.LLST41:
	.byte	0x7
	.8byte	.LVL5
	.8byte	.LVL6
	.byte	0x4
	.byte	0x91
	.byte	0xd8,0x78
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL6
	.8byte	.LVL8
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST42:
	.byte	0x7
	.8byte	.LVL5
	.8byte	.LVL7
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x89
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL7
	.8byte	.LVL8
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST49:
	.byte	0x7
	.8byte	.LVL21
	.8byte	.LVL26
	.byte	0x1
	.byte	0x6c
	.byte	0x7
	.8byte	.LVL27
	.8byte	.LFE10
	.byte	0x1
	.byte	0x6c
	.byte	0
.LLST50:
	.byte	0x7
	.8byte	.LVL21
	.8byte	.LVL24
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL24
	.8byte	.LFE10
	.byte	0x4
	.byte	0xa
	.2byte	0x2000
	.byte	0x9f
	.byte	0
.LLST51:
	.byte	0x7
	.8byte	.LVL21
	.8byte	.LVL24
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST52:
	.byte	0x7
	.8byte	.LVL21
	.8byte	.LVL24
	.byte	0x4
	.byte	0xa
	.2byte	0x2000
	.byte	0x9f
	.byte	0
.LLST53:
	.byte	0x7
	.8byte	.LVL21
	.8byte	.LVL24
	.byte	0x4
	.byte	0xa
	.2byte	0xa000
	.byte	0x9f
	.byte	0
.LLST54:
	.byte	0x7
	.8byte	.LVL21
	.8byte	.LVL22
	.byte	0x6
	.byte	0x85
	.byte	0
	.byte	0x8c
	.byte	0
	.byte	0x22
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL22
	.8byte	.LVL24
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST55:
	.byte	0x7
	.8byte	.LVL21
	.8byte	.LVL23
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x88
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL23
	.8byte	.LVL24
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST15:
	.byte	0x7
	.8byte	.LVL12
	.8byte	.LVL15
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST16:
	.byte	0x7
	.8byte	.LVL12
	.8byte	.LVL15
	.byte	0x2
	.byte	0x38
	.byte	0x9f
	.byte	0
.LLST17:
	.byte	0x7
	.8byte	.LVL12
	.8byte	.LVL15
	.byte	0x4
	.byte	0xa
	.2byte	0x8000
	.byte	0x9f
	.byte	0
.LLST18:
	.byte	0x7
	.8byte	.LVL12
	.8byte	.LVL13
	.byte	0x4
	.byte	0x91
	.byte	0xd0,0x78
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL13
	.8byte	.LVL15
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST19:
	.byte	0x7
	.8byte	.LVL12
	.8byte	.LVL14
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x79
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL14
	.8byte	.LVL15
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST21:
	.byte	0x7
	.8byte	.LVL18
	.8byte	.LVL21
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST22:
	.byte	0x7
	.8byte	.LVL18
	.8byte	.LVL21
	.byte	0x2
	.byte	0x38
	.byte	0x9f
	.byte	0
.LLST23:
	.byte	0x7
	.8byte	.LVL18
	.8byte	.LVL21
	.byte	0x4
	.byte	0x42
	.byte	0x3c
	.byte	0x24
	.byte	0x9f
	.byte	0
.LLST24:
	.byte	0x7
	.8byte	.LVL18
	.8byte	.LVL19
	.byte	0x4
	.byte	0x91
	.byte	0xd4,0x78
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL19
	.8byte	.LVL21
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST25:
	.byte	0x7
	.8byte	.LVL18
	.8byte	.LVL20
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x8e
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL20
	.8byte	.LVL21
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST44:
	.byte	0x7
	.8byte	.LVL15
	.8byte	.LVL18
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST45:
	.byte	0x7
	.8byte	.LVL15
	.8byte	.LVL18
	.byte	0x2
	.byte	0x38
	.byte	0x9f
	.byte	0
.LLST46:
	.byte	0x7
	.8byte	.LVL15
	.8byte	.LVL18
	.byte	0x4
	.byte	0x48
	.byte	0x3c
	.byte	0x24
	.byte	0x9f
	.byte	0
.LLST47:
	.byte	0x7
	.8byte	.LVL15
	.8byte	.LVL16
	.byte	0x4
	.byte	0x91
	.byte	0xcc,0x78
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL16
	.8byte	.LVL18
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST48:
	.byte	0x7
	.8byte	.LVL15
	.8byte	.LVL17
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x8d
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL17
	.8byte	.LVL18
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST57:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL52
	.byte	0x5
	.byte	0x76
	.byte	0
	.byte	0x38
	.byte	0x25
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL52
	.8byte	.LVL53
	.byte	0x7
	.byte	0x76
	.byte	0
	.byte	0x38
	.byte	0x25
	.byte	0x23
	.byte	0x20
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL54
	.8byte	.LFE10
	.byte	0x7
	.byte	0x76
	.byte	0
	.byte	0x38
	.byte	0x25
	.byte	0x23
	.byte	0x20
	.byte	0x9f
	.byte	0
.LLST94:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL53
	.byte	0x1
	.byte	0x56
	.byte	0x7
	.8byte	.LVL54
	.8byte	.LFE10
	.byte	0x1
	.byte	0x56
	.byte	0
.LLST95:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL31
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL31
	.8byte	.LFE10
	.byte	0x4
	.byte	0xa
	.2byte	0x2000
	.byte	0x9f
	.byte	0
.LLST96:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL31
	.byte	0x2
	.byte	0x32
	.byte	0x9f
	.byte	0
.LLST97:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL31
	.byte	0xf
	.byte	0x3
	.8byte	_task_id
	.byte	0x94
	.byte	0x1
	.byte	0x8
	.byte	0xff
	.byte	0x1a
	.byte	0x9f
	.byte	0
.LLST98:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL31
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST99:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL31
	.byte	0x4
	.byte	0xa
	.2byte	0x2000
	.byte	0x9f
	.byte	0
.LLST100:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL31
	.byte	0x4
	.byte	0xa
	.2byte	0x8000
	.byte	0x9f
	.byte	0
.LLST101:
	.byte	0x7
	.8byte	.LVL28
	.8byte	.LVL31
	.byte	0x6
	.byte	0x7a
	.byte	0
	.byte	0x76
	.byte	0
	.byte	0x22
	.byte	0x9f
	.byte	0
.LLST102:
	.byte	0x7
	.8byte	.LVL29
	.8byte	.LVL30
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x89
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL30
	.8byte	.LVL31
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST73:
	.byte	0x7
	.8byte	.LVL35
	.8byte	.LVL37
	.byte	0x2
	.byte	0x31
	.byte	0x9f
	.byte	0
.LLST74:
	.byte	0x7
	.8byte	.LVL35
	.8byte	.LVL37
	.byte	0x4
	.byte	0xa
	.2byte	0xea31
	.byte	0x9f
	.byte	0
.LLST75:
	.byte	0x7
	.8byte	.LVL35
	.8byte	.LVL36
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x86
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL36
	.8byte	.LVL37
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST78:
	.byte	0x7
	.8byte	.LVL33
	.8byte	.LVL35
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST79:
	.byte	0x7
	.8byte	.LVL33
	.8byte	.LVL35
	.byte	0x3
	.byte	0x8
	.byte	0x4c
	.byte	0x9f
	.byte	0
.LLST81:
	.byte	0x7
	.8byte	.LVL33
	.8byte	.LVL34
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x83
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL34
	.8byte	.LVL35
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST83:
	.byte	0x7
	.8byte	.LVL32
	.8byte	.LVL33
	.byte	0x2
	.byte	0x32
	.byte	0x9f
	.byte	0
.LLST84:
	.byte	0x7
	.8byte	.LVL32
	.8byte	.LVL33
	.byte	0x4
	.byte	0xa
	.2byte	0x1c8
	.byte	0x9f
	.byte	0
.LLST85:
	.byte	0x7
	.8byte	.LVL32
	.8byte	.LVL33
	.byte	0x4
	.byte	0x40
	.byte	0x3d
	.byte	0x24
	.byte	0x9f
	.byte	0
.LLST86:
	.byte	0x7
	.8byte	.LVL32
	.8byte	.LVL33
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST87:
	.byte	0x7
	.8byte	.LVL32
	.8byte	.LVL33
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST103:
	.byte	0x7
	.8byte	.LVL46
	.8byte	.LVL53
	.byte	0x1
	.byte	0x56
	.byte	0x7
	.8byte	.LVL54
	.8byte	.LFE10
	.byte	0x1
	.byte	0x56
	.byte	0
.LLST104:
	.byte	0x7
	.8byte	.LVL46
	.8byte	.LVL49
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL49
	.8byte	.LFE10
	.byte	0x4
	.byte	0xa
	.2byte	0x2000
	.byte	0x9f
	.byte	0
.LLST105:
	.byte	0x7
	.8byte	.LVL46
	.8byte	.LVL49
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST106:
	.byte	0x7
	.8byte	.LVL46
	.8byte	.LVL49
	.byte	0x4
	.byte	0xa
	.2byte	0x2000
	.byte	0x9f
	.byte	0
.LLST107:
	.byte	0x7
	.8byte	.LVL46
	.8byte	.LVL49
	.byte	0x4
	.byte	0x4c
	.byte	0x3c
	.byte	0x24
	.byte	0x9f
	.byte	0
.LLST108:
	.byte	0x7
	.8byte	.LVL46
	.8byte	.LVL47
	.byte	0x6
	.byte	0x7d
	.byte	0
	.byte	0x76
	.byte	0
	.byte	0x22
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL47
	.8byte	.LVL49
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST109:
	.byte	0x7
	.8byte	.LVL46
	.8byte	.LVL48
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x87
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL48
	.8byte	.LVL49
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST59:
	.byte	0x7
	.8byte	.LVL40
	.8byte	.LVL43
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST60:
	.byte	0x7
	.8byte	.LVL40
	.8byte	.LVL43
	.byte	0x2
	.byte	0x38
	.byte	0x9f
	.byte	0
.LLST61:
	.byte	0x7
	.8byte	.LVL40
	.8byte	.LVL43
	.byte	0x4
	.byte	0x4a
	.byte	0x3c
	.byte	0x24
	.byte	0x9f
	.byte	0
.LLST62:
	.byte	0x7
	.8byte	.LVL40
	.8byte	.LVL41
	.byte	0x4
	.byte	0x91
	.byte	0xbc,0x78
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL41
	.8byte	.LVL43
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST63:
	.byte	0x7
	.8byte	.LVL40
	.8byte	.LVL42
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x79
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL42
	.8byte	.LVL43
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST65:
	.byte	0x7
	.8byte	.LVL37
	.8byte	.LVL40
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST66:
	.byte	0x7
	.8byte	.LVL37
	.8byte	.LVL40
	.byte	0x2
	.byte	0x38
	.byte	0x9f
	.byte	0
.LLST67:
	.byte	0x7
	.8byte	.LVL37
	.8byte	.LVL40
	.byte	0x4
	.byte	0x48
	.byte	0x3c
	.byte	0x24
	.byte	0x9f
	.byte	0
.LLST68:
	.byte	0x7
	.8byte	.LVL37
	.8byte	.LVL38
	.byte	0x4
	.byte	0x91
	.byte	0xc0,0x78
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL38
	.8byte	.LVL40
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST69:
	.byte	0x7
	.8byte	.LVL37
	.8byte	.LVL39
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x75
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL39
	.8byte	.LVL40
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST89:
	.byte	0x7
	.8byte	.LVL43
	.8byte	.LVL46
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST90:
	.byte	0x7
	.8byte	.LVL43
	.8byte	.LVL46
	.byte	0x2
	.byte	0x38
	.byte	0x9f
	.byte	0
.LLST91:
	.byte	0x7
	.8byte	.LVL43
	.8byte	.LVL46
	.byte	0x4
	.byte	0xa
	.2byte	0xa000
	.byte	0x9f
	.byte	0
.LLST92:
	.byte	0x7
	.8byte	.LVL43
	.8byte	.LVL44
	.byte	0x4
	.byte	0x91
	.byte	0xc4,0x78
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL44
	.8byte	.LVL46
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST93:
	.byte	0x7
	.8byte	.LVL43
	.8byte	.LVL45
	.byte	0x6
	.byte	0x80
	.byte	0
	.byte	0x8d
	.byte	0
	.byte	0x21
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL45
	.8byte	.LVL46
	.byte	0x1
	.byte	0x5c
	.byte	0
.LLST110:
	.byte	0x7
	.8byte	.LVL49
	.8byte	.LVL51
	.byte	0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LLST111:
	.byte	0x7
	.8byte	.LVL49
	.8byte	.LVL51
	.byte	0x2
	.byte	0x38
	.byte	0x9f
	.byte	0
.LLST113:
	.byte	0x7
	.8byte	.LVL49
	.8byte	.LVL50
	.byte	0x4
	.byte	0x91
	.byte	0xc8,0x78
	.byte	0x9f
	.byte	0x7
	.8byte	.LVL50
	.8byte	.LVL51
	.byte	0x1
	.byte	0x5b
	.byte	0
.LLST114:
	.byte	0x7
	.8byte	.LVL49
	.8byte	.LVL51
	.byte	0x1
	.byte	0x5f
	.byte	0
.Ldebug_loc3:
	.section	.debug_aranges,"",@progbits
	.4byte	0x2c
	.2byte	0x2
	.4byte	.Ldebug_info0
	.byte	0x8
	.byte	0
	.2byte	0
	.2byte	0
	.8byte	.Ltext0
	.8byte	.Letext0-.Ltext0
	.8byte	0
	.8byte	0
	.section	.debug_rnglists,"",@progbits
.Ldebug_ranges0:
	.4byte	.Ldebug_ranges3-.Ldebug_ranges2
.Ldebug_ranges2:
	.2byte	0x5
	.byte	0x8
	.byte	0
	.4byte	0
.LLRL2:
	.byte	0x6
	.8byte	.LBB61
	.8byte	.LBE61
	.byte	0x6
	.8byte	.LBB122
	.8byte	.LBE122
	.byte	0x6
	.8byte	.LBB123
	.8byte	.LBE123
	.byte	0x6
	.8byte	.LBB196
	.8byte	.LBE196
	.byte	0x6
	.8byte	.LBB198
	.8byte	.LBE198
	.byte	0
.LLRL4:
	.byte	0x6
	.8byte	.LBB62
	.8byte	.LBE62
	.byte	0x6
	.8byte	.LBB113
	.8byte	.LBE113
	.byte	0
.LLRL14:
	.byte	0x6
	.8byte	.LBB66
	.8byte	.LBE66
	.byte	0x6
	.8byte	.LBB78
	.8byte	.LBE78
	.byte	0x6
	.8byte	.LBB110
	.8byte	.LBE110
	.byte	0x6
	.8byte	.LBB115
	.8byte	.LBE115
	.byte	0
.LLRL20:
	.byte	0x6
	.8byte	.LBB72
	.8byte	.LBE72
	.byte	0x6
	.8byte	.LBB108
	.8byte	.LBE108
	.byte	0x6
	.8byte	.LBB112
	.8byte	.LBE112
	.byte	0x6
	.8byte	.LBB117
	.8byte	.LBE117
	.byte	0
.LLRL26:
	.byte	0x6
	.8byte	.LBB79
	.8byte	.LBE79
	.byte	0x6
	.8byte	.LBB107
	.8byte	.LBE107
	.byte	0x6
	.8byte	.LBB109
	.8byte	.LBE109
	.byte	0x6
	.8byte	.LBB114
	.8byte	.LBE114
	.byte	0
.LLRL27:
	.byte	0x6
	.8byte	.LBB80
	.8byte	.LBE80
	.byte	0x6
	.8byte	.LBB97
	.8byte	.LBE97
	.byte	0x6
	.8byte	.LBB100
	.8byte	.LBE100
	.byte	0
.LLRL32:
	.byte	0x6
	.8byte	.LBB85
	.8byte	.LBE85
	.byte	0x6
	.8byte	.LBB98
	.8byte	.LBE98
	.byte	0x6
	.8byte	.LBB101
	.8byte	.LBE101
	.byte	0
.LLRL36:
	.byte	0x6
	.8byte	.LBB87
	.8byte	.LBE87
	.byte	0x6
	.8byte	.LBB88
	.8byte	.LBE88
	.byte	0x6
	.8byte	.LBB89
	.8byte	.LBE89
	.byte	0
.LLRL37:
	.byte	0x6
	.8byte	.LBB92
	.8byte	.LBE92
	.byte	0x6
	.8byte	.LBB99
	.8byte	.LBE99
	.byte	0
.LLRL43:
	.byte	0x6
	.8byte	.LBB102
	.8byte	.LBE102
	.byte	0x6
	.8byte	.LBB111
	.8byte	.LBE111
	.byte	0x6
	.8byte	.LBB116
	.8byte	.LBE116
	.byte	0
.LLRL56:
	.byte	0x6
	.8byte	.LBB124
	.8byte	.LBE124
	.byte	0x6
	.8byte	.LBB197
	.8byte	.LBE197
	.byte	0x6
	.8byte	.LBB199
	.8byte	.LBE199
	.byte	0
.LLRL58:
	.byte	0x6
	.8byte	.LBB125
	.8byte	.LBE125
	.byte	0x6
	.8byte	.LBB168
	.8byte	.LBE168
	.byte	0x6
	.8byte	.LBB179
	.8byte	.LBE179
	.byte	0x6
	.8byte	.LBB187
	.8byte	.LBE187
	.byte	0
.LLRL64:
	.byte	0x6
	.8byte	.LBB131
	.8byte	.LBE131
	.byte	0x6
	.8byte	.LBB175
	.8byte	.LBE175
	.byte	0x6
	.8byte	.LBB178
	.8byte	.LBE178
	.byte	0x6
	.8byte	.LBB186
	.8byte	.LBE186
	.byte	0
.LLRL70:
	.byte	0x6
	.8byte	.LBB133
	.8byte	.LBE133
	.byte	0x6
	.8byte	.LBB134
	.8byte	.LBE134
	.byte	0
.LLRL71:
	.byte	0x6
	.8byte	.LBB138
	.8byte	.LBE138
	.byte	0x6
	.8byte	.LBB174
	.8byte	.LBE174
	.byte	0x6
	.8byte	.LBB176
	.8byte	.LBE176
	.byte	0x6
	.8byte	.LBB177
	.8byte	.LBE177
	.byte	0x6
	.8byte	.LBB185
	.8byte	.LBE185
	.byte	0
.LLRL72:
	.byte	0x6
	.8byte	.LBB139
	.8byte	.LBE139
	.byte	0x6
	.8byte	.LBB153
	.8byte	.LBE153
	.byte	0x6
	.8byte	.LBB154
	.8byte	.LBE154
	.byte	0x6
	.8byte	.LBB163
	.8byte	.LBE163
	.byte	0x6
	.8byte	.LBB167
	.8byte	.LBE167
	.byte	0
.LLRL76:
	.byte	0x6
	.8byte	.LBB141
	.8byte	.LBE141
	.byte	0x6
	.8byte	.LBB142
	.8byte	.LBE142
	.byte	0x6
	.8byte	.LBB143
	.8byte	.LBE143
	.byte	0
.LLRL77:
	.byte	0x6
	.8byte	.LBB148
	.8byte	.LBE148
	.byte	0x6
	.8byte	.LBB162
	.8byte	.LBE162
	.byte	0x6
	.8byte	.LBB166
	.8byte	.LBE166
	.byte	0
.LLRL82:
	.byte	0x6
	.8byte	.LBB155
	.8byte	.LBE155
	.byte	0x6
	.8byte	.LBB164
	.8byte	.LBE164
	.byte	0x6
	.8byte	.LBB165
	.8byte	.LBE165
	.byte	0
.LLRL88:
	.byte	0x6
	.8byte	.LBB169
	.8byte	.LBE169
	.byte	0x6
	.8byte	.LBB180
	.8byte	.LBE180
	.byte	0x6
	.8byte	.LBB188
	.8byte	.LBE188
	.byte	0
.Ldebug_ranges3:
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF58:
	.string	"load_data"
.LASF37:
	.string	"_task_id"
.LASF14:
	.string	"uintptr_t"
.LASF13:
	.string	"uint64_t"
.LASF8:
	.string	"__uint8_t"
.LASF49:
	.string	"rs1_"
.LASF18:
	.string	"long long unsigned int"
.LASF15:
	.string	"long long int"
.LASF2:
	.string	"signed char"
.LASF47:
	.string	"task_id"
.LASF53:
	.string	"cfg_base_addr"
.LASF7:
	.string	"long int"
.LASF50:
	.string	"rs2_"
.LASF35:
	.string	"spadoffset_0"
.LASF39:
	.string	"spadoffset_1"
.LASF43:
	.string	"spadoffset_2"
.LASF61:
	.string	"kernel_deriche"
.LASF23:
	.string	"float_4"
.LASF24:
	.string	"float_5"
.LASF25:
	.string	"float_6"
.LASF26:
	.string	"float_7"
.LASF27:
	.string	"float_8"
.LASF28:
	.string	"float_9"
.LASF6:
	.string	"unsigned int"
.LASF10:
	.string	"long unsigned int"
.LASF46:
	.string	"iob_ens"
.LASF59:
	.string	"fused"
.LASF5:
	.string	"short unsigned int"
.LASF55:
	.string	"store"
.LASF56:
	.string	"dram_ptr"
.LASF48:
	.string	"dep_type"
.LASF16:
	.string	"long double"
.LASF9:
	.string	"__uint64_t"
.LASF62:
	.string	"load_cfg"
.LASF45:
	.string	"float"
.LASF33:
	.string	"int_14"
.LASF41:
	.string	"int_15"
.LASF51:
	.string	"execute"
.LASF3:
	.string	"unsigned char"
.LASF4:
	.string	"short int"
.LASF11:
	.string	"__uintptr_t"
.LASF54:
	.string	"cfg_num"
.LASF17:
	.string	"char"
.LASF52:
	.string	"config"
.LASF36:
	.string	"roffset_0"
.LASF40:
	.string	"roffset_1"
.LASF44:
	.string	"roffset_2"
.LASF57:
	.string	"spad_ptr"
.LASF19:
	.string	"arg_0"
.LASF20:
	.string	"arg_1"
.LASF21:
	.string	"arg_2"
.LASF22:
	.string	"arg_3"
.LASF60:
	.string	"GNU C17 12.2.0 -mcmodel=medany -mtune=rocket -mabi=lp64d -misa-spec=2.2 -march=rv64imafdc -g -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns"
.LASF34:
	.string	"dramoffset_0"
.LASF38:
	.string	"dramoffset_1"
.LASF42:
	.string	"dramoffset_2"
.LASF12:
	.string	"uint8_t"
.LASF29:
	.string	"float_10"
.LASF30:
	.string	"float_11"
.LASF31:
	.string	"float_12"
.LASF32:
	.string	"float_13"
	.section	.debug_line_str,"MS",@progbits,1
.LASF1:
	.string	"/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/medley/deriche/deriche_mini/IR/3_cgra_exes"
.LASF0:
	.string	"deriche_kernel_exe.c"
	.ident	"GCC: (g2ee5e430018) 12.2.0"
