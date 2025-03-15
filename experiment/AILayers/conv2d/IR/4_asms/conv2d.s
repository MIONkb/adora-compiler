	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.file	"LLVMDialectModule"
	.globl	conv2d                          # -- Begin function conv2d
	.p2align	1
	.type	conv2d,@function
conv2d:                                 # @conv2d
	.cfi_startproc
# %bb.0:                                # %.preheader2
	mv	a6, a1
	li	a1, 0
	addi	a5, a6, 116
.Lpcrel_hi0:
	auipc	a2, %pcrel_hi(.L__constant_6xf32)
	addi	t0, a2, %pcrel_lo(.Lpcrel_hi0)
	li	a4, 57
	lui	a2, 3
	addiw	t1, a2, 1168
	li	a7, 5
.LBB0_1:                                # %.preheader1
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	mv	a2, a1
	slli	a1, a1, 2
	add	a1, a1, t0
	flw	fa5, 0(a1)
	li	a3, -1
	mv	a1, a5
.LBB0_2:                                # %.preheader
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	fsw	fa5, -116(a1)
	fsw	fa5, -112(a1)
	fsw	fa5, -108(a1)
	fsw	fa5, -104(a1)
	fsw	fa5, -100(a1)
	fsw	fa5, -96(a1)
	fsw	fa5, -92(a1)
	fsw	fa5, -88(a1)
	fsw	fa5, -84(a1)
	fsw	fa5, -80(a1)
	fsw	fa5, -76(a1)
	fsw	fa5, -72(a1)
	fsw	fa5, -68(a1)
	fsw	fa5, -64(a1)
	fsw	fa5, -60(a1)
	fsw	fa5, -56(a1)
	fsw	fa5, -52(a1)
	fsw	fa5, -48(a1)
	fsw	fa5, -44(a1)
	fsw	fa5, -40(a1)
	fsw	fa5, -36(a1)
	fsw	fa5, -32(a1)
	fsw	fa5, -28(a1)
	fsw	fa5, -24(a1)
	fsw	fa5, -20(a1)
	fsw	fa5, -16(a1)
	fsw	fa5, -12(a1)
	fsw	fa5, -8(a1)
	fsw	fa5, -4(a1)
	fsw	fa5, 0(a1)
	fsw	fa5, 4(a1)
	fsw	fa5, 8(a1)
	fsw	fa5, 12(a1)
	fsw	fa5, 16(a1)
	fsw	fa5, 20(a1)
	fsw	fa5, 24(a1)
	fsw	fa5, 28(a1)
	fsw	fa5, 32(a1)
	fsw	fa5, 36(a1)
	fsw	fa5, 40(a1)
	fsw	fa5, 44(a1)
	fsw	fa5, 48(a1)
	fsw	fa5, 52(a1)
	fsw	fa5, 56(a1)
	fsw	fa5, 60(a1)
	fsw	fa5, 64(a1)
	fsw	fa5, 68(a1)
	fsw	fa5, 72(a1)
	fsw	fa5, 76(a1)
	fsw	fa5, 80(a1)
	fsw	fa5, 84(a1)
	fsw	fa5, 88(a1)
	fsw	fa5, 92(a1)
	fsw	fa5, 96(a1)
	fsw	fa5, 100(a1)
	fsw	fa5, 104(a1)
	fsw	fa5, 108(a1)
	fsw	fa5, 112(a1)
	addi	a3, a3, 1
	addi	a1, a1, 232
	bltu	a3, a4, .LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	addi	a1, a2, 1
	add	a5, a5, t1
	bltu	a2, a7, .LBB0_1
# %bb.4:
.Lpcrel_hi1:
	auipc	a1, %pcrel_hi(.L__constant_6x3x7x7xf32)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi1)
	mv	a2, a6
	tail	conv2d_kernel_0@plt
.Lfunc_end0:
	.size	conv2d, .Lfunc_end0-conv2d
	.cfi_endproc
                                        # -- End function
	.type	.L__constant_6x3x7x7xf32,@object # @__constant_6x3x7x7xf32
	.section	.rodata,"a",@progbits
	.p2align	4, 0x0
.L__constant_6x3x7x7xf32:
	.word	0xbce3f300                      # float -0.0278258324
	.word	0xbc9914e9                      # float -0.0186867286
	.word	0x3c9c0f4d                      # float 0.0190502647
	.word	0xbd046193                      # float -0.0323196165
	.word	0xbceb0b58                      # float -0.0286919326
	.word	0x3d924219                      # float 0.0714151338
	.word	0x3bdad7ed                      # float 0.00667857239
	.word	0x3d96c2c9                      # float 0.0736137107
	.word	0x3cbd362f                      # float 0.0230971258
	.word	0x3ccbc42d                      # float 0.0248738173
	.word	0x3da4cf38                      # float 0.0804733633
	.word	0xbcd0a990                      # float -0.0254714787
	.word	0xbd8a657e                      # float -0.0675763935
	.word	0xbd720509                      # float -0.0590868331
	.word	0xbcba3644                      # float -0.022730954
	.word	0x3d7e8a35                      # float 0.0621435232
	.word	0xbd4faeaa                      # float -0.050703682
	.word	0x3d9f9366                      # float 0.077917859
	.word	0x3d0e12fc                      # float 0.0346860737
	.word	0x3d8d300a                      # float 0.0689392835
	.word	0xbda5ec83                      # float -0.0810175165
	.word	0xbd8c6d3b                      # float -0.0685677156
	.word	0xbd89b896                      # float -0.067246601
	.word	0xbd88ca1f                      # float -0.0667917654
	.word	0xbd43df04                      # float -0.0478201061
	.word	0x3d94dbf4                      # float 0.0726851522
	.word	0x3d6ee17d                      # float 0.0583205111
	.word	0x3da2266b                      # float 0.0791748389
	.word	0x3d732f77                      # float 0.0593714379
	.word	0x3b6541cd                      # float 0.00349818473
	.word	0xbc2ffcb4                      # float -0.0107414015
	.word	0x3d592bfd                      # float 0.0530204661
	.word	0x3d77f319                      # float 0.0605345704
	.word	0xbd76f155                      # float -0.0602887459
	.word	0x3da1e1c9                      # float 0.0790439322
	.word	0x3d2401ec                      # float 0.0400408953
	.word	0xbbd51004                      # float -0.00650215335
	.word	0x3c6c23d5                      # float 0.0144128399
	.word	0x3c18fe00                      # float 0.00933790206
	.word	0x3ceb4b0f                      # float 0.0287223142
	.word	0x3d7f6f1e                      # float 0.062361829
	.word	0xbd3462fe                      # float -0.0440397188
	.word	0xbce70f55                      # float -0.0282055531
	.word	0x3d0f6897                      # float 0.035011854
	.word	0xbc21599e                      # float -0.00984802655
	.word	0x3d997d66                      # float 0.0749462098
	.word	0xbd9c6142                      # float -0.0763573796
	.word	0xbd843539                      # float -0.0645546392
	.word	0xbb94c690                      # float -0.00454027206
	.word	0x3d6ca019                      # float 0.0577698685
	.word	0xbd46f09b                      # float -0.048569303
	.word	0x3d7a7998                      # float 0.061151117
	.word	0x3d9ba759                      # float 0.0760027841
	.word	0x3cb6977a                      # float 0.0222890265
	.word	0xbcd6450a                      # float -0.0261559673
	.word	0x3d280b7e                      # float 0.0410265848
	.word	0xbd21a5a2                      # float -0.0394646004
	.word	0xbbeab35c                      # float -0.00716249458
	.word	0x3d8b58e7                      # float 0.0680406615
	.word	0x3d472f5f                      # float 0.048629161
	.word	0x3cf2340f                      # float 0.0295658391
	.word	0xbd68a0ac                      # float -0.0567938536
	.word	0x3cc9885a                      # float 0.0246011503
	.word	0x3c614936                      # float 0.013750365
	.word	0xbd2e5fdc                      # float -0.0425718874
	.word	0xbd430fa8                      # float -0.0476223528
	.word	0xbd8c09be                      # float -0.0683779567
	.word	0xbd7620d8                      # float -0.060089916
	.word	0x3d0da864                      # float 0.0345844179
	.word	0xbd31f05c                      # float -0.0434421152
	.word	0x3cf7c4c6                      # float 0.0302451961
	.word	0xbb43ca17                      # float -0.00298750936
	.word	0xbca4d7c0                      # float -0.0201224089
	.word	0x3ccc37f5                      # float 0.0249290261
	.word	0xbb3f8c2d                      # float -0.00292278384
	.word	0x3d4b174a                      # float 0.0495827571
	.word	0x3d288a3c                      # float 0.0411474556
	.word	0xbce8e24e                      # float -0.028428223
	.word	0x3d454739                      # float 0.0481636263
	.word	0x3d991280                      # float 0.0747423172
	.word	0x3d14ca70                      # float 0.0363258719
	.word	0xbd916499                      # float -0.0709926561
	.word	0xbd4cf741                      # float -0.0500404872
	.word	0x3d2dba88                      # float 0.0424142182
	.word	0x3af9148c                      # float 0.00190033158
	.word	0xbc9ac7e1                      # float -0.0188941378
	.word	0xbd5904e7                      # float -0.0529831909
	.word	0xbd0176d0                      # float -0.0316074491
	.word	0xbd1486ac                      # float -0.0362612456
	.word	0xbca8ce47                      # float -0.0206061732
	.word	0xbd35632e                      # float -0.0442840382
	.word	0x3d6d1e8a                      # float 0.0578904524
	.word	0x3d9a87b0                      # float 0.0754541159
	.word	0x3d3e9c3f                      # float 0.0465357266
	.word	0xbd7eeb2e                      # float -0.0622360036
	.word	0x3d72116b                      # float 0.0590986423
	.word	0xbd92f3c3                      # float -0.0717540011
	.word	0x3d825d08                      # float 0.0636540055
	.word	0xbc1b0f39                      # float -0.00946407858
	.word	0xbba9cb23                      # float -0.00518168649
	.word	0x3c24c1ea                      # float 0.0100559983
	.word	0x3d808ec7                      # float 0.0627723262
	.word	0xbb754cbe                      # float -0.00374297751
	.word	0x3a7e2a63                      # float 9.69564717E-4
	.word	0xbc966783                      # float -0.0183599051
	.word	0x3da3c52a                      # float 0.0799659044
	.word	0xbd4a474f                      # float -0.0493844114
	.word	0xbcb57298                      # float -0.0221493691
	.word	0x3cd7bbb8                      # float 0.0263346285
	.word	0xbd28aaa6                      # float -0.041178368
	.word	0x3da17d37                      # float 0.0788521096
	.word	0xbd810b65                      # float -0.0630100146
	.word	0x3d1abbd1                      # float 0.0377767719
	.word	0xbc8add40                      # float -0.0169512033
	.word	0x3cc6b392                      # float 0.0242555477
	.word	0xbd665bd6                      # float -0.0562399253
	.word	0xbd7b0c19                      # float -0.0612908341
	.word	0x3d581258                      # float 0.052751869
	.word	0xbbe4ba74                      # float -0.00698023475
	.word	0xbcd68f8f                      # float -0.0261915009
	.word	0xbda3c562                      # float -0.0799663216
	.word	0x3da3a82a                      # float 0.0799105912
	.word	0x3d9e768f                      # float 0.07737457
	.word	0x3c544bac                      # float 0.0129574947
	.word	0x3d6706eb                      # float 0.0564030819
	.word	0xbd9b4409                      # float -0.0758133605
	.word	0xbbe85f0a                      # float -0.00709140766
	.word	0xbd4df10a                      # float -0.0502787009
	.word	0xbd6acc1d                      # float -0.0573235638
	.word	0x3d8cf7db                      # float 0.0688321218
	.word	0x3b1ec898                      # float 0.00242284499
	.word	0xbd859a70                      # float -0.0652359724
	.word	0xbd8cabf6                      # float -0.0686873645
	.word	0xbd518fa8                      # float -0.0511623919
	.word	0xbd92a959                      # float -0.0716120675
	.word	0xbccd6203                      # float -0.0250711497
	.word	0xbbcd10d0                      # float -0.00625810772
	.word	0x3da4b4cf                      # float 0.08042299
	.word	0xbd84715d                      # float -0.0646693483
	.word	0x3b638836                      # float 0.00347186392
	.word	0xbbcf8805                      # float -0.00633335346
	.word	0x3c750531                      # float 0.014954851
	.word	0x3d35d540                      # float 0.0443928242
	.word	0x3d47c766                      # float 0.0487741455
	.word	0x3c25f7c9                      # float 0.0101298774
	.word	0xbd9babd2                      # float -0.076011315
	.word	0x3d52b838                      # float 0.0514452159
	.word	0xbc4cafe6                      # float -0.0124931093
	.word	0x3be74a92                      # float 0.00705845002
	.word	0xbd144571                      # float -0.036199037
	.word	0xbda59e88                      # float -0.0808687806
	.word	0xbca29eda                      # float -0.019851137
	.word	0x3d8463b8                      # float 0.0646433234
	.word	0x3c1e7877                      # float 0.00967227574
	.word	0x3d85151c                      # float 0.0649816691
	.word	0xbd667e50                      # float -0.0562728047
	.word	0xbac0e641                      # float -0.00147170585
	.word	0x3d0f098e                      # float 0.0349212214
	.word	0xbd1a1490                      # float -0.0376172662
	.word	0x3d56ab06                      # float 0.0524091944
	.word	0xbc186350                      # float -0.00930102169
	.word	0xbd188ba7                      # float -0.0372425579
	.word	0xbd3b0142                      # float -0.0456554964
	.word	0x3d7225e8                      # float 0.0591181815
	.word	0x3d8fe677                      # float 0.0702637956
	.word	0x3c960c5a                      # float 0.0183164366
	.word	0xbd2bf05a                      # float -0.041977264
	.word	0xbd51988e                      # float -0.0511708781
	.word	0x3d375c4f                      # float 0.0447657667
	.word	0x3c93028a                      # float 0.0179455467
	.word	0xbc074827                      # float -0.00825694855
	.word	0x3c8027a9                      # float 0.0156439114
	.word	0xbc2eb52f                      # float -0.0106633147
	.word	0xbd0d608e                      # float -0.0345159099
	.word	0xbd6bcbd9                      # float -0.0575674511
	.word	0x3c8cf850                      # float 0.0172082484
	.word	0x3d0b2b17                      # float 0.0339766406
	.word	0xbd96b405                      # float -0.0735855475
	.word	0xbd01b572                      # float -0.0316671804
	.word	0x3c2350e7                      # float 0.00996801909
	.word	0x3d82b75b                      # float 0.0638262853
	.word	0xbc5caf21                      # float -0.0134694884
	.word	0xbd5b245d                      # float -0.0535014756
	.word	0x3d0ab894                      # float 0.0338674337
	.word	0x3cf1dc5d                      # float 0.0295240227
	.word	0xbda2b164                      # float -0.0794399082
	.word	0xbc2962b8                      # float -0.0103384778
	.word	0xbd422a82                      # float -0.0474038199
	.word	0x3c7d3348                      # float 0.0154541209
	.word	0xbd79809a                      # float -0.0609136596
	.word	0xbc9d09cf                      # float -0.0191697162
	.word	0xbc9fcc77                      # float -0.0195066761
	.word	0xbae8aac1                      # float -0.00177510839
	.word	0x3da128cb                      # float 0.0786910876
	.word	0xbd51a069                      # float -0.0511783697
	.word	0xbd001930                      # float -0.0312740207
	.word	0x3d80116f                      # float 0.0625332519
	.word	0x3da7fe3c                      # float 0.0820278823
	.word	0x3d70f0d4                      # float 0.0588234216
	.word	0x3d76fac4                      # float 0.0602977425
	.word	0x3d917341                      # float 0.0710206106
	.word	0x3d8be060                      # float 0.068299055
	.word	0x3d48fef7                      # float 0.0490712784
	.word	0x3adabb64                      # float 0.00166879268
	.word	0x3cb78f53                      # float 0.0224072095
	.word	0xbbb7260a                      # float -0.00558925141
	.word	0xbd8d75a4                      # float -0.0690720379
	.word	0xbbaac995                      # float -0.00521201873
	.word	0x3c78ff99                      # float 0.015197658
	.word	0x3d5334ce                      # float 0.0515640303
	.word	0x3d45c852                      # float 0.0482867435
	.word	0xbd607540                      # float -0.0547993183
	.word	0xbd95f0bf                      # float -0.073213093
	.word	0x3c5baa4a                      # float 0.0134072993
	.word	0x3d6c8813                      # float 0.057746958
	.word	0x3d4913ca                      # float 0.0490911379
	.word	0xbc859455                      # float -0.0163060818
	.word	0x3cb77ba7                      # float 0.0223978292
	.word	0x3b138202                      # float 0.00225079106
	.word	0x3bd348bb                      # float 0.00644787913
	.word	0x3c167451                      # float 0.00918300543
	.word	0xbc989efc                      # float -0.0186304972
	.word	0x3a958af1                      # float 0.00114092056
	.word	0xbda3f1b7                      # float -0.0800508782
	.word	0xbc5ba2a9                      # float -0.0134054804
	.word	0xbd6bebe2                      # float -0.0575980023
	.word	0x3d5c436e                      # float 0.0537752435
	.word	0xbd8a1230                      # float -0.0674175024
	.word	0x3da2781e                      # float 0.0793306679
	.word	0x3ca469a4                      # float 0.0200699046
	.word	0xbcb50125                      # float -0.0220952723
	.word	0x3d50d47b                      # float 0.0509838872
	.word	0x3d3051cb                      # float 0.0430467539
	.word	0x3d1cfd5d                      # float 0.0383275636
	.word	0xbd660222                      # float -0.0561543778
	.word	0x3d83c72e                      # float 0.0643447489
	.word	0xbd542f42                      # float -0.0518028811
	.word	0xbd5c63f5                      # float -0.053806264
	.word	0x3ccf47c5                      # float 0.0253027771
	.word	0x3d3780f6                      # float 0.0448007211
	.word	0x3da0c70e                      # float 0.0785046667
	.word	0xbb344767                      # float -0.00275083794
	.word	0x3d9cccd3                      # float 0.0765625462
	.word	0x3d6a2f55                      # float 0.0571740456
	.word	0xbd916fa6                      # float -0.0710137337
	.word	0x3d2f8949                      # float 0.0428555347
	.word	0x3cca7c04                      # float 0.0247173384
	.word	0xbbc62f12                      # float -0.00604809169
	.word	0x3d9ed452                      # float 0.0775534064
	.word	0xbd9642d8                      # float -0.0733696818
	.word	0x3c84fb5b                      # float 0.0162331369
	.word	0xbbaf16eb                      # float -0.00534330821
	.word	0x3d7717c9                      # float 0.0603254177
	.word	0x3d8cc72a                      # float 0.0687392503
	.word	0xbd0f1903                      # float -0.0349359624
	.word	0x3da8afe8                      # float 0.0823667645
	.word	0x3cdb085d                      # float 0.0267373864
	.word	0xbda56acf                      # float -0.0807701274
	.word	0x3c9cbb5f                      # float 0.0191323142
	.word	0xbd84ff66                      # float -0.0649402589
	.word	0xbc118894                      # float -0.00888266041
	.word	0x3d8ba225                      # float 0.0681803599
	.word	0xbcf9a0e2                      # float -0.0304722227
	.word	0x3abb838b                      # float 0.00143061706
	.word	0xbd27d0cd                      # float -0.0409706123
	.word	0xbd9d7cb6                      # float -0.0768980235
	.word	0xbbb8a1f0                      # float -0.00563453883
	.word	0x3d69f25f                      # float 0.0571159087
	.word	0xbd315d0e                      # float -0.0433016345
	.word	0x3d07bbdc                      # float 0.033138141
	.word	0x3d667c7a                      # float 0.0562710539
	.word	0x3d869743                      # float 0.0657181963
	.word	0xbda1ff06                      # float -0.0790996999
	.word	0xbccae04f                      # float -0.0247651618
	.word	0xbc2130b6                      # float -0.00983827374
	.word	0x3d2f2cdf                      # float 0.0427674018
	.word	0xbc229b54                      # float -0.00992472842
	.word	0x3d3e8321                      # float 0.046511773
	.word	0x3d8cef7e                      # float 0.0688161701
	.word	0x3d1b27a9                      # float 0.0378796197
	.word	0x3d603c20                      # float 0.0547448397
	.word	0x3c27c44e                      # float 0.0102396738
	.word	0x3da88d81                      # float 0.0823011472
	.word	0xbd3afec3                      # float -0.045653116
	.word	0x3d96db73                      # float 0.0736607537
	.word	0x3ce2d551                      # float 0.0276896078
	.word	0x3cf5b4aa                      # float 0.0299933739
	.word	0xbd916bc5                      # float -0.0710063353
	.word	0x3d9e7bdc                      # float 0.0773846805
	.word	0x3d8214f9                      # float 0.0635165647
	.word	0x3d8cffda                      # float 0.0688473731
	.word	0xbca3a10d                      # float -0.0199742559
	.word	0xbd6de78f                      # float -0.0580821596
	.word	0x3d69dda8                      # float 0.0570961535
	.word	0x3c291cb2                      # float 0.0103217829
	.word	0xbd894628                      # float -0.0670283437
	.word	0xbd3b4223                      # float -0.0457173698
	.word	0xbc282c5a                      # float -0.0102644805
	.word	0x3d2583a8                      # float 0.0404087603
	.word	0x3da51f36                      # float 0.0806259364
	.word	0xbd86fef6                      # float -0.0659159869
	.word	0x3d17f144                      # float 0.0370953232
	.word	0x3c0e646a                      # float 0.00869093276
	.word	0xbd60c60f                      # float -0.0548763834
	.word	0xbd6ecf38                      # float -0.0583030879
	.word	0xbcdc67be                      # float -0.0269049369
	.word	0xbd5ea36f                      # float -0.0543550812
	.word	0xbc7ce577                      # float -0.0154355681
	.word	0xbcea5385                      # float -0.0286042783
	.word	0x3d382eac                      # float 0.0449663848
	.word	0xbcc865dd                      # float -0.0244626347
	.word	0xbd951443                      # float -0.0727925524
	.word	0xbd3f39b0                      # float -0.0466858745
	.word	0xbd3c9447                      # float -0.0460398458
	.word	0x3d660227                      # float 0.0561543964
	.word	0xbd6c37d8                      # float -0.0576704443
	.word	0x3d4a109b                      # float 0.0493322425
	.word	0x3d7e7894                      # float 0.062126711
	.word	0xbcb94f54                      # float -0.0226208344
	.word	0xbbcef633                      # float -0.00631597033
	.word	0xbcd5210c                      # float -0.0260167345
	.word	0x3d1c2d52                      # float 0.0381291583
	.word	0xbc10db55                      # float -0.00884135533
	.word	0xbd9b03e0                      # float -0.0756909847
	.word	0xbcd2fd3e                      # float -0.0257555209
	.word	0xbd48ba57                      # float -0.0490058325
	.word	0xbd99df00                      # float -0.0751323699
	.word	0xbcdeea01                      # float -0.0272111911
	.word	0xbd9c3d85                      # float -0.0762892142
	.word	0xbc8180eb                      # float -0.0158085432
	.word	0x3d861f65                      # float 0.0654895678
	.word	0xbd40dae5                      # float -0.0470837541
	.word	0xbcf4a5c3                      # float -0.0298641976
	.word	0xbc80e782                      # float -0.0157353915
	.word	0x3c9c9bfd                      # float 0.0191173498
	.word	0x3d0aca60                      # float 0.0338844061
	.word	0xbcaa9e23                      # float -0.0208273586
	.word	0x3d9a3e87                      # float 0.0753145739
	.word	0xbda13f0d                      # float -0.078733541
	.word	0x3d81ba0d                      # float 0.063343145
	.word	0x3cc53758                      # float 0.0240742415
	.word	0x3d8a4228                      # float 0.0675089955
	.word	0x3d0404b2                      # float 0.0322310403
	.word	0xbcd44509                      # float -0.0259118248
	.word	0xbda2446e                      # float -0.0792320818
	.word	0x3c8e9e55                      # float 0.017409483
	.word	0xbd99738a                      # float -0.0749274045
	.word	0xbd6895a1                      # float -0.0567833222
	.word	0xbcb3a57f                      # float -0.0219295006
	.word	0xbcf97482                      # float -0.0304510631
	.word	0x3d4f6b5c                      # float 0.0506394953
	.word	0x3d043edf                      # float 0.032286521
	.word	0x3d6910bd                      # float 0.0569007285
	.word	0x3d9b59be                      # float 0.0758547634
	.word	0xbd98ae5a                      # float -0.0745512992
	.word	0xbd7a7b14                      # float -0.0611525327
	.word	0xbd20fefa                      # float -0.0393056646
	.word	0xbd8c80b4                      # float -0.0686048567
	.word	0x3d00adb6                      # float 0.0314156637
	.word	0xbd7b4076                      # float -0.0613407716
	.word	0xbb929c8d                      # float -0.00447422871
	.word	0xbc5bf221                      # float -0.0134244272
	.word	0xb98f22f0                      # float -2.73011159E-4
	.word	0xbd2b896b                      # float -0.0418790989
	.word	0x3d2372f2                      # float 0.0399045423
	.word	0xbd159b85                      # float -0.0365252681
	.word	0xbc0988ad                      # float -0.00839440245
	.word	0x3c43dffc                      # float 0.0119552575
	.word	0x3c75acca                      # float 0.0149948094
	.word	0xbda2c389                      # float -0.0794745162
	.word	0x3d9f118b                      # float 0.0776701793
	.word	0x3b872919                      # float 0.00412477227
	.word	0x3bbb734b                      # float 0.0057205311
	.word	0xbd721693                      # float -0.0591035597
	.word	0xbd70397d                      # float -0.0586485751
	.word	0x3d76a64c                      # float 0.0602171868
	.word	0x3d7d5739                      # float 0.0618507601
	.word	0xbd57163f                      # float -0.0525114499
	.word	0x3cf0f62b                      # float 0.029414257
	.word	0x3d37db17                      # float 0.0448866747
	.word	0xbcfd2548                      # float -0.0309015661
	.word	0x3d909519                      # float 0.0705968812
	.word	0xbda849e0                      # float -0.0821721553
	.word	0x3cd922c5                      # float 0.0265058372
	.word	0xbbb04a51                      # float -0.00537995296
	.word	0xbca432a6                      # float -0.0200436823
	.word	0x3d854722                      # float 0.0650770813
	.word	0xbd44c7db                      # float -0.0480421595
	.word	0x3d5f14ad                      # float 0.0544630773
	.word	0xbb6a4404                      # float -0.00357461069
	.word	0x3cc77311                      # float 0.0243468601
	.word	0xbcd327d8                      # float -0.0257758349
	.word	0xba8b8bcc                      # float -0.00106465211
	.word	0xbd48c167                      # float -0.0490125678
	.word	0xbd2d8b72                      # float -0.0423693135
	.word	0x3d2b1479                      # float 0.0417675711
	.word	0xbc36040b                      # float -0.0111093624
	.word	0x3d1a18bd                      # float 0.0376212485
	.word	0x3d595d9e                      # float 0.0530677959
	.word	0x3da4126e                      # float 0.0801132768
	.word	0x3c9451d5                      # float 0.0181054268
	.word	0x3d77e7e1                      # float 0.0605238713
	.word	0x3da4fab7                      # float 0.0805563256
	.word	0xbd543ef1                      # float -0.0518178381
	.word	0xbb9c3e95                      # float -0.00476820255
	.word	0x3d1ed554                      # float 0.0387776643
	.word	0xbc7575f0                      # float -0.0149817318
	.word	0x3d875f30                      # float 0.0660995245
	.word	0xbc6a888e                      # float -0.0143147837
	.word	0x3d45ca2d                      # float 0.048288513
	.word	0xbd6e1fd7                      # float -0.0581358336
	.word	0xbd4e9f5c                      # float -0.0504449457
	.word	0xbd26c2e9                      # float -0.0407132246
	.word	0x3d7ce0be                      # float 0.0617377684
	.word	0x3cd6a585                      # float 0.0262019727
	.word	0x3cad8a3d                      # float 0.0211840812
	.word	0x3bb45372                      # float 0.00550311152
	.word	0x3cc9a9e6                      # float 0.0246171467
	.word	0x3d731cf0                      # float 0.0593537688
	.word	0x3d3cd6f9                      # float 0.0461034514
	.word	0xbd0a95f0                      # float -0.0338343978
	.word	0xbcd2f021                      # float -0.025749268
	.word	0xbda5aa72                      # float -0.0808915049
	.word	0x3d8a86e3                      # float 0.0676400885
	.word	0x3d60d9c0                      # float 0.0548951626
	.word	0x3d840fb2                      # float 0.0644830614
	.word	0x3d692608                      # float 0.0569210351
	.word	0xbd91bc0e                      # float -0.0711594671
	.word	0xbcaa1680                      # float -0.020762682
	.word	0x3bdd4beb                      # float 0.00675343489
	.word	0xbd5e253e                      # float -0.0542347357
	.word	0x3d822812                      # float 0.0635529906
	.word	0xbda14885                      # float -0.0787516012
	.word	0x3d176c77                      # float 0.0369686745
	.word	0xbda723d2                      # float -0.0816112906
	.word	0x3b7d3792                      # float 0.00386378588
	.word	0x3c8d484b                      # float 0.017246386
	.word	0xbbe4604e                      # float -0.00696948823
	.word	0xbc7fa561                      # float -0.0156033942
	.word	0xbc57a76e                      # float -0.0131624769
	.word	0x3d8ff10c                      # float 0.0702839791
	.word	0x3d32140e                      # float 0.0434761569
	.word	0xbbb64c8b                      # float -0.00556332385
	.word	0x3d9c4fdc                      # float 0.0763241946
	.word	0x3c2dc2d2                      # float 0.0106055308
	.word	0x3ceec7d4                      # float 0.0291480199
	.word	0x3d3d7d28                      # float 0.0462619364
	.word	0xbcdc7e4d                      # float -0.0269156937
	.word	0x3d4d7b2e                      # float 0.0501663014
	.word	0x3d4dfb8c                      # float 0.0502887219
	.word	0x3d7a5d31                      # float 0.0611240305
	.word	0x3c44c2b8                      # float 0.0120093152
	.word	0x3d8a5897                      # float 0.0675517842
	.word	0xbad43167                      # float -0.00161890395
	.word	0xbd615c85                      # float -0.0550198741
	.word	0xbd8ed23e                      # float -0.0697369426
	.word	0xbd4d1305                      # float -0.0500669666
	.word	0xbd8f1e62                      # float -0.0698821694
	.word	0x3cbd9b33                      # float 0.0231452938
	.word	0xbd04b0cc                      # float -0.0323951691
	.word	0x3b1eb059                      # float 0.00242139981
	.word	0x3d6e9c6f                      # float 0.0582546555
	.word	0x3b2cb67a                      # float 0.00263538817
	.word	0x3b52fb9d                      # float 0.00321934302
	.word	0x3d6c8c11                      # float 0.0577507652
	.word	0xbc92067c                      # float -0.0178253576
	.word	0xbd3dc2cf                      # float -0.0463283621
	.word	0x3d18cb12                      # float 0.0373030379
	.word	0xbd3773a7                      # float -0.044788029
	.word	0x3da28010                      # float 0.0793458223
	.word	0xbd9ee6e6                      # float -0.0775888413
	.word	0xbd42d5cf                      # float -0.047567185
	.word	0x3c5c3b0a                      # float 0.0134418104
	.word	0x3c12365d                      # float 0.00892409403
	.word	0x3c3611d2                      # float 0.0111126471
	.word	0xbd19dff1                      # float -0.0375670828
	.word	0x3bfdb3c9                      # float 0.00774237933
	.word	0xbce5dad7                      # float -0.0280584525
	.word	0x3d9ff7b4                      # float 0.0781091749
	.word	0xbd946108                      # float -0.0724506974
	.word	0x3cd7c2d6                      # float 0.0263380222
	.word	0x3b93c79f                      # float 0.00450988067
	.word	0x3d8e0cd7                      # float 0.0693604276
	.word	0x3cd9cce7                      # float 0.0265869629
	.word	0xbda7e74a                      # float -0.0819841176
	.word	0xbd2c04ee                      # float -0.0419968888
	.word	0x3d5fae3b                      # float 0.0546095185
	.word	0x3c924f1b                      # float 0.0178599861
	.word	0x3d488432                      # float 0.0489541963
	.word	0xbd089458                      # float -0.0333445966
	.word	0x3cce4343                      # float 0.0251785573
	.word	0xbcb29122                      # float -0.0217977203
	.word	0xbd2c7de5                      # float -0.0421122499
	.word	0x3c5039ed                      # float 0.0127091231
	.word	0x3bf561db                      # float 0.00748847192
	.word	0xbd7390a3                      # float -0.0594641082
	.word	0xbd98cb6d                      # float -0.0746067539
	.word	0x3d16dad1                      # float 0.0368297733
	.word	0x391d2f01                      # float 1.49901971E-4
	.word	0xbd868959                      # float -0.0656916574
	.word	0xbca3038a                      # float -0.0198991485
	.word	0xbc9c5b7d                      # float -0.0190865938
	.word	0xbcd696c2                      # float -0.0261949338
	.word	0x3c3c5b05                      # float 0.0114963101
	.word	0x3d899f17                      # float 0.067197971
	.word	0xbcf92900                      # float -0.0304150581
	.word	0xbda129aa                      # float -0.0786927491
	.word	0x3bee0fc7                      # float 0.0072650644
	.word	0x3da84033                      # float 0.0821537002
	.word	0xbd5d5023                      # float -0.0540315025
	.word	0x3d73dabb                      # float 0.0595347695
	.word	0xbc54e3ee                      # float -0.0129937958
	.word	0x3d96acb5                      # float 0.0735716
	.word	0xbd5c2cb3                      # float -0.053753566
	.word	0xbcadac1e                      # float -0.0212002359
	.word	0x3d788510                      # float 0.0606737733
	.word	0x3d519971                      # float 0.0511717238
	.word	0xbc8da9d9                      # float -0.0172929037
	.word	0xba1c8f54                      # float -5.97228529E-4
	.word	0xbd867030                      # float -0.0656436682
	.word	0x3d7901e8                      # float 0.0607928335
	.word	0xbb009293                      # float -0.0019618615
	.word	0x3b0eafa2                      # float 0.00217721658
	.word	0xbcd76d8d                      # float -0.0262973551
	.word	0x3c46e0da                      # float 0.0121385697
	.word	0x3d5d8246                      # float 0.0540793166
	.word	0xbc988390                      # float -0.0186174214
	.word	0xbd9379b9                      # float -0.0720095113
	.word	0x3d49a462                      # float 0.0492290333
	.word	0x3d002daa                      # float 0.0312935486
	.word	0x3d82b665                      # float 0.0638244525
	.word	0x3c649cd8                      # float 0.0139534101
	.word	0xbd098d4c                      # float -0.0335820168
	.word	0x3d96a58a                      # float 0.0735579282
	.word	0xbd331f78                      # float -0.0437311828
	.word	0x3da2923e                      # float 0.0793804973
	.word	0xbcac955c                      # float -0.0210673138
	.word	0x3c2f743e                      # float 0.0107088666
	.word	0x3d8d95ce                      # float 0.069133386
	.word	0x3b3acc46                      # float 0.00285031041
	.word	0x3d6dc525                      # float 0.0580493398
	.word	0x3d0df7a2                      # float 0.0346599892
	.word	0x3da8592f                      # float 0.0822013542
	.word	0xbd01a6eb                      # float -0.031653326
	.word	0xbbbe968e                      # float -0.00581628736
	.word	0x3c4d3d79                      # float 0.0125268633
	.word	0xbc50fd7c                      # float -0.0127557479
	.word	0xbd6260b1                      # float -0.0552679934
	.word	0xbce3900c                      # float -0.0277786478
	.word	0xbcfa2060                      # float -0.0305330157
	.word	0xbda40359                      # float -0.0800845101
	.word	0xbd1fcae8                      # float -0.0390118659
	.word	0x3c639d73                      # float 0.0138925193
	.word	0xbd36b877                      # float -0.0446095131
	.word	0xbb08bb19                      # float -0.0020863472
	.word	0xbc35b44a                      # float -0.0110903475
	.word	0xbd9f2240                      # float -0.0777020454
	.word	0x3d204ad8                      # float 0.0391338766
	.word	0xbcce1792                      # float -0.0251577236
	.word	0xbda18966                      # float -0.078875348
	.word	0x3c47a8f2                      # float 0.0121862758
	.word	0xbd8769fa                      # float -0.066120103
	.word	0xbc8f724e                      # float -0.0175105594
	.word	0x3d8d9218                      # float 0.069126308
	.word	0xbb3a4654                      # float -0.00284232665
	.word	0xbc32f19c                      # float -0.010921862
	.word	0xbd60af88                      # float -0.0548548996
	.word	0xbd899b0f                      # float -0.067190282
	.word	0xbd728eaf                      # float -0.0592181049
	.word	0xbd3ff8d7                      # float -0.0468681715
	.word	0x3bb85888                      # float 0.00562578812
	.word	0x3d8c33b4                      # float 0.0684579908
	.word	0x3d8bb254                      # float 0.0682112276
	.word	0xbd7e46a6                      # float -0.0620790944
	.word	0x3b5dd208                      # float 0.00338471122
	.word	0xbc13f429                      # float -0.00903038029
	.word	0x3b1ec9c0                      # float 0.00242291391
	.word	0x3cf832a8                      # float 0.0302975923
	.word	0x3d99bf5f                      # float 0.0750720426
	.word	0x3d2d0bc7                      # float 0.0422475599
	.word	0x3d2a34e2                      # float 0.0415543392
	.word	0x3c959126                      # float 0.0182576887
	.word	0x3c6edf8a                      # float 0.0145796631
	.word	0x3d319f95                      # float 0.0433650799
	.word	0xbd0206d2                      # float -0.0317447856
	.word	0x3d240fed                      # float 0.0400542505
	.word	0xbd0810da                      # float -0.0332191959
	.word	0x3c9feea1                      # float 0.0195229668
	.word	0xbd2e9e19                      # float -0.0426312424
	.word	0x3d810a32                      # float 0.0630077273
	.word	0xbd258f8b                      # float -0.0404200964
	.word	0x3bcb2e26                      # float 0.00620056968
	.word	0x3d58e6b5                      # float 0.0529543944
	.word	0x3da27df4                      # float 0.079341799
	.word	0xbda23063                      # float -0.0791938528
	.word	0x3d00877e                      # float 0.0313792154
	.word	0x3d668570                      # float 0.0562795997
	.word	0xbd564277                      # float -0.0523094796
	.word	0x3d2530ee                      # float 0.0403298661
	.word	0xbcc2e75b                      # float -0.0237919595
	.word	0xbd4ec50b                      # float -0.0504808836
	.word	0x3baea4ef                      # float 0.00532972021
	.word	0x3da6a699                      # float 0.0813724473
	.word	0x3c6c1c3f                      # float 0.0144110313
	.word	0xbd0c04e0                      # float -0.0341843367
	.word	0xbc46aee9                      # float -0.0121266628
	.word	0xbcbb9e64                      # float -0.022902675
	.word	0xbd7c75c7                      # float -0.0616357587
	.word	0x3cab9978                      # float 0.020947203
	.word	0x3d03f013                      # float 0.0322113745
	.word	0x3d52044b                      # float 0.0512736253
	.word	0x3cb5ebfb                      # float 0.0222072508
	.word	0xbccb6d0a                      # float -0.0248322673
	.word	0x3d4d6765                      # float 0.0501474328
	.word	0x3d8ea23f                      # float 0.0696453974
	.word	0xbd7164e8                      # float -0.0589341223
	.word	0x3d7f457c                      # float 0.0623221248
	.word	0xbd5f02bf                      # float -0.0544459783
	.word	0x3d8c29ec                      # float 0.0684393346
	.word	0x3d1f4028                      # float 0.0388795435
	.word	0xbcc8c506                      # float -0.0245080106
	.word	0xbca1e1c5                      # float -0.0197609756
	.word	0xbda3d908                      # float -0.080003798
	.word	0x3d973f82                      # float 0.0738516003
	.word	0xbc004980                      # float -0.00783002376
	.word	0x3cc4113a                      # float 0.0239339955
	.word	0x3da318c3                      # float 0.0796370729
	.word	0xbd67f03d                      # float -0.0566255935
	.word	0x3d48e72e                      # float 0.0490485951
	.word	0xbd5abb02                      # float -0.0534010008
	.word	0x3c0952a7                      # float 0.00838152226
	.word	0xbcd4b73a                      # float -0.0259662755
	.word	0x3d9183e2                      # float 0.0710523278
	.word	0xbb9139a6                      # float -0.00443192106
	.word	0xbb7ccf27                      # float -0.00385756209
	.word	0x3d672d9f                      # float 0.056439992
	.word	0xbd29360e                      # float -0.0413113162
	.word	0xbd90547f                      # float -0.0704736635
	.word	0x3c6f8f04                      # float 0.0146214999
	.word	0xbc80e83b                      # float -0.0157357361
	.word	0x3c9dfa9b                      # float 0.0192845371
	.word	0x3cf4f190                      # float 0.0299003422
	.word	0x3d3d23ed                      # float 0.0461768396
	.word	0x3d6de420                      # float 0.0580788851
	.word	0xbd8ab371                      # float -0.0677250698
	.word	0xbd99c5d2                      # float -0.0750843436
	.word	0xbca9804f                      # float -0.0206910651
	.word	0x3bba262a                      # float 0.00568081904
	.word	0xbcb2c5b6                      # float -0.0218227915
	.word	0x3d15b1ad                      # float 0.036546398
	.word	0x3d4f37a5                      # float 0.0505901761
	.word	0x3cf0f2f3                      # float 0.0294127222
	.word	0xbd7dcd06                      # float -0.0619631037
	.word	0x3da65e9f                      # float 0.0812351629
	.word	0xbd106b90                      # float -0.0352588296
	.word	0x3d9fed2c                      # float 0.0780890882
	.word	0x3cca788d                      # float 0.0247156862
	.word	0xbd06361b                      # float -0.0327664427
	.word	0x3a704dd4                      # float 9.16687073E-4
	.word	0x391563af                      # float 1.42468823E-4
	.word	0xbd9c5781                      # float -0.0763387755
	.word	0x3d200548                      # float 0.0390675366
	.word	0x3c3663ed                      # float 0.0111322226
	.word	0xbbed8561                      # float -0.00724856602
	.word	0x3cf90096                      # float 0.0303957872
	.word	0xbd176532                      # float -0.0369617417
	.word	0xbd55a1ae                      # float -0.0521561429
	.word	0x3d8e0289                      # float 0.0693407729
	.word	0x3d9ea0e0                      # float 0.0774552822
	.word	0x3c9eb819                      # float 0.019374894
	.word	0x3d158c39                      # float 0.0365106799
	.word	0x3d5507bc                      # float 0.0520093292
	.word	0x3cc5769b                      # float 0.0241044071
	.word	0x3d9c2d02                      # float 0.0762577206
	.word	0xbd0ea013                      # float -0.0348206274
	.word	0xbb94122e                      # float -0.00451876875
	.word	0xbce3f081                      # float -0.0278246421
	.word	0xbce68178                      # float -0.0281379074
	.word	0xbd31759d                      # float -0.0433250554
	.word	0xbd843f1f                      # float -0.0645735189
	.word	0xbbb090c1                      # float -0.00538834976
	.word	0xbcf7406b                      # float -0.0301820841
	.word	0xbd84cd76                      # float -0.0648450106
	.word	0xbd86a7f6                      # float -0.0657500476
	.word	0x3d871eba                      # float 0.065976575
	.word	0xbc0830e8                      # float -0.00831244141
	.word	0x3bde4b1c                      # float 0.00678385607
	.word	0xbd261865                      # float -0.0405506082
	.word	0xbd72ed56                      # float -0.0593083724
	.word	0xbd326d8d                      # float -0.043561507
	.word	0xbc964e41                      # float -0.0183478612
	.word	0xbd0ead14                      # float -0.0348330289
	.word	0x3d277c9f                      # float 0.0408903323
	.word	0x3b2b24f8                      # float 0.00261145644
	.word	0xbd428327                      # float -0.0474883579
	.word	0x3d05e0d8                      # float 0.0326851308
	.word	0xba8342c7                      # float -0.0010014408
	.word	0xbd99abef                      # float -0.0750349686
	.word	0xbce230b0                      # float -0.0276111066
	.word	0x3d521b56                      # float 0.0512956008
	.word	0x3d7b8100                      # float 0.0614023209
	.word	0xbd39a550                      # float -0.0453236699
	.word	0xbd5594eb                      # float -0.0521439724
	.word	0xbd92d1dc                      # float -0.0716893374
	.word	0x3d9569cc                      # float 0.0729556977
	.word	0x3d1d62cd                      # float 0.0384243019
	.word	0xbc976ad3                      # float -0.0184835549
	.word	0xbc093cd6                      # float -0.00837632082
	.word	0xbd5dd137                      # float -0.0541546009
	.word	0x3d8cb372                      # float 0.0687016398
	.word	0xbd753ccd                      # float -0.0598724373
	.word	0x3cd084e1                      # float 0.0254539866
	.word	0x3da02e88                      # float 0.0782137513
	.word	0x3da4103d                      # float 0.080109097
	.word	0x3d037fe5                      # float 0.0321043916
	.word	0x3d3ee900                      # float 0.0466089249
	.word	0x3d53ee67                      # float 0.0517410301
	.word	0xbd104ddf                      # float -0.0352305137
	.word	0x3c826860                      # float 0.0159189105
	.word	0x3d865d37                      # float 0.0656074807
	.word	0xbd111f2f                      # float -0.0354301296
	.word	0x3d8498f8                      # float 0.0647448897
	.word	0xbd2a0011                      # float -0.0415039696
	.word	0x3cbf88d0                      # float 0.023380667
	.word	0xbd69f35c                      # float -0.0571168512
	.word	0xbd725c32                      # float -0.0591699556
	.word	0xbd3a9fd7                      # float -0.0455625914
	.word	0xbda62d29                      # float -0.0811408236
	.word	0x3d30038d                      # float 0.0429721363
	.word	0x3d172969                      # float 0.0369047262
	.word	0x3cf12bcc                      # float 0.0294398293
	.word	0x3c3a7c3b                      # float 0.0113821579
	.word	0xbda6ab4e                      # float -0.0813814253
	.word	0xbda3aadd                      # float -0.0799157395
	.word	0x3c1acdbc                      # float 0.00944846495
	.word	0x3c51a134                      # float 0.0127947815
	.word	0x3b906066                      # float 0.00440602284
	.word	0xbcec3757                      # float -0.0288349818
	.word	0x3cc24f68                      # float 0.0237195045
	.word	0xbd7ef6b3                      # float -0.0622469894
	.word	0x3cbcb97f                      # float 0.0230376702
	.word	0xbcba2132                      # float -0.0227209069
	.word	0x3ccc5c64                      # float 0.024946399
	.word	0x3d2d8013                      # float 0.0423584692
	.word	0x3d3917b5                      # float 0.0451886244
	.word	0x3d07083e                      # float 0.0329668447
	.word	0x3da8d62b                      # float 0.0824397429
	.word	0xbc3b5b4b                      # float -0.0114353402
	.word	0xbb291643                      # float -0.00258006225
	.word	0xbc82d905                      # float -0.0159726236
	.word	0xbd85b83a                      # float -0.0652927905
	.word	0x3d6d432e                      # float 0.0579253957
	.word	0xbd05d4ad                      # float -0.0326735266
	.word	0xbcf60fce                      # float -0.0300368331
	.word	0x3d895c60                      # float 0.0670707225
	.word	0x3bda78bf                      # float 0.00666722609
	.word	0xbd26f741                      # float -0.0407631435
	.word	0xbd79d1a6                      # float -0.060990952
	.word	0xbd6b7157                      # float -0.0574811362
	.word	0x3ca401ad                      # float 0.0200203303
	.word	0xbd89fe33                      # float -0.0673793778
	.word	0xbacd9b8b                      # float -0.00156866142
	.word	0xbd20882f                      # float -0.0391923748
	.word	0x3d2afc89                      # float 0.0417447425
	.word	0xbced0d30                      # float -0.0289369524
	.word	0x3d0421f7                      # float 0.0322589539
	.word	0xbb1250ac                      # float -0.00223259162
	.word	0x3c9cfab7                      # float 0.0191625189
	.word	0xbd5d9129                      # float -0.0540935136
	.word	0xbd751bf5                      # float -0.059841115
	.word	0x3d3fc3bc                      # float 0.0468175262
	.word	0x3ce91c99                      # float 0.0284560192
	.word	0x3c3c7a86                      # float 0.0115038212
	.word	0x3ceb10c4                      # float 0.0286945179
	.word	0x3d663047                      # float 0.0561983846
	.word	0x3cfc9cec                      # float 0.0308365449
	.word	0xbcfe0356                      # float -0.0310074501
	.word	0xbc2c2755                      # float -0.0105074244
	.word	0x3c63242a                      # float 0.0138636027
	.word	0x3d00493e                      # float 0.0313198492
	.word	0x3c62c2cc                      # float 0.0138403885
	.word	0x3d8606be                      # float 0.0654425472
	.word	0xbd26e9c4                      # float -0.04075028
	.word	0x3d4bc272                      # float 0.0497459844
	.word	0x3d97eee1                      # float 0.0741860941
	.word	0xbd67de55                      # float -0.0566085167
	.word	0x3c2ac2c7                      # float 0.0104224151
	.word	0x3d674e85                      # float 0.0564713664
	.word	0x3cb99dee                      # float 0.0226583146
	.word	0x3caeb090                      # float 0.0213244259
	.word	0xbd6c2096                      # float -0.0576482639
	.word	0xbd54d7e3                      # float -0.0519636981
	.word	0x3cc787bf                      # float 0.024356721
	.word	0x3d8f8ecb                      # float 0.0700965747
	.word	0x3da655e1                      # float 0.0812184885
	.word	0xbbf8d880                      # float -0.00759416819
	.word	0x3d94bfab                      # float 0.0726312026
	.word	0x3d868a7f                      # float 0.0656938478
	.word	0xbd7b0f98                      # float -0.0612941682
	.word	0x3b72d914                      # float 0.00370556582
	.word	0xbd831400                      # float -0.0640029907
	.word	0x3d1d4398                      # float 0.0383945405
	.word	0x3d46701d                      # float 0.0484467633
	.word	0xbd9b98ab                      # float -0.0759747848
	.word	0xbda1ad92                      # float -0.0789443403
	.word	0x3d8d9654                      # float 0.0691343843
	.word	0xbd41258c                      # float -0.0471549481
	.word	0x3d71761f                      # float 0.0589505397
	.word	0xbda42868                      # float -0.0801551938
	.word	0x398b848a                      # float 2.66108953E-4
	.word	0x3d88a12a                      # float 0.0667136461
	.word	0xbd9bd3a6                      # float -0.0760872811
	.word	0x3d2e8baa                      # float 0.0426136628
	.word	0xb8dd157c                      # float -1.0542103E-4
	.word	0x3c158082                      # float 0.00912487693
	.word	0x3d9f7e6e                      # float 0.0778778642
	.word	0x3d992bc4                      # float 0.0747905075
	.word	0x3d6c0f44                      # float 0.0576317459
	.word	0x3ccf79a6                      # float 0.0253265612
	.word	0x3c7d0ad3                      # float 0.0154444752
	.word	0xbd4c26c7                      # float -0.0498416685
	.word	0x3d377daa                      # float 0.044797577
	.word	0x3d874227                      # float 0.0660441443
	.word	0xbd2cc9ac                      # float -0.0421845168
	.word	0x3d41c847                      # float 0.04731014
	.word	0x3d75e879                      # float 0.0600361563
	.word	0x3d45e7c9                      # float 0.0483167507
	.word	0xbd73f89c                      # float -0.0595632643
	.word	0xbd94b122                      # float -0.072603479
	.word	0xbd86c1af                      # float -0.0657991096
	.word	0x3c363949                      # float 0.0111220563
	.word	0xbd8d7e79                      # float -0.0690888837
	.word	0xbc9c0977                      # float -0.0190474819
	.word	0xbd7be114                      # float -0.0614939481
	.word	0x3d208eac                      # float 0.0391985625
	.word	0x3cf455e2                      # float 0.0298261084
	.word	0xbced2a08                      # float -0.0289507061
	.word	0xbda04bc0                      # float -0.0782694816
	.word	0xbc81f83e                      # float -0.0158654414
	.word	0xbda02a54                      # float -0.0782057344
	.word	0xbda0e4f0                      # float -0.0785616636
	.word	0xbd80b803                      # float -0.0628509745
	.word	0xbd271e07                      # float -0.0408001207
	.word	0x3cf3279f                      # float 0.0296819787
	.word	0xbce1a3d0                      # float -0.0275439322
	.word	0x3d871506                      # float 0.0659580678
	.word	0xbd878ac2                      # float -0.0661826283
	.word	0x3cb417f0                      # float 0.0219840705
	.word	0x3c74abde                      # float 0.0149335545
	.word	0x3d0c374d                      # float 0.0342324264
	.word	0x3d985df5                      # float 0.0743979588
	.word	0xbd757b25                      # float -0.0599318929
	.word	0xbcb943c2                      # float -0.0226153173
	.word	0xbd865a0c                      # float -0.0656014382
	.word	0xbc30408b                      # float -0.0107575757
	.word	0xbd5e8c22                      # float -0.0543328598
	.word	0xbd3e78e7                      # float -0.0465020202
	.word	0xbb4eedde                      # float -0.00315748854
	.word	0x3c202ea2                      # float 0.00977674312
	.word	0x3ba6436b                      # float 0.00507395482
	.word	0x3d9d50cb                      # float 0.0768142566
	.word	0x3c64d94d                      # float 0.0139678242
	.word	0xbb064d60                      # float -0.00204928964
	.word	0x3cf11216                      # float 0.0294275694
	.word	0xbd2abe0e                      # float -0.0416851565
	.word	0x3b4916dc                      # float 0.00306837913
	.word	0xbd9faa95                      # float -0.0779620781
	.word	0xbd3c4d19                      # float -0.0459719636
	.word	0x3d8e27c5                      # float 0.0694117919
	.word	0xbd4bc62d                      # float -0.049749542
	.word	0xbd549fb3                      # float -0.0519101135
	.word	0x3d88d21d                      # float 0.0668070093
	.word	0xbd0683fc                      # float -0.0328407139
	.word	0xbd82e13f                      # float -0.0639061853
	.word	0xbce293b4                      # float -0.027658321
	.word	0x3d800178                      # float 0.0625028014
	.word	0x3d937fd3                      # float 0.0720211491
	.word	0x3d5cfe69                      # float 0.0539535619
	.word	0xbd4ac3d5                      # float -0.0495031662
	.word	0xbd8c1832                      # float -0.0684055239
	.word	0xbd1eb017                      # float -0.0387421511
	.word	0xbcd844ad                      # float -0.0263999347
	.size	.L__constant_6x3x7x7xf32, 3528

	.type	.L__constant_6xf32,@object      # @__constant_6xf32
	.p2align	4, 0x0
.L__constant_6xf32:
	.word	0xbd8a5966                      # float -0.0675533265
	.word	0xbda11039                      # float -0.0786442235
	.word	0xbd9c980e                      # float -0.0764618963
	.word	0xbb1653b2                      # float -0.00229380699
	.word	0xbda3b47a                      # float -0.0799340755
	.word	0xbd9715aa                      # float -0.0737717897
	.size	.L__constant_6xf32, 24

	.section	".note.GNU-stack","",@progbits
