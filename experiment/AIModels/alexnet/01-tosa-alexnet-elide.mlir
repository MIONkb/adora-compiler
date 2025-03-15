module attributes {torch.debug_module_name = "AlexNet"} {
  func.func @forward(%arg0: tensor<1x3x227x227xf32>) -> tensor<1x1000xf32> {
    %0 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<1000x4096xf32>}> : () -> tensor<1000x4096xf32>
    %1 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<4096x4096xf32>}> : () -> tensor<4096x4096xf32>
    %2 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<4096x9216xf32>}> : () -> tensor<4096x9216xf32>
    %3 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<256x256x3x3xf32>}> : () -> tensor<256x256x3x3xf32>
    %4 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %5 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<256x384x3x3xf32>}> : () -> tensor<256x384x3x3xf32>
    %6 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<256xf32>}> : () -> tensor<256xf32>
    %7 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<384x192x3x3xf32>}> : () -> tensor<384x192x3x3xf32>
    %8 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<384xf32>}> : () -> tensor<384xf32>
    %9 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<192x64x5x5xf32>}> : () -> tensor<192x64x5x5xf32>
    %10 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<192xf32>}> : () -> tensor<192xf32>
    %11 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<64x3x11x11xf32>}> : () -> tensor<64x3x11x11xf32>
    %12 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<64xf32>}> : () -> tensor<64xf32>
    %13 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<4xi32>}> : () -> tensor<4xi32>
    %14 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<4xi32>}> : () -> tensor<4xi32>
    %15 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %16 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<1x4096xf32>}> : () -> tensor<1x4096xf32>
    %17 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<1x4096xf32>}> : () -> tensor<1x4096xf32>
    %18 = "tosa.const"() <{value = dense_resource<__elided__> : tensor<1x1000xf32>}> : () -> tensor<1x1000xf32>
    %19 = "tosa.transpose"(%arg0, %13) : (tensor<1x3x227x227xf32>, tensor<4xi32>) -> tensor<1x227x227x3xf32>
    %20 = "tosa.transpose"(%11, %13) : (tensor<64x3x11x11xf32>, tensor<4xi32>) -> tensor<64x11x11x3xf32>
    %21 = "tosa.conv2d"(%19, %20, %12) <{dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 4, 4>}> : (tensor<1x227x227x3xf32>, tensor<64x11x11x3xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
    %22 = "tosa.transpose"(%21, %14) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
    %23 = "tosa.clamp"(%22) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %24 = "tosa.transpose"(%23, %13) : (tensor<1x64x56x56xf32>, tensor<4xi32>) -> tensor<1x56x56x64xf32>
    %25 = "tosa.max_pool2d"(%24) <{kernel = array<i64: 3, 3>, pad = array<i64: 0, -1, 0, -1>, stride = array<i64: 2, 2>}> : (tensor<1x56x56x64xf32>) -> tensor<1x27x27x64xf32>
    %26 = "tosa.transpose"(%9, %13) : (tensor<192x64x5x5xf32>, tensor<4xi32>) -> tensor<192x5x5x64xf32>
    %27 = "tosa.conv2d"(%25, %26, %10) <{dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>}> : (tensor<1x27x27x64xf32>, tensor<192x5x5x64xf32>, tensor<192xf32>) -> tensor<1x27x27x192xf32>
    %28 = "tosa.transpose"(%27, %14) : (tensor<1x27x27x192xf32>, tensor<4xi32>) -> tensor<1x192x27x27xf32>
    %29 = "tosa.clamp"(%28) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x192x27x27xf32>) -> tensor<1x192x27x27xf32>
    %30 = "tosa.transpose"(%29, %13) : (tensor<1x192x27x27xf32>, tensor<4xi32>) -> tensor<1x27x27x192xf32>
    %31 = "tosa.max_pool2d"(%30) <{kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x27x27x192xf32>) -> tensor<1x13x13x192xf32>
    %32 = "tosa.transpose"(%7, %13) : (tensor<384x192x3x3xf32>, tensor<4xi32>) -> tensor<384x3x3x192xf32>
    %33 = "tosa.conv2d"(%31, %32, %8) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x13x13x192xf32>, tensor<384x3x3x192xf32>, tensor<384xf32>) -> tensor<1x13x13x384xf32>
    %34 = "tosa.transpose"(%33, %14) : (tensor<1x13x13x384xf32>, tensor<4xi32>) -> tensor<1x384x13x13xf32>
    %35 = "tosa.clamp"(%34) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x384x13x13xf32>) -> tensor<1x384x13x13xf32>
    %36 = "tosa.transpose"(%35, %13) : (tensor<1x384x13x13xf32>, tensor<4xi32>) -> tensor<1x13x13x384xf32>
    %37 = "tosa.transpose"(%5, %13) : (tensor<256x384x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x384xf32>
    %38 = "tosa.conv2d"(%36, %37, %6) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x13x13x384xf32>, tensor<256x3x3x384xf32>, tensor<256xf32>) -> tensor<1x13x13x256xf32>
    %39 = "tosa.transpose"(%38, %14) : (tensor<1x13x13x256xf32>, tensor<4xi32>) -> tensor<1x256x13x13xf32>
    %40 = "tosa.clamp"(%39) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x256x13x13xf32>) -> tensor<1x256x13x13xf32>
    %41 = "tosa.transpose"(%40, %13) : (tensor<1x256x13x13xf32>, tensor<4xi32>) -> tensor<1x13x13x256xf32>
    %42 = "tosa.transpose"(%3, %13) : (tensor<256x256x3x3xf32>, tensor<4xi32>) -> tensor<256x3x3x256xf32>
    %43 = "tosa.conv2d"(%41, %42, %4) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x13x13x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>) -> tensor<1x13x13x256xf32>
    %44 = "tosa.transpose"(%43, %14) : (tensor<1x13x13x256xf32>, tensor<4xi32>) -> tensor<1x256x13x13xf32>
    %45 = "tosa.clamp"(%44) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x256x13x13xf32>) -> tensor<1x256x13x13xf32>
    %46 = "tosa.transpose"(%45, %13) : (tensor<1x256x13x13xf32>, tensor<4xi32>) -> tensor<1x13x13x256xf32>
    %47 = "tosa.max_pool2d"(%46) <{kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x13x13x256xf32>) -> tensor<1x6x6x256xf32>
    %48 = "tosa.avg_pool2d"(%47) <{acc_type = f32, kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x6x6x256xf32>) -> tensor<1x6x6x256xf32>
    %49 = "tosa.transpose"(%48, %14) : (tensor<1x6x6x256xf32>, tensor<4xi32>) -> tensor<1x256x6x6xf32>
    %50 = "tosa.transpose"(%2, %15) : (tensor<4096x9216xf32>, tensor<2xi32>) -> tensor<9216x4096xf32>
    %51 = "tosa.reshape"(%49) <{new_shape = array<i64: 1, 1, 9216>}> : (tensor<1x256x6x6xf32>) -> tensor<1x1x9216xf32>
    %52 = "tosa.reshape"(%50) <{new_shape = array<i64: 1, 9216, 4096>}> : (tensor<9216x4096xf32>) -> tensor<1x9216x4096xf32>
    %53 = "tosa.matmul"(%51, %52) : (tensor<1x1x9216xf32>, tensor<1x9216x4096xf32>) -> tensor<1x1x4096xf32>
    %54 = "tosa.reshape"(%53) <{new_shape = array<i64: 1, 4096>}> : (tensor<1x1x4096xf32>) -> tensor<1x4096xf32>
    %55 = "tosa.add"(%54, %16) : (tensor<1x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %56 = "tosa.clamp"(%55) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %57 = "tosa.transpose"(%1, %15) : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
    %58 = "tosa.reshape"(%56) <{new_shape = array<i64: 1, 1, 4096>}> : (tensor<1x4096xf32>) -> tensor<1x1x4096xf32>
    %59 = "tosa.reshape"(%57) <{new_shape = array<i64: 1, 4096, 4096>}> : (tensor<4096x4096xf32>) -> tensor<1x4096x4096xf32>
    %60 = "tosa.matmul"(%58, %59) : (tensor<1x1x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x1x4096xf32>
    %61 = "tosa.reshape"(%60) <{new_shape = array<i64: 1, 4096>}> : (tensor<1x1x4096xf32>) -> tensor<1x4096xf32>
    %62 = "tosa.add"(%61, %17) : (tensor<1x4096xf32>, tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %63 = "tosa.clamp"(%62) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %64 = "tosa.transpose"(%0, %15) : (tensor<1000x4096xf32>, tensor<2xi32>) -> tensor<4096x1000xf32>
    %65 = "tosa.reshape"(%63) <{new_shape = array<i64: 1, 1, 4096>}> : (tensor<1x4096xf32>) -> tensor<1x1x4096xf32>
    %66 = "tosa.reshape"(%64) <{new_shape = array<i64: 1, 4096, 1000>}> : (tensor<4096x1000xf32>) -> tensor<1x4096x1000xf32>
    %67 = "tosa.matmul"(%65, %66) : (tensor<1x1x4096xf32>, tensor<1x4096x1000xf32>) -> tensor<1x1x1000xf32>
    %68 = "tosa.reshape"(%67) <{new_shape = array<i64: 1, 1000>}> : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
    %69 = "tosa.add"(%68, %18) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %69 : tensor<1x1000xf32>
  }
}

