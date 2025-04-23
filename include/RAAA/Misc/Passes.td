//===- Passes.td - Pattern Match Optimizations for ADORA -*- tablegen -*----===//
//===----------------------------------------------------------------------===//
//
// Defines language-specific pattern match optimizations for ADORA using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_MISC_PASSES
#define ADORA_MISC_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Basic analysis passes
//===----------------------------------------------------------------------===//

def TestPrintOpNesting : Pass<"adora-test-print-nesting"> {
  let summary = "Print op nesting information";
  let description = [{
      Print nested intructure and metadata info associated with mlir operations.
  }];
  let constructor = "mlir::ADORA::createTestPrintOpNestingPass()";
  let options = [
    Option<"varName", "option-name", "uint64_t", /*default=*/"512",
           "Description of the option">,
  ];
}


//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------------------------------===//

#endif // ADORA_MISC_PASSES
