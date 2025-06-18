//===- ADORAKernelCallToLLVM.cpp - ADORA to LLVM dialect conversion -------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Affine/Utils.h"

#include "mlir/IR/Attributes.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

#include <iostream>
#include <algorithm>
#include <functional>

#include "./LowerPassDetail.h"
#include "RAAA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "RAAA/Dialect/ADORA/IR/ADORA.h"

#define DEBUG_TYPE "ADORA_lowering"

// using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::ADORA;
// using Value = mlir::Value;

#define PASS_NAME "ADORA-convert-kernelcall-to-llvm"
static constexpr StringRef varargsAttrName = "func.varargs";
static constexpr StringRef linkageAttrName = "llvm.linkage";
static constexpr StringRef barePtrAttrName = "llvm.bareptr";

/// Return `true` if the `op` should use bare pointer calling convention.
static bool oldshouldUseBarePtrCallConv(Operation *op,
                                     const LLVMTypeConverter *typeConverter)
{
  return (op && op->hasAttr(barePtrAttrName)) ||
         typeConverter->getOptions().useBarePtrCallConv;
}

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
/// attributes.
static void filterFuncAttributes(func::FuncOp func, bool filterArgAndResAttrs,
                                 SmallVectorImpl<NamedAttribute> &result)
{
  for (const NamedAttribute &attr : func->getAttrs())
  {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == func.getFunctionTypeAttrName() ||
        attr.getName() == linkageAttrName ||
        attr.getName() == varargsAttrName ||
        attr.getName() == LLVM::LLVMDialect::getReadnoneAttrName() ||
        (filterArgAndResAttrs &&
         (attr.getName() == func.getArgAttrsAttrName() ||
          attr.getName() == func.getResAttrsAttrName())))
      continue;
    result.push_back(attr);
  }
}

namespace
{

  struct ADORAFuncOpConversionBase : public ConvertOpToLLVMPattern<func::FuncOp>
  {
  protected:
    using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

    // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
    // to this legalization pattern.
    template <typename FuncTy>
    bool IsArgUsedByBlockLoad(BlockArgument arg, FuncTy func) const
    {
      auto result = func.walk([&](ADORA::DataBlockLoadOp ldop) -> WalkResult {
        if(ldop.getOriginalMemref() == arg)
          return WalkResult::interrupt();
        else
          return WalkResult::advance(); 
      });
      if (result.wasInterrupted())
        return true;
      return false;
    }

    LLVM::LLVMFuncOp
    convertFuncOpToLLVMFuncOp(func::FuncOp funcOp,
                              ConversionPatternRewriter &rewriter) const
    {
      // Convert the original function arguments. They are converted using the
      // LLVMTypeConverter provided to this legalization pattern.
      auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("func.varargs");
      TypeConverter::SignatureConversion result(funcOp.getNumArguments());
      auto llvmType = getTypeConverter()->convertFunctionSignature(
          funcOp.getFunctionType(), varargsAttr && varargsAttr.getValue(),
          oldshouldUseBarePtrCallConv(funcOp, getTypeConverter()), result);
      if (!llvmType)
        return nullptr;

      // Propagate argument/result attributes to all converted arguments/result
      // obtained after converting a given original argument/result.
      SmallVector<NamedAttribute, 4> attributes;
      filterFuncAttributes(funcOp, /*filterArgAndResAttrs=*/true, attributes);
      if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs())
      {
        assert(!resAttrDicts.empty() && "expected array to be non-empty");
        auto newResAttrDicts =
            (funcOp.getNumResults() == 1)
                ? resAttrDicts
                : rewriter.getArrayAttr(rewriter.getDictionaryAttr({}));
        attributes.push_back(
            rewriter.getNamedAttr(funcOp.getResAttrsAttrName(), newResAttrDicts));
      }
      if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs())
      {
        SmallVector<Attribute, 4> newArgAttrs(
            cast<LLVM::LLVMFunctionType>(llvmType).getNumParams());
        for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i)
        {
          // Some LLVM IR attribute have a type attached to them. During FuncOp ->
          // LLVMFuncOp conversion these types may have changed. Account for that
          // change by converting attributes' types as well.
          SmallVector<NamedAttribute, 4> convertedAttrs;
          auto attrsDict = cast<DictionaryAttr>(argAttrDicts[i]);
          convertedAttrs.reserve(attrsDict.size());
          for (const NamedAttribute &attr : attrsDict)
          {
            const auto convert = [&](const NamedAttribute &attr)
            {
              return TypeAttr::get(getTypeConverter()->convertType(
                  cast<TypeAttr>(attr.getValue()).getValue()));
            };
            if (attr.getName().getValue() ==
                LLVM::LLVMDialect::getByValAttrName())
            {
              convertedAttrs.push_back(rewriter.getNamedAttr(
                  LLVM::LLVMDialect::getByValAttrName(), convert(attr)));
            }
            else if (attr.getName().getValue() ==
                     LLVM::LLVMDialect::getByRefAttrName())
            {
              convertedAttrs.push_back(rewriter.getNamedAttr(
                  LLVM::LLVMDialect::getByRefAttrName(), convert(attr)));
            }
            else if (attr.getName().getValue() ==
                     LLVM::LLVMDialect::getStructRetAttrName())
            {
              convertedAttrs.push_back(rewriter.getNamedAttr(
                  LLVM::LLVMDialect::getStructRetAttrName(), convert(attr)));
            }
            else if (attr.getName().getValue() ==
                     LLVM::LLVMDialect::getInAllocaAttrName())
            {
              convertedAttrs.push_back(rewriter.getNamedAttr(
                  LLVM::LLVMDialect::getInAllocaAttrName(), convert(attr)));
            }
            else
            {
              convertedAttrs.push_back(attr);
            }
          }
          auto mapping = result.getInputMapping(i);
          assert(mapping && "unexpected deletion of function argument");
          // Only attach the new argument attributes if there is a one-to-one
          // mapping from old to new types. Otherwise, attributes might be
          // attached to types that they do not support.
          if (mapping->size == 1)
          {
            newArgAttrs[mapping->inputNo] =
                DictionaryAttr::get(rewriter.getContext(), convertedAttrs);
            continue;
          }
          // TODO: Implement custom handling for types that expand to multiple
          // function arguments.
          for (size_t j = 0; j < mapping->size; ++j)
            newArgAttrs[mapping->inputNo + j] =
                DictionaryAttr::get(rewriter.getContext(), {});
        }
        attributes.push_back(rewriter.getNamedAttr(
            funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(newArgAttrs)));
      }

      // Create an LLVM function, use external linkage by default until MLIR
      // functions have linkage.
      LLVM::Linkage linkage = LLVM::Linkage::External;
      if (funcOp->hasAttr(linkageAttrName))
      {
        auto attr =
            dyn_cast<mlir::LLVM::LinkageAttr>(funcOp->getAttr(linkageAttrName));
        if (!attr)
        {
          funcOp->emitError() << "Contains " << linkageAttrName
                              << " attribute not of type LLVM::LinkageAttr";
          return nullptr;
        }
        linkage = attr.getLinkage();
      }

      // Create a memory effect attribute corresponding to readnone.
      StringRef readnoneAttrName = LLVM::LLVMDialect::getReadnoneAttrName();
      LLVM::MemoryEffectsAttr memoryAttr = {};
      if (funcOp->hasAttr(readnoneAttrName))
      {
        auto attr = funcOp->getAttrOfType<UnitAttr>(readnoneAttrName);
        if (!attr)
        {
          funcOp->emitError() << "Contains " << readnoneAttrName
                              << " attribute not of type UnitAttr";
          return nullptr;
        }
        memoryAttr = LLVM::MemoryEffectsAttr::get(rewriter.getContext(),
                                                  {LLVM::ModRefInfo::NoModRef,
                                                   LLVM::ModRefInfo::NoModRef,
                                                   LLVM::ModRefInfo::NoModRef});
      }
      auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
          funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
          /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr,
          attributes);
      // If the memory attribute was created, add it to the function.
      if (memoryAttr)
        newFuncOp.setMemoryAttr(memoryAttr);
      llvm::errs() << "[debug] newFuncOp before inlineRegionBefore" << newFuncOp <<"\n";
      llvm::errs() << "[debug] funcOp before inlineRegionBefore" << funcOp <<"\n";
      // This function is not a kernel
      if (!funcOp.getOperation()->hasAttr("Kernel"))
      {
        // llvm::errs() << "[debug] not a kernel func\n";
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                    newFuncOp.end());
        llvm::errs() << "[debug] after newFuncOp inlineRegionBefore" << newFuncOp <<"\n";
        for(auto arg : newFuncOp.getBody().getArguments()){
          llvm::errs() << "[debug] arg:" << arg << " type: " << arg.getType() << "\n";
          if(isa<MemRefType>(arg.getType()) && IsArgUsedByBlockLoad<LLVM::LLVMFuncOp>(arg, newFuncOp)){
            llvm::errs() << "[debug] yes:" << arg <<"\n";
            auto convertedType = getTypeConverter()->convertType(arg.getType());
            rewriter.setInsertionPointToStart(&(newFuncOp.getBody().front()));
            auto cast0 = rewriter.create<UnrealizedConversionCastOp>(newFuncOp.getLoc(),
                                                convertedType, arg);
            auto cast1 = rewriter.create<UnrealizedConversionCastOp>(newFuncOp.getLoc(),
                                                arg.getType(), cast0.getOutputs()[0]);                                                
            // newFuncOp.getBody()->front().push_front(cast);
            llvm::errs() << "[debug] cast0:" << cast0 <<"\n";
            llvm::errs() << "[debug] cast1:" << cast1 <<"\n";
            llvm::errs() << "[debug] newFuncOp after cast:" << newFuncOp <<"\n";
            // rewriter.replaceUsesOfBlockArgument(arg, cast1.getOutputs()[0]);
            // replaceUsesOfWith(arg, cast1.getOutputs()[0]);
            newFuncOp.walk([&](ADORA::DataBlockLoadOp ldop){
              ldop.getOperation()->replaceUsesOfWith(arg, cast1.getOutputs()[0]);
            });

            llvm::errs() << "[debug] newFuncOp after cast:" << newFuncOp <<"\n";
          }
        }
        llvm::errs() << "[debug] newFuncOp before convertRegionTypes" << newFuncOp <<"\n";
        FailureOr<Block *> block = rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter, &result);
        if (failed(block)){
          return nullptr;
        }
        // if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
        //                                        &result)))
        //   return nullptr;
      }
      llvm::errs() << "[debug] newFuncOp after convertRegionTypes" << newFuncOp <<"\n";
      // rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
      //                             newFuncOp.end());


      return newFuncOp;
    }
  };

  /// FuncOp legalization pattern that converts MemRef arguments to bare pointers
  /// to the MemRef element type. This will impact the calling convention and ABI.
  struct KernelFuncOpConversion : public ADORAFuncOpConversionBase
  {
    using ADORAFuncOpConversionBase::ADORAFuncOpConversionBase;
    SmallVector<::llvm::StringRef, 8> KernelNameVec;

    LogicalResult
    matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {

      // TODO: bare ptr conversion could be handled by argument materialization
      // and most of the code below would go away. But to do this, we would need a
      // way to distinguish between FuncOp and other regions in the
      // addArgumentMaterialization hook.

      // ::llvm::StringRef NameToFind = funcop.getSymName();
      //   // llvm::errs() <<"\nfuncop :" << NameToFind;
      //   auto it = std::find(KernelNameVec.begin(), KernelNameVec.end(), NameToFind);
      //   if (it != KernelNameVec.end()) { // FuncOp is a kernel
      //     llvm::errs() << "[debug] found a kernel func:\n";
      //     funcop.dump();
      //   }

      // llvm::errs() << "[debug] found a kernel func:\n";
      // funcOp.dump();

      // Store the type of memref-typed arguments before the conversion so that we
      // can promote them to MemRef descriptor at the beginning of the function.
      SmallVector<Type, 8> oldArgTypes =
          llvm::to_vector<8>(funcOp.getFunctionType().getInputs());

      funcOp.dump();
      auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
      llvm::errs() << "[debug] after convertFuncOpToLLVMFuncOp newFuncOp:" << newFuncOp <<"\n";
      if (!newFuncOp)
        return failure();
      if (newFuncOp.getBody().empty())
      {
        rewriter.eraseOp(funcOp);
        return success();
      }

      // Promote bare pointers from memref arguments to memref descriptors at the
      // beginning of the function so that all the memrefs in the function have a
      // uniform representation.
      Block *entryBlock = &newFuncOp.getBody().front();
      auto blockArgs = entryBlock->getArguments();
      assert(blockArgs.size() == oldArgTypes.size() &&
             "The number of arguments and types doesn't match");

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);
      for (auto it : llvm::zip(blockArgs, oldArgTypes))
      {
        BlockArgument arg = std::get<0>(it);
        Type argTy = std::get<1>(it);

        // Unranked memrefs are not supported in the bare pointer calling
        // convention. We should have bailed out before in the presence of
        // unranked memrefs.
        assert(!argTy.isa<UnrankedMemRefType>() &&
               "Unranked memref is not supported");
        auto memrefTy = argTy.dyn_cast<MemRefType>();
        if (!memrefTy)
          continue;

        // Replace barePtr with a placeholder (undef), promote barePtr to a ranked
        // or unranked memref descriptor and replace placeholder with the last
        // instruction of the memref descriptor.
        // TODO: The placeholder is needed to avoid replacing barePtr uses in the
        // MemRef descriptor instructions. We may want to have a utility in the
        // rewriter to properly handle this use case.
        Location loc = funcOp.getLoc();
        auto placeholder = rewriter.create<LLVM::UndefOp>(
            loc, getTypeConverter()->convertType(memrefTy));
        llvm::errs() << "[debug] before replaceUsesOfBlockArgument newFuncOp:" << newFuncOp <<"\n";
        rewriter.replaceUsesOfBlockArgument(arg, placeholder);
        llvm::errs() << "[debug] after replaceUsesOfBlockArgument newFuncOp:" << newFuncOp <<"\n";

        Value desc = MemRefDescriptor::fromStaticShape(
            rewriter, loc, *getTypeConverter(), memrefTy, arg);
        // llvm::errs() << "[debug] desc:" << desc <<"\n";
        // llvm::errs() << "[debug] placeholder:" << placeholder <<"\n";
        rewriter.replaceOp(placeholder, {desc});
      }
      // replaceOpWithIf


      rewriter.eraseOp(funcOp);
      llvm::errs() << "[debug] newFuncOp:" << newFuncOp <<"\n";

      return success();
    }
  };

  bool isAllUnrealizedConversionCastOp(const mlir::ValueRange &operands)
  {
    for (const mlir::Value oper : operands)
    {
      auto op = oper.getDefiningOp();
      if (llvm::isa<mlir::UnrealizedConversionCastOp>(op))
      {
        continue;
      }
      else
      {
        return false;
      }
    }
    return true;
  }

  // A CallOp automatically promotes MemRefType to a sequence of alloca/store and
  // passes the pointer to the MemRef across function boundaries.
  template <typename CallOpType>
  struct KernelCallOpInterfaceLowering : public ConvertOpToLLVMPattern<CallOpType>
  {
    using ConvertOpToLLVMPattern<CallOpType>::ConvertOpToLLVMPattern;
    using Super = KernelCallOpInterfaceLowering<CallOpType>;
    using Base = ConvertOpToLLVMPattern<CallOpType>;

    LogicalResult
    matchAndRewrite(CallOpType callOp, typename CallOpType::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
      // llvm::errs() << "[debug] callOp: " << callOp << "\n";
      // for(auto oper : callOp->getOperands()){
      //   llvm::errs() << "[debug] callOp->getOperands(): " << oper << "\n";
      // }
      // for(auto oper : adaptor.getOperands()){
      //   llvm::errs() << "[debug] adaptor.getOperands(): " << oper << "\n";
      // }
      // Pack the result types into a struct.
      Type packedResult = nullptr;
      unsigned numResults = callOp.getNumResults();
      auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

      if (numResults != 0)
      {
        if (!(packedResult =
                  this->getTypeConverter()->packFunctionResults(resultTypes)))
          return failure();
      }

      // assert(adaptor.getOperands().size()==1);
      LLVM::CallOp newOp;
      SmallVector<Value, 4> promoted;
      const mlir::ValueRange operands = adaptor.getOperands();
      if (isAllUnrealizedConversionCastOp(operands))
      {
        for (auto oper : adaptor.getOperands())
        {
          auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(oper.getDefiningOp());
          auto llvmOperand = cast.getInputs()[0];
          // llvm::errs() << "[debug] llvmOperand: " << llvmOperand << "\n";
          promoted.push_back(llvmOperand);
        }
      }
      else
      {
        promoted = this->getTypeConverter()->promoteOperands(
            callOp.getLoc(), /*opOperands=*/callOp->getOperands(),
            adaptor.getOperands(), rewriter, /*useBarePtrCallConv=*/false);

        // assert(promoted.size() == 1);
      }

      newOp = rewriter.create<LLVM::CallOp>(
          callOp.getLoc(), packedResult ? TypeRange(packedResult) : TypeRange(),
          promoted, callOp->getAttrs());

      SmallVector<mlir::Value, 4> results;
      if (numResults < 2)
      {
        // If < 2 results, packing did not do anything and we can just return.
        results.append(newOp.result_begin(), newOp.result_end());
      }
      else
      {
        // Otherwise, it had been converted to an operation producing a structure.
        // Extract individual results from the structure and return them as list.
        results.reserve(numResults);
        for (unsigned i = 0; i < numResults; ++i)
        {
          results.push_back(rewriter.create<LLVM::ExtractValueOp>(
              callOp.getLoc(), newOp->getResult(0), i));
        }
      }

      if (this->getTypeConverter()->getOptions().useBarePtrCallConv)
      {
        // For the bare-ptr calling convention, promote memref results to
        // descriptors.
        assert(results.size() == resultTypes.size() &&
               "The number of arguments and types doesn't match");
        this->getTypeConverter()->promoteBarePtrsToDescriptors(
            rewriter, callOp.getLoc(), resultTypes, results);
      }
      else if (failed(this->copyUnrankedDescriptors(rewriter, callOp.getLoc(),
                                                    resultTypes, results,
                                                    /*toDynamic=*/false)))
      {
        return failure();
      }

      rewriter.replaceOp(callOp, results);
      // llvm::errs() <<"[debug] old callOp:";
      // callOp.dump();
      // llvm::errs() <<"[debug] new callOp:";
      // newOp.dump();

      return success();
    }
  };

  struct KernelCallOpLowering : public KernelCallOpInterfaceLowering<ADORA::KernelCallOp>
  {
    using Super::Super;
  };

  // struct CallIndirectOpLowering
  //     : public KernelCallOpInterfaceLowering<func::CallIndirectOp> {
  //   using Super::Super;
  // };

  struct UnrealizedConversionCastOpLowering
      : public ConvertOpToLLVMPattern<UnrealizedConversionCastOp>
  {
    using ConvertOpToLLVMPattern<
        UnrealizedConversionCastOp>::ConvertOpToLLVMPattern;

    LogicalResult
    matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
      SmallVector<Type> convertedTypes;
      llvm::errs() <<"[debug] UnrealizedConversionCastOp op:" << op << "\n";
      if (succeeded(typeConverter->convertTypes(op.getOutputs().getTypes(),
                                                convertedTypes)) &&
          convertedTypes == adaptor.getInputs().getTypes())
      {
        rewriter.replaceOp(op, adaptor.getInputs());
        return success();
      }

      convertedTypes.clear();
      if (succeeded(typeConverter->convertTypes(adaptor.getInputs().getTypes(),
                                                convertedTypes)) &&
          convertedTypes == op.getOutputs().getType())
      {
        rewriter.replaceOp(op, adaptor.getInputs());
        return success();
      }
      return failure();
    }
  };

  // Special lowering pattern for `ReturnOps`.  Unlike all other operations,
  // `ReturnOp` interacts with the function signature and must have as many
  // operands as the function has return values.  Because in LLVM IR, functions
  // can only return 0 or 1 value, we pack multiple values into a structure type.
  // Emit `UndefOp` followed by `InsertValueOp`s to create such structure if
  // necessary before returning it
  struct ReturnOpLowering : public ConvertOpToLLVMPattern<func::ReturnOp>
  {
    using ConvertOpToLLVMPattern<func::ReturnOp>::ConvertOpToLLVMPattern;

    LogicalResult
    matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
      Location loc = op.getLoc();
      unsigned numArguments = op.getNumOperands();
      SmallVector<Value, 4> updatedOperands;

      auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
      bool useBarePtrCallConv =
          oldshouldUseBarePtrCallConv(funcOp, this->getTypeConverter());
      if (useBarePtrCallConv)
      {
        // For the bare-ptr calling convention, extract the aligned pointer to
        // be returned from the memref descriptor.
        for (auto it : llvm::zip(op->getOperands(), adaptor.getOperands()))
        {
          Type oldTy = std::get<0>(it).getType();
          Value newOperand = std::get<1>(it);
          if (isa<MemRefType>(oldTy) && getTypeConverter()->canConvertToBarePtr(
                                            cast<BaseMemRefType>(oldTy)))
          {
            MemRefDescriptor memrefDesc(newOperand);
            newOperand = memrefDesc.allocatedPtr(rewriter, loc);
          }
          else if (isa<UnrankedMemRefType>(oldTy))
          {
            // Unranked memref is not supported in the bare pointer calling
            // convention.
            return failure();
          }
          updatedOperands.push_back(newOperand);
        }
      }
      else
      {
        updatedOperands = llvm::to_vector<4>(adaptor.getOperands());
        (void)copyUnrankedDescriptors(rewriter, loc, op.getOperands().getTypes(),
                                      updatedOperands,
                                      /*toDynamic=*/true);
      }

      // If ReturnOp has 0 or 1 operand, create it and return immediately.
      if (numArguments <= 1)
      {
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
            op, TypeRange(), updatedOperands, op->getAttrs());
        return success();
      }

      // Otherwise, we need to pack the arguments into an LLVM struct type before
      // returning.
      auto packedType = getTypeConverter()->packFunctionResults(
          op.getOperandTypes(), useBarePtrCallConv);
      if (!packedType)
      {
        return rewriter.notifyMatchFailure(op, "could not convert result types");
      }

      Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
      for (auto [idx, operand] : llvm::enumerate(updatedOperands))
      {
        packed = rewriter.create<LLVM::InsertValueOp>(loc, packed, operand, idx);
      }
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), packed,
                                                  op->getAttrs());
      return success();
    }
  };

  // BlockLoadOpLowering: Custom pattern for lowering ADORA::DataBlockLoadOp to LLVM IR.
  // Inherits from ConvertOpToLLVMPattern for conversion within MLIR dialects.
  // matchAndRewrite: Tries to match and lower ADORA::DataBlockLoadOp instances.
  // Extracts relevant operands and invokes getOffsetedElementPtr for element pointer calculation.
  // getOffsetedElementPtr: Calculates the element pointer with necessary offsets and strides.
  // Iterates through indices and strides, creating LLVM IR instructions.
  // Computes the final element pointer and returns it.

  // This pattern assists in converting high-level MLIR operations to LLVM IR,
  // contributing to the translation of MLIR for LLVM-based backends.
  struct BlockLoadOpLowering : public ConvertOpToLLVMPattern<ADORA::DataBlockLoadOp>
  {
  public:
    using ConvertOpToLLVMPattern<ADORA::DataBlockLoadOp>::ConvertOpToLLVMPattern;

    Value getOffsetedElementPtr(
        Location loc, MemRefType type, MemRefDescriptor memRefDescriptor, SmallVector<Value, 8> indices,
        ConversionPatternRewriter &rewriter) const
    {

      auto [strides, offset] = getStridesAndOffset(type);

      // MemRefDescriptor memRefDescriptor = MemRefDescriptor::fromStaticShape(
      // rewriter,loc, *getTypeConverter(), type, input);

      // MemRefDescriptor memRefDescriptor = input.getDefiningOp<MemRefDescriptor>();
      // llvm::DebugFlag = true;
      LLVM_DEBUG(llvm::errs() << "[debug] memRefDescriptor: " << memRefDescriptor << "\n");
      // Use a canonical representation of the start address so that later
      // optimizations have a longer sequence of instructions to CSE.
      // If we don't do that we would sprinkle the memref.offset in various
      // position of the different address computations.
      Value base =
          memRefDescriptor.bufferPtr(rewriter, loc, *getTypeConverter(), type);

      llvm::errs() << "[debug] base: " << base << "\n";

      Value index;
      for (int i = 0, e = indices.size(); i < e; ++i)
      {
        mlir::Value increment = indices[i];
        mlir::UnrealizedConversionCastOp cast;
        if (increment.getType() == rewriter.getIndexType())
        {
          cast = rewriter.create<UnrealizedConversionCastOp>(increment.getLoc(),
                                                             rewriter.getI64Type(), increment);
          LLVM_DEBUG(llvm::errs() << "[debug] cast: " << cast << "\n");
          increment = cast.getOutputs()[0];
        }
        assert(increment.getType() == rewriter.getI64Type());

        LLVM_DEBUG(llvm::errs() << "[debug]  increment: " << increment << ", type:" << increment.getType() << "\n");
        if (strides[i] != 1)
        { // Skip if stride is 1.
          Value stride = ShapedType::isDynamic(strides[i])
                             ? memRefDescriptor.stride(rewriter, loc, i)
                             : createIndexAttrConstant(rewriter, loc, getIndexType(), strides[i]);
          LLVM_DEBUG(llvm::errs() << "[debug]  stride: " << stride << "\n");
          increment = rewriter.create<LLVM::MulOp>(loc, increment, stride);
          LLVM_DEBUG(llvm::errs() << "[debug]  MulOp: " << increment << "\n");
        }
        index =
            index ? rewriter.create<LLVM::AddOp>(loc, index, increment) : increment;
      }

      Type elementPtrType = memRefDescriptor.getElementPtrType();
      return index ? rewriter.create<LLVM::GEPOp>(
                         loc, elementPtrType,
                         getTypeConverter()->convertType(type.getElementType()),
                         base, index)
                   : base;
    }


    MemRefDescriptor buildMemrefDscrfromStaticShape(
      ConversionPatternRewriter &rewriter, Location loc, const LLVMTypeConverter &typeConverter,
      MemRefType type, Value memory) const
    {
      assert(type.hasStaticShape() && "unexpected dynamic shape");

      // Extract all strides and offsets and verify they are static.
      auto [strides, offset] = getStridesAndOffset(type);
      assert(!ShapedType::isDynamic(offset) &&
         "expected static offset");
      assert(!llvm::any_of(strides, ShapedType::isDynamic) &&
         "expected static strides");

      auto convertedType = typeConverter.convertType(type);
      assert(convertedType && "unexpected failure in memref type conversion");
      llvm::errs() << "[debug] convertedType: " << convertedType << "\n";

      auto descr = MemRefDescriptor::undef(rewriter, loc, convertedType);
      descr.setAllocatedPtr(rewriter, loc, memory);
      descr.setAlignedPtr(rewriter, loc, memory);
      descr.setConstantOffset(rewriter, loc, offset);

      // Fill in sizes and strides
      for (unsigned i = 0, e = type.getRank(); i != e; ++i) {
        descr.setConstantSize(rewriter, loc, i, type.getDimSize(i));
        descr.setConstantStride(rewriter, loc, i, strides[i]);
      }
      return descr;
    }

    LogicalResult matchAndRewrite(ADORA::DataBlockLoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
      // Expand affine map from 'affineLoadOp'.
      // SmallVector<Value> indices(op.getMapOperands());
      auto resultOperands =
          ::mlir::affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), op.getMapOperands());
      llvm::errs() << "[debug] op: " << op << "\n";
      // for(auto oper : *resultOperands){
      //   llvm::errs() << "[debug] oper: " << oper << "\n";
      // }
      if (!resultOperands)
        return failure();
      auto type = op.getMemRefType();

      /// Follow the process of lowering memref loadop to llvm
      // memref::LoadOp loadop = rewriter.create<memref::LoadOp>(op.getLoc(), op.getOriginalMemref(), *resultOperands);
      // llvm::errs() << "[debug] loadop: " << loadop << "\n";
      llvm::errs() << "[debug] getMemref: " << op.getOriginalMemref()
                   << " type:" << op.getOriginalMemref().getType() << "\n";
      Value input;
      mlir::UnrealizedConversionCastOp memref_cast =
          op.getOriginalMemref().getDefiningOp<mlir::UnrealizedConversionCastOp>();

      if (memref_cast)
      {
        input = memref_cast.getInputs()[0];
        LLVM_DEBUG(llvm::errs() << "[debug] input: " << input << "\n");

        MemRefDescriptor memRefDescriptor = MemRefDescriptor(input);
        Value dataPtr =
            getOffsetedElementPtr(op.getLoc(), type, memRefDescriptor, resultOperands.value(), rewriter);

        LLVM_DEBUG(llvm::errs() << "[debug] dataPtr: " << dataPtr);

        rewriter.replaceOp(op, dataPtr);
      }
      else
      {
        // /// TODO: check this
        // input = op.getOriginalMemref();
        // auto placeholder = rewriter.create<LLVM::UndefOp>(
        //     op.getLoc(), getTypeConverter()->convertType(type));
        // rewriter.replaceUsesOfBlockArgument(input, placeholder);
        
        MemRefDescriptor memRefDescriptor = buildMemrefDscrfromStaticShape(
            rewriter,op.getLoc(), *getTypeConverter(), type, input);
        // rewriter.replaceOp(placeholder, {memRefDescriptor});

        // llvm::errs() << "[debug] placeholder: " << placeholder << "\n";
        llvm::errs() << "[debug] input: " << input << "\n";
        llvm::errs() << "[debug] memRefDescriptor: " << memRefDescriptor << "\n";
        Value dataPtr =
            getOffsetedElementPtr(op.getLoc(), type, memRefDescriptor, resultOperands.value(), rewriter);
        llvm::errs() << "[debug] dataPtr: " << dataPtr << "\n";
        rewriter.replaceOp(op, dataPtr);
      }

      // getOperation().dump();

      return success();
    }
  };

  struct BlockStoreOpLowering : public ConvertOpToLLVMPattern<ADORA::DataBlockStoreOp>
  {
  public:
    using ConvertOpToLLVMPattern<ADORA::DataBlockStoreOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(ADORA::DataBlockStoreOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
      /// DataBlockStoreOp can be erased directly because the mapping result contains
      /// data write back.
      rewriter.eraseOp(op);
      return success();
    }
  };

} // namespace

namespace
{
  /// A pass converting Func operations into the LLVM IR dialect.
  struct ConvertKernelCallToLLVMPass
      : public ConvertKernelCallToLLVMBase<ConvertKernelCallToLLVMPass>
  {
    ConvertKernelCallToLLVMPass() = default;

    SmallVector<::llvm::StringRef, 8> KernelNameVec;

    void runOnOperation() override
    {

      // if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
      //         this->dataLayout, [this](const Twine &message) {
      //           getOperation().emitError() << message.str();
      //         }))) {
      //   signalPassFailure();
      //   return;
      // // }

      ModuleOp m = getOperation();

      // m.walk([this](ADORA::KernelCallOp callop) {
      //   // llvm::errs() <<"[debug] Callee: "<<callop.callee()<< "\n";
      //   KernelNameVec.push_back(callop.callee());
      // });

      const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

      LowerToLLVMOptions options(&getContext(),
                                 dataLayoutAnalysis.getAtOrAbove(m));
      options.useBarePtrCallConv = true;
      // if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      //   options.overrideIndexBitwidth(indexBitwidth);
      // options.dataLayout = llvm::DataLayout(this->dataLayout);

      LLVMTypeConverter typeConverter(&getContext(), options,
                                      &dataLayoutAnalysis);

      RewritePatternSet patterns(&getContext());
      patterns.add<KernelCallOpLowering>(typeConverter);
      patterns.add<KernelFuncOpConversion>(typeConverter);
      patterns.add<ReturnOpLowering>(typeConverter);
      patterns.add<BlockLoadOpLowering>(typeConverter);
      patterns.add<BlockStoreOpLowering>(typeConverter);
      // patterns.add<UnrealizedConversionCastOpLowering>(typeConverter);
      // populateFuncToLLVMFuncOpConversionPattern(typeConverter, patterns);
      // populateFuncToLLVMConversionPatterns(typeConverter, patterns);

      // TODO: Remove these in favor of their dedicated conversion passes.
      arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
      cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
      mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

      LLVMConversionTarget target(getContext());

      if (failed(applyPartialConversion(m, target, std::move(patterns))))
      {
        signalPassFailure();
      }
      m.dump();
      // m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
      //            StringAttr::get(m.getContext(), this->dataLayout));
    }
  };
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createConvertKernelCallToLLVMPass()
{
  return std::make_unique<ConvertKernelCallToLLVMPass>();
}
