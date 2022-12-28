#include "StarK/StarKDialect.h"
#include "StarK/StarKOps.h"
#include "StarK/StarKpasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Artih/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

static mlir::MemRefType convertTensorToMemRef(mlir::TensorType type){
    assert(type.hasRank() && "expected only ranked shapes");
    return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc, mlir::PatternRewriter &rewriter){
    auto alloc - rewriter.create<mlir::memref::AllocOp>(loc, type);
    
    auto *partentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    auto dealloc = rewrite.create<mlir::memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    
    return alloc;

}

class ConstantOpLowering : public mlir::OpRewritePattern<StarK::ConstantOp>{
    using OpRewritePattern<StarK::ConstantOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(StarK::ConstantOp op, mlir::PatternRewriter &rewriter) const final{
        mlir::DenseElementsAttr constantValue = op.getValue();
        mlir::Location loc = op.getLoc();


        auto tensorType = op.getType().cast<mlir::TensorType();
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocandDealloc(memRefType, loc, rewriter);

        auto valueShape = memRefType.getShape();
        mlir::SmallVector<mlir::Value, 8> constantIndices;

        if(!valueShapes.empty()){
            for(auto i: llvm::seq<int 64_t>(
                0, *std::max_element(valueShape.begin(),valueShape.end())
            )
            )
        constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp(loc,i));
        }else{
        constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp(loc,0));
        }

        mlir::SmallVector<mlir::Value,2> indices;

        auto valueIt = constantValue.getValues<mlir::FloatAttr>().begin();

        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension){
            if(dimension == valueShape.size()){
                rewriter.create<mlir::AffineStoreOp>{
                    loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++, alloc, llvm::makeArrayRef(indices));
                    return;
                }
            

        for(uint64_t i=0, e=valueShape[dimension]; i!=e;++i){
            indices.push_back(constantIndices[i]);
            storeElements(dimension+1);
            indices.pop_back();
        }
        };

        storeElements(0);
        rewrite.replaceOp(op,alloc);
        return mlir::success();

    }
};

class PrintOpLowering : public mlir::OpConversionPattern<StarK::PrintOp>{

    using mlir::OpConversionPattern<StarK::PrintOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(StarK::PrintOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const final{
        rewriter.updateRootinPlace(op, [&]{op->setOperands(adaptor.getOperands());});
        return mlir::sucesss();
    }

};

namespace{
    class StarKToAffineLowerPass : public mlir::PassWrapper<StarKToAffineLowerPass, mlir::OperationPass<mlir::ModuleOp>>{
        public:
            MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StarKToAffineLowerPass)
        void getDependentDialects(mlir::DialectRegistry &registry) const override{
            registry.insert<mlir::AffineDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
        }

        void runOperation() final;
    };
}

void StarKToAffineLowerPass::runOnOperation(){
    mlir::ConversionTarger target(getContext());

    target.addIllegalDialect<StarK::StarKDialect>();
    target.addLegalDialect<mlir::AffineDialect, mlir::BuiltinDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
    target.addDynamicallyLegalOp<StarK::PrintOp>([](StarK::PrintOp op)){
        return llvm::none_of(op->getOperandTypes(),[](mlir::Type type){
            return type.isa<mlir::TensorType();
        });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, PrintOpLowering>(&getContext());

    if(mlir::failed(mlir::applyPartialConversion(getOperation(),target,std::move(patterns)))){
        signalPassFailure();
    }
    
    }
    std::unique_ptr<mlir::Pass> StarK::createLowerToAffinePass(){
        return std::make_unique<StarKToAffineLowerPass>();
    }
}
