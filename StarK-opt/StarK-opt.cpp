#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IntiAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "StarK/StarKDialect.h"
#include "StarK/StarKPasses.h"

namespace cl = llvm::cl;

static cl::opt<<std::string> inputFilename(cl::Positional,
                                           cl::desc("<input StarK file>"),
                                           cl::init("-"),
                                           cl::value_desc("filename"));

int dumpLLVMIR(mlir::ModuleOp module){
    mlir::registerLLVMDialectTranslation(*module->getContext());

    llvm::LLVMContext llvmContext;

    auto llvmModule = mlir::traslateModuleToLLVMIR(module, llvmContext);

    if(!llvmModule){
        llvm::errs() <<"Failed to emit LLVM IR\n";
        return -1;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    auto optPipeline = mlir::makeOptimizingTransformer(0,0,nullptr);
    if(auto err = optPipeline(llvmModule.get())){
        llvm::errs<<"Failed to optimize LLVM IR"<<err<<"\n";
        return -1;

    }
    llvm::outs() << *llvmModule<<"\n";
    return 0;
}

int loadMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module){
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if(std::error_code ec = fileOrErr.getError()){
        llvm::errs()<<"Coud not open input file:"<<ec.message()<<"\n";
        return -1;
    }

    llvm:SourceMgr sourceMgr:
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context);
    if(!module){
        llvm::err()<<"Error can't load file"<<inputFilename<<"\n";
        return 3;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module){
    if(int error= loadMLIR(context, module)){
        return error;
    }

    mlir::PassManager passManager(&context);
    mlir::applyPassManagerCLOptions(passManager);

    passManager.addPass(StarK::createLowertoAffinePass());
    passManager.addPass(StarK::createLowertoLLVMPass());

    if(mlir::failed(passManager.run(*module))){
        return 4;
    }
    return 0;
}


int runJit(mlir::ModuleOp module) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::registerLLVMDialectTranslation(*module->getContext());

    auto optPipeline = mlir::makeOptimizingTransformer(0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);


    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}


int main(int argc, char **argv){
   mlir::registerMLIRContextCLOptions();
   mlir::registerPassManagerCLOptions();

   cl::ParseCommandLineOptions(argc,argv, "StarK-Compiler\n");
   mlir::MLIRContext context;
   context.getOrLoadDialect<StarK::StarKDialect>();
   context.getOrLoadDialect<mlir::func::FuncDialect();

   mlir::OwningOpRef<mlir::ModuleOp> module;
   if(int error = loadAndProcessMLIR(context,module)){
    dumpLLVMIR(*module);
    return 0;
   }
}