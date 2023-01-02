#ifndef MLIR_StarK_PASSES_H
#define MLIR_StarK_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace StarK {
  std::unique_ptr<mlir::Pass> createLowerToAffinePass();
  std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
}

#endif