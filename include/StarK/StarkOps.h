#ifndef STARK_OPS_H
#define STARK_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInferfaces.h"


#define GET_OPS_CLASSES
#include "StarK/StarkOps.h.inc"

#endif