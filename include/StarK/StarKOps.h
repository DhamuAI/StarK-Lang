#ifndef STARK_STARKOPS_H
#define STARK_STARKOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


#define GET_OPS_CLASSES
#include "StarK/StarKOps.h.inc"


#endif