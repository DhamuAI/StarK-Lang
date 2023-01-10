#ifndef STARK_STARKOPS_H
#define STARK_STARKOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ShapeInferenceOpInterface.hpp"
#include "mlir/IR/Value.h"




#define GET_OP_CLASSES
#include "StarK/StarKOps.h.inc"


#endif