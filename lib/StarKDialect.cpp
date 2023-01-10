#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "StarK/StarKDialect.h"
#include "StarK/StarKOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Traits.h"

using namespace mlir;
using namespace StarK;

#include "StarK/StarKOpsDialect.cpp.inc"

void StarKDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "StarK/StarKOps.cpp.inc"
      >();
}

Type getBroadcastedRankedType(
    Type type1, Type type2, Type elementType = nullptr) {
  if (type1.isa<RankedTensorType>() && type2.isa<RankedTensorType>())
    return OpTrait::util::getBroadcastedType(type1, type2, elementType);
  if (type1.isa<MemRefType>() && type2.isa<MemRefType>()) {
    // Construct RankedTensorType(s).
    if (!elementType)
      elementType = type1.cast<MemRefType>().getElementType();
    RankedTensorType ty1 =
        RankedTensorType::get(type1.cast<MemRefType>().getShape(), elementType);
    RankedTensorType ty2 =
        RankedTensorType::get(type2.cast<MemRefType>().getShape(), elementType);
    // Compute a broadcasted type.
    Type outputType = OpTrait::util::getBroadcastedType(ty1, ty2);
    // Construct a MemRefType.
    return MemRefType::get(
        outputType.cast<RankedTensorType>().getShape(), elementType);
  } else
    return {};
}


void StarK::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  StarK::ConstantOp::build(builder, state, dataType, dataAttribute);
}


void StarK::AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::DenseElementsAttr A, mlir::DenseElementsAttr B ){
  auto lhsTy = A.getType();
  auto rhsTy = B.getType();
  auto resultType = getBroadcastedRankedType(lhsTy, rhsTy);
  // auto shapedType = resultType.dyn_cast_or_null<ShapedType()>;
  // if(!shapedType || !shapedType.hasStaticShape())
  //   resultType = UnrankedTensorType::get(lhsTy.cast<ShapedType>().getElementType());
  // StarK::AddOp::build(builder,state, resultType, lhsTy, rhsTy);
  
}

mlir::Operation *StarKDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
    return builder.create<StarK::ConstantOp>(loc, type,
                                      value.cast<mlir::DenseElementsAttr>());
}
