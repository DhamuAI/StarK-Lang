#include <StarK/StarKOpsDialect.cpp.inc"

void StarKDialect::intialize(){
    addOperations<
    #define GET_OP_LIST
    #include "StarK/StarKOps.cpp.inc"
    >();
}

void hello::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value){
    auto dataType = RankedTensorType::get({}, builder.getF64Type());
    auto dataAttribute = DenseElementsAttr::get(dataType, value);
    StarK::ConstantOp::build(builder, state, dataType, dataAttribute);

}

mlir::Operation *StarKDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc){
    return builder.create<StarK::ConstantOp>(loc,type, value.cast<mlir::DenseElementsAttr>());
                                                   }