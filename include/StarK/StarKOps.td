#ifndef STARK_OPS
#define STARK_OPS

include "StarKDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ShapeInferenceOpInterface.td"



def ConstantOp : StarK_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}

def PrintOp : StarK_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }]; 
      
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def AddOp : StarK_Op<"add",[Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]>{

  let hasCanonicalizer =1;
  let summary = "StarK add operation";
  let description = [{
    "It performs element-wise addition"
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);

  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$A, "mlir::DenseElementsAttr":$B),[{
      // auto lhsTy = A.getType();
      // auto rhsTy = B.getType();
      // auto resultType = getBroadcastedRankedType(lhsTy, rhsTy);
      // auto shapedType = resultType.dyn_cast_or_null<ShapedType()>;
      // if(!shapedType || !shapedType.hasStaticShape())
      //   resultType = UnrankedTensorType::get(lhsTy.cast<ShapedType>().getElementType());
      build($_builder, $_state, A, B);
    }]>,
  ];

  let extraClassDeclaration = [{
    static int getNumberOfOperands(){
      return 2;
    }
    static int getNumberOfResults(){
      return 1;
    }
  }];
  let hasVerifier =1;
}
#endif //STARK_OPS

