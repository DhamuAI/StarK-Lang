
#ifdef SHAPE_INFERENCE_OP_INTERFACE
#else
#define SHAPE_INFERENCE_OP_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can ShapeInferenceOpInterface.hpp.incb
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "mlir::LogicalResult", "inferShapes",
                    (ins "std::function<void(mlir::Region&)>":$shapeInferenceFunc)>
  ];
  
  let cppNamespace = "::mlir";
}

#endif // SHAPE_INFERENCE_OP_INTERFACE
