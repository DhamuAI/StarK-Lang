#ifndef STARK_OPS
#define STARK_OPS

#include "StarKDialect.td"
#include "mlir/Interfaces/SideEffectInterfaces.td"


def ConstantOp : StarK_Op<"constant", [Pure]>{
    let summary = "constant";

    let description =[{
        Constant tensors storage
    }];

    let builders = {
        OpBuilder<(ins "mlir::DenseElementAttr":$value),[{
            build($_builder, $_state, value.getType(),value);
        }]>,
        OpBuilder<ins "double":$value>
    };

    let arguments = (ins F64ElementsAttr:$value);
    let results =(outs F64Tensor);
}

def PrintOp : StarK_op<"print",[Pure]>{
    let summary = "print operation"

    let description = [{
        The "print" builtin operation
    }];
    let arguments = (ins AnyTypeOf<[F64Tensor,F64Memref]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";

}

#endif //STARK_OPS

