#ifndef STARK_DIALECT
#define STARK_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// StarK dialect definition.
//===----------------------------------------------------------------------===//

def StarK_Dialect : Dialect{
    let name = "StarK";

    let summary = "StarK Dialect provides 'Hi_intro' to MLIR";

    let description = [{"StarK Dialect have printable tensor const op";
    }];

    let cppNamespace = "::StarK";
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base StarK operation definition.
//===----------------------------------------------------------------------===//

class StarK_Op<string mnemonic, list<Trait> traits =[]>:Op<StarK_Dialect, mnemonic, traits>;

#endif // STARK_DIALECT