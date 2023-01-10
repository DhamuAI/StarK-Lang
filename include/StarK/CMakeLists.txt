
add_mlir_interface(ShapeInferenceOpInterface)

add_mlir_dialect(StarKOps StarK)

add_mlir_doc(StarKDialect StarKDialect StarK/ -gen-dialect-doc)
add_mlir_doc(StarKOps StarKOps StarK/ -gen-op-doc)

