configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(StarK_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        StarK-opt
        )

add_lit_testsuite(check-StarK "Running the StarK regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${StarK_TEST_DEPENDS}
        )
set_target_properties(check-StarK PROPERTIES FOLDER "Tests")

add_lit_testsuites(StarK ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${StarK_TEST_DEPENDS})