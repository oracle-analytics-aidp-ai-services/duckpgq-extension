PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=duckpgq
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

.PHONY: test-csr-ownership test-csr-finalization test-csr-source-inventory
test-csr-ownership:
	./build/release/extension/duckpgq/duckpgq_csr_ownership_test

test-csr-finalization:
	./build/release/extension/duckpgq/duckpgq_csr_finalization_test

test-csr-source-inventory:
	python3 "$(PROJ_DIR)test/native/check_csr_compute_inventory.py"

test_release_internal: test-csr-ownership test-csr-finalization test-csr-source-inventory
