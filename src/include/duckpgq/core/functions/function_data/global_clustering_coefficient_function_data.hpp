//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/functions/function_data/global_clustering_coefficient_function_data.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb/main/client_context.hpp"
#include "duckpgq/common.hpp"

namespace duckdb {

struct GlobalClusteringCoefficientFunctionData final : FunctionData {
	ClientContext &context;
	int32_t csr_id;
	double transitivity;   // Cached graph-level transitivity (same value for every vertex).
	std::mutex state_lock; // Lock guarding the compute-once-and-cache.
	bool state_initialized;

	GlobalClusteringCoefficientFunctionData(ClientContext &context, int32_t csr_id);
	static unique_ptr<FunctionData> GlobalClusteringCoefficientBind(ClientContext &context,
	                                                                ScalarFunction &bound_function,
	                                                                vector<unique_ptr<Expression>> &arguments);

	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other_p) const override;
};

} // namespace duckdb
