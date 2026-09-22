//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/functions/function_data/strongly_connected_component_function_data.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb/main/client_context.hpp"
#include "duckpgq/common.hpp"

namespace duckdb {

struct StronglyConnectedComponentFunctionData final : FunctionData {
	ClientContext &context;
	int32_t csr_id;
	vector<int64_t> component; // Component id per vertex (size v_size)
	std::mutex state_lock;     // Lock for state
	bool state_initialized;

	StronglyConnectedComponentFunctionData(ClientContext &context, int32_t csr_id);
	StronglyConnectedComponentFunctionData(ClientContext &context, int32_t csr_id, const vector<int64_t> &component);
	static unique_ptr<FunctionData> StronglyConnectedComponentBind(ClientContext &context,
	                                                               ScalarFunction &bound_function,
	                                                               vector<unique_ptr<Expression>> &arguments);

	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other_p) const override;
};

} // namespace duckdb
