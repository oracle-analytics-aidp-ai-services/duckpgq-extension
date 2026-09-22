//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/functions/function_data/single_source_shortest_path_function_data.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb/main/client_context.hpp"
#include "duckpgq/common.hpp"

namespace duckdb {

struct SingleSourceShortestPathFunctionData final : FunctionData {
	ClientContext &context;
	int32_t csr_id;
	vector<int64_t> dist;  // BFS hop distance per vertex (size v_size); -1 = unreached
	std::mutex state_lock; // Lock for state
	bool state_initialized;

	SingleSourceShortestPathFunctionData(ClientContext &context, int32_t csr_id);
	SingleSourceShortestPathFunctionData(ClientContext &context, int32_t csr_id, const vector<int64_t> &dist);
	static unique_ptr<FunctionData> SingleSourceShortestPathBind(ClientContext &context, ScalarFunction &bound_function,
	                                                             vector<unique_ptr<Expression>> &arguments);

	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other_p) const override;
};

} // namespace duckdb
