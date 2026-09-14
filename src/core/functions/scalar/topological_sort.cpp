#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/topological_sort_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>
#include <algorithm>
#include <functional>
#include <queue>
#include <vector>

namespace duckdb {

static void TopologicalSortFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<TopologicalSortFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing topological sort.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t) * 2, "topological_sort");

	// Compute the topological order once and cache the result.
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			// Kahn's algorithm (iterative, BFS-style) over the directed CSR.
			// vsize includes 2 padding slots, so the real vertex count is
			// v_size - 2 and real vertices satisfy `u + 2 < v_size`.
			info.order.assign(v_size, -1);

			vector<int64_t> indeg(v_size, 0);
			// Compute in-degree of every real vertex.
			for (size_t u = 0; u + 2 < v_size; u++) {
				int64_t end = (u + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
				for (int64_t edge_idx = v[u]; edge_idx < end; edge_idx++) {
					int64_t w = e[edge_idx];
					indeg[w]++;
				}
			}

			// Min-heap so that, among vertices with in-degree 0, the smallest
			// vertex id is processed first (deterministic output).
			std::priority_queue<int64_t, vector<int64_t>, std::greater<int64_t>> q;
			for (size_t u = 0; u + 2 < v_size; u++) {
				if (indeg[u] == 0) {
					q.push(static_cast<int64_t>(u));
				}
			}

			int64_t counter = 0;
			while (!q.empty()) {
				int64_t u = q.top();
				q.pop();
				info.order[u] = counter++;
				int64_t end = (static_cast<size_t>(u) + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
				for (int64_t edge_idx = v[u]; edge_idx < end; edge_idx++) {
					int64_t w = e[edge_idx];
					if (--indeg[w] == 0) {
						q.push(w);
					}
				}
			}

			// Any real vertex never popped is part of a cycle: leave order = -1.
			info.state_initialized = true;
		}
	}

	// Get the source vector for the current DataChunk
	auto &src = args.data[1];
	UnifiedVectorFormat vdata_src;
	src.ToUnifiedFormat(args.size(), vdata_src);
	auto src_data = reinterpret_cast<const int64_t *>(vdata_src.data);

	// Create result vector
	ValidityMask &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);

	// Output the topological order corresponding to each source rowid in the DataChunk
	for (idx_t i = 0; i < args.size(); i++) {
		auto id_pos = vdata_src.sel->get_index(i);
		if (!vdata_src.validity.RowIsValid(id_pos)) {
			result_validity.SetInvalid(i);
			continue; // Skip invalid rows
		}
		auto node_id = src_data[id_pos];
		if (node_id < 0 || static_cast<size_t>(node_id) >= v_size) {
			result_validity.SetInvalid(i);
			continue;
		}
		result_data[i] = info.order[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterTopologicalSortScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("topological_sort", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::BIGINT, TopologicalSortFunction,
	                                       TopologicalSortFunctionData::TopologicalSortBind));
}

} // namespace duckdb
