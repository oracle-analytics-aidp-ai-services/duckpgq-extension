#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/single_source_shortest_path_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>
#include <queue>
#include <vector>

namespace duckdb {

static void SingleSourceShortestPathFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<SingleSourceShortestPathFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing single source shortest path.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t) * 2, "single_source_shortest_path");

	// Compute the BFS hop distances from the source once and cache the result.
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			// The source is a CONSTANT argument (same for every row): read it once.
			int64_t source = args.data[2].GetValue(0).GetValue<int64_t>();

			vector<int64_t> dist(v_size, -1); // -1 = unreached
			if (source >= 0 && static_cast<size_t>(source) < v_size) {
				std::queue<int64_t> frontier;
				dist[source] = 0;
				frontier.push(source);
				while (!frontier.empty()) {
					int64_t u = frontier.front();
					frontier.pop();
					// Out-neighbors of u: e[v[u] .. v[u+1])
					int64_t end_edge =
					    (static_cast<size_t>(u) + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
					for (int64_t idx = v[u]; idx < end_edge; idx++) {
						int64_t w = e[idx];
						if (dist[w] == -1) {
							dist[w] = dist[u] + 1;
							frontier.push(w);
						}
					}
				}
			}

			info.dist = std::move(dist);
			info.state_initialized = true;
		}
	}

	// Get the source (rowid) vector for the current DataChunk
	auto &src = args.data[1];
	UnifiedVectorFormat vdata_src;
	src.ToUnifiedFormat(args.size(), vdata_src);
	auto src_data = reinterpret_cast<const int64_t *>(vdata_src.data);

	// Create result vector
	ValidityMask &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);

	// Output the distance corresponding to each vertex rowid in the DataChunk
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
		result_data[i] = info.dist[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterSingleSourceShortestPathScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction(
	    "single_source_shortest_path", {LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::BIGINT},
	    LogicalType::BIGINT, SingleSourceShortestPathFunction,
	    SingleSourceShortestPathFunctionData::SingleSourceShortestPathBind));
}

} // namespace duckdb
