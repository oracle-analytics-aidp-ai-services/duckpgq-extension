#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/closeness_centrality_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>

#include <queue>

namespace duckdb {

static void ClosenessCentralityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<ClosenessCentralityFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing closeness centrality.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(double) * 2, "closeness_centrality");

	// State initialization and whole-graph computation (only once)
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			info.closeness.resize(v_size, 0.0);

			for (size_t source = 0; source < v_size; source++) {
				// BFS over the undirected CSR from `source`
				vector<int64_t> dist(v_size, -1);
				std::queue<int64_t> frontier;
				dist[source] = 0;
				frontier.push(static_cast<int64_t>(source));

				int64_t reachable = 0; // vertices reachable excluding source
				int64_t sum_distances = 0;

				while (!frontier.empty()) {
					int64_t u = frontier.front();
					frontier.pop();

					auto start_edge = v[u];
					auto end_edge =
					    (static_cast<size_t>(u) + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
					for (int64_t j = start_edge; j < end_edge; j++) {
						int64_t neighbor = e[j];
						if (dist[neighbor] == -1) {
							dist[neighbor] = dist[u] + 1;
							reachable++;
							sum_distances += dist[neighbor];
							frontier.push(neighbor);
						}
					}
				}

				info.closeness[source] =
				    (sum_distances > 0) ? static_cast<double>(reachable) / static_cast<double>(sum_distances) : 0.0;
			}

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
	auto result_data = FlatVector::GetData<double_t>(result);

	// Output the closeness value corresponding to each source ID in the DataChunk
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
		result_data[i] = info.closeness[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterClosenessCentralityScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("closeness_centrality", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::DOUBLE, ClosenessCentralityFunction,
	                                       ClosenessCentralityFunctionData::ClosenessCentralityBind));
}

} // namespace duckdb
