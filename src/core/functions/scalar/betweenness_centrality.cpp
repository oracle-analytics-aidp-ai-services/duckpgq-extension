#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/betweenness_centrality_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/functions/table/betweenness_centrality.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

static void BetweennessCentralityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<BetweennessCentralityFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing betweenness centrality.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(double) * 5 + (idx_t)e.size() * sizeof(int64_t),
	                           "betweenness_centrality");

	// Compute betweenness centrality once (Brandes' algorithm) and cache it.
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			info.betweenness.assign(v_size, 0.0);

			// The CSR is allocated with two extra padding slots (vsize = #vertices + 2);
			// only the first `num_vertices` entries are real graph vertices. Iterating the
			// padding slots as BFS sources would inject spurious dependencies, so restrict
			// the source loop to real vertices.
			const size_t num_vertices = (v_size >= 2) ? v_size - 2 : v_size;

			vector<double_t> sigma(v_size, 0.0);
			vector<int64_t> dist(v_size, -1);
			vector<double_t> delta(v_size, 0.0);
			vector<vector<int64_t>> pred(v_size);
			vector<int64_t> stack_order;
			stack_order.reserve(v_size);
			vector<int64_t> queue;
			queue.reserve(v_size);

			for (size_t s = 0; s < num_vertices; s++) {
				// Reset single-source state
				for (size_t i = 0; i < v_size; i++) {
					pred[i].clear();
					sigma[i] = 0.0;
					dist[i] = -1;
					delta[i] = 0.0;
				}
				stack_order.clear();
				queue.clear();

				// BFS from source s
				sigma[s] = 1.0;
				dist[s] = 0;
				queue.push_back(static_cast<int64_t>(s));
				size_t head = 0;
				while (head < queue.size()) {
					int64_t cur = queue[head++];
					stack_order.push_back(cur);
					auto start_edge = v[cur];
					auto end_edge = (static_cast<size_t>(cur) + 1 < v_size)
					                    ? v[cur + 1]
					                    : static_cast<int64_t>(e.size());
					for (int64_t j = start_edge; j < end_edge; j++) {
						int64_t w = e[j];
						// First time we reach w
						if (dist[w] < 0) {
							dist[w] = dist[cur] + 1;
							queue.push_back(w);
						}
						// Shortest path to w via cur?
						if (dist[w] == dist[cur] + 1) {
							sigma[w] += sigma[cur];
							pred[w].push_back(cur);
						}
					}
				}

				// Accumulation in reverse BFS order
				for (size_t idx = stack_order.size(); idx-- > 0;) {
					int64_t w = stack_order[idx];
					for (int64_t pv : pred[w]) {
						delta[pv] += (sigma[pv] / sigma[w]) * (1.0 + delta[w]);
					}
					if (w != static_cast<int64_t>(s)) {
						info.betweenness[w] += delta[w];
					}
				}
			}

			// Undirected graph: every shortest path counted twice
			for (size_t i = 0; i < v_size; i++) {
				info.betweenness[i] /= 2.0;
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

	// Output the betweenness value corresponding to each source ID in the DataChunk
	for (idx_t i = 0; i < args.size(); i++) {
		auto id_pos = vdata_src.sel->get_index(i);
		if (!vdata_src.validity.RowIsValid(id_pos)) {
			result_validity.SetInvalid(i);
			continue; // Skip invalid rows
		}
		auto node_id = src_data[id_pos];
		if (node_id < 0 || node_id >= v_size) {
			result_validity.SetInvalid(i);
			continue;
		}
		result_data[i] = info.betweenness[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterBetweennessCentralityScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("betweenness_centrality", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::DOUBLE, BetweennessCentralityFunction,
	                                       BetweennessCentralityFunctionData::BetweennessCentralityBind));
}

} // namespace duckdb
