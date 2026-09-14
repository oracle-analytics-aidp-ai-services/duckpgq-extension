#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/louvain_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>
#include <unordered_map>

namespace duckdb {

static void LouvainFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<LouvainFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before running louvain.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t) * 4, "louvain");

	// State initialization and computation (only once). Both the resize/init
	// and the computation run inside the lock because DuckDB shares bind data
	// across parallel scan threads.
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			// Per-vertex degree (each undirected edge counts once per endpoint).
			vector<int64_t> degree(v_size, 0);
			double two_m = 0.0; // 2 * m, i.e. the sum of all degrees
			for (size_t u = 0; u + 2 < v_size; u++) {
				auto start_edge = v[u];
				auto end_edge = (u + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
				degree[u] = end_edge - start_edge;
				two_m += static_cast<double>(degree[u]);
			}

			// Initialize: each vertex in its own community; community total
			// degree equals the vertex degree.
			info.community.assign(v_size, 0);
			vector<int64_t> comm_total_degree(v_size, 0);
			for (size_t u = 0; u + 2 < v_size; u++) {
				info.community[u] = static_cast<int64_t>(u);
				comm_total_degree[u] = degree[u];
			}

			if (two_m > 0.0) {
				const int64_t max_passes = 20;
				// Map from neighbor community id -> sum of edge weights from u
				// into that community (weight 1 per edge, unweighted graph).
				std::unordered_map<int64_t, int64_t> neighbor_weight;
				for (int64_t pass = 0; pass < max_passes; pass++) {
					bool moved = false;
					for (size_t u = 0; u + 2 < v_size; u++) {
						int64_t k_u = degree[u];
						if (k_u == 0) {
							continue; // isolated vertex, nothing to move
						}
						auto start_edge = v[u];
						auto end_edge = (u + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());

						// Remove u from its current community.
						int64_t old_comm = info.community[u];
						comm_total_degree[old_comm] -= k_u;

						// Accumulate edge weights from u to each neighbor
						// community.
						neighbor_weight.clear();
						for (int64_t j = start_edge; j < end_edge; j++) {
							int64_t neighbor = e[j];
							if (neighbor == static_cast<int64_t>(u)) {
								continue; // ignore self-loops
							}
							neighbor_weight[info.community[neighbor]]++;
						}

						// Evaluate modularity gain of moving u into each
						// candidate community. Gain (up to a positive scaling
						// constant) is:
						//   k_in(C) - k_u * sum_tot(C) / (2*m)
						// where k_in(C) is the weight of edges from u to C and
						// sum_tot(C) is the total degree of C (with u removed).
						int64_t best_comm = old_comm;
						double best_gain = 0.0; // staying put (gain 0) is the baseline
						for (auto &entry : neighbor_weight) {
							int64_t cand_comm = entry.first;
							int64_t k_in = entry.second;
							double gain = static_cast<double>(k_in) -
							              static_cast<double>(k_u) *
							                  static_cast<double>(comm_total_degree[cand_comm]) / two_m;
							if (gain > best_gain || (gain == best_gain && cand_comm < best_comm)) {
								best_gain = gain;
								best_comm = cand_comm;
							}
						}

						// Insert u into the chosen community.
						info.community[u] = best_comm;
						comm_total_degree[best_comm] += k_u;
						if (best_comm != old_comm) {
							moved = true;
						}
					}
					if (!moved) {
						break;
					}
				}
			}

			// Deterministic relabeling: each community id becomes the smallest
			// vertex id it contains.
			vector<int64_t> comm_min(v_size, -1);
			for (size_t u = 0; u + 2 < v_size; u++) {
				int64_t c = info.community[u];
				if (comm_min[c] == -1 || static_cast<int64_t>(u) < comm_min[c]) {
					comm_min[c] = static_cast<int64_t>(u);
				}
			}
			for (size_t u = 0; u + 2 < v_size; u++) {
				info.community[u] = comm_min[info.community[u]];
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
	auto result_data = FlatVector::GetData<int64_t>(result);

	// Output the community corresponding to each source ID in the DataChunk
	for (idx_t i = 0; i < args.size(); i++) {
		auto id_pos = vdata_src.sel->get_index(i);
		if (!vdata_src.validity.RowIsValid(id_pos)) {
			result_validity.SetInvalid(i);
			continue;
		}
		auto node_id = src_data[id_pos];
		if (node_id < 0 || static_cast<size_t>(node_id) >= v_size) {
			result_validity.SetInvalid(i);
			continue;
		}
		result_data[i] = info.community[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterLouvainScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("louvain", {LogicalType::INTEGER, LogicalType::BIGINT}, LogicalType::BIGINT,
	                                       LouvainFunction, LouvainFunctionData::LouvainBind));
}

} // namespace duckdb
