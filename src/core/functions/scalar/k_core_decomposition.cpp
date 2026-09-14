#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/k_core_decomposition_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/functions/table/k_core_decomposition.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

static void KCoreDecompositionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<KCoreDecompositionFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before running k-core decomposition.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t) * 3, "k_core_decomposition");

	// State initialization and computation (only once)
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			// vsize = num_vertices + 2 (padding); real vertices are those with u + 2 < v_size.
			// Batagelj-Zaversnik O(V+E) core decomposition over the real vertices
			// (vsize = num_vertices + 2 padding slots).
			const size_t n = (v_size >= 2) ? v_size - 2 : 0;
			info.core.assign(v_size, 0);
			if (n > 0) {
				vector<int64_t> deg(n, 0);
				int64_t max_deg = 0;
				for (size_t u = 0; u < n; u++) {
					int64_t start_edge = v[u];
					int64_t end_edge = (u + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
					deg[u] = end_edge - start_edge;
					if (deg[u] > max_deg) {
						max_deg = deg[u];
					}
				}

				// Bucket-sort vertices by current degree into `vert`, tracking each
				// vertex's position (`pos`) and the start index of each degree bin (`bin`).
				vector<int64_t> bin(static_cast<size_t>(max_deg) + 2, 0);
				for (size_t u = 0; u < n; u++) {
					bin[deg[u]]++;
				}
				int64_t start = 0;
				for (int64_t d = 0; d <= max_deg; d++) {
					int64_t cnt = bin[d];
					bin[d] = start;
					start += cnt;
				}
				vector<int64_t> vert(n, 0);
				vector<int64_t> pos(n, 0);
				for (size_t u = 0; u < n; u++) {
					pos[u] = bin[deg[u]];
					vert[pos[u]] = static_cast<int64_t>(u);
					bin[deg[u]]++;
				}
				for (int64_t d = max_deg; d >= 1; d--) {
					bin[d] = bin[d - 1];
				}
				bin[0] = 0;

				// Process vertices in non-decreasing degree order; a vertex's core
				// number is its residual degree when it is processed.
				for (size_t i = 0; i < n; i++) {
					int64_t node = vert[i];
					info.core[node] = deg[node];
					int64_t start_edge = v[node];
					int64_t end_edge =
					    (static_cast<size_t>(node) + 1 < v_size) ? v[node + 1] : static_cast<int64_t>(e.size());
					for (int64_t j = start_edge; j < end_edge; j++) {
						int64_t u = e[j];
						if (u < 0 || static_cast<size_t>(u) >= n) {
							continue;
						}
						if (deg[u] > deg[node]) {
							int64_t du = deg[u];
							int64_t pu = pos[u];
							int64_t pw = bin[du];
							int64_t w = vert[pw];
							if (u != w) {
								pos[u] = pw;
								vert[pu] = w;
								pos[w] = pu;
								vert[pw] = u;
							}
							bin[du]++;
							deg[u]--;
						}
					}
				}
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

	// Output the core number corresponding to each source ID in the DataChunk
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
		result_data[i] = info.core[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterKCoreDecompositionScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("k_core_decomposition", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::BIGINT, KCoreDecompositionFunction,
	                                       KCoreDecompositionFunctionData::KCoreDecompositionBind));
}

} // namespace duckdb
