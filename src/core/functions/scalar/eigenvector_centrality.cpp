#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/eigenvector_centrality_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/functions/table/eigenvector_centrality.hpp>
#include <duckpgq/core/utils/duckpgq_bitmap.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>

#include <cmath>

namespace duckdb {

static void EigenvectorCentralityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<EigenvectorCentralityFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing eigenvector centrality.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(double) * 3, "eigenvector_centrality");
	duckpgq_state->FinalizeCSRForRead(info.csr_id, csr, info.context);

	// Compute once and cache. Initialization AND computation must run under the
	// lock: DuckDB parallelizes the scan and shares this bind data across threads,
	// so an unlocked resize() of the state vector races and corrupts the heap
	// (manifests as a segfault only once the graph is large enough to parallelize).
	if (!info.converged) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety

		if (!info.state_initialized) {
			info.centrality.resize(v_size, 1.0 / static_cast<double>(v_size)); // Initial value for each node
			info.max_iterations = 100;
			info.convergence_threshold = 1e-8;
			info.state_initialized = true;
			info.converged = false;
			info.iteration_count = 0;
		}

		if (!info.converged) {
			vector<double_t> next(v_size, 0.0);
			while (info.iteration_count < info.max_iterations && !info.converged) {
				fill(next.begin(), next.end(), 0.0);

				// y[u] = sum over neighbors n of u of x[n].
				// Restrict to real vertices: vsize = num_vertices + 2 padding slots whose
				// edge offsets are unspecified in the undirected CSR, so reading their
				// neighbor ranges would index e[] out of bounds.
				for (size_t i = 0; i + 2 < v_size; i++) {
					auto start_edge = v[i];
					auto end_edge = (i + 1 < v_size) ? v[i + 1] : static_cast<int64_t>(e.size());
					double_t sum = 0.0;
					for (int64_t j = start_edge; j < end_edge; j++) {
						sum += info.centrality[e[j]];
					}
					next[i] = sum;
				}

				// L2-normalize y (guard against zero norm)
				double_t norm = 0.0;
				for (size_t i = 0; i < v_size; i++) {
					norm += next[i] * next[i];
				}
				norm = std::sqrt(norm);
				if (norm > 0.0) {
					for (size_t i = 0; i < v_size; i++) {
						next[i] /= norm;
					}
				}

				// Check convergence
				double_t max_delta = 0.0;
				for (size_t i = 0; i < v_size; i++) {
					max_delta = std::max(max_delta, std::abs(next[i] - info.centrality[i]));
				}

				info.centrality.swap(next);
				info.iteration_count++;
				if (max_delta < info.convergence_threshold) {
					info.converged = true;
				}
			}
			info.converged = true;
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

	// Output the centrality value corresponding to each source ID in the DataChunk
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
		result_data[i] = info.centrality[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterEigenvectorCentralityScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("eigenvector_centrality", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::DOUBLE, EigenvectorCentralityFunction,
	                                       EigenvectorCentralityFunctionData::EigenvectorCentralityBind));
}

} // namespace duckdb
