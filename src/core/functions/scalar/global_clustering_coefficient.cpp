#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/global_clustering_coefficient_function_data.hpp"
#include "duckpgq/core/utils/duckpgq_bitmap.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"

#include <duckpgq/core/functions/scalar.hpp>

namespace duckdb {

static void GlobalClusteringCoefficientFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<GlobalClusteringCoefficientFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing global clustering coefficient.");
	}

	int64_t *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t), "global_clustering_coefficient");

	// Compute the single graph-level transitivity once and cache it. The same
	// value is then returned for every source rowid in this and later chunks.
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety.
		if (!info.state_initialized) {
			DuckPGQBitmap neighbors(v_size);

			// Bookkeeping over the UNDIRECTED CSR:
			//   ordered_connected_pairs = sum over every real vertex u of the number
			//     of ORDERED connected neighbor pairs of u (the exact quantity that
			//     triangle_count computes per vertex as `count`). Summed over all
			//     vertices this counts each triangle 6 times (3 apex choices x 2
			//     orderings), so num_triangles = ordered_connected_pairs / 6.
			//   num_paths_of_length_2 = sum over u of C(deg_u, 2) = deg_u*(deg_u-1)/2.
			// transitivity = (3 * num_triangles) / num_paths_of_length_2.
			int64_t ordered_connected_pairs = 0;
			int64_t num_paths_of_length_2 = 0;

			// Real vertices only (v_size = num_vertices + 2 padding).
			for (size_t u = 0; u + 2 < v_size; u++) {
				int64_t start = v[u];
				int64_t end = (u + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
				int64_t degree = end - start;

				// Paths of length 2 centered at u: choose 2 of u's neighbors.
				num_paths_of_length_2 += degree * (degree - 1) / 2;

				if (degree < 2) {
					continue;
				}

				neighbors.reset();
				for (int64_t offset = start; offset < end; offset++) {
					neighbors.set(e[offset]);
				}

				// Count ordered connected neighbor pairs among u's neighbors.
				for (int64_t offset = start; offset < end; offset++) {
					int64_t neighbor = e[offset];
					int64_t nstart = v[neighbor];
					int64_t nend = (static_cast<size_t>(neighbor) + 1 < v_size)
					                   ? v[neighbor + 1]
					                   : static_cast<int64_t>(e.size());
					for (int64_t offset2 = nstart; offset2 < nend; offset2++) {
						int is_connected = neighbors.test(e[offset2]);
						ordered_connected_pairs += is_connected; // 1 if connected, else 0.
					}
				}
			}

			// Each triangle is counted 6 times in ordered_connected_pairs.
			int64_t num_triangles = ordered_connected_pairs / 6;
			info.transitivity = (num_paths_of_length_2 > 0)
			                        ? (3.0 * static_cast<double>(num_triangles)) /
			                              static_cast<double>(num_paths_of_length_2)
			                        : 0.0;
			info.state_initialized = true;
		}
	}

	auto &src = args.data[1];
	UnifiedVectorFormat vdata_src;
	src.ToUnifiedFormat(args.size(), vdata_src);
	auto src_data = reinterpret_cast<const int64_t *>(vdata_src.data);

	ValidityMask &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<double>(result);

	// Return the same graph-level transitivity for every source rowid.
	for (idx_t n = 0; n < args.size(); n++) {
		auto src_sel = vdata_src.sel->get_index(n);
		if (!vdata_src.validity.RowIsValid(src_sel)) {
			result_validity.SetInvalid(n);
			continue;
		}
		int64_t src_node = src_data[src_sel];
		if (src_node < 0 || static_cast<size_t>(src_node) >= v_size) {
			result_validity.SetInvalid(n);
			continue;
		}
		result_data[n] = info.transitivity;
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterGlobalClusteringCoefficientScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(
	    ScalarFunction("global_clustering_coefficient", {LogicalType::INTEGER, LogicalType::BIGINT},
	                   LogicalType::DOUBLE, GlobalClusteringCoefficientFunction,
	                   GlobalClusteringCoefficientFunctionData::GlobalClusteringCoefficientBind));
}

} // namespace duckdb
