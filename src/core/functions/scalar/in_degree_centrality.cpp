#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/in_degree_centrality_function_data.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"

#include <duckpgq/core/functions/scalar.hpp>

namespace duckdb {

static void InDegreeCentralityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<InDegreeCentralityFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing in-degree centrality.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t), "in_degree_centrality");

	// Compute the in-degrees once and cache the result. The CSR only stores
	// out-edges, so we accumulate incoming counts by scanning all out-edges.
	// Init + compute must happen INSIDE the lock since DuckDB shares bind data
	// across parallel scan threads.
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			vector<int64_t> indeg(v_size, 0);
			// Real vertex count is v_size - 2 (padding). Iterate real vertices u
			// and bump the in-degree of every out-neighbor w.
			int64_t real_count = static_cast<int64_t>(v_size) - 2;
			for (int64_t u = 0; u < real_count; u++) {
				int64_t start = v[u];
				int64_t end = (static_cast<size_t>(u) + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
				for (int64_t idx = start; idx < end; idx++) {
					int64_t w = e[idx];
					if (w >= 0 && static_cast<size_t>(w) < v_size) {
						indeg[w]++;
					}
				}
			}
			info.indeg = std::move(indeg);
			info.state_initialized = true;
		}
	}

	auto &src = args.data[1];
	UnifiedVectorFormat vdata_src;
	src.ToUnifiedFormat(args.size(), vdata_src);
	auto src_data = reinterpret_cast<const int64_t *>(vdata_src.data);

	ValidityMask &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);

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
		result_data[n] = info.indeg[src_node];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterInDegreeCentralityScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("in_degree_centrality", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::BIGINT, InDegreeCentralityFunction,
	                                       InDegreeCentralityFunctionData::InDegreeCentralityBind));
}

} // namespace duckdb
