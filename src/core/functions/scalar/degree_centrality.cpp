#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/degree_centrality_function_data.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"

#include <duckpgq/core/functions/scalar.hpp>

namespace duckdb {

static void DegreeCentralityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<DegreeCentralityFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing degree centrality.");
	}

	int64_t *v = reinterpret_cast<int64_t *>(csr->v);
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t), "degree_centrality");

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
		result_data[n] = v[src_node + 1] - v[src_node];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterDegreeCentralityScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("degree_centrality", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::BIGINT, DegreeCentralityFunction,
	                                       DegreeCentralityFunctionData::DegreeCentralityBind));
}

} // namespace duckdb
