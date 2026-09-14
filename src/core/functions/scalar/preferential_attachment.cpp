#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/preferential_attachment_function_data.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"

#include <duckpgq/core/functions/scalar.hpp>

namespace duckdb {

static void PreferentialAttachmentFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<PreferentialAttachmentFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");
	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing preferential attachment.");
	}

	int64_t *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;

	UnifiedVectorFormat vdata_a, vdata_b;
	args.data[1].ToUnifiedFormat(args.size(), vdata_a);
	args.data[2].ToUnifiedFormat(args.size(), vdata_b);
	auto a_data = reinterpret_cast<const int64_t *>(vdata_a.data);
	auto b_data = reinterpret_cast<const int64_t *>(vdata_b.data);

	ValidityMask &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);

	auto degree = [&](int64_t node) -> int64_t {
		int64_t start_edge = v[node];
		int64_t end_edge = (static_cast<size_t>(node) + 1 < v_size) ? v[node + 1] : static_cast<int64_t>(e.size());
		return end_edge - start_edge;
	};

	for (idx_t i = 0; i < args.size(); i++) {
		auto a_pos = vdata_a.sel->get_index(i);
		auto b_pos = vdata_b.sel->get_index(i);
		if (!vdata_a.validity.RowIsValid(a_pos) || !vdata_b.validity.RowIsValid(b_pos)) {
			result_validity.SetInvalid(i);
			continue;
		}
		int64_t a = a_data[a_pos];
		int64_t b = b_data[b_pos];
		if (a < 0 || b < 0 || static_cast<size_t>(a) >= v_size || static_cast<size_t>(b) >= v_size) {
			result_validity.SetInvalid(i);
			continue;
		}
		result_data[i] = degree(a) * degree(b);
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterPreferentialAttachmentScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("preferential_attachment",
	                                       {LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::BIGINT},
	                                       LogicalType::BIGINT, PreferentialAttachmentFunction,
	                                       PreferentialAttachmentFunctionData::PreferentialAttachmentBind));
}

} // namespace duckdb
