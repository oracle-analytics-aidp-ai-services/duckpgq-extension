#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/triangle_count_function_data.hpp"
#include "duckpgq/core/utils/duckpgq_bitmap.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"

#include <duckpgq/core/functions/scalar.hpp>

namespace duckdb {

static void TriangleCountFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<TriangleCountFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing triangle count.");
	}

	int64_t *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t), "triangle_count");
	duckpgq_state->FinalizeCSRForRead(info.csr_id, csr, info.context);

	auto &src = args.data[1];
	UnifiedVectorFormat vdata_src;
	src.ToUnifiedFormat(args.size(), vdata_src);
	auto src_data = reinterpret_cast<const int64_t *>(vdata_src.data);

	ValidityMask &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);

	DuckPGQBitmap neighbors(v_size);

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
		int64_t number_of_edges = v[src_node + 1] - v[src_node];
		if (number_of_edges < 2) {
			result_data[n] = 0;
			continue;
		}

		neighbors.reset();
		for (int64_t offset = v[src_node]; offset < v[src_node + 1]; offset++) {
			neighbors.set(e[offset]);
		}

		// Count ordered connected neighbor pairs among src_node's neighbors.
		int64_t count = 0;
		for (int64_t offset = v[src_node]; offset < v[src_node + 1]; offset++) {
			int64_t neighbor = e[offset];
			for (int64_t offset2 = v[neighbor]; offset2 < v[neighbor + 1]; offset2++) {
				int is_connected = neighbors.test(e[offset2]);
				count += is_connected; // Add 1 if connected, 0 otherwise
			}
		}

		// Each triangle through src_node is counted twice (once per ordering of
		// the neighbor pair), so the number of triangles is count / 2.
		result_data[n] = count / 2;
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterTriangleCountScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("triangle_count", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::BIGINT, TriangleCountFunction,
	                                       TriangleCountFunctionData::TriangleCountBind));
}

} // namespace duckdb
