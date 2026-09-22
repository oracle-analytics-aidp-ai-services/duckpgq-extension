#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/jaccard_similarity_function_data.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"

#include <duckpgq/core/functions/scalar.hpp>
#include <unordered_set>

namespace duckdb {

static void JaccardSimilarityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<JaccardSimilarityFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");
	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing jaccard similarity.");
	}

	duckpgq_state->FinalizeCSRForRead(info.csr_id, csr, info.context);

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
	auto result_data = FlatVector::GetData<double_t>(result);

	auto neighbors = [&](int64_t node, std::unordered_set<int64_t> &out) {
		int64_t start_edge = v[node];
		int64_t end_edge = (static_cast<size_t>(node) + 1 < v_size) ? v[node + 1] : static_cast<int64_t>(e.size());
		for (int64_t j = start_edge; j < end_edge; j++) {
			out.insert(e[j]);
		}
	};

	std::unordered_set<int64_t> na, nb;
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
		na.clear();
		nb.clear();
		neighbors(a, na);
		neighbors(b, nb);
		int64_t intersection = 0;
		for (auto x : na) {
			if (nb.count(x)) {
				intersection++;
			}
		}
		int64_t union_size = static_cast<int64_t>(na.size() + nb.size()) - intersection;
		result_data[i] = (union_size > 0) ? static_cast<double>(intersection) / static_cast<double>(union_size) : 0.0;
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterJaccardSimilarityScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction(
	    "jaccard_similarity", {LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE,
	    JaccardSimilarityFunction, JaccardSimilarityFunctionData::JaccardSimilarityBind));
}

} // namespace duckdb
