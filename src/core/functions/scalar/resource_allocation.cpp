#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/resource_allocation_function_data.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"

#include <duckpgq/core/functions/scalar.hpp>
#include <unordered_set>

namespace duckdb {

static void ResourceAllocationFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<ResourceAllocationFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");
	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing resource allocation.");
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

	auto degree = [&](int64_t node) -> int64_t {
		std::unordered_set<int64_t> dn;
		neighbors(node, dn);
		return static_cast<int64_t>(dn.size());
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
		double resource_allocation = 0.0;
		for (auto w : nb) {
			if (na.count(w)) {
				int64_t deg_w = degree(w);
				if (deg_w == 0) {
					continue;
				}
				resource_allocation += 1.0 / static_cast<double>(deg_w);
			}
		}
		result_data[i] = resource_allocation;
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterResourceAllocationScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction(
	    "resource_allocation", {LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE,
	    ResourceAllocationFunction, ResourceAllocationFunctionData::ResourceAllocationBind));
}

} // namespace duckdb
