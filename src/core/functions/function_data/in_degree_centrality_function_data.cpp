#include "duckpgq/core/functions/function_data/in_degree_centrality_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

InDegreeCentralityFunctionData::InDegreeCentralityFunctionData(ClientContext &ctx, int32_t csr)
    : context(ctx), csr_id(csr), state_initialized(false) {
}

InDegreeCentralityFunctionData::InDegreeCentralityFunctionData(ClientContext &ctx, int32_t csr,
                                                               const vector<int64_t> &indeg_p)
    : context(ctx), csr_id(csr), indeg(indeg_p), state_initialized(false) {
}

unique_ptr<FunctionData>
InDegreeCentralityFunctionData::InDegreeCentralityBind(ClientContext &context, ScalarFunction &bound_function,
                                                       vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}

	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);

	return make_uniq<InDegreeCentralityFunctionData>(context, csr_id);
}

unique_ptr<FunctionData> InDegreeCentralityFunctionData::Copy() const {
	auto result = make_uniq<InDegreeCentralityFunctionData>(context, csr_id);
	result->indeg = indeg; // Deep copy of in-degree vector
	result->state_initialized = state_initialized;
	// Note: state_lock is not copied as mutexes are not copyable
	return std::move(result);
}

bool InDegreeCentralityFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<InDegreeCentralityFunctionData>();
	if (csr_id != other.csr_id) {
		return false;
	}
	if (indeg != other.indeg) {
		return false;
	}
	if (state_initialized != other.state_initialized) {
		return false;
	}
	return true;
}

} // namespace duckdb
