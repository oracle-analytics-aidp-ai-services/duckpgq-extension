#include "duckpgq/core/functions/function_data/closeness_centrality_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

// Constructor
ClosenessCentralityFunctionData::ClosenessCentralityFunctionData(ClientContext &ctx, int32_t csr)
    : context(ctx), csr_id(csr), state_initialized(false) {
}

unique_ptr<FunctionData>
ClosenessCentralityFunctionData::ClosenessCentralityBind(ClientContext &context, ScalarFunction &bound_function,
                                                         vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}

	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);

	return make_uniq<ClosenessCentralityFunctionData>(context, csr_id);
}

// Copy method
unique_ptr<FunctionData> ClosenessCentralityFunctionData::Copy() const {
	auto result = make_uniq<ClosenessCentralityFunctionData>(context, csr_id);
	result->closeness = closeness; // Deep copy of closeness vector
	result->state_initialized = state_initialized;
	// Note: state_lock is not copied as mutexes are not copyable
	return std::move(result);
}

// Equals method
bool ClosenessCentralityFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<ClosenessCentralityFunctionData>();
	if (csr_id != other.csr_id) {
		return false;
	}
	if (closeness != other.closeness) {
		return false;
	}
	if (state_initialized != other.state_initialized) {
		return false;
	}
	return true;
}

} // namespace duckdb
