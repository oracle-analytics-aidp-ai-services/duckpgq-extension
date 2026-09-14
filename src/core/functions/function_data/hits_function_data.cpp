#include "duckpgq/core/functions/function_data/hits_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

// Constructor
HitsFunctionData::HitsFunctionData(ClientContext &ctx, int32_t csr)
    : context(ctx), csr_id(csr), state_initialized(false) {
}

unique_ptr<FunctionData> HitsFunctionData::HitsBind(ClientContext &context, ScalarFunction &bound_function,
                                                    vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}

	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);

	return make_uniq<HitsFunctionData>(context, csr_id);
}

// Copy method
unique_ptr<FunctionData> HitsFunctionData::Copy() const {
	auto result = make_uniq<HitsFunctionData>(context, csr_id);
	result->authority = authority; // Deep copy of authority vector
	result->hub = hub;             // Deep copy of hub vector
	result->state_initialized = state_initialized;
	// Note: state_lock is not copied as mutexes are not copyable
	return std::move(result);
}

// Equals method
bool HitsFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<HitsFunctionData>();
	if (csr_id != other.csr_id) {
		return false;
	}
	if (authority != other.authority) {
		return false;
	}
	if (hub != other.hub) {
		return false;
	}
	if (state_initialized != other.state_initialized) {
		return false;
	}
	return true;
}

} // namespace duckdb
