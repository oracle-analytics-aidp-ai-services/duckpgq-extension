#include "duckpgq/core/functions/function_data/label_propagation_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

// Constructor
LabelPropagationFunctionData::LabelPropagationFunctionData(ClientContext &ctx, int32_t csr)
    : context(ctx), csr_id(csr), state_initialized(false) {
}

LabelPropagationFunctionData::LabelPropagationFunctionData(ClientContext &ctx, int32_t csr,
                                                           const vector<int64_t> &label_p)
    : context(ctx), csr_id(csr), label(label_p), state_initialized(false) {
}

unique_ptr<FunctionData> LabelPropagationFunctionData::LabelPropagationBind(ClientContext &context,
                                                                            ScalarFunction &bound_function,
                                                                            vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}

	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);

	return make_uniq<LabelPropagationFunctionData>(context, csr_id);
}

// Copy method
unique_ptr<FunctionData> LabelPropagationFunctionData::Copy() const {
	auto result = make_uniq<LabelPropagationFunctionData>(context, csr_id);
	result->label = label; // Deep copy of label vector
	result->state_initialized = state_initialized;
	// Note: state_lock is not copied as mutexes are not copyable
	return std::move(result);
}

// Equals method
bool LabelPropagationFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<LabelPropagationFunctionData>();
	if (csr_id != other.csr_id) {
		return false;
	}
	if (label != other.label) {
		return false;
	}
	if (state_initialized != other.state_initialized) {
		return false;
	}
	return true;
}

} // namespace duckdb
