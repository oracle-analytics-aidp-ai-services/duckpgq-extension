#include "duckpgq/core/functions/function_data/harmonic_centrality_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

// Constructor
HarmonicCentralityFunctionData::HarmonicCentralityFunctionData(ClientContext &ctx, int32_t csr)
    : context(ctx), csr_id(csr), state_initialized(false) {
}

unique_ptr<FunctionData>
HarmonicCentralityFunctionData::HarmonicCentralityBind(ClientContext &context, ScalarFunction &bound_function,
                                                       vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}

	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);

	return make_uniq<HarmonicCentralityFunctionData>(context, csr_id);
}

// Copy method
unique_ptr<FunctionData> HarmonicCentralityFunctionData::Copy() const {
	auto result = make_uniq<HarmonicCentralityFunctionData>(context, csr_id);
	result->harmonic = harmonic; // Deep copy of harmonic vector
	result->state_initialized = state_initialized;
	// Note: state_lock is not copied as mutexes are not copyable
	return std::move(result);
}

// Equals method
bool HarmonicCentralityFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<HarmonicCentralityFunctionData>();
	if (csr_id != other.csr_id) {
		return false;
	}
	if (harmonic != other.harmonic) {
		return false;
	}
	if (state_initialized != other.state_initialized) {
		return false;
	}
	return true;
}

} // namespace duckdb
