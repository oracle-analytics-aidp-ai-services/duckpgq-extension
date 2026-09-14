#include "duckpgq/core/functions/function_data/eigenvector_centrality_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

// Constructor
EigenvectorCentralityFunctionData::EigenvectorCentralityFunctionData(ClientContext &ctx, int32_t csr)
    : context(ctx), csr_id(csr), max_iterations(100), convergence_threshold(1e-8), iteration_count(0),
      state_initialized(false), converged(false) {
}

unique_ptr<FunctionData>
EigenvectorCentralityFunctionData::EigenvectorCentralityBind(ClientContext &context, ScalarFunction &bound_function,
                                                             vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}

	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);

	return make_uniq<EigenvectorCentralityFunctionData>(context, csr_id);
}

// Copy method
unique_ptr<FunctionData> EigenvectorCentralityFunctionData::Copy() const {
	auto result = make_uniq<EigenvectorCentralityFunctionData>(context, csr_id);
	result->centrality = centrality; // Deep copy of centrality vector
	result->max_iterations = max_iterations;
	result->convergence_threshold = convergence_threshold;
	result->iteration_count = iteration_count;
	result->state_initialized = state_initialized;
	result->converged = converged;
	// Note: state_lock is not copied as mutexes are not copyable
	return std::move(result);
}

// Equals method
bool EigenvectorCentralityFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<EigenvectorCentralityFunctionData>();
	if (csr_id != other.csr_id) {
		return false;
	}
	if (centrality != other.centrality) {
		return false;
	}
	if (max_iterations != other.max_iterations) {
		return false;
	}
	if (convergence_threshold != other.convergence_threshold) {
		return false;
	}
	if (iteration_count != other.iteration_count) {
		return false;
	}
	if (state_initialized != other.state_initialized) {
		return false;
	}
	if (converged != other.converged) {
		return false;
	}
	return true;
}

} // namespace duckdb
