#include "duckpgq/core/functions/function_data/common_neighbors_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

CommonNeighborsFunctionData::CommonNeighborsFunctionData(ClientContext &context, int32_t csr_id)
    : context(context), csr_id(csr_id) {
}

unique_ptr<FunctionData> CommonNeighborsFunctionData::CommonNeighborsBind(ClientContext &context,
                                                                          ScalarFunction &bound_function,
                                                                          vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}
	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);
	return make_uniq<CommonNeighborsFunctionData>(context, csr_id);
}

unique_ptr<FunctionData> CommonNeighborsFunctionData::Copy() const {
	return make_uniq<CommonNeighborsFunctionData>(context, csr_id);
}

bool CommonNeighborsFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<CommonNeighborsFunctionData>();
	return other.csr_id == csr_id;
}

} // namespace duckdb
