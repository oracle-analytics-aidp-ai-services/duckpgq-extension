#include "duckpgq/core/functions/function_data/triangle_count_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

TriangleCountFunctionData::TriangleCountFunctionData(ClientContext &context, int32_t csr_id)
    : context(context), csr_id(csr_id) {
}

unique_ptr<FunctionData> TriangleCountFunctionData::TriangleCountBind(ClientContext &context,
                                                                      ScalarFunction &bound_function,
                                                                      vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}

	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);
	return make_uniq<TriangleCountFunctionData>(context, csr_id);
}

unique_ptr<FunctionData> TriangleCountFunctionData::Copy() const {
	return make_uniq<TriangleCountFunctionData>(context, csr_id);
}

bool TriangleCountFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<TriangleCountFunctionData>();
	return other.csr_id == csr_id;
}

} // namespace duckdb
