#include "duckpgq/core/functions/function_data/preferential_attachment_function_data.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <duckpgq/core/utils/duckpgq_utils.hpp>

namespace duckdb {

PreferentialAttachmentFunctionData::PreferentialAttachmentFunctionData(ClientContext &context, int32_t csr_id)
    : context(context), csr_id(csr_id) {
}

unique_ptr<FunctionData>
PreferentialAttachmentFunctionData::PreferentialAttachmentBind(ClientContext &context, ScalarFunction &bound_function,
                                                              vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Id must be constant.");
	}
	int32_t csr_id = ExpressionExecutor::EvaluateScalar(context, *arguments[0]).GetValue<int32_t>();
	auto duckpgq_state = GetDuckPGQState(context);
	duckpgq_state->ScheduleDelete(csr_id);
	return make_uniq<PreferentialAttachmentFunctionData>(context, csr_id);
}

unique_ptr<FunctionData> PreferentialAttachmentFunctionData::Copy() const {
	return make_uniq<PreferentialAttachmentFunctionData>(context, csr_id);
}

bool PreferentialAttachmentFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<PreferentialAttachmentFunctionData>();
	return other.csr_id == csr_id;
}

} // namespace duckdb
