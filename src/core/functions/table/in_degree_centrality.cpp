#include "duckpgq/core/functions/table/in_degree_centrality.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckpgq_extension.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"

#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckpgq/core/utils/compressed_sparse_row.hpp"
#include <duckdb/parser/query_node/select_node.hpp>
#include <duckdb/parser/tableref/subqueryref.hpp>
#include <duckpgq/core/functions/table.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>

namespace duckdb {

// Main binding function
unique_ptr<TableRef> InDegreeCentralityFunction::InDegreeCentralityBindReplace(ClientContext &context,
                                                                               TableFunctionBindInput &input) {
	auto vertex_table = StringUtil::Lower(StringValue::Get(input.inputs[0]));
	auto vertex_id = StringUtil::Lower(StringValue::Get(input.inputs[1]));
	auto edge_table = StringUtil::Lower(StringValue::Get(input.inputs[2]));
	auto src_col = StringUtil::Lower(StringValue::Get(input.inputs[3]));
	auto dst_col = StringUtil::Lower(StringValue::Get(input.inputs[4]));
	(void)context;

	auto edge_pg_entry = MakeEdgeSpec(vertex_table, vertex_id, edge_table, src_col, dst_col);

	auto select_node = CreateSelectNode(edge_pg_entry, "in_degree_centrality", "in_degree_centrality");

	select_node->cte_map.map["csr_cte"] = CreateDirectedCSRCTE(edge_pg_entry, "src", "edge", "dst");

	auto subquery = make_uniq<SelectStatement>();
	subquery->node = std::move(select_node);

	auto result = make_uniq<SubqueryRef>(std::move(subquery));
	result->alias = "in_degree_centrality";
	return std::move(result);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreTableFunctions::RegisterInDegreeCentralityTableFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(InDegreeCentralityFunction());
}

} // namespace duckdb
