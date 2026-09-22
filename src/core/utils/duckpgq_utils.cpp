#include "duckpgq/core/utils/duckpgq_utils.hpp"
#include "duckpgq/common.hpp"
#include <atomic>
#include "duckdb/parser/statement/copy_statement.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include "duckdb/parser/property_graph_table.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckpgq/core/utils/compressed_sparse_row.hpp"

namespace duckdb {

// Function to get DuckPGQState from ClientContext
shared_ptr<DuckPGQState> GetDuckPGQState(ClientContext &context, bool throw_not_found_error) {
	auto lookup = context.registered_state->Get<DuckPGQState>("duckpgq");
	if (lookup) {
		return lookup;
	}
	if (throw_not_found_error) {
		throw Exception(ExceptionType::INVALID, "Registered DuckPGQ state not found");
	}
	shared_ptr<DuckPGQState> state = make_shared_ptr<DuckPGQState>();
	context.registered_state->Insert("duckpgq", state);
	return state;
}

void CheckAlgorithmMemoryBudget(ClientContext &context, idx_t estimated_bytes, const string &algorithm_name) {
	auto &buffer_manager = BufferManager::GetBufferManager(context);
	idx_t max_memory = buffer_manager.GetMaxMemory();
	// max_memory == 0 (or the sentinel max) means effectively unlimited.
	if (max_memory == 0 || max_memory == DConstants::INVALID_INDEX) {
		return;
	}
	if (estimated_bytes > max_memory) {
		throw OutOfMemoryException(
		    "DuckPGQ %s needs approximately %s of working memory, which exceeds the configured memory_limit of %s. "
		    "Increase memory_limit (e.g. SET memory_limit='8GB') or run on a smaller graph.",
		    algorithm_name, StringUtil::BytesToHumanReadableString(estimated_bytes),
		    StringUtil::BytesToHumanReadableString(max_memory));
	}
}

// Build an edge-spec directly from table-function arguments (no property-graph
// registration). The CSR builder + CreateSelectNode read these fields exactly as
// they did for a registered PropertyGraphTable; here we populate them from the
// caller-supplied (vertex_table, vertex_id, edge_table, src_col, dst_col).
// Source and destination vertices are the same vertex table; the CSR builders
// disambiguate the self-join via the "src"/"dst" bindings.
// Process-wide CSR id allocator. Each MakeEdgeSpec call (i.e. each algorithm
// invocation) gets a fresh id, so multiple graph-algorithm calls in one query —
// or concurrent queries — never share a CSR slot in DuckPGQState::csr_list.
static int32_t GetNextCsrId() {
	static std::atomic<int32_t> counter {0};
	return counter.fetch_add(1, std::memory_order_relaxed);
}

shared_ptr<PropertyGraphTable> MakeEdgeSpec(const string &vertex_table, const string &vertex_id,
                                            const string &edge_table, const string &src_col, const string &dst_col) {
	auto vsrc = make_shared_ptr<PropertyGraphTable>();
	vsrc->table_name = vertex_table;
	vsrc->is_vertex_table = true;
	auto vdst = make_shared_ptr<PropertyGraphTable>();
	vdst->table_name = vertex_table;
	vdst->is_vertex_table = true;

	auto e = make_shared_ptr<PropertyGraphTable>();
	e->table_name = edge_table;
	e->table_name_alias = edge_table;
	e->source_fk = {src_col};
	e->destination_fk = {dst_col};
	e->source_pk = {vertex_id};
	e->destination_pk = {vertex_id};
	e->source_reference = vertex_table;
	e->destination_reference = vertex_table;
	e->source_pg_table = std::move(vsrc);
	e->destination_pg_table = std::move(vdst);
	e->csr_id = GetNextCsrId();
	return e;
}

// Function to create the SELECT node
unique_ptr<SelectNode> CreateSelectNode(const shared_ptr<PropertyGraphTable> &edge_pg_entry,
                                        const string &function_name, const string &function_alias) {
	auto select_node = make_uniq<SelectNode>();
	std::vector<unique_ptr<ParsedExpression>> select_expression;

	select_expression.emplace_back(
	    make_uniq<ColumnRefExpression>(edge_pg_entry->source_pk[0], edge_pg_entry->source_reference));

	auto cte_col_ref = make_uniq<ColumnRefExpression>("temp", "__x");

	vector<unique_ptr<ParsedExpression>> function_children;
	function_children.push_back(make_uniq<ConstantExpression>(Value::INTEGER(edge_pg_entry->csr_id)));
	function_children.push_back(make_uniq<ColumnRefExpression>("rowid", edge_pg_entry->source_reference));
	auto function = make_uniq<FunctionExpression>(function_name, std::move(function_children));

	std::vector<unique_ptr<ParsedExpression>> addition_children;
	addition_children.emplace_back(std::move(cte_col_ref));
	addition_children.emplace_back(std::move(function));

	auto addition_function = make_uniq<FunctionExpression>("add", std::move(addition_children));
	addition_function->alias = function_alias;
	select_expression.emplace_back(std::move(addition_function));
	select_node->select_list = std::move(select_expression);

	auto src_base_ref = edge_pg_entry->source_pg_table->CreateBaseTableRef();

	auto temp_cte_select_subquery = CreateCountCTESubquery();

	auto cross_join_ref = make_uniq<JoinRef>(JoinRefType::CROSS);
	cross_join_ref->left = std::move(src_base_ref);
	cross_join_ref->right = std::move(temp_cte_select_subquery);

	select_node->from_table = std::move(cross_join_ref);

	return select_node;
}

unique_ptr<SelectNode> CreateParamSelectNode(const shared_ptr<PropertyGraphTable> &edge_pg_entry,
                                             const string &function_name, const string &function_alias,
                                             const Value &extra_arg) {
	auto select_node = make_uniq<SelectNode>();
	std::vector<unique_ptr<ParsedExpression>> select_expression;

	select_expression.emplace_back(
	    make_uniq<ColumnRefExpression>(edge_pg_entry->source_pk[0], edge_pg_entry->source_reference));

	auto cte_col_ref = make_uniq<ColumnRefExpression>("temp", "__x");

	vector<unique_ptr<ParsedExpression>> function_children;
	function_children.push_back(make_uniq<ConstantExpression>(Value::INTEGER(edge_pg_entry->csr_id)));
	function_children.push_back(make_uniq<ColumnRefExpression>("rowid", edge_pg_entry->source_reference));
	function_children.push_back(make_uniq<ConstantExpression>(extra_arg));
	auto function = make_uniq<FunctionExpression>(function_name, std::move(function_children));

	std::vector<unique_ptr<ParsedExpression>> addition_children;
	addition_children.emplace_back(std::move(cte_col_ref));
	addition_children.emplace_back(std::move(function));
	auto addition_function = make_uniq<FunctionExpression>("add", std::move(addition_children));
	addition_function->alias = function_alias;
	select_expression.emplace_back(std::move(addition_function));
	select_node->select_list = std::move(select_expression);

	auto src_base_ref = edge_pg_entry->source_pg_table->CreateBaseTableRef();
	auto temp_cte_select_subquery = CreateCountCTESubquery();

	auto cross_join_ref = make_uniq<JoinRef>(JoinRefType::CROSS);
	cross_join_ref->left = std::move(src_base_ref);
	cross_join_ref->right = std::move(temp_cte_select_subquery);
	select_node->from_table = std::move(cross_join_ref);

	return select_node;
}

// WARNING: this emits an O(V^2) cross-join over the vertex table (every a<b
// pair), so the pairwise similarity functions (jaccard/cosine/overlap/common-
// neighbors/adamic-adar/preferential-attachment/resource-allocation) are
// quadratic in vertex count and NOT bounded by CheckAlgorithmMemoryBudget.
// Intended for small/medium graphs or filtered use (e.g. WHERE sim > t LIMIT k).
unique_ptr<SelectNode> CreatePairwiseSelectNode(const shared_ptr<PropertyGraphTable> &edge_pg_entry,
                                                const string &function_name, const string &function_alias) {
	auto select_node = make_uniq<SelectNode>();
	const string &pk = edge_pg_entry->source_pk[0];

	std::vector<unique_ptr<ParsedExpression>> select_expression;
	auto a_pk = make_uniq<ColumnRefExpression>(pk, "a");
	a_pk->alias = "node1";
	auto b_pk = make_uniq<ColumnRefExpression>(pk, "b");
	b_pk->alias = "node2";
	select_expression.emplace_back(std::move(a_pk));
	select_expression.emplace_back(std::move(b_pk));

	vector<unique_ptr<ParsedExpression>> function_children;
	function_children.push_back(make_uniq<ConstantExpression>(Value::INTEGER(edge_pg_entry->csr_id)));
	function_children.push_back(make_uniq<ColumnRefExpression>("rowid", "a"));
	function_children.push_back(make_uniq<ColumnRefExpression>("rowid", "b"));
	auto function = make_uniq<FunctionExpression>(function_name, std::move(function_children));

	std::vector<unique_ptr<ParsedExpression>> addition_children;
	addition_children.emplace_back(make_uniq<ColumnRefExpression>("temp", "__x"));
	addition_children.emplace_back(std::move(function));
	auto addition_function = make_uniq<FunctionExpression>("add", std::move(addition_children));
	addition_function->alias = function_alias;
	select_expression.emplace_back(std::move(addition_function));
	select_node->select_list = std::move(select_expression);

	auto a_ref = edge_pg_entry->source_pg_table->CreateBaseTableRef();
	a_ref->alias = "a";
	auto b_ref = edge_pg_entry->source_pg_table->CreateBaseTableRef();
	b_ref->alias = "b";

	auto inner_join = make_uniq<JoinRef>(JoinRefType::CROSS);
	inner_join->left = std::move(a_ref);
	inner_join->right = std::move(b_ref);

	auto outer_join = make_uniq<JoinRef>(JoinRefType::CROSS);
	outer_join->left = std::move(inner_join);
	outer_join->right = CreateCountCTESubquery();
	select_node->from_table = std::move(outer_join);

	// Only emit each unordered pair once.
	select_node->where_clause =
	    make_uniq<ComparisonExpression>(ExpressionType::COMPARE_LESSTHAN, make_uniq<ColumnRefExpression>("rowid", "a"),
	                                    make_uniq<ColumnRefExpression>("rowid", "b"));

	return select_node;
}

unique_ptr<BaseTableRef> CreateBaseTableRef(const string &table_name, const string &alias) {
	auto base_table_ref = make_uniq<BaseTableRef>();
	base_table_ref->table_name = table_name;
	if (!alias.empty()) {
		base_table_ref->alias = alias;
	}
	return base_table_ref;
}

unique_ptr<ColumnRefExpression> CreateColumnRefExpression(const string &column_name, const string &table_name,
                                                          const string &alias) {
	unique_ptr<ColumnRefExpression> column_ref;
	if (table_name.empty()) {
		column_ref = make_uniq<ColumnRefExpression>(column_name);
	} else {
		column_ref = make_uniq<ColumnRefExpression>(column_name, table_name);
	}
	if (!alias.empty()) {
		column_ref->alias = alias;
	}
	return column_ref;
}

} // namespace duckdb
