//===----------------------------------------------------------------------===//
//                         DuckPGQ (stock-DuckDB build)
//
// src/core/functions/table/graph_match.cpp
//
// Ported from the fork's extension/duckpgq/src/core/functions/table/match.cpp.
// The relational-rewrite logic (ProcessPathList / AddEdgeJoins / EdgeType* /
// CreateMatchJoinExpression / bounded-path expansion) is pure (no catalog /
// binder / ClientContext coupling) and is reproduced essentially unchanged.
//
// What is adapted: the fork's MatchBindReplace pulled a parsed MatchExpression
// and a registered CreatePropertyGraphInfo out of DuckPGQState. Here,
// GraphMatchBindReplace instead:
//   1. reads the table-function args (pattern, vertex_table, vertex_id,
//      edge_table, src, dst),
//   2. parses the pattern string into a MatchExpression (pattern_parser),
//   3. SYNTHESIZES a one-vertex-table / one-edge-table CreatePropertyGraphInfo,
//      populated exactly like MakeEdgeSpec and keyed by the parsed labels so
//      FindGraphTable resolves, then
//   4. runs the ported rewrite to return a SubqueryRef.
//
// Variable-length edges (-[e:E]->{lo,hi}) are lowered via the relational
// bounded-path expansion (AddBoundedPathExpansion) only — the CSR/shortestpath
// route is intentionally not wired here (see TODO at the bottom).
//===----------------------------------------------------------------------===//

#include "duckpgq/core/functions/table/graph_match.hpp"
#include "duckpgq/core/parser/pattern_parser.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/common_table_expression_info.hpp"

#include <duckpgq/core/functions/table.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>

#include <algorithm>

namespace duckdb {

namespace {

//! Largest upper bound a quantified hop may ask for.
//!
//! This bounds RECURSION DEPTH, and nothing else. It is emphatically NOT a bound
//! on the result: one row per walk means the output grows with the graph's
//! branching factor, so on a 20-vertex complete digraph `{1,6}` is around 993
//! million rows. It used to bound the result as a side effect, because the outer
//! DISTINCT capped output at the number of distinct endpoint pairs; that
//! DISTINCT is gone by design, and this constant did not become a replacement
//! for it. A caller that needs a row bound needs its own LIMIT.
constexpr int64_t MAX_BOUNDED_PATH_EXPANSION_UPPER = 16;

unique_ptr<ParsedExpression> BuildConjunction(vector<unique_ptr<ParsedExpression>> &conditions) {
	unique_ptr<ParsedExpression> where_clause;
	for (auto &condition : conditions) {
		if (where_clause) {
			where_clause = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(where_clause),
			                                                std::move(condition));
		} else {
			where_clause = std::move(condition);
		}
	}
	return where_clause;
}

void CrossJoinTableRef(unique_ptr<TableRef> &from_clause, unique_ptr<TableRef> table_ref) {
	if (from_clause) {
		auto join = make_uniq<JoinRef>(JoinRefType::CROSS);
		join->left = std::move(from_clause);
		join->right = std::move(table_ref);
		from_clause = std::move(join);
	} else {
		from_clause = std::move(table_ref);
	}
}

//! `<binding>.<label_column> = '<label>'` — the filter that makes a written
//! label mean something.
//!
//! Without this the label was parsed, used to look up the element's table, and
//! then dropped. With one vertex table and one edge table behind graph_match
//! that lookup always succeeded, so every label resolved and none filtered:
//! `(a:Person)-[x:knows]->(b:Person)` returned every edge in the graph. Measured
//! on a three-vertex fixture with two classes and two predicates, the pattern
//! `(a:zzz)-[x:qqq]->(b:zzz)` — labels that exist nowhere — returned all rows.
//! CASE-SENSITIVE, DELIBERATELY, AND ASYMMETRIC WITH THE RELATION-NAME RULE.
//!
//! Two readings of a written label coexist here, and they fold case differently
//! because they compare different kinds of thing:
//!
//!   * Against a RELATION NAME (the no-label-column rule) the comparison is
//!     case-insensitive, because SQL identifiers fold. `(a:STUDENT)` over the
//!     `Student` table matches.
//!   * Against a COLUMN VALUE — this function — the comparison is exact, because
//!     it is a data value and `'Person' <> 'person'` in SQL. `(a:PERSON)` against
//!     rows holding `'person'` matches nothing.
//!
//! So a caller moving from the relation-name form to the label-column form can
//! lose every row to a spelling that was previously accepted. That is recorded
//! rather than smoothed over: case-folding the value comparison would mean
//! inventing a collation for someone else's data, and folding the identifier
//! comparison would contradict SQL. Callers that generate patterns should emit
//! the label spelling their data holds.
unique_ptr<ParsedExpression> BuildLabelFilter(const string &binding, const string &label_column, const string &label) {
	auto column = make_uniq<ColumnRefExpression>(label_column, binding);
	auto value = make_uniq<ConstantExpression>(Value(label));
	return make_uniq<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(column), std::move(value));
}

} // namespace

shared_ptr<PropertyGraphTable> GraphMatchFunction::FindGraphTable(const string &label,
                                                                 CreatePropertyGraphInfo &pg_table) {
	const auto graph_table_entry = pg_table.label_map.find(label);
	if (graph_table_entry == pg_table.label_map.end()) {
		throw BinderException("graph_match: the label '%s' is not present in the pattern's tables", label);
	}
	return graph_table_entry->second;
}

shared_ptr<PropertyGraphTable> GraphMatchFunction::ResolveByRole(bool is_vertex, CreatePropertyGraphInfo &pg_table) {
	// RESOLVED BY ROLE, NOT BY LABEL. There is exactly one vertex relation and one
	// edge relation behind graph_match, so the element's position in the pattern
	// already determines its relation and a label lookup adds nothing.
	//
	// Resolving by label instead made one name unusable as both a vertex and an
	// edge label, because a single `label_map` holds one entry per name. That cost
	// a real capability twice over. A vertex class and an edge class sharing a
	// name (`type`, `member`, `owns`) is ordinary in the mappings this feature
	// serves. And for a SELF-GRAPH — one table passed as both the vertex and the
	// edge relation — the no-label-column rule says a label must name its
	// relation, so the only legal spelling was `(a:t)-[e:t]->(b:t)`, which the
	// collision refusal then rejected: labels were unusable on such a graph
	// altogether.
	if (is_vertex) {
		return pg_table.vertex_tables[0];
	}
	return pg_table.edge_tables[0];
}

unique_ptr<ParsedExpression> GraphMatchFunction::CreateMatchJoinExpression(vector<string> vertex_keys,
                                                                          vector<string> edge_keys,
                                                                          const string &vertex_alias,
                                                                          const string &edge_alias) {
	vector<unique_ptr<ParsedExpression>> conditions;
	if (vertex_keys.size() != edge_keys.size()) {
		throw BinderException("Vertex columns and edge columns size mismatch");
	}
	for (idx_t i = 0; i < vertex_keys.size(); i++) {
		auto vertex_colref = make_uniq<ColumnRefExpression>(vertex_keys[i], vertex_alias);
		auto edge_colref = make_uniq<ColumnRefExpression>(edge_keys[i], edge_alias);
		conditions.push_back(make_uniq<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(vertex_colref),
		                                                     std::move(edge_colref)));
	}
	return BuildConjunction(conditions);
}

PathElement *GraphMatchFunction::GetPathElement(const unique_ptr<PathReference> &path_reference) {
	if (path_reference->path_reference_type == PGQPathReferenceType::PATH_ELEMENT) {
		return reinterpret_cast<PathElement *>(path_reference.get());
	}
	if (path_reference->path_reference_type == PGQPathReferenceType::SUBPATH) {
		return nullptr;
	}
	throw InternalException("Unknown path reference type detected");
}

SubPath *GraphMatchFunction::GetSubPath(const unique_ptr<PathReference> &path_reference) {
	if (path_reference->path_reference_type == PGQPathReferenceType::PATH_ELEMENT) {
		return nullptr;
	}
	if (path_reference->path_reference_type == PGQPathReferenceType::SUBPATH) {
		return reinterpret_cast<SubPath *>(path_reference.get());
	}
	throw InternalException("Unknown path reference type detected");
}

unique_ptr<ParsedExpression> GraphMatchFunction::CreateWhereClause(vector<unique_ptr<ParsedExpression>> &conditions) {
	return BuildConjunction(conditions);
}

void GraphMatchFunction::CheckEdgeTableConstraints(const string &src_reference, const string &dst_reference,
                                                  const shared_ptr<PropertyGraphTable> &edge_table) {
	if (src_reference != edge_table->source_reference) {
		throw BinderException("Label %s is not registered as a source reference for edge pattern of table %s",
		                      src_reference, edge_table->table_name);
	}
	if (dst_reference != edge_table->destination_reference) {
		throw BinderException("Label %s is not registered as a destination reference for edge pattern of table %s",
		                      src_reference, edge_table->table_name);
	}
}

void GraphMatchFunction::EdgeTypeAny(const shared_ptr<PropertyGraphTable> &edge_table, const string &edge_binding,
                                    const string &prev_binding, const string &next_binding,
                                    vector<unique_ptr<ParsedExpression>> &conditions,
                                    unique_ptr<TableRef> &from_clause) {
	// (SELECT src, dst, * FROM edge UNION ALL SELECT dst, src, * FROM edge) edge_binding
	auto src_dst_select_node = make_uniq<SelectNode>();
	src_dst_select_node->from_table = edge_table->CreateBaseTableRef(edge_binding);
	{
		vector<unique_ptr<ParsedExpression>> src_dst_children;
		src_dst_children.push_back(make_uniq<ColumnRefExpression>(edge_table->source_fk[0], edge_binding));
		src_dst_children.push_back(make_uniq<ColumnRefExpression>(edge_table->destination_fk[0], edge_binding));
		src_dst_children.push_back(make_uniq<StarExpression>());
		src_dst_select_node->select_list = std::move(src_dst_children);
	}

	auto dst_src_select_node = make_uniq<SelectNode>();
	dst_src_select_node->from_table = edge_table->CreateBaseTableRef(edge_binding);
	{
		vector<unique_ptr<ParsedExpression>> dst_src_children;
		dst_src_children.push_back(make_uniq<ColumnRefExpression>(edge_table->destination_fk[0], edge_binding));
		dst_src_children.push_back(make_uniq<ColumnRefExpression>(edge_table->source_fk[0], edge_binding));
		dst_src_children.push_back(make_uniq<StarExpression>());
		dst_src_select_node->select_list = std::move(dst_src_children);
	}

	auto union_node = make_uniq<SetOperationNode>();
	union_node->setop_type = SetOperationType::UNION;
	union_node->setop_all = true;
	union_node->children.push_back(std::move(src_dst_select_node));
	union_node->children.push_back(std::move(dst_src_select_node));
	auto union_select = make_uniq<SelectStatement>();
	union_select->node = std::move(union_node);
	auto union_subquery = make_uniq<SubqueryRef>(std::move(union_select));
	union_subquery->alias = edge_binding;

	CrossJoinTableRef(from_clause, std::move(union_subquery));

	auto src_left_expr =
	    CreateMatchJoinExpression(edge_table->source_pk, edge_table->source_fk, prev_binding, edge_binding);
	auto dst_left_expr =
	    CreateMatchJoinExpression(edge_table->destination_pk, edge_table->destination_fk, next_binding, edge_binding);
	auto combined_left_expr = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
	                                                           std::move(src_left_expr), std::move(dst_left_expr));
	conditions.push_back(std::move(combined_left_expr));
}

void GraphMatchFunction::EdgeTypeLeft(const shared_ptr<PropertyGraphTable> &edge_table, const string &next_table_name,
                                     const string &prev_table_name, const string &edge_binding,
                                     const string &prev_binding, const string &next_binding,
                                     vector<unique_ptr<ParsedExpression>> &conditions) {
	CheckEdgeTableConstraints(next_table_name, prev_table_name, edge_table);
	conditions.push_back(
	    CreateMatchJoinExpression(edge_table->source_pk, edge_table->source_fk, next_binding, edge_binding));
	conditions.push_back(
	    CreateMatchJoinExpression(edge_table->destination_pk, edge_table->destination_fk, prev_binding, edge_binding));
}

void GraphMatchFunction::EdgeTypeRight(const shared_ptr<PropertyGraphTable> &edge_table, const string &next_table_name,
                                      const string &prev_table_name, const string &edge_binding,
                                      const string &prev_binding, const string &next_binding,
                                      vector<unique_ptr<ParsedExpression>> &conditions) {
	CheckEdgeTableConstraints(prev_table_name, next_table_name, edge_table);
	conditions.push_back(
	    CreateMatchJoinExpression(edge_table->source_pk, edge_table->source_fk, prev_binding, edge_binding));
	conditions.push_back(
	    CreateMatchJoinExpression(edge_table->destination_pk, edge_table->destination_fk, next_binding, edge_binding));
}

void GraphMatchFunction::EdgeTypeLeftRight(const shared_ptr<PropertyGraphTable> &edge_table, const string &edge_binding,
                                          const string &prev_binding, const string &next_binding,
                                          vector<unique_ptr<ParsedExpression>> &conditions,
                                          case_insensitive_map_t<shared_ptr<PropertyGraphTable>> &alias_map,
                                          int32_t &extra_alias_counter) {
	auto src_left_expr =
	    CreateMatchJoinExpression(edge_table->source_pk, edge_table->source_fk, next_binding, edge_binding);
	auto dst_left_expr =
	    CreateMatchJoinExpression(edge_table->destination_pk, edge_table->destination_fk, prev_binding, edge_binding);
	auto combined_left_expr = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
	                                                           std::move(src_left_expr), std::move(dst_left_expr));

	const auto additional_edge_alias = edge_binding + std::to_string(extra_alias_counter);
	extra_alias_counter++;
	alias_map[additional_edge_alias] = edge_table;

	auto src_right_expr =
	    CreateMatchJoinExpression(edge_table->source_pk, edge_table->source_fk, prev_binding, additional_edge_alias);
	auto dst_right_expr = CreateMatchJoinExpression(edge_table->destination_pk, edge_table->destination_fk,
	                                                next_binding, additional_edge_alias);
	auto combined_right_expr = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
	                                                            std::move(src_right_expr), std::move(dst_right_expr));

	auto combined_expr = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(combined_left_expr),
	                                                      std::move(combined_right_expr));
	conditions.push_back(std::move(combined_expr));
}

void GraphMatchFunction::AddEdgeJoins(const shared_ptr<PropertyGraphTable> &edge_table,
                                     const shared_ptr<PropertyGraphTable> &previous_vertex_table,
                                     const shared_ptr<PropertyGraphTable> &next_vertex_table, PGQMatchType edge_type,
                                     const string &edge_binding, const string &prev_binding,
                                     const string &next_binding, vector<unique_ptr<ParsedExpression>> &conditions,
                                     case_insensitive_map_t<shared_ptr<PropertyGraphTable>> &alias_map,
                                     int32_t &extra_alias_counter, unique_ptr<TableRef> &from_clause) {
	if (edge_type != PGQMatchType::MATCH_EDGE_ANY) {
		alias_map[edge_binding] = edge_table;
	}
	switch (edge_type) {
	case PGQMatchType::MATCH_EDGE_ANY:
		EdgeTypeAny(edge_table, edge_binding, prev_binding, next_binding, conditions, from_clause);
		break;
	case PGQMatchType::MATCH_EDGE_LEFT:
		EdgeTypeLeft(edge_table, next_vertex_table->table_name, previous_vertex_table->table_name, edge_binding,
		             prev_binding, next_binding, conditions);
		break;
	case PGQMatchType::MATCH_EDGE_RIGHT:
		EdgeTypeRight(edge_table, next_vertex_table->table_name, previous_vertex_table->table_name, edge_binding,
		              prev_binding, next_binding, conditions);
		break;
	case PGQMatchType::MATCH_EDGE_LEFT_RIGHT:
		EdgeTypeLeftRight(edge_table, edge_binding, prev_binding, next_binding, conditions, alias_map,
		                  extra_alias_counter);
		break;
	default:
		throw InternalException("Unknown match type found");
	}
}

bool GraphMatchFunction::BoundNeedsExpansion(const SubPath &subpath) {
	// A WRITTEN BOUND THAT IS NOT EXACTLY `{1,1}` NEEDS THE RECURSION.
	//
	// One predicate in one place, deliberately. This test existed twice — here and
	// at the ProcessPathList call site — and the two spellings disagreed: the call
	// site said `upper > 1`, which sent `{0,1}` down the ordinary single-edge join
	// path. That path has nowhere to put a lower bound of zero, so the zero-length
	// match was dropped and `{0,1}` silently answered `{1,1}`. Oracle answers
	// `{0,n}` by including each vertex paired with itself (measured: `{0,2}` on a
	// four-vertex chain returns 9 rows = 4 + 3 + 2), so the rows were simply
	// missing. Two copies of a routing predicate is how that got in.
	return subpath.lower != 1 || subpath.upper != 1;
}

bool GraphMatchFunction::CanExpandBoundedSubpath(const shared_ptr<PropertyGraphTable> &edge_table, SubPath *subpath,
                                                PGQMatchType edge_type) {
	if (edge_type != PGQMatchType::MATCH_EDGE_RIGHT) {
		return false;
	}
	if (!BoundNeedsExpansion(*subpath)) {
		return false;
	}
	if (subpath->upper > MAX_BOUNDED_PATH_EXPANSION_UPPER) {
		return false;
	}
	if (edge_table->source_reference != edge_table->destination_reference) {
		return false;
	}
	if (edge_table->source_pk.size() != edge_table->destination_pk.size()) {
		return false;
	}
	return edge_table->source_fk.size() == edge_table->source_pk.size() &&
	       edge_table->destination_fk.size() == edge_table->destination_pk.size();
}

void GraphMatchFunction::AddBoundedPathExpansion(const shared_ptr<PropertyGraphTable> &edge_table, SubPath *subpath,
                                                const string &prev_binding, const string &edge_binding,
                                                const string &next_binding,
                                                vector<unique_ptr<ParsedExpression>> &conditions,
                                                unique_ptr<TableRef> &from_clause, int32_t &extra_alias_counter,
                                                const string &edge_label, const string &edge_label_column) {
	// A RECURSIVE CTE, not a union of fixed-length branches. This shape is a
	// correctness requirement, not a tidiness preference.
	//
	// The previous shape emitted one branch per length, and each branch
	// cross-joined the edge table to itself once per hop: length 7 meant seven
	// identical scans of one table in a single plan. DuckDB v1.5.0's
	// `common_subplan` optimizer mis-plans that, and the result is a WRONG ROW
	// COUNT rather than an error. Measured on the fixture in
	// test/sql/scalar/graph_match.test (5 vertices, 8 edges), against walk counts
	// derived independently by iterated self-join:
	//
	//     bound    expected   emitted
	//     {1,7}         338       204     short by 134
	//     {6,7}         217        83     short by 134
	//     {7,7}         134       154     20 spurious rows
	//     {8,9}         565         0     returned nothing
	//     {9,9}         350      2300     6.5x over-count
	//
	// `SET disabled_optimizers='common_subplan'` makes every one of those correct,
	// which is what identifies the optimizer as the cause. `{7,7}` is a SINGLE
	// branch, so the trigger is repeated identical scans WITHIN one plan, not
	// sharing across branches.
	//
	// THE OPTIMIZER BUG IS FIXED UPSTREAM, IN 1.5.1. It is present only in 1.5.0,
	// which is the version this extension is pinned to build against. Reduced to
	// standalone SQL — two UNION ALL branches of 6 and 7 self-joins over a
	// 5-edge, 4-vertex fixture, no extension involved — the count is 12 on 1.5.0
	// and the correct 10 on 1.5.1, 1.5.2, 1.5.3, 1.5.4 and 1.5.5. So there is
	// nothing to report upstream; the exposure is entirely the engine pin.
	//
	// That leaves two ways to be correct, and this is the one that does not need
	// anybody's permission. Bumping the engine pin would also fix it, but the
	// pinned DuckDB version and the built artifact are a single unit that has to
	// move together, and it changes every other consumer of that pin too — a
	// decision for whoever owns it, not a side effect of a graph fix.
	//
	// The recursion is worth having regardless of the engine, because it does not
	// depend on optimizer behaviour at all: it scans the edge table once per step
	// instead of once per hop per branch, so there are no identical subtrees to
	// merge. A hop cap was rejected because the breakage is not a function of the
	// bound — the same sweep over a 4-vertex chain and a 2-vertex cycle is correct
	// at every bound up to 16 while the 5-vertex graph above breaks at 7 — so no
	// limit could be shown safe, only unobserved. A wrapper subquery around each
	// edge scan also produced correct answers and was rejected too: it perturbs
	// the planner into a different choice without removing the ambiguity.
	//
	// The emitted shape, for bounds {lo,hi}:
	//
	//   WITH RECURSIVE <alias>_walk(__duckpgq_src_0, __duckpgq_dst_0, __duckpgq_len) AS (
	//       SELECT v.<vid>, v.<vid>, 0 FROM <vertex_table> v
	//     UNION ALL
	//       SELECT w.__duckpgq_src_0, e.<dst_fk>, w.__duckpgq_len + 1
	//         FROM <alias>_walk w
	//         JOIN <vertex_table> iv ON iv.<vid> = w.__duckpgq_dst_0
	//         JOIN <edge_table> e ON e.<src_fk> = w.__duckpgq_dst_0
	//              [AND e.<label_col> = '<label>']
	//        WHERE w.__duckpgq_len < hi
	//   )
	//   SELECT __duckpgq_src_0, __duckpgq_dst_0 FROM <alias>_walk
	//    WHERE __duckpgq_len BETWEEN lo AND hi
	//
	// UNION ALL, never UNION: one row per walk is the whole point, and UNION would
	// reintroduce the endpoint-pair collapse this commit removed.
	//
	// The length-0 base row is emitted whatever `lower` is, and filtered out by
	// the final BETWEEN when `lower > 0`. That is what makes `{0,n}` fall out
	// rather than needing its own branch.
	auto path_alias = edge_binding + "_bounded_path_" + std::to_string(static_cast<uint32_t>(extra_alias_counter++));
	const auto cte_name = path_alias + "_walk";

	// Single-column keys only. graph_match synthesizes its own property graph from
	// scalar arguments (see MakeEdgeSpec and GraphMatchBindReplace), so every key
	// here is exactly one column; a composite key cannot arise. Asserted rather
	// than assumed, because the recursion below carries one src and one dst column
	// and would silently drop the rest.
	if (edge_table->source_pk.size() != 1 || edge_table->destination_pk.size() != 1 ||
	    edge_table->source_fk.size() != 1 || edge_table->destination_fk.size() != 1) {
		throw InternalException("graph_match: bounded path expansion requires single-column keys, got %llu/%llu",
		                        static_cast<unsigned long long>(edge_table->source_pk.size()),
		                        static_cast<unsigned long long>(edge_table->destination_pk.size()));
	}
	const auto &vertex_key = edge_table->source_pk[0];
	const auto &edge_src_fk = edge_table->source_fk[0];
	const auto &edge_dst_fk = edge_table->destination_fk[0];
	const string src_column = "__duckpgq_src_0";
	const string dst_column = "__duckpgq_dst_0";
	const string len_column = "__duckpgq_len";

	// --- base term: every vertex, paired with itself, at length 0 ---------------
	const auto base_vertex_alias = path_alias + "_v0";
	auto base = make_uniq<SelectNode>();
	base->from_table = edge_table->source_pg_table->CreateBaseTableRef(base_vertex_alias);
	{
		auto s = make_uniq<ColumnRefExpression>(vertex_key, base_vertex_alias);
		s->alias = src_column;
		base->select_list.push_back(std::move(s));
		auto d = make_uniq<ColumnRefExpression>(vertex_key, base_vertex_alias);
		d->alias = dst_column;
		base->select_list.push_back(std::move(d));
		auto z = make_uniq<ConstantExpression>(Value::BIGINT(0));
		z->alias = len_column;
		base->select_list.push_back(std::move(z));
	}

	// --- recursive term: one more edge --------------------------------------------
	const auto walk_alias = path_alias + "_w";
	const auto step_edge_alias = path_alias + "_e";
	const auto step_vertex_alias = path_alias + "_iv";
	auto step = make_uniq<SelectNode>();
	{
		auto walk_ref = make_uniq<BaseTableRef>();
		walk_ref->table_name = cte_name;
		walk_ref->alias = walk_alias;
		step->from_table = std::move(walk_ref);
		// The intermediate vertex is joined, matching what the branch shape did:
		// it required each in-between vertex to exist in the vertex relation.
		CrossJoinTableRef(step->from_table,
		                  edge_table->destination_pg_table->CreateBaseTableRef(step_vertex_alias));
		CrossJoinTableRef(step->from_table, edge_table->CreateBaseTableRef(step_edge_alias));

		auto s = make_uniq<ColumnRefExpression>(src_column, walk_alias);
		s->alias = src_column;
		step->select_list.push_back(std::move(s));
		auto d = make_uniq<ColumnRefExpression>(edge_dst_fk, step_edge_alias);
		d->alias = dst_column;
		step->select_list.push_back(std::move(d));
		// Built with push_back rather than a braced list: an initializer_list
		// copies its elements, and unique_ptr is not copyable.
		vector<unique_ptr<ParsedExpression>> len_args;
		len_args.push_back(make_uniq<ColumnRefExpression>(len_column, walk_alias));
		len_args.push_back(make_uniq<ConstantExpression>(Value::BIGINT(1)));
		auto next_len = make_uniq<FunctionExpression>("+", std::move(len_args));
		next_len->alias = len_column;
		step->select_list.push_back(std::move(next_len));

		vector<unique_ptr<ParsedExpression>> step_conditions;
		step_conditions.push_back(make_uniq<ComparisonExpression>(
		    ExpressionType::COMPARE_EQUAL, make_uniq<ColumnRefExpression>(vertex_key, step_vertex_alias),
		    make_uniq<ColumnRefExpression>(dst_column, walk_alias)));
		step_conditions.push_back(make_uniq<ComparisonExpression>(
		    ExpressionType::COMPARE_EQUAL, make_uniq<ColumnRefExpression>(edge_src_fk, step_edge_alias),
		    make_uniq<ColumnRefExpression>(dst_column, walk_alias)));
		// A LABEL ON A QUANTIFIED HOP CONSTRAINS EVERY STEP.
		//
		// One condition in the recursive term is applied on every iteration, which
		// is what "every step" means here — so `-[t:transfer]->{1,3}` is three
		// transfers, not a path whose first edge happens to be one. The length-0
		// base term traverses no edge and carries no filter, which is right:
		// there is no edge for the label to describe.
		if (!edge_label_column.empty() && !edge_label.empty()) {
			step_conditions.push_back(BuildLabelFilter(step_edge_alias, edge_label_column, edge_label));
		}
		// The recursion terminates on length, not on the graph, so a cyclic graph
		// cannot spin: every iteration increases `len`, and this stops it at the
		// upper bound.
		step_conditions.push_back(make_uniq<ComparisonExpression>(
		    ExpressionType::COMPARE_LESSTHAN, make_uniq<ColumnRefExpression>(len_column, walk_alias),
		    make_uniq<ConstantExpression>(Value::BIGINT(subpath->upper))));
		step->where_clause = BuildConjunction(step_conditions);
	}

	auto recursive_node = make_uniq<RecursiveCTENode>();
	recursive_node->ctename = cte_name;
	recursive_node->union_all = true;
	recursive_node->left = std::move(base);
	recursive_node->right = std::move(step);
	recursive_node->aliases = {src_column, dst_column, len_column};

	auto cte_select = make_uniq<SelectStatement>();
	cte_select->node = std::move(recursive_node);

	auto cte_info = make_uniq<CommonTableExpressionInfo>();
	cte_info->aliases = {src_column, dst_column, len_column};
	cte_info->query = std::move(cte_select);

	// --- the endpoint projection over the walk, bounded by length ----------------
	auto endpoint_node = make_uniq<SelectNode>();
	{
		auto walk_ref = make_uniq<BaseTableRef>();
		walk_ref->table_name = cte_name;
		endpoint_node->from_table = std::move(walk_ref);
		auto s = make_uniq<ColumnRefExpression>(src_column);
		s->alias = src_column;
		endpoint_node->select_list.push_back(std::move(s));
		auto d = make_uniq<ColumnRefExpression>(dst_column);
		d->alias = dst_column;
		endpoint_node->select_list.push_back(std::move(d));

		vector<unique_ptr<ParsedExpression>> bounds;
		bounds.push_back(make_uniq<ComparisonExpression>(
		    ExpressionType::COMPARE_GREATERTHANOREQUALTO, make_uniq<ColumnRefExpression>(len_column),
		    make_uniq<ConstantExpression>(Value::BIGINT(subpath->lower))));
		bounds.push_back(make_uniq<ComparisonExpression>(
		    ExpressionType::COMPARE_LESSTHANOREQUALTO, make_uniq<ColumnRefExpression>(len_column),
		    make_uniq<ConstantExpression>(Value::BIGINT(subpath->upper))));
		endpoint_node->where_clause = BuildConjunction(bounds);
	}
	endpoint_node->cte_map.map[cte_name] = std::move(cte_info);

	auto endpoint_select = make_uniq<SelectStatement>();
	endpoint_select->node = std::move(endpoint_node);
	auto endpoint_subquery = make_uniq<SubqueryRef>(std::move(endpoint_select));
	endpoint_subquery->alias = path_alias;
	CrossJoinTableRef(from_clause, std::move(endpoint_subquery));

	conditions.push_back(CreateMatchJoinExpression(edge_table->source_pk, vector<string> {src_column},
	                                              prev_binding, path_alias));
	conditions.push_back(CreateMatchJoinExpression(edge_table->destination_pk, vector<string> {dst_column},
	                                              next_binding, path_alias));
}

//! The label as WRITTEN, or empty when the element carried none.
//!
//! An unlabelled element is given [PATTERN_UNLABELLED_VERTEX] or
//! [PATTERN_UNLABELLED_EDGE] by the parser so the
//! synthesized property graph can resolve it. Those reserved names must never
//! reach a row filter, or every unlabelled pattern would compare its label
//! column against a string no row holds and return nothing.
static string WrittenLabel(const PathElement *element) {
	if (!element || element->label == PATTERN_UNLABELLED_VERTEX || element->label == PATTERN_UNLABELLED_EDGE) {
		return string();
	}
	return element->label;
}

void GraphMatchFunction::ProcessPathList(vector<unique_ptr<PathReference>> &path_list,
                                        vector<unique_ptr<ParsedExpression>> &conditions,
                                        unique_ptr<SelectNode> &final_select_node,
                                        case_insensitive_map_t<shared_ptr<PropertyGraphTable>> &alias_map,
                                        CreatePropertyGraphInfo &pg_table, int32_t &extra_alias_counter,
                                        MatchExpression &original_ref, const string &vertex_label_column,
                                        const string &edge_label_column) {
	// A VARIABLE NAMES ONE ELEMENT. Reusing an edge variable was accepted and
	// silently answered a different question.
	//
	// Both hops of `(a)-[e]->(b)-[e]->(c)` emitted joins against the same alias,
	// so the conditions became `b.id = e.src AND b.id = e.dst` and the answer
	// collapsed to self-loops only: measured, one row on a graph where distinct
	// variables give four. With labels it produces contradictory filters on one
	// alias and returns nothing. SQL/PGQ requires distinct element variables in a
	// path pattern, and DFL's own lowering refuses this for the same reason, so
	// the two agree rather than one silently answering.
	//
	// Collected before any join is emitted so the message names the cause rather
	// than a consequence ("Table \"e\" does not have a column named ...").
	{
		case_insensitive_set_t vertex_names;
		case_insensitive_set_t edge_names;
		for (idx_t i = 0; i < path_list.size(); i++) {
			auto *element = GetPathElement(path_list[i]);
			if (!element) {
				auto *sub = GetSubPath(path_list[i]);
				if (!sub || sub->path_list.empty()) {
					continue;
				}
				element = GetPathElement(sub->path_list[0]);
				if (!element) {
					continue;
				}
			}
			if (element->variable_binding.empty()) {
				continue;
			}
			const auto is_vertex = element->match_type == PGQMatchType::MATCH_VERTEX;
			auto &own = is_vertex ? vertex_names : edge_names;
			auto &other = is_vertex ? edge_names : vertex_names;
			if (other.find(element->variable_binding) != other.end()) {
				throw BinderException("graph_match: '%s' names both a vertex and an edge in the same pattern; "
				                      "give them different names",
				                      element->variable_binding);
			}
			// A repeated VERTEX variable is a cycle and is legal; a repeated EDGE
			// variable is not, because one edge cannot be two hops.
			if (!is_vertex && !own.insert(element->variable_binding).second) {
				throw BinderException("graph_match: edge variable '%s' appears on more than one hop; one edge "
				                      "cannot be two hops, so give each hop its own name",
				                      element->variable_binding);
			}
			if (is_vertex) {
				own.insert(element->variable_binding);
			}
		}
	}

	PathElement *previous_vertex_element = GetPathElement(path_list[0]);
	if (!previous_vertex_element) {
		throw NotImplementedException("graph_match: a path may not begin with a quantified subpath");
	}
	auto previous_vertex_table = ResolveByRole(/*is_vertex=*/true, pg_table);
	alias_map[previous_vertex_element->variable_binding] = previous_vertex_table;
	if (!vertex_label_column.empty() && !WrittenLabel(previous_vertex_element).empty()) {
		conditions.push_back(BuildLabelFilter(previous_vertex_element->variable_binding, vertex_label_column,
		                                      WrittenLabel(previous_vertex_element)));
	}

	for (idx_t idx_j = 1; idx_j < path_list.size(); idx_j = idx_j + 2) {
		PathElement *next_vertex_element = GetPathElement(path_list[idx_j + 1]);
		if (!next_vertex_element) {
			throw NotImplementedException("graph_match: a quantified subpath may not appear on a vertex");
		}
		if (next_vertex_element->match_type != PGQMatchType::MATCH_VERTEX ||
		    previous_vertex_element->match_type != PGQMatchType::MATCH_VERTEX) {
			throw BinderException("Vertex and edge patterns must be alternated.");
		}
		auto next_vertex_table = ResolveByRole(/*is_vertex=*/true, pg_table);
		alias_map[next_vertex_element->variable_binding] = next_vertex_table;
		if (!vertex_label_column.empty() && !WrittenLabel(next_vertex_element).empty()) {
			conditions.push_back(BuildLabelFilter(next_vertex_element->variable_binding, vertex_label_column,
			                                      WrittenLabel(next_vertex_element)));
		}

		PathElement *edge_element = GetPathElement(path_list[idx_j]);
		if (!edge_element) {
			// Quantified (variable-length) edge — a SubPath wrapping one edge element.
			auto edge_subpath = reinterpret_cast<SubPath *>(path_list[idx_j].get());
			if (edge_subpath->path_list.size() > 1) {
				throw NotImplementedException("graph_match: nested subpaths are not supported");
			}
			edge_element = GetPathElement(edge_subpath->path_list[0]);
			auto edge_table = ResolveByRole(/*is_vertex=*/false, pg_table);

			if (BoundNeedsExpansion(*edge_subpath)) {
				if (CanExpandBoundedSubpath(edge_table, edge_subpath, edge_element->match_type)) {
					AddBoundedPathExpansion(edge_table, edge_subpath, previous_vertex_element->variable_binding,
					                        edge_element->variable_binding, next_vertex_element->variable_binding,
					                        conditions, final_select_node->from_table, extra_alias_counter,
					                        WrittenLabel(edge_element), edge_label_column);
				} else {
					// TODO: wire the CSR/shortestpath route (AddPathFinding) for
					// left/undirected or large-bound variable-length edges. The CSR
					// scalar functions (create_csr_*, iterativelength, shortestpath) are
					// registered, so this is a contained follow-up; for now only
					// right-directed, bounded (upper <= 16) var-length is lowered.
					throw NotImplementedException(
					    "graph_match: variable-length edges are only supported as right-directed with bounded "
					    "upper limit <= %lld (use -[e:E]->{lo,hi})",
					    static_cast<long long>(MAX_BOUNDED_PATH_EXPANSION_UPPER));
				}
			} else {
				AddEdgeJoins(edge_table, previous_vertex_table, next_vertex_table, edge_element->match_type,
				             edge_element->variable_binding, previous_vertex_element->variable_binding,
				             next_vertex_element->variable_binding, conditions, alias_map, extra_alias_counter,
				             final_select_node->from_table);
				if (!edge_label_column.empty() && !WrittenLabel(edge_element).empty()) {
					conditions.push_back(BuildLabelFilter(edge_element->variable_binding, edge_label_column,
					                                      WrittenLabel(edge_element)));
				}
			}
		} else {
			auto edge_table = ResolveByRole(/*is_vertex=*/false, pg_table);
			AddEdgeJoins(edge_table, previous_vertex_table, next_vertex_table, edge_element->match_type,
			             edge_element->variable_binding, previous_vertex_element->variable_binding,
			             next_vertex_element->variable_binding, conditions, alias_map, extra_alias_counter,
			             final_select_node->from_table);
			if (!edge_label_column.empty() && !WrittenLabel(edge_element).empty()) {
				conditions.push_back(BuildLabelFilter(edge_element->variable_binding, edge_label_column,
				                                      WrittenLabel(edge_element)));
			}
		}
		previous_vertex_element = next_vertex_element;
		previous_vertex_table = next_vertex_table;
	}
	(void)original_ref;
}

//------------------------------------------------------------------------------
// BindReplace: synthesize the property graph from args, parse, rewrite.
//------------------------------------------------------------------------------
unique_ptr<TableRef> GraphMatchFunction::GraphMatchBindReplace(ClientContext &context,
                                                              TableFunctionBindInput &input) {
	// graph_match(pattern, vertex_table, vertex_id, edge_table, src, dst)
	auto pattern = StringValue::Get(input.inputs[0]);
	auto vertex_table = StringUtil::Lower(StringValue::Get(input.inputs[1]));
	auto vertex_id = StringUtil::Lower(StringValue::Get(input.inputs[2]));
	auto edge_table = StringUtil::Lower(StringValue::Get(input.inputs[3]));
	auto src_col = StringUtil::Lower(StringValue::Get(input.inputs[4]));
	auto dst_col = StringUtil::Lower(StringValue::Get(input.inputs[5]));
	(void)context;

	// The columns a written label is compared against. Named parameters, so a
	// caller that writes no labels needs no change, and a caller that does write
	// them says where they live rather than having them quietly ignored.
	string vertex_label_column;
	string edge_label_column;
	for (auto &named : input.named_parameters) {
		const auto is_vertex = StringUtil::CIEquals(named.first, "vertex_label_column");
		const auto is_edge = StringUtil::CIEquals(named.first, "edge_label_column");
		if (!is_vertex && !is_edge) {
			continue;
		}
		// NULL IS A USER ERROR, NOT AN INTERNAL ONE.
		//
		// `StringValue::Get` on a NULL throws InternalException, which prints a
		// stack trace and asks the user to file a bug — and in a build configured
		// with CRASH_ON_ASSERT (which is how DuckDB's own CI builds) it calls
		// abort() and takes the process down. A value the caller typed must not be
		// able to do that.
		if (named.second.IsNull()) {
			throw BinderException("graph_match: %s must be a column name, not NULL", named.first);
		}
		auto value = StringUtil::Lower(StringValue::Get(named.second));
		// EMPTY IS NOT ABSENT. Every test below is `column.empty()`, so an empty
		// string silently selected the relation-name reading and then reported
		// "no vertex_label_column was given" to a caller who had given one.
		if (value.empty()) {
			throw BinderException("graph_match: %s was given as an empty string; pass a column name, or omit it",
			                      named.first);
		}
		if (is_vertex) {
			vertex_label_column = std::move(value);
		} else {
			edge_label_column = std::move(value);
		}
	}

	// 1. Parse the pattern string into a MatchExpression + the distinct labels.
	vector<string> vertex_labels;
	vector<string> edge_labels;
	vector<ParsedPatternLabel> label_refs;
	auto match_expr = ParseGraphPattern(pattern, vertex_labels, edge_labels, label_refs);

	// A WRITTEN LABEL MUST MEAN SOMETHING. IT USED TO MEAN NOTHING.
	//
	// graph_match is handed one vertex table and one edge table, so a label never
	// selects BETWEEN relations here. Previously that made every label
	// self-fulfilling: it was used to look up the element's table, the lookup was
	// keyed by the labels just parsed so it always succeeded, and no filter was
	// ever emitted. Measured on a two-class fixture, `(a:zzz)-[x:qqq]->(b:zzz)` —
	// labels present nowhere in the data — returned every edge in the graph.
	//
	// There are two honest readings of a label here, and which one applies depends
	// on whether the caller said where labels live:
	//
	//   * A label COLUMN was named. The label is a row-level class, so it becomes
	//     a filter (see ProcessPathList and AddBoundedPathExpansion). This is the
	//     property-graph reading, and what a VKG caller wants.
	//   * No label column was named. Then the only label this relation has is the
	//     relation itself, so the label must NAME it — `(a:Student)` over the
	//     `Student` vertex table. Anything else is refused rather than ignored.
	//
	// The second reading is what keeps `(a:Student)-[e:know]->(b:Student)` over
	// tables `Student` / `know` working without a label column, while still
	// refusing `(a:zzz)`.
	// THE READING IS A PROPERTY OF THE CALL, NOT OF EACH ELEMENT KIND.
	//
	// This loop used to decide per element: a label whose own column was absent
	// fell back to the relation-name rule independently of the other kind. In the
	// ASYMMETRIC mode — one column given, the other not — that reintroduced
	// exactly the silent ignore this whole change exists to remove. Measured on a
	// three-person fixture with `knows` and `hates` edges in a table named
	// `knows`:
	//
	//   graph_match('(a:person)-[e:knows]->(b:person)', 'person','id','knows',...,
	//               vertex_label_column = 'kind')
	//     -> 3 rows, two of them `hates` edges
	//   ...with edge_label_column = 'kind' as well
	//     -> 1 row
	//
	// Three constraints written, two applied, no diagnostic. The edge label
	// `knows` happened to name the edge relation, so it was accepted under the
	// other reading and no filter was emitted. The same switch fires in reverse:
	// `(a)-[e:knows]->(b)` over a table named `knows` answers 5 rows without
	// `edge_label_column` and 2 with it — one pattern, two answers, both quiet.
	//
	// So once EITHER column is supplied, labels are row values for the whole
	// call, and a written label of the other kind without its column is refused
	// rather than reinterpreted.
	const auto any_label_column = !vertex_label_column.empty() || !edge_label_column.empty();
	for (auto &ref : label_refs) {
		const auto &column = ref.is_vertex ? vertex_label_column : edge_label_column;
		if (!column.empty()) {
			continue;
		}
		const auto *own_param = ref.is_vertex ? "vertex_label_column" : "edge_label_column";
		const auto *other_param = ref.is_vertex ? "edge_label_column" : "vertex_label_column";
		const auto *kind = ref.is_vertex ? "vertex" : "edge";
		if (any_label_column) {
			throw BinderException(
			    "graph_match: the pattern writes the %s label '%s' on '%s', and %s was given, so labels are read "
			    "as row values for this call — but no %s was given, and this label would be silently dropped. "
			    "Pass %s := '<column>', or remove the label.",
			    kind, ref.label, ref.binding, other_param, own_param, own_param);
		}
		const auto &relation = ref.is_vertex ? vertex_table : edge_table;
		if (StringUtil::CIEquals(ref.label, relation)) {
			continue;
		}
		throw BinderException(
		    "graph_match: the pattern writes the %s label '%s' on '%s', but no %s was given and '%s' is not the "
		    "name of the %s relation ('%s'). Either pass %s := '<column>' — the column holding each row's label, "
		    "so the label can filter rows — or label the element after the relation it comes from.",
		    kind, ref.label, ref.binding, own_param, ref.label, kind, relation, own_param);
	}

	// 2. Synthesize a CreatePropertyGraphInfo: one vertex table + one edge table,
	//    populated exactly like MakeEdgeSpec, keyed by the parsed labels so
	//    FindGraphTable resolves. Source and destination vertices are the same
	//    vertex table; the CSR/join builders disambiguate via the bindings.
	// `vertex_id` MUST BE UNIQUE PER ROW. Nothing here can check it cheaply, so it
	// is stated instead. A duplicate id multiplies rows: it duplicates the
	// endpoints of every match, and inside a quantified hop it multiplies again at
	// each intermediate vertex. The old outer DISTINCT absorbed both effects, so
	// this is a consequence of returning one row per walk. A denormalised source
	// table or a view over one is the easy way to hit it.
	auto edge_spec = MakeEdgeSpec(vertex_table, vertex_id, edge_table, src_col, dst_col);
	edge_spec->main_label = edge_labels.empty() ? string(PATTERN_UNLABELLED_EDGE) : edge_labels[0];

	auto vertex_spec = make_shared_ptr<PropertyGraphTable>();
	vertex_spec->table_name = vertex_table;
	vertex_spec->is_vertex_table = true;
	vertex_spec->main_label = vertex_labels.empty() ? string(PATTERN_UNLABELLED_VERTEX) : vertex_labels[0];
	vertex_spec->source_pk = {vertex_id};
	vertex_spec->destination_pk = {vertex_id};

	CreatePropertyGraphInfo pg_info("__graph_match");
	pg_info.vertex_tables.push_back(vertex_spec);
	pg_info.edge_tables.push_back(edge_spec);
	// EVERY label in the pattern maps to the one table of its kind.
	//
	// There is exactly one vertex table and one edge table behind graph_match, so
	// a label never selects BETWEEN tables here — it selects rows, which the
	// filters in ProcessPathList now do. Registering every parsed label is what
	// lifts the old "exactly one vertex label and one edge label per call"
	// refusal: a cross-class pattern such as
	// `(o:Order)-[e:placedBy]->(c:Customer)` is two vertex labels over one vertex
	// relation, which Oracle answers and this refused outright.
	// The label map is populated for the ported rewrite's benefit, but elements
	// are NOT resolved through it — see `ResolveByRole`. A name used as both a
	// vertex and an edge label therefore collides here harmlessly: whichever entry
	// wins is never consulted.
	for (auto &label : vertex_labels) {
		pg_info.label_map[label] = vertex_spec;
	}
	for (auto &label : edge_labels) {
		pg_info.label_map[label] = edge_spec;
	}

	// 3. Run the ported rewrite.
	vector<unique_ptr<ParsedExpression>> conditions;
	auto final_select_node = make_uniq<SelectNode>();
	case_insensitive_map_t<shared_ptr<PropertyGraphTable>> alias_map;
	int32_t extra_alias_counter = 0;

	for (auto &path_pattern : match_expr->path_patterns) {
		ProcessPathList(path_pattern->path_elements, conditions, final_select_node, alias_map, pg_info,
		                extra_alias_counter, *match_expr, vertex_label_column, edge_label_column);
	}

	// Cross-join all the vertex/edge relations encountered.
	for (auto &table_alias_entry : alias_map) {
		auto table_ref = table_alias_entry.second->CreateBaseTableRef();
		table_ref->alias = table_alias_entry.first;
		if (final_select_node->from_table) {
			auto new_root = make_uniq<JoinRef>(JoinRefType::CROSS);
			new_root->left = std::move(final_select_node->from_table);
			new_root->right = std::move(table_ref);
			final_select_node->from_table = std::move(new_root);
		} else {
			final_select_node->from_table = std::move(table_ref);
		}
	}

	// Projection. Because the result is wrapped in a SubqueryRef, the inner
	// relation aliases (a, b, e, ...) are not visible to the outer query, so we
	// expose a stable, collision-free set of output columns named
	// "<binding>_<column>" for the key columns of every bound relation:
	//   - vertex binding  -> "<binding>_<vertex_id>"
	//   - edge binding    -> "<binding>_<src>" and "<binding>_<dst>"
	// (Undirected/ANY edges are not bound and therefore not projected.) This is
	// the contract DFL codegen selects from. Ordered by binding name for
	// determinism.
	vector<string> ordered_bindings;
	ordered_bindings.reserve(alias_map.size());
	for (auto &entry : alias_map) {
		ordered_bindings.push_back(entry.first);
	}
	std::sort(ordered_bindings.begin(), ordered_bindings.end());
	for (auto &binding : ordered_bindings) {
		const auto &tbl = alias_map[binding];
		if (tbl->is_vertex_table) {
			auto colref = make_uniq<ColumnRefExpression>(tbl->source_pk[0], binding);
			colref->alias = binding + "_" + tbl->source_pk[0];
			final_select_node->select_list.push_back(std::move(colref));
		} else {
			auto src_ref = make_uniq<ColumnRefExpression>(tbl->source_fk[0], binding);
			src_ref->alias = binding + "_" + tbl->source_fk[0];
			final_select_node->select_list.push_back(std::move(src_ref));
			auto dst_ref = make_uniq<ColumnRefExpression>(tbl->destination_fk[0], binding);
			dst_ref->alias = binding + "_" + tbl->destination_fk[0];
			final_select_node->select_list.push_back(std::move(dst_ref));
		}
	}
	if (final_select_node->select_list.empty()) {
		final_select_node->select_list.push_back(make_uniq<StarExpression>());
	}
	final_select_node->where_clause = CreateWhereClause(conditions);

	auto subquery = make_uniq<SelectStatement>();
	subquery->node = std::move(final_select_node);
	auto result = make_uniq<SubqueryRef>(std::move(subquery), "graph_match");
	return std::move(result);
}

//------------------------------------------------------------------------------
// Register
//------------------------------------------------------------------------------
void CoreTableFunctions::RegisterGraphMatchTableFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(GraphMatchFunction());
}

} // namespace duckdb
