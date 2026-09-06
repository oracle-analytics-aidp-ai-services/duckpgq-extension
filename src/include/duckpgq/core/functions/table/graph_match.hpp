//===----------------------------------------------------------------------===//
//                         DuckPGQ (stock-DuckDB build)
//
// duckpgq/core/functions/table/graph_match.hpp
//
// graph_match(pattern, vertex_table, vertex_id, edge_table, src, dst,
//             vertex_label_column := ..., edge_label_column := ...):
// a table function that takes a PGQ pattern STRING + table-mapping args, parses
// the pattern (pattern_parser), synthesizes a CreatePropertyGraphInfo from the
// args + parsed labels, then runs the ported match-rewrite to lower the pattern
// to relational joins — returning the matched rows.
//
// The rewrite logic is ported (essentially unchanged) from the fork's
// extension/duckpgq/src/core/functions/table/match.cpp; only MatchBindReplace is
// adapted to read function args + a synthesized property graph instead of
// reading a registered property graph out of DuckPGQState.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckpgq/common.hpp"

#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_property_graph_info.hpp"
#include "duckdb/parser/path_element.hpp"
#include "duckdb/parser/path_pattern.hpp"
#include "duckdb/parser/subpath_element.hpp"
#include "duckdb/parser/property_graph_table.hpp"
#include "duckdb/parser/tableref/matchref.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"

namespace duckdb {

struct GraphMatchFunction : public TableFunction {
public:
	GraphMatchFunction() {
		name = "graph_match";
		// graph_match(pattern, vertex_table, vertex_id, edge_table, src, dst)
		arguments = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
		             LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
		// Optional: the columns a pattern label is compared against. Named rather
		// than positional so a caller that writes no labels is unaffected.
		//
		// A pattern that writes a label without supplying the matching column is
		// refused UNLESS the label names the relation it comes from — see
		// GraphMatchBindReplace, which spells out both readings. The earlier
		// version of this comment omitted the exemption and so contradicted the
		// implementation and the first test in the suite, which relies on it.
		named_parameters["vertex_label_column"] = LogicalType::VARCHAR;
		named_parameters["edge_label_column"] = LogicalType::VARCHAR;
		bind_replace = GraphMatchBindReplace;
	}

	static unique_ptr<TableRef> GraphMatchBindReplace(ClientContext &context, TableFunctionBindInput &input);

	// --- Ported rewrite (see match.cpp provenance in the .cpp) ---
	static shared_ptr<PropertyGraphTable> FindGraphTable(const string &label, CreatePropertyGraphInfo &pg_table);

	//! The relation an element belongs to, from its KIND rather than its label.
	//! There is one vertex relation and one edge relation, so the label never
	//! selected between them; resolving by label merely made one name unusable as
	//! both a vertex and an edge label.
	static shared_ptr<PropertyGraphTable> ResolveByRole(bool is_vertex, CreatePropertyGraphInfo &pg_table);

	//! Does this written bound need the recursive expansion rather than a plain
	//! single-edge join? The one place this is decided.
	static bool BoundNeedsExpansion(const SubPath &subpath);

	static unique_ptr<ParsedExpression> CreateMatchJoinExpression(vector<string> vertex_keys, vector<string> edge_keys,
	                                                              const string &vertex_alias, const string &edge_alias);

	static PathElement *GetPathElement(const unique_ptr<PathReference> &path_reference);
	static SubPath *GetSubPath(const unique_ptr<PathReference> &path_reference);

	static unique_ptr<ParsedExpression> CreateWhereClause(vector<unique_ptr<ParsedExpression>> &conditions);

	static void EdgeTypeAny(const shared_ptr<PropertyGraphTable> &edge_table, const string &edge_binding,
	                        const string &prev_binding, const string &next_binding,
	                        vector<unique_ptr<ParsedExpression>> &conditions, unique_ptr<TableRef> &from_clause);
	static void EdgeTypeLeft(const shared_ptr<PropertyGraphTable> &edge_table, const string &next_table_name,
	                         const string &prev_table_name, const string &edge_binding, const string &prev_binding,
	                         const string &next_binding, vector<unique_ptr<ParsedExpression>> &conditions);
	static void EdgeTypeRight(const shared_ptr<PropertyGraphTable> &edge_table, const string &next_table_name,
	                          const string &prev_table_name, const string &edge_binding, const string &prev_binding,
	                          const string &next_binding, vector<unique_ptr<ParsedExpression>> &conditions);
	static void EdgeTypeLeftRight(const shared_ptr<PropertyGraphTable> &edge_table, const string &edge_binding,
	                              const string &prev_binding, const string &next_binding,
	                              vector<unique_ptr<ParsedExpression>> &conditions,
	                              case_insensitive_map_t<shared_ptr<PropertyGraphTable>> &alias_map,
	                              int32_t &extra_alias_counter);

	static void CheckEdgeTableConstraints(const string &src_reference, const string &dst_reference,
	                                       const shared_ptr<PropertyGraphTable> &edge_table);

	static void AddEdgeJoins(const shared_ptr<PropertyGraphTable> &edge_table,
	                         const shared_ptr<PropertyGraphTable> &previous_vertex_table,
	                         const shared_ptr<PropertyGraphTable> &next_vertex_table, PGQMatchType edge_type,
	                         const string &edge_binding, const string &prev_binding, const string &next_binding,
	                         vector<unique_ptr<ParsedExpression>> &conditions,
	                         case_insensitive_map_t<shared_ptr<PropertyGraphTable>> &alias_map,
	                         int32_t &extra_alias_counter, unique_ptr<TableRef> &from_clause);

	static bool CanExpandBoundedSubpath(const shared_ptr<PropertyGraphTable> &edge_table, SubPath *subpath,
	                                     PGQMatchType edge_type);
	//! [edge_label] is the label written on the quantified hop, or empty. It is
	//! applied to EVERY step of the expansion, because a quantified edge is not a
	//! relation in the outer query and so cannot be filtered from outside.
	static void AddBoundedPathExpansion(const shared_ptr<PropertyGraphTable> &edge_table, SubPath *subpath,
	                                    const string &prev_binding, const string &edge_binding,
	                                    const string &next_binding, vector<unique_ptr<ParsedExpression>> &conditions,
	                                    unique_ptr<TableRef> &from_clause, int32_t &extra_alias_counter,
	                                    const string &edge_label, const string &edge_label_column);

	static void ProcessPathList(vector<unique_ptr<PathReference>> &path_list,
	                            vector<unique_ptr<ParsedExpression>> &conditions, unique_ptr<SelectNode> &select_node,
	                            case_insensitive_map_t<shared_ptr<PropertyGraphTable>> &alias_map,
	                            CreatePropertyGraphInfo &pg_table, int32_t &extra_alias_counter,
	                            MatchExpression &original_ref, const string &vertex_label_column,
	                            const string &edge_label_column);
};

} // namespace duckdb
