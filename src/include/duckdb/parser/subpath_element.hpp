//===----------------------------------------------------------------------===//
//                         DuckPGQ (stock-DuckDB build)
//
// duckdb/parser/subpath_element.hpp
//
// Vendored, lean copy of the fork's SubPath AST node (a parenthesized/quantified
// sub-pattern, e.g. variable-length `-[e:E]->{1,3}`). Only fields the
// match-rewrite reads are kept; Serialize/Deserialize/Equals/ToString omitted.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/parser/path_reference.hpp"
#include "duckdb/parser/parsed_expression.hpp"

namespace duckdb {

enum class PGQPathMode : uint8_t { NONE, WALK, SIMPLE, TRAIL, ACYCLIC };

class SubPath : public PathReference {
public:
	vector<unique_ptr<PathReference>> path_list;

	unique_ptr<ParsedExpression> where_clause;

	PGQPathMode path_mode;

	bool single_bind;
	int64_t lower, upper;

	string path_variable;

public:
	explicit SubPath(PGQPathReferenceType path_reference_type = PGQPathReferenceType::SUBPATH)
	    : PathReference(path_reference_type) {
		path_mode = PGQPathMode::NONE;
		single_bind = true;
		lower = 1;
		upper = 1;
	}

	unique_ptr<PathReference> Copy() override {
		auto result = make_uniq<SubPath>(path_reference_type);
		for (auto &child : path_list) {
			result->path_list.push_back(child->Copy());
		}
		if (where_clause) {
			result->where_clause = where_clause->Copy();
		}
		result->path_mode = path_mode;
		result->single_bind = single_bind;
		result->lower = lower;
		result->upper = upper;
		result->path_variable = path_variable;
		return std::move(result);
	}
};

} // namespace duckdb
