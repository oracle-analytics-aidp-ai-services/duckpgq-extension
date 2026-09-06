//===----------------------------------------------------------------------===//
//                         DuckPGQ (stock-DuckDB build)
//
// duckpgq/core/parser/pattern_parser.hpp
//
// A small, bounded recursive-descent parser for the DFL-emitted PGQ pattern
// subset. The full SQL/PGQ grammar lives in libpg_query's bison and cannot run
// on stock DuckDB, so graph_match takes the pattern as a plain string and this
// parser turns it into the (vendored) MatchExpression AST that the ported
// match-rewrite consumes.
//
// Supported grammar (whitespace-insensitive):
//   pattern   := node (edge node)*
//   node      := '(' var (':' Label)? ')'
//   edge      := '-' '[' var (':' Label)? ']' '->' '{' lo ',' hi '}'?     (right)
//             |  '<-' '[' var (':' Label)? ']' '-'                         (left)
//             |  '-' '[' var (':' Label)? ']' '-'                         (any/undirected)
//   var/Label := [A-Za-z_][A-Za-z0-9_]*
// Variable-length quantifier '{lo,hi}' is only accepted on a right edge.
// Anything else raises a specific ParserException.
//
// A LABEL IS OPTIONAL, on a vertex and on an edge alike. It used to be required,
// which made an unlabelled pattern such as `(a)-[t]->(b)` a parse error even
// though SQL/PGQ and Oracle both accept one. An element with no written label is given the reserved internal
// label below so the synthesized property graph still resolves it, and nothing
// is filtered for it. See [ParsedPatternLabel] for how a WRITTEN label is
// carried out to the caller, which is what makes it filter rows instead of
// being silently ignored.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckpgq/common.hpp"
#include "duckdb/parser/tableref/matchref.hpp"

namespace duckdb {

//! The reserved labels given to a pattern element that carries no written label.
//!
//! TWO of them, one per element kind. This mattered when elements were resolved
//! through a single `label_map`: one shared reserved name meant the edge
//! registration overwrote the vertex one, and an unlabelled `(a)-[e]->(b)`
//! resolved `a` to the EDGE relation, so a column the vertex relation had and the
//! edge relation did not could not be found. Elements now resolve by ROLE
//! (`GraphMatchFunction::ResolveByRole`), so the two names are no longer
//! load-bearing — they are kept because one name for two kinds is confusing to
//! read, not because a collision would break anything.
//!
//! Chosen so no plausible user label collides with them; a pattern that writes
//! one is rejected.
extern const char *const PATTERN_UNLABELLED_VERTEX;
extern const char *const PATTERN_UNLABELLED_EDGE;

//! One label as WRITTEN in the pattern, tied to the variable it was written on.
//!
//! The distinct-label vectors alone were not enough to make a label mean
//! anything: they say WHICH labels appeared, not which variable each belongs to,
//! so the caller could resolve the element's table but could not emit a row
//! filter. Without that filter a written label was accepted and then silently
//! ignored, and `(a:Person)-[x:knows]->(b:Person)` returned every edge in the
//! graph, including edges of other types between vertices of other classes.
struct ParsedPatternLabel {
	//! The pattern variable the label was written on (`a` in `(a:Person)`).
	string binding;
	//! The label exactly as written.
	string label;
	//! Vertex element, as opposed to an edge element.
	bool is_vertex;
};

//! Parse a single PGQ path pattern string into a MatchExpression. The set of
//! distinct labels encountered (vertex + edge) is returned in [out_*_labels] so
//! the caller can synthesize the matching CreatePropertyGraphInfo, and every
//! WRITTEN label is returned in [out_label_refs] tied to its variable so the
//! caller can emit the row filter that makes the label mean something. Throws
//! ParserException with a specific message on unsupported / malformed input.
unique_ptr<MatchExpression> ParseGraphPattern(const string &pattern, vector<string> &out_vertex_labels,
                                              vector<string> &out_edge_labels,
                                              vector<ParsedPatternLabel> &out_label_refs);

} // namespace duckdb
