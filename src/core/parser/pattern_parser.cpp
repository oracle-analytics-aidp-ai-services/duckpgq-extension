//===----------------------------------------------------------------------===//
//                         DuckPGQ (stock-DuckDB build)
//
// src/core/parser/pattern_parser.cpp
//
// Bounded recursive-descent parser for the DFL-emitted PGQ pattern subset.
// See pattern_parser.hpp for the supported grammar. The grammar is intentionally
// tiny: it builds the vendored MatchExpression AST directly so the ported
// match-rewrite (graph_match.cpp) can run without libpg_query's PGQ bison.
//===----------------------------------------------------------------------===//

#include "duckpgq/core/parser/pattern_parser.hpp"

#include "duckdb/common/string_util.hpp"

#include "duckdb/common/exception/parser_exception.hpp"
#include "duckdb/parser/path_element.hpp"
#include "duckdb/parser/subpath_element.hpp"
#include "duckdb/parser/path_pattern.hpp"

#include <set>

namespace duckdb {

namespace {

class PatternParser {
public:
	explicit PatternParser(const string &pattern) : src(pattern), pos(0) {
	}

	unique_ptr<MatchExpression> Parse(vector<string> &out_vertex_labels, vector<string> &out_edge_labels,
	                                  vector<ParsedPatternLabel> &out_label_refs) {
		auto match_expr = make_uniq<MatchExpression>();
		auto path_pattern = make_uniq<PathPattern>();
		label_refs = &out_label_refs;

		// pattern := node (edge node)*
		path_pattern->path_elements.push_back(ParseNode(out_vertex_labels));
		SkipWhitespace();
		while (!AtEnd()) {
			path_pattern->path_elements.push_back(ParseEdge(out_edge_labels));
			path_pattern->path_elements.push_back(ParseNode(out_vertex_labels));
			SkipWhitespace();
		}

		match_expr->path_patterns.push_back(std::move(path_pattern));
		return match_expr;
	}

private:
	const string &src;
	idx_t pos;
	//! Every WRITTEN label, tied to its variable, for the caller's row filters.
	vector<ParsedPatternLabel> *label_refs = nullptr;

	//! Record a written label, and refuse the reserved internal one.
	//!
	//! The reserved name is what an unlabelled element is given so the
	//! synthesized property graph can still resolve it. If an author could write
	//! it too, their element would be indistinguishable from an unlabelled one
	//! and would silently lose its filter — the exact failure this whole change
	//! removes — so writing it is an error rather than a coincidence.
	void RecordLabel(const string &binding, const string &label, bool is_vertex) {
		if (label.empty()) {
			return;
		}
		// Compared case-INSENSITIVELY, matching every other label comparison in
		// this feature (`label_map` is a case-insensitive map, and the
		// relation-name rule uses StringUtil::CIEquals). A case-sensitive test let
		// `(a:__DUCKPGQ_UNLABELLED_EDGE)` past this guard and surface later as a
		// confusing message about an unrelated rule.
		if (StringUtil::CIEquals(label, PATTERN_UNLABELLED_VERTEX) ||
		    StringUtil::CIEquals(label, PATTERN_UNLABELLED_EDGE)) {
			Fail("label '" + label + "' is reserved for internal use; choose another label");
		}
		label_refs->push_back(ParsedPatternLabel {binding, label, is_vertex});
	}

	bool AtEnd() const {
		return pos >= src.size();
	}

	char Peek() const {
		return AtEnd() ? '\0' : src[pos];
	}

	char PeekAt(idx_t offset) const {
		return (pos + offset >= src.size()) ? '\0' : src[pos + offset];
	}

	void SkipWhitespace() {
		while (!AtEnd() && std::isspace(static_cast<unsigned char>(src[pos]))) {
			pos++;
		}
	}

	[[noreturn]] void Fail(const string &what) const {
		throw ParserException("graph_match: invalid pattern '%s' near offset %llu: %s", src,
		                      static_cast<unsigned long long>(pos), what);
	}

	void Expect(char c, const string &what) {
		SkipWhitespace();
		if (Peek() != c) {
			Fail(what + " (expected '" + string(1, c) + "')");
		}
		pos++;
	}

	static bool IsIdentStart(char c) {
		return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
	}

	static bool IsIdentChar(char c) {
		return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
	}

	string ParseIdentifier(const string &what) {
		SkipWhitespace();
		if (!IsIdentStart(Peek())) {
			Fail(what);
		}
		idx_t start = pos;
		while (!AtEnd() && IsIdentChar(src[pos])) {
			pos++;
		}
		return src.substr(start, pos - start);
	}

	// Parse the inner "var (':' Label)?" of a node or edge token. Returns the
	// variable binding; sets [label] to the parsed label (empty if none).
	string ParseVarAndLabel(string &label) {
		string variable = ParseIdentifier("expected a variable name");
		label = "";
		SkipWhitespace();
		if (Peek() == ':') {
			pos++;
			label = ParseIdentifier("expected a label after ':'");
		}
		return variable;
	}

	// node := '(' var (':' Label)? ')'
	unique_ptr<PathReference> ParseNode(vector<string> &out_vertex_labels) {
		Expect('(', "expected '(' to start a vertex pattern");
		string label;
		string variable = ParseVarAndLabel(label);
		Expect(')', "expected ')' to close a vertex pattern");

		// A missing label is legal. SQL/PGQ and Oracle both accept `(a)`, and DFL
		// emits that spelling, so requiring one made a correct pattern a parse
		// error. The reserved label stands in so the synthesized property graph
		// resolves the element; no filter is emitted for it.
		RecordLabel(variable, label, /*is_vertex=*/true);
		const auto resolved_label = label.empty() ? string(PATTERN_UNLABELLED_VERTEX) : label;

		auto element = make_uniq<PathElement>(PGQPathReferenceType::PATH_ELEMENT);
		element->match_type = PGQMatchType::MATCH_VERTEX;
		element->variable_binding = variable;
		element->label = resolved_label;
		out_vertex_labels.push_back(resolved_label);
		return std::move(element);
	}

	// edge := '-[' var (':' Label)? ']->' '{lo,hi}'?    (right, optional var-length)
	//       | '<-[' var (':' Label)? ']-'               (left)
	//       | '-[' var (':' Label)? ']-'                (any / undirected)
	unique_ptr<PathReference> ParseEdge(vector<string> &out_edge_labels) {
		SkipWhitespace();
		bool left_arrow = false;
		if (Peek() == '<') {
			// '<-['
			if (PeekAt(1) != '-') {
				Fail("expected '<-' to start a left edge pattern");
			}
			left_arrow = true;
			pos += 2; // consume '<-'
		} else if (Peek() == '-') {
			pos += 1; // consume '-'
		} else {
			Fail("expected an edge pattern starting with '-' or '<-'");
		}

		Expect('[', "expected '[' to start the edge body");
		string label;
		string variable = ParseVarAndLabel(label);
		Expect(']', "expected ']' to close the edge body");

		SkipWhitespace();
		bool right_arrow = false;
		if (Peek() == '-') {
			if (PeekAt(1) == '>') {
				right_arrow = true;
				pos += 2; // consume '->'
			} else {
				pos += 1; // consume '-'
			}
		} else {
			Fail("expected '-' or '->' to close the edge pattern");
		}

		PGQMatchType match_type;
		if (left_arrow && right_arrow) {
			Fail("edge pattern cannot be both left- and right-directed");
		} else if (left_arrow) {
			match_type = PGQMatchType::MATCH_EDGE_LEFT;
		} else if (right_arrow) {
			match_type = PGQMatchType::MATCH_EDGE_RIGHT;
		} else {
			match_type = PGQMatchType::MATCH_EDGE_ANY;
		}

		// Optional variable-length quantifier '{lo,hi}'.
		int64_t lower = 1;
		int64_t upper = 1;
		bool var_length = false;
		SkipWhitespace();
		if (Peek() == '{') {
			if (match_type != PGQMatchType::MATCH_EDGE_RIGHT) {
				Fail("variable-length quantifier is only supported on a right-directed edge (-[..]->{lo,hi})");
			}
			var_length = true;
			pos++; // consume '{'
			lower = ParseInteger("expected a lower bound in '{lo,hi}'");
			Expect(',', "expected ',' in '{lo,hi}'");
			upper = ParseInteger("expected an upper bound in '{lo,hi}'");
			Expect('}', "expected '}' to close '{lo,hi}'");
			if (lower < 0 || upper < lower) {
				Fail("variable-length bounds must satisfy 0 <= lo <= hi");
			}
		}

		RecordLabel(variable, label, /*is_vertex=*/false);
		const auto resolved_label = label.empty() ? string(PATTERN_UNLABELLED_EDGE) : label;
		out_edge_labels.push_back(resolved_label);

		auto element = make_uniq<PathElement>(PGQPathReferenceType::PATH_ELEMENT);
		element->match_type = match_type;
		element->variable_binding = variable;
		element->label = resolved_label;

		if (!var_length) {
			return std::move(element);
		}

		// Wrap a variable-length edge in a SubPath carrying the {lo,hi} bounds,
		// matching the shape ProcessPathList expects for path-finding.
		auto subpath = make_uniq<SubPath>(PGQPathReferenceType::SUBPATH);
		subpath->lower = lower;
		subpath->upper = upper;
		subpath->single_bind = false;
		subpath->path_list.push_back(std::move(element));
		return std::move(subpath);
	}

	int64_t ParseInteger(const string &what) {
		SkipWhitespace();
		if (!std::isdigit(static_cast<unsigned char>(Peek()))) {
			Fail(what);
		}
		idx_t start = pos;
		while (!AtEnd() && std::isdigit(static_cast<unsigned char>(src[pos]))) {
			pos++;
		}
		// `stoll` throws `std::out_of_range` on a number too large for int64, which
		// escapes as `Invalid Error: stoll` and contradicts this parser's promise
		// that anything malformed raises a specific ParserException. Caught so the
		// bound gets the same treatment as every other bad input.
		const auto digits = src.substr(start, pos - start);
		try {
			return std::stoll(digits);
		} catch (const std::out_of_range &) {
			Fail("quantifier bound '" + digits + "' is too large");
		} catch (const std::invalid_argument &) {
			Fail("quantifier bound '" + digits + "' is not a number");
		}
	}
};

// Deduplicate while preserving order.
void Unique(vector<string> &labels) {
	std::set<string> seen;
	vector<string> result;
	for (auto &label : labels) {
		if (seen.insert(label).second) {
			result.push_back(label);
		}
	}
	labels = std::move(result);
}

} // namespace

const char *const PATTERN_UNLABELLED_VERTEX = "__duckpgq_unlabelled_vertex";
const char *const PATTERN_UNLABELLED_EDGE = "__duckpgq_unlabelled_edge";

unique_ptr<MatchExpression> ParseGraphPattern(const string &pattern, vector<string> &out_vertex_labels,
                                              vector<string> &out_edge_labels,
                                              vector<ParsedPatternLabel> &out_label_refs) {
	out_vertex_labels.clear();
	out_edge_labels.clear();
	out_label_refs.clear();
	PatternParser parser(pattern);
	auto result = parser.Parse(out_vertex_labels, out_edge_labels, out_label_refs);
	Unique(out_vertex_labels);
	Unique(out_edge_labels);
	return result;
}

} // namespace duckdb
