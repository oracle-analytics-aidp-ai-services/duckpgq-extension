//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/functions/table/personalized_pagerank.hpp
//
//
//===----------------------------------------------------------------------===//

#include "duckpgq/common.hpp"

namespace duckdb {

class PersonalizedPageRankFunction : public TableFunction {
public:
	PersonalizedPageRankFunction() {
		name = "personalized_pagerank";
		// personalized_pagerank(vertex_table, vertex_id_col, edge_table, src_col, dst_col, source_vertex)
		arguments = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
		             LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT};
		bind_replace = PersonalizedPageRankBindReplace;
	}

	static unique_ptr<TableRef> PersonalizedPageRankBindReplace(ClientContext &context, TableFunctionBindInput &input);
};

struct PersonalizedPageRankData : TableFunctionData {
	static unique_ptr<FunctionData> PersonalizedPageRankBind(ClientContext &context, TableFunctionBindInput &input,
	                                                         vector<LogicalType> &return_types, vector<string> &names) {
		auto result = make_uniq<PersonalizedPageRankData>();
		result->pg_name = StringValue::Get(input.inputs[0]);
		result->node_table = StringValue::Get(input.inputs[1]);
		result->edge_table = StringValue::Get(input.inputs[2]);
		result->source = BigIntValue::Get(input.inputs[3]);
		return_types.emplace_back(LogicalType::BIGINT);
		return_types.emplace_back(LogicalType::DOUBLE);
		names.emplace_back("rowid");
		names.emplace_back("personalized_pagerank");
		return std::move(result);
	}

	string pg_name;
	string node_table;
	string edge_table;
	int64_t source;
};

struct PersonalizedPageRankScanState : GlobalTableFunctionState {
	static unique_ptr<GlobalTableFunctionState> Init(ClientContext &context, TableFunctionInitInput &input) {
		auto result = make_uniq<PersonalizedPageRankScanState>();
		return std::move(result);
	}

	bool finished = false;
};

} // namespace duckdb
