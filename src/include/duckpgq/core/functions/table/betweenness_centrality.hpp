//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/functions/table/betweenness_centrality.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckpgq/common.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

class BetweennessCentralityFunction : public TableFunction {
public:
	BetweennessCentralityFunction() {
		name = "betweenness_centrality";
		arguments = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
		             LogicalType::VARCHAR};
		bind_replace = BetweennessCentralityBindReplace;
	}

	static unique_ptr<TableRef> BetweennessCentralityBindReplace(ClientContext &context, TableFunctionBindInput &input);
};

struct BetweennessCentralityData : TableFunctionData {
	static unique_ptr<FunctionData> BetweennessCentralityBind(ClientContext &context, TableFunctionBindInput &input,
	                                                          vector<LogicalType> &return_types,
	                                                          vector<string> &names) {
		auto result = make_uniq<BetweennessCentralityData>();
		result->pg_name = StringValue::Get(input.inputs[0]);
		result->node_table = StringValue::Get(input.inputs[1]);
		result->edge_table = StringValue::Get(input.inputs[2]);
		return_types.emplace_back(LogicalType::BIGINT);
		return_types.emplace_back(LogicalType::DOUBLE);
		names.emplace_back("rowid");
		names.emplace_back("betweenness_centrality");
		return std::move(result);
	}

	string pg_name;
	string node_table;
	string edge_table;
};

struct BetweennessCentralityScanState : GlobalTableFunctionState {
	static unique_ptr<GlobalTableFunctionState> Init(ClientContext &context, TableFunctionInitInput &input) {
		auto result = make_uniq<BetweennessCentralityScanState>();
		return std::move(result);
	}

	bool finished = false;
};

} // namespace duckdb
