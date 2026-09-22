//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/functions/table/eigenvector_centrality.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckpgq/common.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

class EigenvectorCentralityFunction : public TableFunction {
public:
	EigenvectorCentralityFunction() {
		name = "eigenvector_centrality";
		arguments = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
		             LogicalType::VARCHAR};
		bind_replace = EigenvectorCentralityBindReplace;
	}

	static unique_ptr<TableRef> EigenvectorCentralityBindReplace(ClientContext &context, TableFunctionBindInput &input);
};

struct EigenvectorCentralityData : TableFunctionData {
	static unique_ptr<FunctionData> EigenvectorCentralityBind(ClientContext &context, TableFunctionBindInput &input,
	                                                          vector<LogicalType> &return_types,
	                                                          vector<string> &names) {
		auto result = make_uniq<EigenvectorCentralityData>();
		result->pg_name = StringValue::Get(input.inputs[0]);
		result->node_table = StringValue::Get(input.inputs[1]);
		result->edge_table = StringValue::Get(input.inputs[2]);
		return_types.emplace_back(LogicalType::BIGINT);
		return_types.emplace_back(LogicalType::DOUBLE);
		names.emplace_back("rowid");
		names.emplace_back("eigenvector_centrality");
		return std::move(result);
	}

	string pg_name;
	string node_table;
	string edge_table;
};

struct EigenvectorCentralityScanState : GlobalTableFunctionState {
	static unique_ptr<GlobalTableFunctionState> Init(ClientContext &context, TableFunctionInitInput &input) {
		auto result = make_uniq<EigenvectorCentralityScanState>();
		return std::move(result);
	}

	bool finished = false;
};

} // namespace duckdb
