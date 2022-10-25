#pragma once
#include "pch.h"
#include "StanceModel.h"
#include "StanceModelDefinitions.h"
#include <iostream>

class StanceCSV : public Output {
public:
	StanceCSV(Construct& cst) {
		auto model = cst.model_manager.get_model_by_name("Stance Model");
		StanceModel* stance_model = dynamic_cast<StanceModel*>(model);
		stance_out = stance_model->stance_net;
		output.open("stance_output.csv");
		for (auto &node: *stance_out->source_nodeset) {
			output << "," << node.name;
		}
		output << std::endl;

		output << "initial" << ",";
		for (int i = 0; i < stance_out->row_size; i++) {
			output << stance_out->examine(i, 0) << ",";
		}
		output << std::endl;
	}

	void process(unsigned int _t) {
		output << "t_" << _t << ",";
		for (int i = 0; i < stance_out->row_size; i++) {
			output << stance_out->examine(i, 0) << ",";
		}
		output << std::endl;
	}

	~StanceCSV() {
		output.close();
	}

	std::ofstream output;
	Graph<float>* stance_out;
};