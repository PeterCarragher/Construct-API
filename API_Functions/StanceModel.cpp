
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <vector>

#include "pch.h"
#include "StanceModelDefinitions.h"
#include "StanceModel.h"

#include <random>

#define USE_EIGEN_NOT_CONSTRUCT

typedef Eigen::Triplet<double> T;


// ------------ CONFIGURATION ------------ //
bool flip_most_influential = TRUE;
bool flip_strongest_stance = TRUE;
bool flip_least_susceptible = TRUE;



Eigen::SparseMatrix<float> networkToSparseMatrix(Graph<float>* G) {
	assert(G->col_size == G->row_size);
	int n = G->row_size;
	std::vector<T> tripletList;
	tripletList.reserve(n * n);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			// self loops in network?
			auto influence = G->examine(i, j);
			if (influence > 0.0)
				tripletList.push_back(T(i, j, influence));
		}
	}

	auto mat = Eigen::SparseMatrix<float>();
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	return mat;
}

Eigen::MatrixXd networkToDenseMatrix(Graph<float>* G) {
	assert(G->col_size == G->row_size);
	int n = G->row_size;
	Eigen::MatrixXd mat(n, n);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			// self loops in network?			
			mat(i, j) = G->examine(i, j);
		}
	}

	return mat;
}

// Write the contents of the Stance & influence matrices back to the construct network objects
// WARNING: this is expensive and should only be done for debugging purposes or on toy examples
#ifdef USE_EIGEN_NOT_CONSTRUCT
void StanceModel::syncNetworks(int timestamp) {
	int n = agents->size();
	for (int i = 0; i < n; i++) {
		stance_net->at(i, 0, stance_vec(i));

		for (int j = 0; j < n; j++) {
			influence_net->at(i, j, influence_mat.coeff(i, j));
		}
	}
}
#endif

StanceModel::StanceModel(dynet::ParameterMap parameters, Construct* _construct) : Model(_construct, "Stance Model") {
	agents = ns_manager->get_nodeset(nodeset_names::agents);
	int n = agents->size();

	stance_vec = Eigen::VectorXd(n);
	susceptible_vec = Eigen::VectorXd(n);
	std::mt19937 seed(123);
	std::uniform_real_distribution susceptibility_gen(0.0, 0.75);
	std::uniform_real_distribution stance_gen(-0.25, 0.5);
	for (int i = 0; i < n; i++) {
		auto susceptibility = susceptibility_gen(seed);
		susceptibilities.push_back(susceptibility);
		susceptible_vec(i) = susceptibility;

		std::string stance = agents->get_node_by_index(i)->get_attribute("stance");
		float stance_float = std::stof(stance);
		initial_stances.push_back(stance_float);
		stance_vec(i) = stance_float;
	}

	influence_net = graph_manager->load_required(NETWORK_INFLUENCE, agents, agents);

	if (flip_most_influential) {
		float max = 0.0f;
		int most_influential = 0;
		for (int i = 0; i < agents->size(); i++) {
			float influence_sum = 0.0f;
			for (int j = 0; j < agents->size(); j++) {
				if (i == j) {
					continue;
				}
				influence_sum += influence_net->examine(i, j);
			}
			if (influence_sum > max) {
				most_influential = i;
				max = influence_sum;
			}
		}
		this->stanceChange(most_influential, std::abs(this->getStance(most_influential)) * 2);
	}

	if (flip_strongest_stance) {
		int strongest_stance = std::distance(initial_stances.begin(), std::max_element(initial_stances.begin(), initial_stances.end()));
		this->stanceChange(strongest_stance, std::abs(this->getStance(strongest_stance))*2);
	}

	if (flip_least_susceptible) {
		int least_susceptible = std::distance(susceptibilities.begin(), std::min_element(susceptibilities.begin(), susceptibilities.end()));
		this->stanceChange(least_susceptible, std::abs(this->getStance(least_susceptible)) * 2);
	}

	influence_mat = networkToDenseMatrix(influence_net);

	stance_net = graph_manager->load_required(NETWORK_STANCE, agents, ns_manager->get_nodeset("stance"));
}

void StanceModel::stanceChange(int node, float amount) {
	float stance = this->getStance(node);
	float new_stance = stance > 0 ? stance - amount : stance + amount;
	this->setStance(node, new_stance);
}

float StanceModel::getStance(int i) {
	return construct->current_time == 0 ? 
		initial_stances.at(i) : 
		stance_net->examine(i, 0);
}

void StanceModel::setStance(int i, float stance) {
	if (construct->current_time == 0)
		initial_stances[i] = stance;
	else
		stance_net->at(i, 0, stance);
}

void StanceModel::think(void) {

	int n = agents->size();

	#ifdef USE_EIGEN_NOT_CONSTRUCT
		// TODO implement models using Eigen
		auto ones = Eigen::VectorXd::Ones(n);
		auto new_stance = influence_mat * stance_vec;
		stance_vec = (ones - susceptible_vec).cwiseProduct(stance_vec) + susceptible_vec.cwiseProduct(new_stance);
		stance_vec = stance_vec.cwiseMin(1.0).cwiseMax(-1.0);	

		auto new_influence = stance_vec * stance_vec.transpose();
		influence_mat = (1 - lr) * influence_mat + lr * new_influence;

		// enforce influence rows as probability distributions (sum to 1)
		auto norms = ones.cwiseQuotient(influence_mat.rowwise().sum());
		influence_mat = norms.asDiagonal() * influence_mat;
		syncNetworks(construct->current_time);
	#else
		// calculate stances - Friedkin
		for (int i = 0; i < agents->size(); i++) {

			float stance_delta = (1 - susceptibilities.at(i)) * getStance(i);
			for (int j = 0; j < agents->size(); j++) {
				if (i == j) {
					continue;
				}
				stance_delta += susceptibilities.at(i) * influence_net->examine(i, j) * getStance(j);
			}

			auto new_stance = std::clamp(stance_delta, -1.0f, 1.0f);
			stance_net->at(i, 0, new_stance);
		}

		// calculate influence - Hopfield
		for (int i = 0; i < agents->size(); i++) {
			std::vector<float> new_influences;
			for (int j = 0; j < agents->size(); j++) {
				auto current_influence = influence_net->examine(i, j);
				auto new_influence = (1 - structural_learning_rate) * current_influence + structural_learning_rate * getStance(i) * getStance(j);
				new_influences.push_back(std::clamp(new_influence, 0.0f, 1.0f));
			}

			// normalize
			auto inf_sum = std::accumulate(new_influences.begin(), new_influences.end(),0.0);
			std::transform(new_influences.begin(), new_influences.end(), new_influences.begin(), 
				[inf_sum](float& c) { return c / inf_sum; });

			// update
			for (int j = 0; j < agents->size(); j++) {
				influence_net->add_delta(i, j, new_influences.at(j));
			}
		}
	#endif

	syncNetworks(construct->current_time);
}

StanceModel::~StanceModel(void) {
	int final_timestep = construct->time_count - 1;
#ifdef USE_EIGEN_NOT_CONSTRUCT
	syncNetworks(final_timestep);
#endif
}



