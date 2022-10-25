
#ifndef STANCE_MODEL_HH_H
#define STANCE_MODEL_HH_H

#include <vector>
#include <Eigen/SparseCore>

#include "pch.h"

//All models need to inherit the "Model" class in order to be added to the Model Manager
//The Model class allows for inheritance of Interaction Messages and Communication Mediums
//Sets the pointers for the construct, graph and nodeset manager, and the random generator

class StanceModel : public Model
{
public:

	StanceModel(dynet::ParameterMap parameters, Construct* construct);
	~StanceModel();

	void think();

	std::vector<float> susceptibilities;
	std::vector<float> initial_stances;
	float lr = 0.2;
	const Nodeset* agents;

	Graph<float>* influence_net;
	Graph<float>* stance_net;
	Eigen::MatrixXd influence_mat;
	Eigen::VectorXd stance_vec;
	Eigen::VectorXd susceptible_vec;

	//void initialize(void);
	//void update(void);
	//void communicate(InteractionMessageQueue::iterator msg);
	//void cleanup(void);
private:

	float getStance(int i);
	void setStance(int i, float stance);
	void stanceChange(int node, float amount);
	void syncNetworks(int timestamp);

};
#endif



#pragma once
