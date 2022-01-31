#pragma once

#ifdef CONSTRUCTDLL_EXPORTS
#define CONSTRUCT_API __declspec(dllexport)
#else
#define CONSTRUCT_API __declspec(dllimport)
#endif

#include "pch.h"
//Do not edit these functions

namespace dynet {

	//Construct will call this function if it doesn't recongize a model_name
	//All the param elements in the input xml file are dumped in the ParameterMap
	//If the model_name is not reconginized by this function a NULL pointer should be returned
	CONSTRUCT_API Model* create_custom_model(const std::string & model_name, const ParameterMap& parameters, Construct * construct);

	//Construct will call this function if it doesn't recongize an output_name
	//All the param elements in the input xml file are dumped in the ParameterMap
	//If the output_name is not reconginized by this function a NULL pointer should be returned
	CONSTRUCT_API Output* create_custom_output(const std::string& output_name, const ParameterMap& parameters, Construct* construct);

	CONSTRUCT_API media_user* create_custom_media_user(Social_Media* _media, Nodeset::iterator node);
}