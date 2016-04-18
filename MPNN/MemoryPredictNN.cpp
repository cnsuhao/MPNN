#include "StdAfx.h"
#include "MemoryPredictNN.h"
#include <iostream>
#include <fstream>

MemoryPredictNN::MemoryPredictNN(void)
{
}


MemoryPredictNN::~MemoryPredictNN(void)
{
}

bool MemoryPredictNN::OpenMINSTTrainImage(string image, string label)
{
	_minst_train_image.open(image, ios::binary);
	_minst_train_label.open(label, ios::binary);

	_minst_train_image.seekg(4, ios_base::beg);//skip magic number
	_minst_train_image.read((char*)(&_minst_train_image_number), 4);//image number
	_minst_train_image.read((char*)(&_minst_train_image_rows), 4);//image rows
	_minst_train_image.read((char*)(&_minst_train_image_cols), 4);//image cols

	_minst_train_label.seekg(4, ios_base::beg);//skip magic number
	_minst_train_label.read((char*)(&_minst_train_label_number), 4);//image number

	_minst_train_image_buf.resize(_minst_train_image_rows*_minst_train_image_cols);
	_read_minst_train_image_number = 0;
	if(_minst_train_label_number == _minst_train_image_number)
		return true;
	else
		return false;
}
bool MemoryPredictNN::ReadNextMINSTTrainImage()
{
	if(_read_minst_train_image_number > _minst_train_image_number)
		return false;

	if(!_minst_train_image.read(_minst_train_image_buf.data(), _minst_train_image_rows*_minst_train_image_cols))
		return false;
	
	if(!_minst_train_label.read(&_minst_train_image_buf_label,1))
		return false;

	_read_minst_train_image_number++;

	return true;
}
bool MemoryPredictNN::LearnMINSTImage()
{
	
	return false;
}
bool MemoryPredictNN::CloseMINSTTrainImage()
{
	_minst_train_image.close();
	_minst_train_label.close();
	return false;
}
bool MemoryPredictNN::OpenMINSTTestImage(string image, string label)
{
	_minst_test_image.open(image, ios::binary);
	_minst_test_label.open(label, ios::binary);

	_minst_test_image.seekg(4, ios_base::beg);//skip magic number
	_minst_test_image.read((char*)(&_minst_test_image_number), 4);//image number
	_minst_test_image.read((char*)(&_minst_test_image_rows), 4);//image rows
	_minst_test_image.read((char*)(&_minst_test_image_cols), 4);//image cols

	_minst_test_label.seekg(4, ios_base::beg);//skip magic number
	_minst_test_label.read((char*)(&_minst_test_label_number), 4);//image number

	_minst_test_image_buf.resize(_minst_test_image_rows*_minst_test_image_cols);
	_read_minst_test_image_number = 0;
	if(_minst_test_label_number == _minst_test_image_number)
		return true;
	else
		return false;

}
bool MemoryPredictNN::ReadNextMINSTTestImage()
{
	if(_read_minst_test_image_number > _minst_test_image_number)
		return false;

	if(!_minst_test_image.read(_minst_test_image_buf.data(), _minst_test_image_rows*_minst_test_image_cols))
		return false;
	
	if(!_minst_test_label.read(&_minst_test_image_buf_label,1))
		return false;

	_read_minst_test_image_number++;

	return true;

}
bool MemoryPredictNN::RecognativeMINSTImage()
{
	return false;
}
bool MemoryPredictNN::CloseMINSTTestImage()
{
	_minst_train_image.close();
	_minst_train_label.close();

	return false;
}

bool MemoryPredictNN::CreateInputLayer()
{
	NeuronsLayer layer;
	layer._height = (int)floor(_input_layer_height*sqrt(_layer_shrink_ratio));
	layer._width = (int)floor(_input_layer_width*sqrt(_layer_shrink_ratio));
	if(layer._height * layer._width <= 0)
	{
		return false;
	}
	else
	{
		_neurons_layers.push_back(layer);
		return true;
	}
}

bool MemoryPredictNN::CreateLayer()
{
	NeuronsLayer lower_layer = _neurons_layers.back();
	NeuronsLayer layer;
	layer._height = (int)floor(lower_layer._height*sqrt(_layer_shrink_ratio));
	layer._width = (int)floor(lower_layer._width*sqrt(_layer_shrink_ratio));
	if(layer._height * layer._width <= 0)
	{
		return false;
	}
	else
	{
		_neurons_layers.push_back(layer);

		//initalize pattern matrix between lower layer and the current.
		WeightMatrix mat_pattern;
		mat_pattern._rows = lower_layer._width*lower_layer._height;
		mat_pattern._cols = layer._width*layer._height;
		mat_pattern._weights.resize(mat_pattern._rows * mat_pattern._cols);
		vector<float>::iterator it = mat_pattern._weights.begin();
		while(it != mat_pattern._weights.end())
		{
			*it = 0.0f;
		}
		_pattern_matrix.push_back(mat_pattern);
		//intalize predict matrix of the layer.
		WeightMatrix mat_predict;
		mat_predict._rows = layer._width*layer._height;
		mat_predict._cols = layer._width*layer._height;
		mat_predict._weights.resize(mat_predict._rows * mat_predict._cols);
		vector<float>::iterator it = mat_predict._weights.begin();
		while(it != mat_pattern._weights.end())
		{
			*it = 0.0f;
		}
		_prediction_matrix.push_back(mat_predict);
		//initalize inhibition matrix of the layer.
		WeightMatrix mat_inhibition;
		mat_inhibition._rows = layer._width*layer._height;
		mat_inhibition._cols = layer._width*layer._height;
		mat_inhibition._weights.resize(mat_inhibition._rows * mat_inhibition._cols);
		vector<float>::iterator it = mat_inhibition._weights.begin();
		while(it != mat_inhibition._weights.end())
		{
			*it = 0.0f;
		}
		_inhibition_matrix.push_back(mat_predict);


		return true;
	}
}

bool MemoryPredictNN::TrainInputLayer()
{
	NeuronsLayer& input_layer = _neurons_layers.at(0);
	for(int r = 0; r<input_layer._width; r++)
	{
		for(int c=0; c<input_layer._height; c++)
		{
			input_layer._layer_output_vector.at(r*input_layer._width+c)=active_input_neuron(r,c);
		}
	}
}
bool MemoryPredictNN::TrainLayer(int layer)
{

}

float MemoryPredictNN::active_input_neuron(int r, int c)
{
	
	return 0.0f;
}