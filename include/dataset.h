#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream> 

template <typename T, typename U>
struct datastruct 
{   
	std::vector<std::vector<T>> X;
	std::vector<U> y;
	unsigned int cardinality = 0;	
};

// references:
// https://www.cs.toronto.edu/~kriz/cifar.html
// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c/10409376#10409376

namespace uiuc {
	class cfar
	{
	public:

		
		struct datastruct <uint8_t, uint8_t> train_dataset, val_dataset, test_dataset;		
		
		cfar(std::string directory = "",
					 unsigned int Ntrain_num = 40000,
					 unsigned int Ntest_num = 10000,
					 unsigned int Nval_num = 10000);		
		
		// ~cfar()	{}

	private:

		unsigned int train_num = 0;
		unsigned int val_num   = 0;
		unsigned int test_num  = 0;

		std::string train_file_name = "data_batch";
		std::string test_file_name = "test_batch";
		unsigned int col = 3072; // number of column features (32*32*3)
		unsigned int file_samples = 10000; // number of samples in each bin-file
		
		struct datastruct <uint8_t, uint8_t> read_cifar10(std::string file_path, 
														  unsigned int image_num = 10000, 
														  int batch_num = 1, 
														  unsigned int readPosition = 0);

		
	};	
}

namespace uiuc {

	cfar::cfar(std::string directory,
					 unsigned int Ntrain_num,
					 unsigned int Ntest_num,
					 unsigned int Nval_num)
	{
		int train_batch_num = 5;
		int test_batch_num  = 1;
		int val_batch_num   = 5;

		if (Ntrain_num <= 50000 && Ntrain_num > 0)
			train_num = Ntrain_num;
		if (Ntest_num <= 10000 && Ntest_num > 0)
			test_num = Ntest_num;
		if ((Nval_num+Ntrain_num) <= 50000 && Nval_num > 0 && Ntrain_num > 0)
			val_num = Nval_num;

		std::cout << "\ntrain dataset:";
		train_dataset = read_cifar10(directory + '/' + train_file_name, train_num, train_batch_num);
		std::cout << "\ntest dataset:";
		test_dataset  = read_cifar10(directory + '/' + test_file_name,  test_num,  test_batch_num);
		std::cout << "\nvalidation dataset:";
		val_dataset   = read_cifar10(directory + '/' + train_file_name, val_num, val_batch_num, train_num);
		
	}

	struct datastruct <uint8_t, uint8_t> cfar::read_cifar10(std::string file_path, 
															unsigned int image_num, 
															int batch_num, 
															unsigned int readPosition)
	{				
		struct datastruct<uint8_t, uint8_t> cfar10Data;
		cfar10Data.X.resize(image_num, std::vector<uint8_t>(col));
		cfar10Data.y.resize(image_num);
		unsigned int mindex = 0;
		for (int ibatch = 0; ibatch < batch_num; ibatch++)
		{
			if (readPosition >= file_samples)
			{
				readPosition -= file_samples;
				continue;
			}
			if (readPosition < 0)				
				readPosition = 0;
			
			std::string full_file_path = file_path;
			if (batch_num > 1)
				full_file_path += "_" + std::to_string(ibatch + 1);

			std::ifstream file(full_file_path + ".bin", std::ios::binary);
			if (file.is_open())
			{
				std::cout << "\nreading the file " << full_file_path + ".bin" << " ..." << std::endl;

				if (readPosition > 0)					
					file.seekg(readPosition*(col+1), std::ios::beg);
				unsigned int i;
				for (i = mindex; i < image_num; i++)
				{
					if ((i - mindex) == file_samples - readPosition)
						break;

					file.read((char*)&cfar10Data.y[i], 1);
					file.read((char*)&cfar10Data.X[i][0], col);
				}

				file.close();
				readPosition = 0;
				mindex = i;
				if (mindex >= image_num)
					break;
			}
			else
			{
				throw std::runtime_error("cannot open file `" + full_file_path + "`!");
			}
		}

		// std::cout << "\nThe number of samples: " << mindex;
		// std::cout << "\n------------------------------------------------\n";

		cfar10Data.cardinality = mindex;
		return cfar10Data;
	}
}