#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

namespace uiuc {
	class cmath
	{
	public:
			
		template <typename T, typename U, typename V>
		std::vector< T> add(std::vector< U>, V);

		template <typename T, typename U, typename V>
		std::vector< T> add(U, std::vector< V>);

		template <typename T, typename U, typename V>
		std::vector<T> add(std::vector<U> vecx, std::vector< V> vecy);	

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> add(std::vector< std::vector<U>>, std::vector<V>, int axis = 0);

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> add(std::vector< std::vector<U>>, std::vector< std::vector<V>>);

		template <typename T, typename U>
		std::vector< T> mean(std::vector< std::vector<U>>, int axis = 0);
		
		template <typename T, typename U>
		std::vector< T> sum(std::vector< std::vector<U>>, int axis);
		template <typename T, typename U>
		T sum(std::vector< std::vector<U>>);

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> subtract(std::vector< std::vector<U>> xmatrix, std::vector< std::vector<V>> ymatrix);
		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> subtract(std::vector< std::vector<U>>, std::vector<V>, int axis = 0);


		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> multiply(std::vector< std::vector<U>>, std::vector< std::vector<V>>);

		template <typename T>
		std::vector < std::vector< T>>  transpose(std::vector < std::vector< T>>);		

		template <typename T, typename U>
		std::vector< T> var(std::vector< std::vector<U>>, int axis = 0);

		template <typename T>
		std::vector < std::vector <T>> rand(unsigned int, unsigned int, T start, T stop, int seed);

		// vector exp
		template <typename T, typename U>
		std::vector<T> msqrt(std::vector<U> xvec);

		template <typename T, typename U, typename V>
		std::vector< T> divide(std::vector< U>, V);

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> divide(std::vector< std::vector<U>>, V);

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> divide(std::vector< std::vector<U>>, std::vector<V>, int axis = 0);
				
		template <typename T>
		std::vector <T> max(std::vector < std::vector <T>> x, int axis = 0);

		// matrix exp
		template <typename T, typename U>
		std::vector < std::vector <T>> mexp(std::vector < std::vector<U>> X);

		template<typename T>
		std::vector < std::vector <T> > zeros(unsigned int row, unsigned int col);

		template<typename T>
		std::vector <T> zeros(unsigned int row);

	
		template <typename T>
		std::vector < std::vector <T>> randn(unsigned int row, unsigned int col, T mean, T std);

		template<typename T>
		std::vector< T> ones(unsigned int);

		template <typename T, typename U>
		std::vector < std::vector< T>> copy(std::vector < std::vector< U>> xmatrix);

		template <typename T, typename U>
		std::vector < std::vector<T>> power(std::vector < std::vector<U>>, double p = 2);

		

		template <typename T, typename U, typename V>
		std::vector<T> dot(U, std::vector<V>);
		template <typename T, typename U, typename V>
		std::vector<T> dot(std::vector<U>, V);

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> dot(U, std::vector< std::vector<V>>);

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> dot(std::vector< std::vector<U>> xmatrix, V offset);

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> dot(std::vector<U> v, std::vector< std::vector<V>> X, int axis = 0);
		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> dot(std::vector< std::vector<U>> X, std::vector<V> v, int axis = 0);

		template <typename T, typename U, typename V>
		std::vector< std::vector<T>> dot(std::vector< std::vector<U>>, std::vector< std::vector<V>>);
		
	protected:

	private:

	};
};

namespace uiuc {
	
	template <typename T, typename U, typename V>
	std::vector< T> cmath::add(std::vector< U> vecx, V offset)
	{
		std::vector< T> addx;
		addx.resize(vecx.size());
		for (unsigned int i = 0; i < vecx.size(); i++)
			addx[i] = vecx[i] + offset;

		return addx;
	}

	template <typename T, typename U, typename V>
	std::vector< T> cmath::add(U offset, std::vector< V> vecx)
	{
		return add<T, V, U>(vecx, offset);
	}

	template <typename T, typename U, typename V>
	std::vector<T> cmath::add(std::vector<U> vecx, std::vector< V> vecy)
	{			
		unsigned int xsize = vecx.size();		
		std::vector<T> addResult(xsize);
		for (unsigned int i = 0; i < xsize; i++)
			addResult[i] = vecx[i] + vecy[i];
		return addResult;					
	}

	// broadcasting: a 2D matrix is added to a 1D vector
	// axis parameter is used when the input marix is square (the number of rows and columns are equal)
	// in the case of 
	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::add(std::vector< std::vector<U>> X, std::vector<V> x, int axis)
	{
		unsigned int Xrow = X.size();
		unsigned int Xcol = X[0].size();
		unsigned int xnumel = x.size();

		std::vector<std::vector<T>> addX(Xrow, std::vector<T>(Xcol, 0));
		
		if (Xrow == xnumel &&  axis == 0)
		{
			for (unsigned int i = 0; i < Xrow; i++)
				for (unsigned int j = 0; j < Xcol; j++)
					addX[i][j] = X[i][j] + x[i];
		}
		else if (Xcol == xnumel)
		{
			for (unsigned int i = 0; i < Xrow; i++)
				for (unsigned int j = 0; j < Xcol; j++)
					addX[i][j] = X[i][j] + x[j];
		}
		else
		{

		}
		return addX;
	}
	
	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::add(std::vector< std::vector<U>> xmatrix, std::vector< std::vector<V>> ymatrix)
	{
		unsigned int row = xmatrix.size();
		unsigned int col = xmatrix[0].size();
		std::vector<std::vector<T>> addx(row, std::vector<T>(col, 0));

		for (unsigned int i = 0; i < row; i++)
			for (unsigned int j = 0; j < col; j++)
				addx[i][j] = xmatrix[i][j] + ymatrix[i][j];

		return addx;
	}

	template <typename T, typename U>
	std::vector< T> cmath::mean(std::vector< std::vector<U>> xmatrix, int axis)
	{
		std::vector< T> meanx;

		unsigned int row = xmatrix.size();
		unsigned int col = xmatrix[0].size();

		if (axis == 1)
		{
			meanx.resize(row);
			for (unsigned int i = 0; i < row; i++)
			{
				meanx[i] = 0;
				for (unsigned int j = 0; j < col; j++)
					meanx[i] += xmatrix[i][j];

				meanx[i] /= col;
			}
		}
		else if (axis == 0)
		{
			meanx.resize(col);
			for (unsigned int j = 0; j < col; j++)
			{
				meanx[j] = 0;
				for (unsigned int i = 0; i < row; i++)
					meanx[j] += xmatrix[i][j];

				meanx[j] /= row;
			}
		}
		return meanx;
	}

	template <typename T, typename U>
	std::vector< T> cmath::sum(std::vector< std::vector<U>> X, int axis)		
	{
		unsigned row = X.size();
		unsigned col = X[0].size();

		std::vector< T> sumVec;

		if (axis == 0)
		{
			sumVec.resize(col);
			for (unsigned int i = 0; i < col; i++)
			{
				T xsum = 0;
				for (unsigned j = 0; j < row; j++)
					xsum += X[j][i];
				sumVec[i] = xsum;
			}
		}
		else if (axis == 1)
		{
			sumVec.resize(row);
			for (unsigned int i = 0; i < row; i++)
			{
				T xsum = 0;
				for (unsigned j = 0; j < col; j++)
					xsum += X[i][j];
				sumVec[i] = xsum;
			}
		}

		return sumVec;
	}
	
	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::subtract(std::vector< std::vector<U>> xmatrix, std::vector< std::vector<V>> ymatrix)
	{
		unsigned int row = xmatrix.size();
		unsigned int col = xmatrix[0].size();

		std::vector<std::vector<T>> addx(row, std::vector<T>(col, 0));

		for (unsigned int i = 0; i < row; i++)
			for (unsigned int j = 0; j < col; j++)
				addx[i][j] = xmatrix[i][j] - ymatrix[i][j];

		return addx;
	}	

	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::subtract(std::vector< std::vector<U>> X, std::vector<V> x, int axis)
	{
		unsigned int Xrow = X.size();
		unsigned int Xcol = X[0].size();
		unsigned int xnumel = x.size();

		std::vector<std::vector<T>> addX(Xrow, std::vector<T>(Xcol, 0));

		if (Xrow == xnumel && axis == 0)
		{
			for (unsigned int i = 0; i < Xrow; i++)
				for (unsigned int j = 0; j < Xcol; j++)
					addX[i][j] = X[i][j] - x[i];
		}
		else if (Xcol == xnumel)
		{
			for (unsigned int i = 0; i < Xrow; i++)
				for (unsigned int j = 0; j < Xcol; j++)
					addX[i][j] = X[i][j] - x[j];
		}
		else
		{

		}
		return addX;
	}

	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::multiply(std::vector< std::vector<U>> X, std::vector< std::vector<V>> Y)
	{
		unsigned int row = X.size();
		unsigned int col = Y[0].size();
		unsigned int rowcol = X[0].size();

		if (X[0].size() != Y.size())
		{
			std::cout << "the size of the matrix is not consistant!";			
		}
		
		std::vector<std::vector<T>> Z(row, std::vector<T>(col, 0));

		unsigned int i, j, k;
		for (i = 0; i < row; i++)
			for (j = 0; j < col; j++)
				for (k = 0; k < rowcol; k++)
					Z[i][j] += T(X[i][k]) * T(Y[k][j]);

		return Z;

	}

	// transopse a 2D matrix
	template <typename T>
	std::vector < std::vector< T>>  cmath::transpose(std::vector < std::vector< T>> A) 
	{
		unsigned int row = A.size();
		unsigned int col = A[0].size();

		std::vector<std::vector<T>>  AT(col, std::vector<T>(row));
		unsigned int i, j;
		for (i = 0; i < row; i++)
			for (j = 0; j < col; j++)
				AT[j][i] = A[i][j];

		return AT;
	}

	
	template <typename T, typename U>
	std::vector< T> cmath::var(std::vector< std::vector<U>> xmatrix, int axis)
	{
		unsigned int row = xmatrix.size();
		unsigned int col = xmatrix[0].size();

		std::vector< T> xmean;
		std::vector< T> xvar;

		T meanx = 0;
		T varx = 0;

		unsigned int i, j;

		if (axis == 1)
		{
			xmean.resize(row);
			xvar.resize(row);
			for (i = 0; i < row; i++)
			{
				if (col == 1)
				{
					xvar[i] = 0;
					continue;
				}
				meanx = 0;
				for (j = 0; j < col; j++)
					meanx += xmatrix[i][j];
				meanx /= col;

				varx = 0;
				for (j = 0; j < col; j++)
					varx += pow(xmatrix[i][j] - meanx, 2);
				xvar[i] = varx / (col - 1);

				xmean[i] = meanx;
			}
		}
		else if (axis == 0)
		{
			xmean.resize(col);
			xvar.resize(col);
			for (j = 0; j < col; j++)
			{
				if (row == 1)
				{
					xvar[j] = 0;
					continue;
				}

				meanx = 0;
				for (i = 0; i < row; i++)
					meanx += xmatrix[i][j];
				meanx /= row;

				varx = 0;
				for (i = 0; i < row; i++)
					varx += pow(xmatrix[i][j] - meanx, 2);
				xvar[j] = varx / (row - 1);

				xmean[j] = meanx;
			}
		}

		return xvar;
	}

	
	template <typename T, typename U>
	std::vector<T> cmath::msqrt(std::vector<U> xvec)
	{
		unsigned int len = xvec.size();

		std::vector<T> X(len, 0);

		unsigned int i, j;
		for (unsigned int i = 0; i < len; i++)			
				X[i] = sqrt(xvec[i]);

		return X;
	}

	template <typename T, typename U, typename V>
	std::vector< T> cmath::divide(std::vector< U> vecx, V offset)
	{
		std::vector< T> addx;
		addx.resize(vecx.size(), 0);
		for (unsigned int i = 0; i < vecx.size(); i++)
			addx[i] = vecx[i] / offset;

		return addx;
	}

	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::divide(std::vector< std::vector<U>> X, V x)
	{
		unsigned int Xrow = X.size();
		unsigned int Xcol = X[0].size();
		

		std::vector<std::vector<T>> divX(Xrow, std::vector<T>(Xcol, 0));
		
		for (unsigned int i = 0; i < Xrow; i++)
			for (unsigned int j = 0; j < Xcol; j++)
				divX[i][j] = T(X[i][j]) / T(x);
		
		return divX;

	}


	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::divide(std::vector< std::vector<U>> X, std::vector<V> x, int axis)
	{
		unsigned int Xrow = X.size();
		unsigned int Xcol = X[0].size();
		unsigned int xnumel = x.size();

		std::vector<std::vector<T>> divX(Xrow, std::vector<T>(Xcol, 0));

		if (Xrow == xnumel && axis == 0)
		{
			for (unsigned int i = 0; i < Xrow; i++)
				for (unsigned int j = 0; j < Xcol; j++)
					divX[i][j] = T(X[i][j]) / T(x[i]);
		}
		else if (Xcol == xnumel)
		{
			for (unsigned int i = 0; i < Xrow; i++)
				for (unsigned int j = 0; j < Xcol; j++)
					divX[i][j] = T(X[i][j]) / T(x[j]);
		}
		else
		{

		}
		return divX;

	}	

	template <typename T>
	std::vector <T> cmath::max(std::vector < std::vector <T>> X, int axis)
	{
		std::vector<T> maxvec;
		unsigned int row = X.size();
		unsigned int col = X[0].size();

		if (axis == 0)
		{
			maxvec.resize(col, 0);
			for (unsigned int j = 0; j < col; j++)
			{
				T maxvalue = X[0][j];
				for (unsigned int i = 1; i < row; i++)
					if (maxvalue < X[i][j])
						maxvalue = X[i][j];
				maxvec[j] = maxvalue;
			}
		}
		else if (axis == 1)
		{
			maxvec.resize(row, 0);
			for (unsigned int i = 0; i < row; i++)
			{
				T maxvalue = X[i][0];
					for (unsigned int j = 1; j < col; j++)
						if (maxvalue < X[i][j])
							maxvalue = X[i][j];
				maxvec[i] = maxvalue;
			}
		}
		
		return maxvec;
	}

	template <typename T, typename U>
	std::vector < std::vector <T>> cmath::mexp(std::vector < std::vector<U>> X)
	{
		unsigned int row = X.size();
		unsigned int col = X[0].size();

		std::vector<std::vector<T>>  Xexp(row, std::vector<U>(col));

		unsigned int i, j;
		for (i = 0; i < row; i++)
			for (j = 0; j < col; j++)
				Xexp[i][j] = exp(X[i][j]);
		return Xexp;
	}

	template<typename T>
	std::vector < std::vector <T> > cmath::zeros(unsigned int row, unsigned int col)
	{
		std::vector<std::vector<T>> zX(row, std::vector<T>(col, 0));
		return zX;
	}

	template<typename T>
	std::vector <T> cmath::zeros(unsigned int row)
	{
		std::vector<T> zX(row, 0);
		return zX;
	}

	template <typename T>
	std::vector < std::vector <T>> cmath::rand(unsigned int row, unsigned int col, T start, T stop, int seed)
	{
		std::vector<std::vector<T>> X(row, std::vector<T>(col, 0));
		
		std::default_random_engine generator;
		if (seed == -1)
		{
			std::random_device rd;
			generator.seed(rd()); //Now this is seeded differently each time.
		}
		else
			generator.seed(unsigned (seed));
		//std::shuffle_order_engine generator;
		std::uniform_real_distribution<T> distribution(start, stop);
		distribution.reset();
		for (unsigned int i = 0; i < row; i++)
			for (unsigned int j = 0; j < col; j++)
				X[i][j] = distribution(generator);
		return X;
	}

	template <typename T>
	std::vector < std::vector <T>> cmath::randn(unsigned int row, unsigned int col, T mean, T std)
	{
		std::vector<std::vector<T>> X(row, std::vector<T>(col, 0));
		std::random_device rd;
		std::default_random_engine generator;
		generator.seed(rd()); //Now this is seeded differently each time.
		//std::shuffle_order_engine generator;
		std::normal_distribution<T> distribution(mean, std);
		distribution.reset();
		for (unsigned int i = 0; i < row; i++)
			for (unsigned int j = 0; j < col; j++)
				X[i][j] = distribution(generator);
		return X;
	}

	template<typename T>
	std::vector< T> cmath::ones(unsigned int row)
	{
		std::vector<T> ov(row, 1);
		return ov;
	}

	template <typename T, typename U>
	std::vector < std::vector< T>> cmath::copy(std::vector < std::vector< U>> xmatrix)
	{
		unsigned int row = xmatrix.size();
		unsigned int col = xmatrix[0].size();

		std::vector<std::vector<T>> cmatrix(row, std::vector<T>(col, 0));
		unsigned int i, j;
		for (i = 0; i < row; i++)
			for (j = 0; j < col; j++)
				cmatrix[i][j] = T(xmatrix[i][j]);
		return cmatrix;
	}

	template <typename T, typename U>
	std::vector < std::vector<T>> cmath::power(std::vector < std::vector<U>> xmatrix, double p)
	{

		unsigned int row = xmatrix.size();
		unsigned int col = xmatrix[0].size();

		std::vector<std::vector<T>> pmatrix(row, std::vector<T>(col, 0));
		unsigned int i, j;
		
		if (p != 0.5)
			for (i = 0; i < row; i++)
				for (j = 0; j < col; j++)
					pmatrix[i][j] = pow(xmatrix[i][j], p);
		else
			for (i = 0; i < row; i++)
				for (j = 0; j < col; j++)
					pmatrix[i][j] = sqrt(xmatrix[i][j]);

		return pmatrix;
	}

	template <typename T, typename U>
	T cmath::sum(std::vector< std::vector<U>> X)
	{
		unsigned row = X.size();
		unsigned col = X[0].size();
		T sumx = 0;

		for (unsigned int i = 0; i < row; i++)
			for (unsigned j = 0; j < col; j++)
				sumx += X[i][j];

		return sumx;
	}

	template <typename T, typename U, typename V>
	std::vector<T> cmath::dot(U offset, std::vector<V> xvec)
	{
		unsigned int row = xvec.size();

		std::vector<T> dotx(row, 0);

		for (unsigned int i = 0; i < row; i++)			
				dotx[i] = offset * xvec[i];

		return dotx;
	}

	template <typename T, typename U, typename V>
	std::vector<T> cmath::dot(std::vector<U> xvec, V offset)
	{
		return dot<T, V, U>(offset, xvec);
	}

	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::dot(U offset, std::vector< std::vector<V>> xmatrix)
	{
		unsigned int row = xmatrix.size();
		unsigned int col = xmatrix[0].size();

		std::vector<std::vector<T>> dotx(row, std::vector<T>(col, 0));

		for (unsigned int i = 0; i < row; i++)
			for (unsigned int j = 0; j < col; j++)
				dotx[i][j] = offset * xmatrix[i][j];

		return dotx;
	}

	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::dot(std::vector< std::vector<U>> xmatrix, V offset)
	{
		return dot<T, V, U> (offset, xmatrix);
	}


	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::dot(std::vector< std::vector<U>> xmatrix, std::vector< std::vector<V>> ymatrix)
	{
		unsigned int row = xmatrix.size();
		unsigned int col = xmatrix[0].size();

		std::vector<std::vector<T>> dotx(row, std::vector<T>(col, 0));

		for (unsigned int i = 0; i < row; i++)
			for (unsigned int j = 0; j < col; j++)
				dotx[i][j] = xmatrix[i][j] * ymatrix[i][j];

		return dotx;
	}

	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::dot(std::vector<U> v, std::vector< std::vector<V>> X, int axis)
	{
		return dot <T, V, U>(X, v, axis);
	}


	template <typename T, typename U, typename V>
	std::vector< std::vector<T>> cmath::dot(std::vector< std::vector<U>> X, std::vector<V> v, int axis)
	{
		unsigned int row = X.size();
		unsigned int col = X[0].size();
		std::vector< std::vector<T>> dotX(row, std::vector<T>(col, 0));

		unsigned int vlen = v.size();
		unsigned int i, j;
		if (vlen == row && axis == 0)
		{
			for (i = 0; i < row; i++)
				for (j = 0; j < col; j++)
					dotX[i][j] = X[i][j] * v[i];
		}
		else if (vlen == col)
		{
			for (i = 0; i < row; i++)
				for (j = 0; j < col; j++)
					dotX[i][j] = X[i][j] * v[j];
		}
		else
			std::cout << "warning!, the dimension of X and v are not consistent" << std::endl;

		return dotX;

	}	
}
