#ifndef MATH_MATRIX_HH_
#define MATH_MATRIX_HH_

#include <Core/CommonHeaders.hh>
#include <Core/OpenMPWrapper.hh>

#include <Math/Blas.hh>
#include <Math/Vector.hh>   		// for matrix-vector operations (Blas 2)
#include <Math/FastVectorOperations.hh>

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <limits>
#include <iostream>				// to use std::cout
#include <typeinfo>
#include <sstream>

namespace Math {

template<typename T>
class Vector;

/*
 * matrix with col-major storage
 * use of BLAS routines
 *
 */

template<typename T>
class Matrix {
	friend class Vector<T>;
	friend class Vector<f64>;
	friend class Vector<f32>;
	friend class Matrix<f64>;
	friend class Matrix<f32>;
	friend class Vector<u32>;
protected:
	// the number of sizeof(T) elements that are actually allocated
	// (may differ from nRows_ * nColumns_ due to lazy resize)
	u64 nAllocatedCells_;
	u32 nRows_;
	u32 nColumns_;
	T *elem_;
protected:
	static bool initialized;
	static s32 maxThreads;
	static s32 _initialize();
	static s32 initialize();
	s32 nThreads_;
public:
	s32 getNumberOfThreads(){ return nThreads_; }
public:
	// iterators
	typedef T* iterator;
	typedef const T* const_iterator;
	iterator begin() { return elem_; }
	const_iterator begin() const { return elem_; }
	iterator end() { return &(elem_[(u64)nRows_ * (u64)nColumns_]); }
	const_iterator end() const { return &(elem_[(u64)nRows_ * (u64)nColumns_]); }
public:
	// constructor with memory allocation
	Matrix(u32 nRows = 0, u32 nColumns = 0);

	// (deep) copy constructor
	Matrix(const Matrix<T> &X);

	// constructor for creating sub-matrices via copyBlockFromMatrix()
	Matrix(const Matrix<T> &X, u32 rowIndexX, u32 colIndexX,
			u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns);

	// destructor
	virtual ~Matrix();

private:
	bool allocate();

private:
	void writeBinaryHeader(Core::IOStream& stream, bool transpose);
	void writeAsciiHeader(Core::IOStream& stream, bool transpose);
	void writeBinary(Core::IOStream& stream, bool transpose);
	void writeAscii(Core::IOStream& stream, bool transpose, bool scientific);
	bool readBinaryHeader(Core::IOStream& stream, bool transpose);
	bool readAsciiHeader(Core::IOStream& stream, bool transpose);
	void read(Core::IOStream& stream, bool transpose = false);

public:
	// file IO
	void write(const std::string& filename, bool transpose = false, bool scientific = false);
	void read(const std::string& filename, bool transpose = false);

public:
	// free memory
	void clear();

	// required for assignment operator
	void swap(Matrix<T> &X);

	// swap matrix and vector (matrix will end up being a single column, former matrix columns are concatenated in the vector)
	void swap(Vector<T> &X);

	void setNumberOfThreads(int nThreads) { nThreads_ = nThreads;};

	// resize & allocate
	// side effect: after resize content is meaningless
	// if reallocate is true enforce reallocation of memory
	virtual void resize(u32 nRows, u32 nColumns, bool reallocate = false);

	// set dimensions to those of X and allocate
	template <typename S>
	void copyStructure(const Matrix<S> &X);

	// returns the number of rows
	u32 nRows() const { return nRows_; }

	// returns the number of columns
	u32 nColumns() const { return nColumns_; }

	// copy method
	// this = X
	// for matrices with same dimensions
	template<typename S>
	void copy(const Matrix<S> &X);

	// copy method
	// this = X
	// array X is assumed to be of size nRows_ * nColumns_
	template<typename S>
	void copy(const S *X, u32 rowOffset = 0, u32 colOffset = 0);

	// copy from std::vector
	template<typename S>
	void copy(const std::vector<S> &X, u32 rowOffset = 0, u32 colOffset = 0);

	// returns the total number of entries
	u32 size() const { return nRows_*nColumns_; }

	// fills the matrix with the given value
	void fill(T value) { std::fill(elem_, elem_ + (u64)nRows_ * (u64)nColumns_, value); }

	// fills the matrix from position (rowA, columnA) to (rowB, columnB) with the value
	void fill(u32 rowA, u32 columnA, u32 rowB, u32 columnB, T value);

	// get reference to element in row i, column j
	T& at(u32 i, u32 j){
		require(i < nRows_);
		require(j < nColumns_);
		return *(elem_ + (u64)j*(u64)nRows_ + i);
	}

	// get const reference to element in row i, column j
	const T& at(u32 i, u32 j) const {
		require(i < nRows_);
		require(j < nColumns_);
		return *(elem_ + (u64)j*(u64)nRows_ + i);
	}

	// convert matrix to string
	std::string toString(bool transpose = false) const;

	// get row with index rowIndex
	void getRow(u32 rowIndex, Math::Vector<T> &row) const;

	// get column with index columnIndex
	void getColumn(u32 columnIndex, Math::Vector<T> &column) const;

	// set row at rowIndex to values in vector row
	void setRow(u32 rowIndex, const Math::Vector<T> &row);

	// set column at columnIndex to values in vector column
	void setColumn(u32 columnIndex, const Math::Vector<T> &column);

	// copy block from matrix to specific position
	void copyBlockFromMatrix(const Math::Matrix<T> &X, u32 rowIndexX, u32 colIndexX, u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns = 1);

	// add block from matrix to specific position
	void addBlockFromMatrix(const Math::Matrix<T> &X, u32 rowIndexX, u32 colIndexX, u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns, T scale = 1.0);

	// this = 0
	void setToZero() { memset(elem_, 0, (u64)nRows_ * (u64)nColumns_ * sizeof(T)); }



public:

	/*
	 * MATH OPERATIONS
	 *
	 */

	/*
	 * Blas1-like methods
	 */

	// this += alpha * X
	template<typename S>
	void add(const Matrix<S> &X, S alpha = 1.0);

	// @return l1-norm of matrix
	T l1norm() const { return Math::mt_asum(nRows_ * nColumns_, elem_, nThreads_); }

	// @return sum of squared matrix entries
	T sumOfSquares() const { return dot (*this); }

	// dot product
	// return this' * X
	// for matrices with multiple columns: interpret matrix as vector
	T dot(const Matrix<T> &X) const;

	// scale elements
	// this *= alpha
	void scale(T alpha);

	/*
	 * Blas2-like methods
	 */

	// rank-1 update: this += alpha * x y^T
	void addOuterProduct(const Vector<T>& x, const Vector<T> &y, T alpha, u32 lda = 0);

	/*
	 * Blas3-like methods
	 */

	// (*this) = (scaleA * matrixA) * matrixB + scaleC * (*this)
	void addMatrixProduct(const Matrix<T> &matrixA, const Matrix<T> &matrixB,
			T scaleC = 0, T scaleA = 1, bool transposeA = false, bool transposeB = false);

	// return sum over all elements
	T sum() const;

	// apply exp to each element of matrix
	void exp();

	// apply log to each element of matrix
	void log();

	// absolute values
	void abs();

	// return index of minimum absolute value in column
	u32 argAbsMin(u32 column) const;

	// return index of maximum absolute value in column
	u32 argAbsMax(u32 column) const;

	// save arg max of each column of *this in the rows of v
	template<typename S>
	void argMax(Vector<S>& v) const;

	// this = this .* X
	void elementwiseMultiplication(const Matrix<T> &X);

	// this = this ./ X
	void elementwiseDivision(const Matrix<T> &X);

	// add constant value c to each element
	void addConstantElementwise(T c);

	// add vector (scaled by alpha) to column with given index
	void addToColumn(const Vector<T> &v, u32 column, T alpha = 1.0);

	// multiply column by scalar alpha
	void multiplyColumnByScalar(u32 column, T alpha);

	// multiply row by scalar alpha
	void multiplyRowByScalar(u32 row, T alpha);

	// add vector (scaled by alpha) to row with given index
	void addToRow(const Vector<T> &v, u32 row, T alpha = 1.0);

	// add vector (scaled by alpha) to each column of the matrix
	void addToAllColumns(const Vector<T> &v, T alpha = 1.0);

	// add vector (scaled by alpha) to each column of the matrix
	void addToAllRows(const Vector<T> &v, T alpha = 1.0);

	// for each i: multiply column i by scalars[i]
	void multiplyColumnsByScalars(const Vector<T> &scalars);

	// for each i: divide column i by scalars[i]
	void divideColumnsByScalars(const Vector<T> &scalars);

	// for each i: multiply row i by scalars[i]
	void multiplyRowsByScalars(const Vector<T> &scalars);

	// for each i: multiply row i by scalars[i]
	void divideRowsByScalars(const Vector<T> &scalars);

private:
	// this = \alpha A^{"", T} * B^{"", T} + \gamma this
	static void _gemm(bool transposeA, bool transposeB, u32 M, u32 N, u32 K, T scaleA, T* matrixA, u32 lda, T* matrixB, u32 ldb, T scaleC, T* matrixC, u32 ldc);
};


template<typename T>
bool Matrix<T>::initialized = false;

template<typename T>
s32 Matrix<T>::maxThreads = 1;

template<typename T>
s32 Matrix<T>::_initialize(){
	if (!initialized){
		initialized = true;

		int value;

		char* svalue;
		svalue = std::getenv("OMP_NUM_THREADS");
		if (svalue != NULL){
			std::string tmp = svalue;
			std::istringstream ss(tmp);
			ss >> value;
			if (ss.fail())
				value = 1;
		}
		else{
			value = 1;
		}

		maxThreads = value;
		Core::omp::set_num_threads(value);
		std::cout << "Maximum number of threads for CPU matrix operations: " << maxThreads << std::endl;

	}
	return maxThreads;
}

template<typename T>
s32 Matrix<T>::initialize(){
	if (!initialized){
		// ensure that initialize is in fact only invoked once
		maxThreads = Matrix<u32>::_initialize();
		initialized = true;
	}
	return maxThreads;
}

/**	Allocate the memory for the matrix.
 *
 * 	Allocate the memory for the matrix. If the size is 0 the pointer
 * 	is ZERO.
 */
template<typename T>
bool Matrix<T>::allocate() {
	if (elem_)
		delete [] elem_;
	elem_ = (u64)nRows_ * (u64)nColumns_ > 0 ? new T[(u64)nRows_*(u64)nColumns_] : 0;
	nAllocatedCells_ = (u64)nRows_ * (u64)nColumns_ > 0 ? (u64)nRows_*(u64)nColumns_ : 0;
	return true;
}

// constructor with allocation
template<typename T>
Matrix<T>::Matrix(u32 nRows, u32 nColumns) :
nAllocatedCells_(0),
nRows_(nRows),
nColumns_(nColumns),
elem_(0),
nThreads_(1)
{
	nThreads_ = initialized ? maxThreads : initialize();
	if ((u64)nRows_* (u64)nColumns_ < 250000)
		nThreads_ = 1;
	allocate();
}

// copy constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T> &X) :
nAllocatedCells_(0),
nRows_(X.nRows_),
nColumns_(X.nColumns_),
elem_(0),
nThreads_(1)
{
	nThreads_ = initialized ? maxThreads : initialize();
	if ((u64)nRows_* (u64)nColumns_ < 250000)
		nThreads_ = 1;
	allocate();
	copy(X);
}

// copy constructor for sub-matrices
template<typename T>
Matrix<T>::Matrix(const Matrix<T> &X, u32 rowIndexX, u32 colIndexX,
		u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns) :
		nAllocatedCells_(0),
		nRows_(nRows),
		nColumns_(nColumns),
		elem_(0),
		nThreads_(initialized ? maxThreads : initialize())
{
	if ((u64)nRows_* (u64)nColumns_ < 250000)
		nThreads_ = 1;
	allocate();
	copyBlockFromMatrix(X, rowIndexX, colIndexX, thisRowIndex, thisColIndex, nRows, nColumns);
}

template<typename T>
Matrix<T>::~Matrix() {
	if (elem_)
		delete [] elem_;
	elem_ = 0;
}

template<typename T>
void Matrix<T>::clear() {
	if (elem_)
		delete [] elem_;
	elem_ = 0;
	nRows_ = 0;
	nColumns_ = 0;
	nAllocatedCells_ = 0;
}


template<typename T>
void Matrix<T>::swap(Matrix<T> &X){
	std::swap(nRows_, X.nRows_);
	std::swap(nColumns_, X.nColumns_);
	std::swap(nAllocatedCells_, X.nAllocatedCells_);
	std::swap(elem_, X.elem_);
}

template<typename T>
void Matrix<T>::swap(Vector<T> &X){
	u32 nRows = X.nRows_;
	X.nRows_ = nRows_ * nColumns_;
	nRows_ = nRows;
	nColumns_ = 1;
	u32 tmpAllocatedCells = X.nAllocatedCells_;
	X.nAllocatedCells_ = (u32)nAllocatedCells_;
	nAllocatedCells_ = (u64)tmpAllocatedCells;
	std::swap(elem_, X.elem_);
}

template<typename T>
void Matrix<T>::resize(u32 nRows, u32 nColumns, bool reallocate) {
	reallocate |= (u64)nRows * (u64)nColumns > nAllocatedCells_;
	nRows_ = nRows;
	nColumns_ = nColumns;
	if (reallocate) {
		allocate();
	}
}

template<typename T> template <typename S>
void Matrix<T>::copyStructure(const Matrix<S> &X) {
	resize(X.nRows(), X.nColumns());
}

template<typename T>
std::string Matrix<T>::toString(bool transpose) const {
	require(nRows_ > 0);
	require(nColumns_ > 0);
	std::stringstream s;
	if (transpose) {
		for (u32 i = 0; i < nColumns_; i++) {
			for (u32 j = 0; j < nRows_; j++) {
				s << at(j,i);
				if (j < nRows_ - 1) s << " ";
			}
			if (i != nColumns_ - 1)
				s << std::endl;
		}
	}
	else {
		for (u32 i = 0; i < nRows_; i++) {
			for (u32 j = 0; j < nColumns_; j++) {
				s << at(i,j);
				if (j < nColumns_ - 1) s << " ";
			}
			if (i != nRows_ - 1)
				s << std::endl;
		}
	}
	return s.str();
}

template<typename T>
void Matrix<T>::exp(){
	mt_vr_exp(nRows_ * nColumns_, elem_ , elem_, nThreads_);
}

template<typename T>
void Matrix<T>::log(){
	vr_log(nRows_ * nColumns_, elem_ , elem_);
}

template<typename T>
void Matrix<T>::abs(){
# pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++)
		elem_[i] = std::abs(elem_[i]);
}

template<typename T>
T Matrix<T>::sum() const {
	T result = 0;
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		result += elem_[i];
	}
	return result;
}

template<typename T>
u32 Matrix<T>::argAbsMin(u32 column) const {
	require_lt(column, nColumns_);
	return Math::iamin(nRows_, elem_ + column * nRows_, 1);
}

template<typename T>
u32 Matrix<T>::argAbsMax(u32 column) const {
	require_lt(column, nColumns_);
	return Math::iamax(nRows_, elem_ + column * nRows_, 1);
}

template<typename T>
template<typename S>
void Matrix<T>::argMax(Vector<S>& v) const {
	require_eq(v.nRows(), nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < nColumns_; i++) {
		T maxVal = at(0, i);
		v.at(i) = 0;
		for (u32 j = 1; j < nRows_; j++){
			if (at(j, i) > maxVal){
				maxVal = at(j, i);
				v.at(i) = j;
			}
		}
	}
}

template<typename T>
template<typename S>
void Matrix<T>::add(const Matrix<S> &X, S alpha){
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	Math::axpy<S,T>(nRows_ * nColumns_, alpha, X.elem_, 1, elem_, 1);
}

template<typename T>
T Matrix<T>::dot(const Matrix<T> &X) const {
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	return Math::mt_dot(nRows_ * nColumns_, elem_, X.elem_, nThreads_);
}

template<typename T>
void Matrix<T>::scale(T alpha){
	Math::mt_scal(nRows_ * nColumns_, alpha, elem_, nThreads_);
}

template<typename T>
template<typename S>
void Matrix<T>::copy(const Matrix<S> &X){
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	Math::copy<S,T>(nRows_ * nColumns_, X.elem_, 1, elem_, 1);
}

template<typename T>
template<typename S>
void Matrix<T>::copy(const S *X, u32 rowOffset, u32 colOffset){
	require_lt(rowOffset, nRows_);
	require_lt(colOffset, nColumns_);
	return Math::copy<S,T>(nRows_ * nColumns_ - colOffset * nRows_ - rowOffset, X, 1, elem_ + colOffset * nRows_ + rowOffset, 1);
}

template<typename T>
template<typename S>
void Matrix<T>::copy(const std::vector<S> &X, u32 rowOffset, u32 colOffset){
	require_lt(rowOffset, nRows_);
	require_lt(colOffset, nColumns_);
	return Math::copy<S,T>(X.size(), &X.at(0), 1, elem_ + colOffset * nRows_ + rowOffset, 1);
}

template<typename T>
void Matrix<T>::fill(u32 rowA, u32 columnA, u32 rowB, u32 columnB, T value) {
	require_lt(rowA, nRows_);
	require_lt(rowB, nRows_);
	require_lt(columnA, nColumns_);
	require_lt(columnB, nColumns_);
	if ( (columnA < columnB) || ((columnA == columnB) && (rowA < rowB)) )
		std::fill(elem_ + columnA * nRows_ + rowA, elem_ + columnB * nRows_ + rowB + 1, value);
}

template<typename T>
void Matrix<T>::elementwiseMultiplication(const Matrix<T> &X){
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	// TODO parallelize
	std::transform(elem_, elem_ + nRows_ * nColumns_, X.elem_, elem_, std::multiplies<T>());
}

template<typename T>
void Matrix<T>::elementwiseDivision(const Matrix<T> &X){
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	std::transform(elem_, elem_ + nRows_ * nColumns_, X.elem_, elem_, std::divides<T>());
}

template<typename T>
void Matrix<T>::addConstantElementwise(T c) {
	std::transform(elem_, elem_ + nRows_ * nColumns_, elem_, std::bind2nd(std::plus<T>(), c));
}

template<typename T>
void Matrix<T>::addToColumn(const Vector<T> &v, u32 column, T alpha) {
	require_lt(column, nColumns_);
	require_eq(v.nRows(), nRows_);
	Math::mt_axpy(nRows_, alpha, v.begin(), elem_ + column * nRows_, nThreads_);
}

template<typename T>
void Matrix<T>::multiplyColumnByScalar(u32 column, T alpha) {
	require_lt(column, nColumns_);
	Math::scal(nRows_, alpha, &at(0, column), 1);
}

template<typename T>
void Matrix<T>::multiplyRowByScalar(u32 row, T alpha) {
	require_lt(row, nRows_);
	Math::scal(nColumns_, alpha, &at(row, 0), nRows_);
}

template<typename T>
void Matrix<T>::addToRow(const Vector<T> &v, u32 row, T alpha) {
	require_lt(row, nRows_);
	require_eq(v.nRows(), nColumns_);
	Math::axpy(nColumns_, alpha, v.begin(), 1, elem_ + row, nRows_);
}

template<typename T>
void Matrix<T>::addToAllColumns(const Vector<T> &v, T alpha){
	require_eq(v.nRows(), nRows_);
# pragma omp parallel for
	for (u32 i = 0; i < nColumns_; i++)
		Math::mt_axpy(nRows_, alpha, v.begin(), elem_ + i*nRows_, nThreads_);
}

template<typename T>
void Matrix<T>::addToAllRows(const Vector<T> &v, T alpha){
	require_eq(v.nRows(), nColumns_);
#pragma omp parallel for
	for (u32 j = 0; j < nColumns_; j++){
		T value = alpha * v.at(j);
		std::transform(elem_ + j*nRows_, elem_ + (j+1)*nRows_, elem_ + j*nRows_, std::bind2nd(std::plus<T>(), value));
	}
}

template<typename T>
void Matrix<T>::multiplyColumnsByScalars(const Vector<T> &scalars){
	require_eq(nColumns_, scalars.size());
#pragma omp parallel for
	for (u32 i = 0; i < nColumns_; i++)
		Math::scal(nRows_, scalars[i], &at(0, i), 1);
}

template<typename T>
void Matrix<T>::divideColumnsByScalars(const Vector<T> &scalars){
	require_eq(nColumns_, scalars.size());
#pragma omp parallel for
	for (u32 i = 0; i < nColumns_; i++)
		Math::scal(nRows_, (T) 1.0 / scalars[i], &at(0, i), 1);
}

template<typename T>
void Matrix<T>::multiplyRowsByScalars(const Vector<T> &scalars){
	require_eq(nRows_, scalars.size());
#pragma omp parallel for
	for (u32 i = 0; i < nRows_; i++)
		Math::scal(nColumns_, scalars.at(i), &at(i, 0), nRows_);
}

template<typename T>
void Matrix<T>::divideRowsByScalars(const Vector<T> &scalars){
	require_eq(nRows_, scalars.size());
#pragma omp parallel for
	for (u32 i = 0; i < nRows_; i++)
		Math::scal(nColumns_, (T) 1.0 / scalars.at(i), &at(i, 0), nRows_);
}

template<typename T>
void Matrix<T>::getRow(u32 rowIndex, Math::Vector<T> &row) const {
	require_lt(rowIndex, nRows_);
	row.resize(nColumns_);
	Math::copy(nColumns_, elem_ + rowIndex, nRows_, row.begin(), 1);
}

template<typename T>
void Matrix<T>::getColumn(u32 columnIndex, Math::Vector<T> &column) const {
	require_lt(columnIndex, nColumns_);
	column.resize(nRows_);
	Math::copy(nRows_, elem_ + columnIndex * nRows_, 1, column.begin(), 1);
}

template<typename T>
void Matrix<T>::setRow(u32 rowIndex, const Math::Vector<T> &row) {
	require_lt(rowIndex, nRows_);
	require_eq(row.size(), nColumns_);
	Math::copy(nColumns_, row.begin(), 1, elem_ + rowIndex, nRows_);
}

template<typename T>
void Matrix<T>::setColumn(u32 columnIndex, const Math::Vector<T> &column) {
	require_lt(columnIndex, nColumns_);
	require_eq(column.size(), nRows_);
	Math::copy(nRows_, column.begin(), 1, elem_ + columnIndex * nRows_, 1);
}

template<typename T>
void Matrix<T>::copyBlockFromMatrix(const Math::Matrix<T> &X, u32 rowIndexX, u32 colIndexX, u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns) {
	require_le(thisColIndex + nColumns, nColumns_);
	require_le(thisRowIndex + nRows, nRows_);
	require_le(colIndexX + nColumns, X.nColumns_);
	require_le(rowIndexX + nRows, X.nRows_);
	for (u32 column = 0; column < nColumns; column++){
		const T *posX =  &X.at(rowIndexX, colIndexX  + column);
		T * posThis = &at(thisRowIndex, thisColIndex + column);
		Math::copy(nRows, posX, 1, posThis, 1);
	}
}

template<typename T>
void Matrix<T>::addBlockFromMatrix(const Math::Matrix<T> &X, u32 rowIndexX, u32 colIndexX, u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns, T scale) {
	require_le(thisColIndex + nColumns, nColumns_);
	require_le(thisRowIndex + nRows, nRows_);
	require_le(colIndexX + nColumns, X.nColumns_);
	require_le(rowIndexX + nRows, X.nRows_);
	for (u32 column = 0; column < nColumns; column++){
		const T *posX =  &X.at(rowIndexX, colIndexX  + column);
		T * posThis = &at(thisRowIndex, thisColIndex + column);
		Math::axpy(nRows, scale, posX, 1, posThis, 1);
	}
}

template<typename T>
void Matrix<T>::addOuterProduct(const Math::Vector<T> &x, const Math::Vector<T> &y, T alpha, u32 lda){
	require_eq(x.size(), nRows_);
	require_eq(y.size(), nColumns_);
	require_le(lda, nRows_);
	if (lda == 0)
		lda = nRows_;
	Math::ger<T>(CblasColMajor, nRows_, nColumns_, alpha, x.begin(), 1, y.begin(), 1, elem_, lda);
}



// c = \alpha a * b + \gamma c
template<typename T>
void Matrix<T>::_gemm(bool transposeA, bool transposeB, u32 M, u32 N, u32 K,
		T scaleA, T* matrixA, u32 lda,
		T* matrixB, u32 ldb,
		T scaleC, T* matrixC, u32 ldc) {
	Math::gemm<T>(CblasColMajor,
			(transposeA ? CblasTrans : CblasNoTrans),
			(transposeB ? CblasTrans : CblasNoTrans),
			M, N, K,
			scaleA, matrixA, lda,
			matrixB, ldb,
			scaleC, matrixC, ldc);
}

// (*this) = (scaleA * matrixA) * matrixB + scaleC * (*this)
template<typename T>
void Matrix<T>::addMatrixProduct(const Matrix<T> &matrixA, const Matrix<T> &matrixB,
		T scaleC, T scaleA, bool transposeA, bool transposeB) {
	// final matrix (this) must be of size matrixProductNRows x matrixProductNColumns
	u32 matrixProductNRows, matrixProductNColumns;

	// boundary check depends on the configuration
	if ( (! transposeA) && (! transposeB) ) {
		require_eq(matrixA.nColumns(), matrixB.nRows());
		matrixProductNRows = matrixA.nRows();
		matrixProductNColumns = matrixB.nColumns();
	} else if ( (! transposeA) && (transposeB) ) {
		require_eq(matrixA.nColumns(), matrixB.nColumns());
		matrixProductNRows = matrixA.nRows();
		matrixProductNColumns = matrixB.nRows();
	} else if ( (transposeA) && (! transposeB) ) {
		require_eq(matrixA.nRows(), matrixB.nRows());
		matrixProductNRows = matrixA.nColumns();
		matrixProductNColumns = matrixB.nColumns();
	} else if ( (transposeA) && (transposeB) ) {
		require_eq(matrixA.nRows(), matrixB.nColumns());
		matrixProductNRows = matrixA.nColumns();
		matrixProductNColumns = matrixB.nRows();
	}
	require_eq(matrixProductNRows, nRows_);
	require_eq(matrixProductNColumns, nColumns_);

	// multiply the matrices
	// example: A(2,297); B(297,7000); C(2,7000); transposeA=false; transposeB=false; M=2; N=7000; K=297; LDA=2; LDB=7000; LDC=2
	Math::gemm<T>(CblasColMajor,
			(transposeA ? CblasTrans : CblasNoTrans), (transposeB ? CblasTrans : CblasNoTrans),
			matrixProductNRows, matrixProductNColumns, (transposeA ? matrixA.nRows() : matrixA.nColumns()),
			(T) scaleA, matrixA.begin(), matrixA.nRows(),
			matrixB.begin(), matrixB.nRows(),
			(T) scaleC, (*this).begin(), matrixProductNRows);
}

template<typename T>
void Matrix<T>::writeBinaryHeader(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	u8 tid;
	if (typeid(T) == typeid(f32)) {
		tid = 0;
	}
	else {
		tid = 1;
	}
	if (transpose)
		stream << tid << nColumns_ << nRows_;
	else
		stream << tid << nRows_ << nColumns_;
}

template<typename T>
void Matrix<T>::writeAsciiHeader(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	if (transpose)
		stream << nColumns_ << " " << nRows_ << Core::IOStream::endl;
	else
		stream << nRows_ << " " << nColumns_ << Core::IOStream::endl;
}

template<typename T>
void Matrix<T>::writeBinary(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	u32 I, J;
	if (transpose) {
		I = nColumns_;
		J = nRows_;
	}
	else {
		I = nRows_;
		J = nColumns_;
	}
	for (u32 i = 0; i < I; i++) {
		for (u32 j = 0; j < J; j++) {
			if (transpose)
				stream << at(j, i);
			else
				stream << at(i, j);
		}
	}
}

template<typename T>
void Matrix<T>::writeAscii(Core::IOStream& stream, bool transpose, bool scientific) {
	require(stream.is_open());
	if (scientific)
		stream << Core::IOStream::scientific;
	if (transpose) {
		for (u32 col = 0; col < nColumns_ - 1; col++) {
			for (u32 row = 0; row < nRows_ - 1; row++) {
				stream << at(row, col) << " ";
			}
			stream << at(nRows_ - 1, col) << Core::IOStream::endl;
		}
		for (u32 row = 0; row < nRows_ - 1; row++) {
			stream << at(row, nColumns_ - 1) << " ";
		}
		stream << at(nRows_ - 1, nColumns_ - 1);
	}
	else {
		for (u32 row = 0; row < nRows_ - 1; row++) {
			for (u32 col = 0; col < nColumns_ - 1; col++) {
				stream << at(row, col) << " ";
			}
			stream << at(row, nColumns_ - 1) << Core::IOStream::endl;
		}
		for (u32 col = 0; col < nColumns_ - 1; col++) {
			stream << at(nRows_ - 1, col) << " ";
		}
		stream << at(nRows_ - 1, nColumns_ - 1);
	}
}

template<typename T>
void Matrix<T>::write(const std::string& filename, bool transpose, bool scientific) {
	if (Core::Utils::isBinary(filename)) {
		Core::BinaryStream stream(filename, std::ios::out);
		writeBinaryHeader(stream, transpose);
		writeBinary(stream, transpose);
		stream.close();
	}
	else if (Core::Utils::isGz(filename)) {
		Core::CompressedStream stream(filename, std::ios::out);
		writeAsciiHeader(stream, transpose);
		writeAscii(stream, transpose, scientific);
		stream.close();
	}
	else {
		Core::AsciiStream stream(filename, std::ios::out);
		writeAsciiHeader(stream, transpose);
		writeAscii(stream, transpose, scientific);
		stream.close();
	}
}

template<typename T>
bool Matrix<T>::readBinaryHeader(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	u8 tid;
	stream >> tid;
	if (((tid == 0) && (typeid(T) != typeid(f32))) || ((tid == 1) && (typeid(T) != typeid(f64)))) {
		return false;
	}
	u32 nRows, nColumns;
	stream >> nRows;
	stream >> nColumns;
	if (transpose)
		resize(nColumns, nRows);
	else
		resize(nRows, nColumns);
	return true;
}

template<typename T>
bool Matrix<T>::readAsciiHeader(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	u32 nRows, nColumns;
	stream >> nRows;
	stream >> nColumns;
	if (transpose)
		resize(nColumns, nRows);
	else
		resize(nRows, nColumns);
	return true;
}

template<typename T>
void Matrix<T>::read(Core::IOStream& stream, bool transpose) {
	if (transpose) {
		for (u32 col = 0; col < nColumns_; col++) {
			for (u32 row = 0; row < nRows_; row++) {
				stream >> at(row, col);
			}
		}
	}
	else {
		for (u32 row = 0; row < nRows_; row++) {
			for (u32 col = 0; col < nColumns_; col++) {
				stream >> at(row, col);
			}
		}
	}
}

template<typename T>
void Matrix<T>::read(const std::string& filename, bool transpose) {
	if (Core::Utils::isBinary(filename)) {
		Core::BinaryStream stream(filename, std::ios::in);
		if (!readBinaryHeader(stream, transpose)) {
			std::cerr << "In file " << filename << ": Header does not match. Abort." << std::endl;
			exit(1);
		}
		read(stream, transpose);
		stream.close();
	}
	else if (Core::Utils::isGz(filename)) {
		Core::CompressedStream stream(filename, std::ios::in);
		readAsciiHeader(stream, transpose);
		read(stream, transpose);
		stream.close();;
	}
	else {
		Core::AsciiStream stream(filename, std::ios::in);
		readAsciiHeader(stream, transpose);
		read(stream, transpose);
		stream.close();
	}
}

} // namespace (Math)


#endif /* MATH_MATRIX_HH_ */
