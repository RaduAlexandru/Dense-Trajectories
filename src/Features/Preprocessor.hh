/*
 * Preprocessor.hh
 *
 *  Created on: Sep 30, 2014
 *      Author: richard
 */

#ifndef FEATURES_PREPROCESSOR_HH_
#define FEATURES_PREPROCESSOR_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Vector.hh>
#include <Math/Matrix.hh>

namespace Features {

/*
 * Base class for feature preprocessors
 */
class Preprocessor
{
private:
	static const Core::ParameterEnum paramType_;
	enum Type { none, vectorSubtraction, vectorDivision, matrixMultiplication };
protected:
	std::string name_;
	u32 inputDimension_;
	u32 outputDimension_;
	bool isInitialized_;
public:
	Preprocessor(const char* name);
	virtual ~Preprocessor() {}
	virtual void initialize(u32 inputDimension);
	u32 inputDimension() const;
	u32 outputDimension() const;
	virtual bool needsContext() { return false; }

	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) = 0;

	static Preprocessor* createPreprocessor(const char* name);
};

/*
 * vector subtraction
 */
class VectorSubtractionPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterString paramVectorFile_;
protected:
	std::string vectorFile_;
	Math::Vector<Float> vector_;
public:
	VectorSubtractionPreprocessor(const char* name);
	virtual ~VectorSubtractionPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * vector division
 */
class VectorDivisionPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterString paramVectorFile_;
protected:
	std::string vectorFile_;
	Math::Vector<Float> vector_;
public:
	VectorDivisionPreprocessor(const char* name);
	virtual ~VectorDivisionPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * matrix multiplication
 */
class MatrixMultiplicationPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterString paramMatrixFile_;
	static const Core::ParameterBool paramTransposeMatrix_;
protected:
	std::string matrixFile_;
	Math::Matrix<Float> matrix_;
	bool transpose_;
public:
	MatrixMultiplicationPreprocessor(const char* name);
	virtual ~MatrixMultiplicationPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

} // namespace

#endif /* FEATURES_PREPROCESSOR_HH_ */
