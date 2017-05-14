/*
 * Example.cc
 *
 *  Created on: Apr 25, 2017
 *      Author: richard
 */

#include "Example.hh"

using namespace ActionRecognition;

/* parameter definition */
const Core::ParameterString Example::paramSomeFile_(
		"filename",    // parameter name
		"",            // default value
		"example");    // prefix (set parameter via --example.some-string=your-string)

const Core::ParameterInt Example::paramSomeInt_(
		"some-int",    // parameter name
		42,            // default value
		"example");    // prefix (set parameter via --example.some-int=11)

const Core::ParameterFloat Example::paramSomeFloat_(
		"some-float",  // parameter name
		0.123,         // default value
		"example");    // prefix (set parameter via --example.some-float=0.1)

/* constructor */
Example::Example() :
		filename_(Core::Configuration::config(paramSomeFile_)),
		someInt_(Core::Configuration::config(paramSomeInt_)),
		someFloat_(Core::Configuration::config(paramSomeFloat_))
{
	require(!filename_.empty()); // make sure paramter filename contains some string
}

void Example::logMessages() {

	// write something to the log (default: std::cout, write log to file by setting parameter --log-file=your-file.log)
	Core::Log::os("This is an example log. The string ") << filename_ << " has been passed as parameter.";
	// write something with indentation in xml-style
	Core::Log::openTag("example");
	Core::Log::os("This is an example log in xml style with an opening and closing tag.");
	Core::Log::closeTag();

	// example: abort execution due to some error (e.g. wrong parameter)
	if (someFloat_ <= 0)
		Core::Error::msg("Parameter some-float must be positive.") << Core::Error::abort;
}

void Example::matrixUsage() {

	Math::Matrix<Float> A(5, 3); // create 5x3 matrix
	A.setToZero();
	A.at(2, 2) = 1.0;
	A.addConstantElementwise(10.0);
	Math::Matrix<Float> B(5, 3);
	B.fill(2.0);
	B.at(0, 0) = 0.0;
	A.elementwiseMultiplication(B);
	A.write("matrix.gz"); // write matrix to file
}

void Example::fileIO() {

	/* example: write a gzipped file */
	Core::Log::openTag("writing example");
	if (!Core::Utils::isGz(filename_)) // if file does not end on .gz
		filename_.append(".gz");
	Core::CompressedStream outStream(filename_, std::ios::out);
	outStream << "some string followed by a newline" << Core::IOStream::endl;
	outStream << someFloat_ << " " << someInt_ << Core::IOStream::endl; // write numbers to the file
	outStream.close();
	Core::Log::closeTag();

	/* example: read a gzipped file */
	Core::Log::openTag("reading example 1");
	Core::CompressedStream inStream(filename_, std::ios::in);
	// read a complete line as a string
	std::string line;
	inStream.getline(line);
	// read the float and int
	Float f;
	inStream >> f;
	u32 i;
	inStream >> i;
	Core::Log::os("Read line \"") << line << "\" and int " << i << " and float " << f;
	inStream.close();
	Core::Log::closeTag();

	/* example: read all lines of a file */
	Core::Log::openTag("reading example 2");
	Core::CompressedStream inStream2(filename_, std::ios::in);
	std::vector<std::string> allLines;
	while (inStream2.getline(line)) {
		Core::Log::os("Line by line reading: ") << line;
	}
	inStream2.close();
	Core::Log::closeTag();
}

/* function */
void Example::run() {
	logMessages();
	matrixUsage();
	fileIO();
}
