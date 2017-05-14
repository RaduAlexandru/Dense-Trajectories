/*
 * Utils.hh
 *
 *  Created on: 24.03.2014
 *      Author: richard
 */

#ifndef CORE_UTILS_HH_
#define CORE_UTILS_HH_

#include <Modules.hh>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <time.h>
#include <execinfo.h>
#include "Types.hh"

#ifdef MODULE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#else
namespace cv {
	typedef u32 Mat; // dummy definition if OpenCV is not used
}
#endif

/* require */

#define _require(a, s) { \
if (!(a)) \
{ \
printf("\n\t\033[0;31mREQUIRE\033[0;0m: %s:%d: %s is not true%s. Abort.\n", __FILE__, __LINE__, #a, s); \
Core::Utils::print_stacktrace(); \
exit(1); \
} }

#define _require2(v, a, b) { \
if (!(v)) \
{ \
std::stringstream s; \
s << " (" << a << " vs. " << b << ")"; \
_require(v, s.str().c_str()); \
} }

#define require(a) { _require(a, ""); }

#define require_eq(a,b) { _require2(a == b, a, b); }
#define require_le(a,b) { _require2(a <= b, a, b); }
#define require_ge(a,b) { _require2(a >= b, a, b); }
#define require_lt(a,b) { _require2(a < b, a, b); }
#define require_gt(a,b) { _require2(a > b, a, b); }

/* end require */

namespace Core {

class Utils {
public:
	static void print_stacktrace();

	// string operations
	static void tokenizeString(std::vector<std::string> & result, const std::string& str, const char* delimiter = " ");
	static void replaceChar(std::string& str, char oldCh, char newCh);
	static void removeAllOf(std::string& str, const char* chars);
	static bool isBinary(const std::string& filename);
	static bool isGz(const std::string& filename);
	static void appendSuffix(std::string& filename, const std::string& suffix);

	static f64 timeDiff(timeval& start, timeval& end);

	/*
	 * some useful converters
	 */
	static void copyCVMatToMemory(const cv::Mat& image, Float* dest);
	static void copyMemoryToCVMat(const Float* src, cv::Mat& image);

	/*
	 * timer to measure performance
	 */
	class Timer
	{
	private:
		struct timeval t;
		u64 startVal_;
		u64 elapsedTime_;
	public:
		Timer();
		/*
		 * run the timer
		 */
		void run();
		/*
		 * interrupt the timer
		 */
		void stop();
		/*
		 * reset the timer
		 */
		void reset();
		/*
		 * @return the elapsed time in seconds
		 */
		Float time();
	};
};

} // namespace

#endif /* CORE_UTILS_HH_ */
