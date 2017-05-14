/*
 * Log.hh
 *
 *  Created on: 25.03.2014
 *      Author: richard
 */

#ifndef CORE_LOG_HH_
#define CORE_LOG_HH_

#include <ostream>
#include <fstream>
#include <string>
#include <vector>
#include "Types.hh"
#include "Parameter.hh"

namespace Core {

class Log
{
private:
	static const ParameterString paramLogFile;

	std::ofstream ofs_; // output file stream
	std::ostream* os_; // output stream for logging
	std::vector<std::string> tags_; // stack containing all currently open tags

	u32 indentationLevel();
	void indent();
	void setOutputFile(const char* filename);

	static Log* theInstance_;
	static Log* getInstance();
	Log();
public:
	static std::ostream& os(const char* msg = "");
	static void openTag(const char* tag, const char* description = "");
	static void openTag(std::string& tag) { Log::openTag(tag.c_str()); }
	static void closeTag();
	static void finalize();
};

} // namespace


#endif /* CORE_LOG_HH_ */
