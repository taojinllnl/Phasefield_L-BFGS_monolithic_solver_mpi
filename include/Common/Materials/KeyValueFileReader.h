//
//  KeyValueFileReader.h
//  main
//
//

#ifndef Common_Materials_KeyValueFileReader_h
#define Common_Materials_KeyValueFileReader_h


#include <filesystem>
#include <map>
#include <set>
#include <string>


namespace PhaseField
{
  /**
   * Read an exact set of numeric key-value assignments from a text file.
   *
   * The accepted syntax is "key = numeric_value;" with optional C++-style
   * line comments. Unknown, duplicate, and missing keys are rejected.
   */
  std::map<std::string, double>
  read_numeric_key_value_file(
    const std::filesystem::path &file,
    const std::set<std::string> &required_keys);

} // namespace PhaseField

#endif /* Common_Materials_KeyValueFileReader_h */
