#include "../../../include/Common/Materials/KeyValueFileReader.h"

#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>


namespace PhaseField
{
  std::map<std::string, double>
  read_numeric_key_value_file(
    const std::filesystem::path &file,
    const std::set<std::string> &required_keys)
  {
    std::ifstream input(file);
    if (!input)
      throw std::runtime_error("Cannot open key-value file: " + file.string());

    std::map<std::string, double> values;
    const std::regex assignment_regex(
      R"(^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*;\s*$)");

    std::string  line;
    unsigned int line_number = 0;
    while (std::getline(input, line))
      {
        ++line_number;

        const std::size_t comment_position = line.find("//");
        if (comment_position != std::string::npos)
          line.erase(comment_position);

        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
          continue;

        std::smatch match;
        if (!std::regex_match(line, match, assignment_regex))
          {
            std::ostringstream message;
            message << "Invalid key-value syntax in " << file.string() << ':'
                    << line_number;
            throw std::runtime_error(message.str());
          }

        const std::string key = match[1].str();
        if (required_keys.count(key) == 0)
          throw std::runtime_error("Unknown key '" + key +
                                   "' in key-value file: " + file.string());

        const double value = std::stod(match[2].str());
        if (!std::isfinite(value))
          throw std::runtime_error("Non-finite value for key '" + key +
                                   "' in key-value file: " + file.string());

        if (!values.emplace(key, value).second)
          throw std::runtime_error("Duplicate key '" + key +
                                   "' in key-value file: " + file.string());
      }

    for (const std::string &key : required_keys)
      if (values.count(key) == 0)
        throw std::runtime_error("Missing key '" + key +
                                 "' in key-value file: " + file.string());

    return values;
  }

} // namespace PhaseField
