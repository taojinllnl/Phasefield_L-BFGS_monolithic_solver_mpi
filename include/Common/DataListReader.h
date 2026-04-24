//
//  DataListReader.h
//  main
//
//

#ifndef DataListReader_h
#define DataListReader_h


#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace common {

class DataListReader
{
public:
    
    template <typename T>
    static void read_keys_values(const std::string& filename,
                          const std::vector<std::string>& parameter_names,
                          std::unordered_map<std::string, T>& values);
};


template <typename T>
void
DataListReader::read_keys_values(const std::string& filename,
                          const std::vector<std::string>& parameter_names,
                          std::unordered_map<std::string, T>& values)
{
    std::ifstream input(filename);
    if (!input)
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;

    // match type:
    // L  = 1.0;
    // C1 = 0.010;
    // W  = 1.0e-6;
    const std::regex assignment_regex(
        R"(^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*;)"
    );

    while (std::getline(input, line))
    {
        // remove comments after "//"
        const std::size_t comment_pos = line.find("//");
        if (comment_pos != std::string::npos)
        {
            line = line.substr(0, comment_pos);
        }

        std::smatch match;
        if (std::regex_search(line, match, assignment_regex))
        {
            const std::string name = match[1].str();
            const T value = static_cast<T>(std::stod(match[2].str()));

            values[name] = value;
        }
    }

    // check whether all required parameters exist
    for (const auto& name : parameter_names)
    {
        if (values.find(name) == values.end())
        {
            throw std::runtime_error("Missing parameter in geo file: " + name);
        }
    }
}


}

#endif /* DataListReader_h */
