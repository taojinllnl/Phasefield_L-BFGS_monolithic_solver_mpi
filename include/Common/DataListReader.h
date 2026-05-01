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
    
    /**
     * Read key-value pairs from a text file.
     *
     * The input file is expected to contain assignments in the form:
     *
     *     key = value;
     *
     * Lines may contain comments introduced by "//". Everything after "//"
     * on the same line will be ignored.
     *
     * Only numeric values are supported. The parsed value is first converted
     * to double using std::stod(), and then cast to type T.
     *
     * @param filename
     *        Name of the file to be read.
     *
     * @param keys
     *        List of required parameter names. After reading the file, the
     *        function checks that every name in this list has been found.
     *
     * @param values
     *        Output map storing the parsed key-value pairs. Existing entries
     *        are overwritten if the same key appears in the file.
     *
     * @throws std::runtime_error
     *         If the file cannot be opened, or if any required parameter is
     *         missing.
     */
    template <typename T>
    static void read_keys_values(const std::string& filename,
                                 const std::vector<std::string>& keys,
                                 std::unordered_map<std::string, T>& values);
};


template <typename T>
void
DataListReader::read_keys_values(const std::string& filename,
                                 const std::vector<std::string>& keys,
                                 std::unordered_map<std::string, T>& values)
{
    std::ifstream input(filename);
    if (!input)
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    values.clear();
    
    const std::unordered_set<std::string> required_keys(keys.begin(),
                                                        keys.end());
    
    if (required_keys.empty())
    {
        return;
    }

    std::string line;

    // match format:
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
            
            // Ignore keys that are not requested.
            if (required_keys.find(name) == required_keys.end())
            {
                continue;
            }
            
            const T value = static_cast<T>(std::stod(match[2].str()));
            
            values.emplace(name, value);
        }
        
        // Stop once all requested parameters have been found.
        if (values.size() == required_keys.size())
        {
            break;
        }
    }

    // check whether all required parameters exist
    for (const auto& key : keys)
    {
        if (values.find(key) == values.end())
        {
            throw std::runtime_error("Missing parameter in geo file: " + key);
        }
    }
}


}

#endif /* DataListReader_h */
