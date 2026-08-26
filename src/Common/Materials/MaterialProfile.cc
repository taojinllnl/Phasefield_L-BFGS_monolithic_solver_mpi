#include "../../../include/Common/Materials/MaterialProfile.h"

#include "../../../include/Common/Materials/KeyValueFileReader.h"

#include <stdexcept>


namespace PhaseField
{
  MaterialProfile
  read_material_profile(const std::filesystem::path &profile_file)
  {
    const auto values = read_numeric_key_value_file(
      profile_file, {"lambda", "mu", "gc", "tensile_strength"});

    if (values.at("lambda") <= 0.0 || values.at("mu") <= 0.0 ||
        values.at("gc") <= 0.0 || values.at("tensile_strength") <= 0.0)
      throw std::runtime_error(
        "All values in a material profile must be positive: " +
        profile_file.string());

    return {values.at("lambda"),
            values.at("mu"),
            values.at("gc"),
            values.at("tensile_strength")};
  }

} // namespace PhaseField
