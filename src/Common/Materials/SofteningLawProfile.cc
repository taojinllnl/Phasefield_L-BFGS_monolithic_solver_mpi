#include "../../../include/Common/Materials/SofteningLawProfile.h"

#include "../../../include/Common/Materials/KeyValueFileReader.h"


namespace PhaseField
{
  SofteningLawProfile
  read_softening_law_profile(const std::filesystem::path &profile_file)
  {
    const auto values =
      read_numeric_key_value_file(profile_file, {"p", "a2", "a3"});

    return {values.at("p"), values.at("a2"), values.at("a3")};
  }

} // namespace PhaseField
