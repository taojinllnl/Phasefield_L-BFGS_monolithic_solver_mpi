//
//  MaterialProfile.h
//  main
//
//

#ifndef Common_Materials_MaterialProfile_h
#define Common_Materials_MaterialProfile_h


#include <filesystem>


namespace PhaseField
{
  /**
   * Material properties that are independent of a simulation case.
   *
   * Case-dependent phase-field parameters, such as the length scale and
   * residual stiffness, are intentionally not stored in this structure.
   */
  struct MaterialProfile
  {
    double lambda;
    double mu;
    double gc;
    double tensile_strength;
  };


  /**
   * Read a strict key-value material profile.
   *
   * The file must define exactly the four MaterialProfile fields using the
   * syntax "key = numeric_value;". C++-style line comments are supported.
   */
  MaterialProfile
  read_material_profile(const std::filesystem::path &profile_file);

} // namespace PhaseField

#endif /* Common_Materials_MaterialProfile_h */
