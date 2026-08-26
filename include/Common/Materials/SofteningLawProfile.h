//
//  SofteningLawProfile.h
//  main
//
//

#ifndef Common_Materials_SofteningLawProfile_h
#define Common_Materials_SofteningLawProfile_h


#include <filesystem>


namespace PhaseField
{
  /**
   * Coefficients that define a cohesive phase-field softening law.
   */
  struct SofteningLawProfile
  {
    double p;
    double a2;
    double a3;
  };


  /**
   * Read a strict key-value softening-law profile.
   */
  SofteningLawProfile
  read_softening_law_profile(const std::filesystem::path &profile_file);

} // namespace PhaseField

#endif /* Common_Materials_SofteningLawProfile_h */
