//
//  QPHUpdateFlag.h
//  main
//

#ifndef QPHUpdateFlag_h
#define QPHUpdateFlag_h

#include <type_traits>

namespace PhaseField
{
  namespace flags
  {

    enum class QPHUpdateFlag : unsigned int
    {
      none = 0u,

      strain_decomp      = 1u << 0,
      stress_positive    = 1u << 1,
      stress             = 1u << 2,
      CCCC               = 1u << 3,
      strain_energy_pos  = 1u << 4,
      strain_energy_neg  = 1u << 5,
      strain_energy_tot  = 1u << 6,
      energy_dissipation = 1u << 7
    };


    enum class QPHNeedFlag : unsigned int
    {
      none = 0u,

      positive_ramp = 1u << 0,
      negative_ramp = 1u << 1,

      u_gradient          = 1u << 2,
      phasefield_value    = 1u << 3,
      phasefield_gradient = 1u << 4,
      previous_phasefield = 1u << 5
    };


    // -----------------------------------------------------------------------------
    // Basic helpers
    // -----------------------------------------------------------------------------

    using QPHUpdateFlagInt = std::underlying_type_t<QPHUpdateFlag>;
    using QPHNeedFlagInt   = std::underlying_type_t<QPHNeedFlag>;


    constexpr inline QPHUpdateFlagInt
    to_int(const QPHUpdateFlag flag) noexcept
    {
      return static_cast<QPHUpdateFlagInt>(flag);
    }


    constexpr inline QPHNeedFlagInt
    to_int(const QPHNeedFlag flag) noexcept
    {
      return static_cast<QPHNeedFlagInt>(flag);
    }


    constexpr inline QPHUpdateFlag
    to_update_flag(const QPHUpdateFlagInt value) noexcept
    {
      return static_cast<QPHUpdateFlag>(value);
    }


    constexpr inline QPHNeedFlag
    to_need_flag(const QPHNeedFlagInt value) noexcept
    {
      return static_cast<QPHNeedFlag>(value);
    }


    // -----------------------------------------------------------------------------
    // Bitwise operators for QPHUpdateFlag
    // -----------------------------------------------------------------------------

    constexpr inline QPHUpdateFlag
    operator|(const QPHUpdateFlag lhs, const QPHUpdateFlag rhs) noexcept
    {
      return to_update_flag(to_int(lhs) | to_int(rhs));
    }


    constexpr inline QPHUpdateFlag &
    operator|=(QPHUpdateFlag &lhs, const QPHUpdateFlag rhs) noexcept
    {
      lhs = lhs | rhs;
      return lhs;
    }


    constexpr inline QPHUpdateFlag
    operator&(const QPHUpdateFlag lhs, const QPHUpdateFlag rhs) noexcept
    {
      return to_update_flag(to_int(lhs) & to_int(rhs));
    }


    constexpr inline QPHUpdateFlag &
    operator&=(QPHUpdateFlag &lhs, const QPHUpdateFlag rhs) noexcept
    {
      lhs = lhs & rhs;
      return lhs;
    }


    constexpr inline QPHUpdateFlag
    operator~(const QPHUpdateFlag flag) noexcept
    {
      return to_update_flag(~to_int(flag));
    }


    constexpr inline bool
    has_flag(const QPHUpdateFlag flags, const QPHUpdateFlag flag) noexcept
    {
      return to_int(flags & flag) != 0u;
    }


    constexpr inline bool
    has_any_flag(const QPHUpdateFlag flags, const QPHUpdateFlag mask) noexcept
    {
      return to_int(flags & mask) != 0u;
    }


    // -----------------------------------------------------------------------------
    // Bitwise operators for QPHNeedFlag
    // -----------------------------------------------------------------------------

    constexpr inline QPHNeedFlag
    operator|(const QPHNeedFlag lhs, const QPHNeedFlag rhs) noexcept
    {
      return to_need_flag(to_int(lhs) | to_int(rhs));
    }


    constexpr inline QPHNeedFlag &
    operator|=(QPHNeedFlag &lhs, const QPHNeedFlag rhs) noexcept
    {
      lhs = lhs | rhs;
      return lhs;
    }


    constexpr inline QPHNeedFlag
    operator&(const QPHNeedFlag lhs, const QPHNeedFlag rhs) noexcept
    {
      return to_need_flag(to_int(lhs) & to_int(rhs));
    }


    constexpr inline QPHNeedFlag &
    operator&=(QPHNeedFlag &lhs, const QPHNeedFlag rhs) noexcept
    {
      lhs = lhs & rhs;
      return lhs;
    }


    constexpr inline QPHNeedFlag
    operator~(const QPHNeedFlag flag) noexcept
    {
      return to_need_flag(~to_int(flag));
    }


    constexpr inline bool
    has_flag(const QPHNeedFlag flags, const QPHNeedFlag flag) noexcept
    {
      return to_int(flags & flag) != 0u;
    }


    constexpr inline bool
    has_any_flag(const QPHNeedFlag flags, const QPHNeedFlag mask) noexcept
    {
      return to_int(flags & mask) != 0u;
    }


    // -----------------------------------------------------------------------------
    // Masks
    // -----------------------------------------------------------------------------

    inline constexpr QPHUpdateFlag all_qph_update_flags =
      QPHUpdateFlag::strain_decomp | QPHUpdateFlag::stress_positive |
      QPHUpdateFlag::stress | QPHUpdateFlag::CCCC |
      QPHUpdateFlag::strain_energy_pos | QPHUpdateFlag::strain_energy_neg |
      QPHUpdateFlag::strain_energy_tot | QPHUpdateFlag::energy_dissipation;


    inline constexpr QPHNeedFlag all_qph_need_flags =
      QPHNeedFlag::positive_ramp | QPHNeedFlag::negative_ramp |
      QPHNeedFlag::u_gradient | QPHNeedFlag::phasefield_value |
      QPHNeedFlag::phasefield_gradient | QPHNeedFlag::previous_phasefield;


    constexpr inline bool
    has_only_known_update_flags(const QPHUpdateFlag flags) noexcept
    {
      return (to_int(flags) & ~to_int(all_qph_update_flags)) == 0u;
    }



    // -----------------------------------------------------------------------------
    // complete_update()
    //   Completes the requested QPH update flags.
    //
    // Dependencies:
    //
    //   stress
    //     -> stress_positive
    //     -> strain_decomp
    //
    //   stress_positive
    //     -> strain_decomp
    //
    //   CCCC
    //     -> strain_decomp
    //
    //   strain_energy_pos
    //     -> strain_decomp
    //
    //   strain_energy_neg
    //     -> strain_decomp
    //
    //   strain_energy_tot
    //     -> strain_energy_pos
    //     -> strain_energy_neg
    //     -> strain_decomp
    //
    //   energy_dissipation
    //     does not require strain decomposition by itself.
    // -----------------------------------------------------------------------------

    constexpr inline QPHUpdateFlag
    complete_update(const QPHUpdateFlag raw_flags) noexcept
    {
      QPHUpdateFlag flags = raw_flags & all_qph_update_flags;

      if (has_flag(flags, QPHUpdateFlag::stress))
        flags |= QPHUpdateFlag::stress_positive;

      if (has_flag(flags, QPHUpdateFlag::strain_energy_tot))
        flags |=
          QPHUpdateFlag::strain_energy_pos | QPHUpdateFlag::strain_energy_neg;

      if (has_flag(flags, QPHUpdateFlag::stress_positive) ||
          has_flag(flags, QPHUpdateFlag::stress) ||
          has_flag(flags, QPHUpdateFlag::CCCC) ||
          has_flag(flags, QPHUpdateFlag::strain_energy_pos) ||
          has_flag(flags, QPHUpdateFlag::strain_energy_neg) ||
          has_flag(flags, QPHUpdateFlag::strain_energy_tot))
        flags |= QPHUpdateFlag::strain_decomp;

      return flags;
    }


    // -----------------------------------------------------------------------------
    // QPHUpdateFlag checking
    // -----------------------------------------------------------------------------

    constexpr inline bool
    is_complete_update(const QPHUpdateFlag flags) noexcept
    {
      return has_only_known_update_flags(flags) &&
             to_int(flags) == to_int(complete_update(flags));
    }


    template <QPHUpdateFlag flags>
    constexpr inline QPHUpdateFlag
    complete_update_checked() noexcept
    {
      static_assert(has_only_known_update_flags(flags),
                    "QPHUpdateFlag contains unknown bits.");

      return complete_update(flags);
    }


    template <QPHUpdateFlag flags>
    constexpr inline void
    assert_complete_update() noexcept
    {
      static_assert(has_only_known_update_flags(flags),
                    "QPHUpdateFlag contains unknown bits.");

      static_assert(
        is_complete_update(flags),
        "Incomplete QPHUpdateFlag dependency. "
        "Use complete_update_checked<raw_flags>() or complete_update(raw_flags).");
    }


    // -----------------------------------------------------------------------------
    // QPHUpdateFlag --> QPHNeedFlag
    // -----------------------------------------------------------------------------

    constexpr inline QPHNeedFlag
    generate_need(const QPHUpdateFlag raw_flags) noexcept
    {
      const QPHUpdateFlag flags = complete_update(raw_flags);

      QPHNeedFlag needs = QPHNeedFlag::none;

      const bool strain_decomp = has_flag(flags, QPHUpdateFlag::strain_decomp);

      const bool stress_positive =
        has_flag(flags, QPHUpdateFlag::stress_positive);

      const bool stress = has_flag(flags, QPHUpdateFlag::stress);

      const bool CCCC = has_flag(flags, QPHUpdateFlag::CCCC);

      const bool strain_energy_pos =
        has_flag(flags, QPHUpdateFlag::strain_energy_pos);

      const bool strain_energy_neg =
        has_flag(flags, QPHUpdateFlag::strain_energy_neg);

      const bool strain_energy_tot =
        has_flag(flags, QPHUpdateFlag::strain_energy_tot);

      const bool energy_dissipation =
        has_flag(flags, QPHUpdateFlag::energy_dissipation);


      if (stress_positive || stress || strain_energy_pos || strain_energy_tot)
        needs |= QPHNeedFlag::positive_ramp;


      if (stress || strain_energy_neg || strain_energy_tot)
        needs |= QPHNeedFlag::negative_ramp;


      if (strain_decomp || stress_positive || stress || CCCC ||
          strain_energy_pos || strain_energy_neg || strain_energy_tot)
        needs |= QPHNeedFlag::u_gradient;


      if (stress || CCCC || strain_energy_tot || energy_dissipation)
        needs |= QPHNeedFlag::phasefield_value;



      if (stress || energy_dissipation)
        needs |= QPHNeedFlag::phasefield_gradient;


      if (energy_dissipation)
        needs |= QPHNeedFlag::previous_phasefield;


      return needs;
    }



  } // namespace flags
} // namespace PhaseField



#endif /* QPHUpdateFlag_h */
