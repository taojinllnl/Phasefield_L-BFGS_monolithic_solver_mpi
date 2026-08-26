//
//  TriaStrategyBase.h
//  main
//
//

#ifndef TriaStrategyBase_h
#define TriaStrategyBase_h


#include "../Traits.h"

namespace grid
{

template <typename Tria>
class TriaStrategyBase
{
public:
    constexpr static int  dim           = Tria::dimension;
    constexpr static int  spacedim      = Tria::space_dimension;
    constexpr static bool isDistribted  = std::is_same_v<Tria, DTria<dim,spacedim>>;
    
    using TCellIter = typename Tria::active_cell_iterator;
    
public:
    virtual ~TriaStrategyBase() = default;
    
    virtual bool setRefineFlag(const TCellIter& cell) const;
};

}

#endif /* TriaStrategyBase_h */
