//
//  DoFStrategyBase.h
//  main
//
//

#ifndef DoFStrategyBase_h
#define DoFStrategyBase_h

#include <deal.II/dofs/dof_handler.h>


#include "../Traits.h"

namespace grid
{

template <typename Tria>
class DoFStrategyBase
{
public:
    constexpr static int  dim           = Tria::dimension;
    constexpr static int  spacedim      = Tria::space_dimension;
    constexpr static bool isDistribted  = std::is_same_v<Tria, DTria<dim,spacedim>>;
    
    using DoFHandler = dealii::DoFHandler<dim, spacedim>;
    using DCellIter = typename DoFHandler::active_cell_iterator;
    
public:
    virtual ~DoFStrategyBase() = default;
    
    virtual bool setRefineFlag(const DCellIter& cell) const;
};

}

#endif /* DoFStrategyBase_h */
