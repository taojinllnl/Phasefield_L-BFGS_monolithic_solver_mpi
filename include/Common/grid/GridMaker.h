//
//  GridMaker.h
//  main
//
//

#ifndef GridMaker_h
#define GridMaker_h

#include <functional>
#include <memory>




#include "../MPIInfo.h"

#include "TriaStrategyBase.h"
#include "DoFStrategyBase.h"

namespace grid
{


template <typename Tria, bool supportRepartioning>
class GridMaker
{
public:
    constexpr static int  dim           = Tria::dimension;
    constexpr static int  spacedim      = Tria::space_dimension;
    constexpr static bool isDistribted  = std::is_same_v<Tria, DTria<dim,spacedim>>;
    
    using DoFHandler = dealii::DoFHandler<dim, spacedim>;
    
    using TCellIter = typename Tria::active_cell_iterator;
    using DCellIter = typename DoFHandler::active_cell_iterator;
private:
    
    
    using TCellSelector = std::function<bool(const TCellIter& cell)>;
    using DCellSelector = std::function<bool(const DCellIter& cell)>;
    
    
    const common::MPIInfo& __mpiInfo;
    

    
    std::unique_ptr<TriaStrategyBase<Tria>>  __triaStrategyPtr{};
    std::unique_ptr<DoFStrategyBase<Tria>>   __dofStrategyPtr{};
    
    
    template <bool hasTCellSelector, bool isDoFBased, typename CellSelectorType>
    unsigned int __refine(Tria& tria,
                          const DoFHandler* dofHandler,
                          const CellSelectorType* CellSelector,
                          const unsigned int refineTimesLimit) const;
    
    

    
    
public:
    
    inline static constexpr auto defaultSelector = [](const auto& /*cell*/)
    {
        return true;
    };

    
    virtual ~GridMaker() = default;
    
    GridMaker(const common::MPIInfo& mpiInfo);
    
//    void bindTriaStrategy(const TriaStrategyBase<Tria>& strategy);
//    bool isBoundTriaStrategy() const;
//    
//    
//    void bindDoFStrategy(const DoFStrategyBase<Tria>& strategy);
//    bool isBoundDoFStrategy() const;
//    
//    
//    unsigned int refine(Tria& tria,
//                        const TCellSelector& triaCellSelector,
//                        const unsigned int refineTimesLimit) const;
//    
//    unsigned int refine(Tria& tria,
//                        const unsigned int refineTimesLimit) const;
//    
//    
//    unsigned int refine(Tria& tria,
//                        const DoFHandler& dof,
//                        const DCellSelector& dofCellSelector,
//                        const unsigned int refineTimesLimit) const;
//    
//    unsigned int refine(Tria& tria,
//                        const DoFHandler& dof,
//                        const unsigned int refineTimesLimit) const;
    
    
    template <typename CellSelectorType = std::decay_t<decltype(defaultSelector)>, bool LocallyOwend=true>
    unsigned int refineInitialMesh(Tria&                   tria,
                                   const TriaStrategyBase<Tria>& strategy,
                                   const unsigned int      refineTimesLimit = 100,
                                   const CellSelectorType& cellSelector = defaultSelector);
    
    
//    template <typename CellSelectorType>
    
//    unsigned int refineByStrategy(Tria&                   tria,
//                                  const DoFHandler*       dofHandler,
//                                  const CellSelectorType* CellSelector,
//                                  const DoFStrategyBase<Tria>& strategy,
//                                  const unsigned int      refineTimesLimit) const;
    
};


template <typename Tria, bool supportRepartioning>
template <typename CellSelectorType, bool LocallyOwend>
unsigned int
GridMaker<Tria, supportRepartioning>
::refineInitialMesh(Tria&                           tria,
                    const TriaStrategyBase<Tria>&   strategy,
                    const unsigned int              refineTimesLimit,
                    const CellSelectorType&         cellSelector)
{
    using namespace ::dealii;
    
    unsigned int refineTimes = 0;
    
    while (refineTimes < refineTimesLimit)
    {
        bool needRefinement = false;
        
        for (const auto& cell : tria.active_cell_iterators())
        {
            if constexpr (LocallyOwend && isDistribted)
            {
                if (!cell->is_locally_owned())
                    continue;
            }
            
            if (!cellSelector(cell))
                continue;
            
            if (strategy.setRefineFlag(cell))
            {
                cell->set_refine_flag();
                needRefinement = true;
            }
        }
        
        if constexpr (isDistribted)
        {
            needRefinement = __mpiInfo.syncFlag(needRefinement);
        }
        
        if (!needRefinement)
            break;
        
        tria.execute_coarsening_and_refinement();
        ++refineTimes;
    }
    
    return refineTimes;
}


template <typename Tria, bool supportRepartioning>
template <bool hasTCellSelector, bool isDoFBased, typename CellSelectorType>
unsigned int
GridMaker<Tria, supportRepartioning>
::__refine(Tria&                    tria,
           const DoFHandler*        dofHandler,
           const CellSelectorType*  cellSelector,
           const unsigned int       refineTimesLimit) const
{
    using namespace ::dealii;
    
    if constexpr(isDoFBased){
        AssertThrow(!__dofStrategyPtr,
                    ExcMessage("GridMaker DoFStrategy has not been bound."));
    } else {
        AssertThrow(!__triaStrategyPtr,
                    ExcMessage("GridMaker TriaStrategy has not been bound."));
    }
    
    if constexpr (hasTCellSelector) {
        AssertThrow(!cellSelector,
                    ExcMessage("cellSelector could not be nullptr."));
    }
    
    
    bool isNotCompleted = true;
    unsigned int refineTimes = 0;
    while (isNotCompleted)
    {
        isNotCompleted = false;
        if constexpr(isDoFBased)
        {
            for (const auto &cell : dofHandler->active_cell_iterators())
            {
                if constexpr (isDistribted)
                {
                    if (!cell->is_locally_owned()) continue;
                }
                
                if constexpr (hasTCellSelector)
                {
                    
                    if (!(*cellSelector)(cell))
                        continue;
                }
                
                if(__dofStrategyPtr->setRefineFlag(cell))
                {
                    cell->set_refine_flag();
                    isNotCompleted = true;
                }
                
            }
        }
        else
        {
            AssertThrow(!__triaStrategyPtr,
                        ExcMessage("GridMaker strategy has not been bound."));
            for (const auto &cell : tria.active_cell_iterators())
            {
                if constexpr (isDistribted) {
                    if (!cell->is_locally_owned()) continue;
                }
                
                if constexpr (hasTCellSelector)
                {
                    
                    if (!(*cellSelector)(cell))
                        continue;
                    
                }
                
                if(__triaStrategyPtr->setRefineFlag(cell))
                {
                    cell->set_refine_flag();
                    isNotCompleted = true;
                }
                
            }
        }
        if constexpr (isDistribted)
        {
            // accumulate local flag over all ranks
            const unsigned int local_flag = isNotCompleted ? 1u : 0u;
            const unsigned int global_flag =
            Utilities::MPI::sum(local_flag, *(__mpiInfo.mpiCommPtr()));
            isNotCompleted = (global_flag > 0u);
        }
        
        if(isNotCompleted)
        {
            tria.execute_coarsening_and_refinement();
            
            if((++refineTimes) >= refineTimesLimit)
            {
                isNotCompleted = false;
            }
        }
    }
    
    return refineTimes;
}



}


#endif /* GridMaker_h */
