//
//  TriaStrategyBase.cpp
//  main
//
//

#include "../../../include/Common/grid/TriaStrategyBase.h"

using namespace grid;

template <typename Tria>
bool 
TriaStrategyBase<Tria>::setRefineFlag(const TCellIter& /*cell*/) const
{
    return true;
}



template class grid::TriaStrategyBase<RTria<1, 1>>;
template class grid::TriaStrategyBase<RTria<1, 2>>;
template class grid::TriaStrategyBase<RTria<1, 3>>;


template class grid::TriaStrategyBase<RTria<2, 2>>;
template class grid::TriaStrategyBase<RTria<2, 3>>;

template class grid::TriaStrategyBase<RTria<3, 3>>;


#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template class grid::TriaStrategyBase<DTria<1, 1>>;
template class grid::TriaStrategyBase<DTria<1, 2>>;
template class grid::TriaStrategyBase<DTria<1, 3>>;


template class grid::TriaStrategyBase<DTria<2, 2>>;
template class grid::TriaStrategyBase<DTria<2, 3>>;

template class grid::TriaStrategyBase<DTria<3, 3>>;
#endif
