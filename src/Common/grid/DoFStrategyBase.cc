//
//  DoFStrategyBase.cpp
//  main
//
#include "../../../include/Common/grid/DoFStrategyBase.h"


using namespace grid;


template <typename Tria>
bool
DoFStrategyBase<Tria>::setRefineFlag(const DCellIter& /*cell*/) const
{
    return true;
}





template class grid::DoFStrategyBase<RTria<1, 1>>;
template class grid::DoFStrategyBase<RTria<1, 2>>;
template class grid::DoFStrategyBase<RTria<1, 3>>;


template class grid::DoFStrategyBase<RTria<2, 2>>;
template class grid::DoFStrategyBase<RTria<2, 3>>;

template class grid::DoFStrategyBase<RTria<3, 3>>;


#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template class grid::DoFStrategyBase<DTria<1, 1>>;
template class grid::DoFStrategyBase<DTria<1, 2>>;
template class grid::DoFStrategyBase<DTria<1, 3>>;


template class grid::DoFStrategyBase<DTria<2, 2>>;
template class grid::DoFStrategyBase<DTria<2, 3>>;

template class grid::DoFStrategyBase<DTria<3, 3>>;
#endif
