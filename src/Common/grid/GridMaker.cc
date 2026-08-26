//
//  GridMaker.cpp
//  main
//
//

#include "../../../include/Common/grid/GridMaker.h"

using namespace grid;


template <typename Tria, bool supportRepartioning>
GridMaker<Tria, supportRepartioning>
::GridMaker(const common::MPIInfo& mpiInfo)
: __mpiInfo(mpiInfo)
, __triaStrategyPtr(nullptr)
, __dofStrategyPtr(nullptr)
{}

//
//template <typename Tria, bool supportRepartioning>
//void
//GridMaker<Tria, supportRepartioning>
//::bindTriaStrategy(const TriaStrategyBase<Tria>& strategy)
//{
//    __triaStrategyPtr.reset();
//    __triaStrategyPtr = std::make_unique<TriaStrategyBase>(strategy);
//}
//
//
//template <typename Tria, bool supportRepartioning>
//bool
//GridMaker<Tria, supportRepartioning>
//::isBoundTriaStrategy() const
//{
//    return __triaStrategyPtr != nullptr;
//}
//
//
//
//template <typename Tria, bool supportRepartioning>
//void
//GridMaker<Tria, supportRepartioning>
//::bindDoFStrategy(const DoFStrategyBase<Tria>& strategy)
//{
//    __dofStrategyPtr.reset();
//    __dofStrategyPtr = std::make_unique<DoFStrategyBase>(strategy);
//}
//
//
//template <typename Tria, bool supportRepartioning>
//bool
//GridMaker<Tria, supportRepartioning>
//::isBoundDoFStrategy() const
//{
//    return __dofStrategyPtr != nullptr;
//}
//
//
//
//
//template <typename Tria, bool supportRepartioning>
//unsigned int
//GridMaker<Tria, supportRepartioning>
//::refine(Tria& tria,
//         const TCellSelector& triaCellSelector,
//         const unsigned int refineTimesLimit) const
//{
//    return __refine<true, false, TCellSelector>(tria,
//                                                nullptr,    // const DoFHandler*
//                                                &triaCellSelector,
//                                                refineTimesLimit);
//}
//
//
//template <typename Tria, bool supportRepartioning>
//unsigned int
//GridMaker<Tria, supportRepartioning>
//::refine(Tria& tria,
//         const unsigned int refineTimesLimit) const
//{
//    return __refine<false, false, TCellSelector>(tria,
//                                                 nullptr,  // const DoFHandler*
//                                                 nullptr,  // const TCellSelector*
//                                                 refineTimesLimit);
//}
//
//
//
//
//template <typename Tria, bool supportRepartioning>
//unsigned int
//GridMaker<Tria, supportRepartioning>
//::refine(Tria& tria,
//         const DoFHandler& dof,
//         const DCellSelector& dofCellSelector,
//         const unsigned int refineTimesLimit) const
//{
//    return __refine<true, true, DCellSelector>(tria,
//                                               dof,
//                                               dofCellSelector,
//                                               refineTimesLimit);
//}
//
//template <typename Tria, bool supportRepartioning>
//unsigned int
//GridMaker<Tria, supportRepartioning>
//::refine(Tria& tria,
//         const DoFHandler& dof,
//         const unsigned int refineTimesLimit) const
//{
//    return __refine<false, true, DCellSelector>(tria,
//                                                &dof,
//                                                nullptr, // const DCellSelector*
//                                                refineTimesLimit);
//}



template class grid::GridMaker<RTria<1, 1>, true>;
template class grid::GridMaker<RTria<1, 2>, true>;
template class grid::GridMaker<RTria<1, 3>, true>;

template class grid::GridMaker<RTria<2, 2>, true>;
template class grid::GridMaker<RTria<2, 3>, true>;

template class grid::GridMaker<RTria<3, 3>, true>;


template class grid::GridMaker<RTria<1, 1>, false>;
template class grid::GridMaker<RTria<1, 2>, false>;
template class grid::GridMaker<RTria<1, 3>, false>;

template class grid::GridMaker<RTria<2, 2>, false>;
template class grid::GridMaker<RTria<2, 3>, false>;

template class grid::GridMaker<RTria<3, 3>, false>;



#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template class grid::GridMaker<DTria<1, 1>, true>;
template class grid::GridMaker<DTria<1, 2>, true>;
template class grid::GridMaker<DTria<1, 3>, true>;

template class grid::GridMaker<DTria<2, 2>, true>;
template class grid::GridMaker<DTria<2, 3>, true>;

template class grid::GridMaker<DTria<3, 3>, true>;



template class grid::GridMaker<DTria<1, 1>, false>;
template class grid::GridMaker<DTria<1, 2>, false>;
template class grid::GridMaker<DTria<1, 3>, false>;

template class grid::GridMaker<DTria<2, 2>, false>;
template class grid::GridMaker<DTria<2, 3>, false>;

template class grid::GridMaker<DTria<3, 3>, false>;
#endif
