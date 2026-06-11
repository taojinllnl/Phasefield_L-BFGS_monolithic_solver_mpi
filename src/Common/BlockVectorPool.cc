//
//  BlockVectorPool.cpp
//  main
//
//


#include "../../include/Common/BlockVectorPool.h"


using namespace ::common;
using namespace ::dealii;

template <typename TraitsType, bool Ghosted>
BlockVectorPool<TraitsType, Ghosted>::BlockVectorPool(
  const MPIInfo   &mpiInfo,
  const BlockDesc &blockDesc)
  : __mpiInfo(mpiInfo)
  , __blockDesc(blockDesc)
{}


template <typename TraitsType, bool Ghosted>
BlockVectorPool<TraitsType, Ghosted>::~BlockVectorPool()
{
  Assert(__lent.empty(),
         ExcMessage(
           "Destroying BlockVectorPool while vectors are still lent."));
}


template <typename TraitsType, bool Ghosted>
typename BlockVectorPool<TraitsType, Ghosted>::BVectorPtr
BlockVectorPool<TraitsType, Ghosted>::__makeVector() const
{
  auto vecPtr = std::make_unique<BVector>(__mpiInfo, __blockDesc, Ghosted);

  vecPtr->initialize();

  return vecPtr;
}


template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::__release(BVector *vecPtr)
{
  AssertThrow(vecPtr != nullptr,
              dealii::ExcMessage("Cannot release a null vector pointer."));

  auto iter = __lent.find(vecPtr);

  AssertThrow(iter != __lent.end(),
              dealii::ExcMessage("Vector was not lent by this pool."));

  __eraseLentPtr(vecPtr);

  (*iter->second) = 0.0;

  __cached.emplace_back(std::move(iter->second));
  __lent.erase(iter);
}

template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::__discard(BVector *vecPtr)
{
  AssertThrow(vecPtr != nullptr,
              dealii::ExcMessage("Cannot discard a null vector pointer."));

  auto iter = __lent.find(vecPtr);

  AssertThrow(iter != __lent.end(),
              dealii::ExcMessage("Vector was not lent by this pool."));

  __eraseLentPtr(vecPtr);
  __lent.erase(iter);
}

template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::__eraseLentPtr(BVector *vec)
{
  auto it = std::find(__lentPtrs.begin(), __lentPtrs.end(), vec);

  AssertThrow(it != __lentPtrs.end(),
              dealii::ExcMessage(
                "Lent vector pointer was not found in __lentPtrs."));

  __lentPtrs.erase(it);
}


template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::initializeCached()
{
  for (BVectorPtr &ptr : __cached)
    {
      ptr->initialize();
    }
}


template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::initializeLent()
{
  AssertThrow(__lentPtrs.size() == __lent.size(),
              dealii::ExcMessage(
                "__lentPtrs and __lent have inconsistent sizes."));

  for (BVector *lentVecPtr : __lentPtrs)
    {
      AssertThrow(lentVecPtr != nullptr,
                  dealii::ExcMessage("Null pointer in __lentPtrs."));

      AssertThrow(__lent.find(lentVecPtr) != __lent.end(),
                  dealii::ExcMessage("Stale pointer in __lentPtrs."));

      lentVecPtr->initialize();
    }
}

template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::initialize()
{
  initializeLent();
  initializeCached();
}


template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::addCachedVectors(const std::size_t n)
{
  while (__cached.size() < n)
    {
      __cached.push_back(__makeVector());
    }
}



template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::clear()
{
  AssertThrow(__lent.empty(),
              ExcMessage(
                "Cannot clear BlockVectorPool while vectors are still lent."));
  __cached.clear();
}


template <typename TraitsType, bool Ghosted>
void
BlockVectorPool<TraitsType, Ghosted>::shrink_to(const std::size_t n_cached)
{
  while (__cached.size() > n_cached)
    __cached.pop_back();
}


template <typename TraitsType, bool Ghosted>
BlockVectorHandle<TraitsType, Ghosted>
BlockVectorPool<TraitsType, Ghosted>::getHandle(const bool setZero)
{
  if (__cached.empty())
    {
      auto ptr = std::make_unique<BVector>(__mpiInfo, __blockDesc, Ghosted);
      ptr->initialize();

      __cached.emplace_back(std::move(ptr));
    }

  BVectorPtr ptr = std::move(__cached.back());
  __cached.pop_back();


  BVector *raw_ptr = ptr.get();

  if (setZero)
    {
      *raw_ptr = 0.0;
      if constexpr (Ghosted)
        raw_ptr->updateRelevance();
    }

  const auto [iter, inserted] = __lent.emplace(raw_ptr, std::move(ptr));
  __lentPtrs.emplace_back(raw_ptr);

  AssertThrow(inserted, ExcMessage("Vector is already registered as lent."));

  return BlockVectorHandle<TraitsType, Ghosted>(*this, raw_ptr);
}



template class common::BlockVectorPool<common::Traits<TagSerial>, false>;
template class common::BlockVectorPool<common::Traits<TagSerial>, true>;


#if defined(HAVE_PETSC) && HAVE_PETSC
template class common::BlockVectorPool<common::Traits<TagPETSc>, false>;
template class common::BlockVectorPool<common::Traits<TagPETSc>, true>;
#endif

#if defined(HAVE_TRILINOS) && HAVE_TRILINOS
template class common::BlockVectorPool<common::Traits<TagTrilinos>, false>;
template class common::BlockVectorPool<common::Traits<TagTrilinos>, true>;
#endif
