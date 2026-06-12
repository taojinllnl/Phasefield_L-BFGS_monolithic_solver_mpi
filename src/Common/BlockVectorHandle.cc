//
//  BlockVectorHandle.cpp
//  main
//
//


#include "../../include/Common/BlockVectorPool.h"


using namespace ::common;
using namespace ::dealii;


template <typename TraitsType, bool Ghosted>
BlockVectorHandle<TraitsType, Ghosted>::BlockVectorHandle(
  BlockVectorPool<TraitsType, Ghosted> &pool,
  BVectorPtr                            vec)
  : __poolPtr(&pool)
  , __vecPtr(std::move(vec))
{}



template <typename TraitsType, bool Ghosted>
BlockVectorHandle<TraitsType, Ghosted>::BlockVectorHandle(
  BlockVectorHandle &&other) noexcept
  : __poolPtr(other.__poolPtr)
  , __vecPtr(std::move(other.__vecPtr))
{
  other.__poolPtr = nullptr;
  other.__vecPtr  = nullptr;
}

template <typename TraitsType, bool Ghosted>
BlockVectorHandle<TraitsType, Ghosted> &
BlockVectorHandle<TraitsType, Ghosted>::operator=(BlockVectorHandle &&other)
{
  if (this != &other)
    {
      __reset();

      __poolPtr = other.__poolPtr;
      __vecPtr  = std::move(other.__vecPtr);

      other.__poolPtr = nullptr;
      other.__vecPtr  = nullptr;
    }

  return *this;
}



template <typename TraitsType, bool Ghosted>
BlockVectorHandle<TraitsType, Ghosted>::~BlockVectorHandle()
{
  __reset();
}



template <typename TraitsType, bool Ghosted>
void
BlockVectorHandle<TraitsType, Ghosted>::__reset()
{
  if (__poolPtr && __vecPtr)
    {
      __poolPtr->__release(__vecPtr);
    }

  __poolPtr = nullptr;
  __vecPtr  = nullptr;
}



template <typename TraitsType, bool Ghosted>
typename BlockVectorHandle<TraitsType, Ghosted>::BVector &
BlockVectorHandle<TraitsType, Ghosted>::get()
{
  AssertThrow(__vecPtr != nullptr, ExcMessage("Invalid BlockVectorHandle."));
  return *__vecPtr;
}


template <typename TraitsType, bool Ghosted>
const typename BlockVectorHandle<TraitsType, Ghosted>::BVector &
BlockVectorHandle<TraitsType, Ghosted>::get() const
{
  AssertThrow(__vecPtr != nullptr, ExcMessage("Invalid BlockVectorHandle."));
  return *__vecPtr;
}


template <typename TraitsType, bool Ghosted>
typename BlockVectorHandle<TraitsType, Ghosted>::BVector &
BlockVectorHandle<TraitsType, Ghosted>::operator*()
{
  return get();
}

template <typename TraitsType, bool Ghosted>
const typename BlockVectorHandle<TraitsType, Ghosted>::BVector &
BlockVectorHandle<TraitsType, Ghosted>::operator*() const
{
  return get();
}



template <typename TraitsType, bool Ghosted>
typename BlockVectorHandle<TraitsType, Ghosted>::BVector *
BlockVectorHandle<TraitsType, Ghosted>::operator->()
{
  return &get();
}


template <typename TraitsType, bool Ghosted>
const typename BlockVectorHandle<TraitsType, Ghosted>::BVector *
BlockVectorHandle<TraitsType, Ghosted>::operator->() const
{
  return &get();
}


template <typename TraitsType, bool Ghosted>
void
BlockVectorHandle<TraitsType, Ghosted>::discard()
{
  if (__poolPtr && __vecPtr)
    {
      __poolPtr->__discard(__vecPtr);
    }

  __poolPtr = nullptr;
  __vecPtr  = nullptr;
}


template class common::BlockVectorHandle<common::Traits<TagSerial>, false>;
template class common::BlockVectorHandle<common::Traits<TagSerial>, true>;


#if defined(HAVE_PETSC) && HAVE_PETSC
template class common::BlockVectorHandle<common::Traits<TagPETSc>, false>;
template class common::BlockVectorHandle<common::Traits<TagPETSc>, true>;
#endif

#if defined(HAVE_TRILINOS) && HAVE_TRILINOS
template class common::BlockVectorHandle<common::Traits<TagTrilinos>, false>;
template class common::BlockVectorHandle<common::Traits<TagTrilinos>, true>;
#endif
