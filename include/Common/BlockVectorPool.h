//
//  BlockVectorPool.h
//  main
//
//

#ifndef BlockVectorPool_h
#define BlockVectorPool_h

#include <memory>
#include <unordered_map>
#include <vector>

#include "BlockDesc.h"
#include "BlockVectorWrapper.h"
#include "MPIInfo.h"


namespace common
{

  template <typename TraitsType, bool Ghosted = true>
  class BlockVectorPool;

  template <typename TraitsType, bool Ghosted = true>
  class BlockVectorHandle;

  template <typename TraitsType, bool Ghosted>
  class BlockVectorPool
  {
  public:
    using BVector    = BlockVectorWrapper<TraitsType>;
    using BVectorPtr = std::unique_ptr<BVector>;

  private:
    const MPIInfo   &__mpiInfo;
    const BlockDesc &__blockDesc;


    std::vector<BVectorPtr>                   __cached;
    std::unordered_map<BVector *, BVectorPtr> __lent;
    std::vector<BVector *>                    __lentPtrs;

    BVectorPtr
    __makeVector() const;

    void
    __release(BVector *vec);
    void
    __discard(BVector *vec);
    void
    __eraseLentPtr(BVector *vec);

  public:
    BlockVectorPool()                        = delete;
    BlockVectorPool(const BlockVectorPool &) = delete;
    BlockVectorPool &
    operator=(const BlockVectorPool &) = delete;

    BlockVectorPool(const MPIInfo &mpiInfo, const BlockDesc &blockDesc);

    ~BlockVectorPool();

    void
    initializeCached();
    void
    initializeLent();
    void
    initialize();

    void
    addCachedVectors(const std::size_t n);
    void
    clear();
    void
    shrink_to(const std::size_t n_cached);

    BlockVectorHandle<TraitsType, Ghosted>
    getHandle(const bool setZero = true);

    friend class BlockVectorHandle<TraitsType, Ghosted>;
  };



  template <typename TraitsType, bool Ghosted>
  class BlockVectorHandle
  {
  public:
    using BVector    = BlockVectorWrapper<TraitsType>;
    using BVectorPtr = BVector *;

  private:
    friend class BlockVectorPool<TraitsType, Ghosted>;

    BlockVectorPool<TraitsType, Ghosted> *__poolPtr = nullptr;
    BVectorPtr                            __vecPtr  = nullptr;


    BlockVectorHandle(BlockVectorPool<TraitsType, Ghosted> &pool,
                      BVectorPtr                            vec);

    void
    __reset();

  public:
    BlockVectorHandle()                          = delete;
    BlockVectorHandle(const BlockVectorHandle &) = delete;
    BlockVectorHandle &
    operator=(const BlockVectorHandle &) = delete;



    BlockVectorHandle(BlockVectorHandle &&other) noexcept;

    BlockVectorHandle &
    operator=(BlockVectorHandle &&other);

    ~BlockVectorHandle();



    BVector &
    get();

    const BVector &
    get() const;

    BVector &
    operator*();

    const BVector &
    operator*() const;


    BVector *
    operator->();

    const BVector *
    operator->() const;



    void
    discard();
  };


} // namespace common

#endif /* BlockVectorPool_h */
