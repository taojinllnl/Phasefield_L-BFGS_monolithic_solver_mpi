//
//  LASolver.hpp
//  main
//
//

#ifndef LASolver_hpp
#define LASolver_hpp

#include <variant>
#include <array>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>

#include "Common/Traits.h"

#include "Common/BlockVectorWrapper.h"
#include "Common/BlockSparseMatrixWrapper.h"

#include "Common/BlockDesc.h"
#include "Common/MPIInfo.h"

#include "MPICGSolver.h"

namespace PhaseField {


/**
 *
 * The class `LASolver` is an integrated interface for selection of different types of linear algebra solvers in serial or MPI modes.
 * It supports:
 * - Sparse direct solver
 * - Iterative solver (conjugate gradient solver)
 *
 *
 */

enum class SolverType
{
    Direct, CG
};


enum class PrecType
{
    None,
    Jacobi,
    BlockJacobi,
    ILU,
    ICC,
    IC,
    ILUT,
    SOR,
    SSOR,
    Chebyshev,
    ParaSails,
};


struct Tol
{
    const unsigned int nIters;
    const double tol;
    Tol(const unsigned int nIters,
        const double tol);
    
};



template <typename LATraits>
class LASolver
{
public:
    using BSMatrix  = ::common::BlockSparseMatrixWrapper<LATraits>;
    using BVector   = ::common::BlockVectorWrapper<LATraits>;
    
private:
    const SolverType    __type;
    const PrecType      __precType;
    
    
    const unsigned int      __u_group_ID;
    const unsigned int      __d_group_ID;
    
    const std::array<Tol, 2> __tolList;
    
    const common::BlockDesc&        __blockDesc;
    const common::MPIInfo&          __mpiInfo;
    
    void __directSolve(BVector & LBFGS_r_vector,
                       const BVector & LBFGS_q_vector,
                       const BSMatrix& tangentMatrix);
    void __cgSolve(BVector & LBFGS_r_vector,
                   const BVector & LBFGS_q_vector,
                   const BSMatrix& tangentMatrix);
    
    
    
public:
    
    virtual ~LASolver() = default;
    
    static SolverType solverType(const std::string& typeName);
    static PrecType   precType(const std::string& typeName);
    
    LASolver(const SolverType&          type,
             const PrecType&            precType,
             const common::BlockDesc&    blockDesc,
             const common::MPIInfo&      mpiInfo,
             const Tol tol_u = Tol(1e6, 1e-12),
             const Tol tol_d = Tol(1e6, 1e-12));
    
    
    LASolver(const std::string&         typeName,
             const std::string&         precTypeName,
             const common::BlockDesc&    blockDesc,
             const common::MPIInfo&      mpiInfo,
             const Tol tol_u = Tol(1e6, 1e-12),
             const Tol tol_d = Tol(1e6, 1e-12));
    
    void solve(BVector & LBFGS_r_vector,
               const BVector & LBFGS_q_vector,
               const BSMatrix& tangentMatrix);
    
};





}
#endif /* LASolver_hpp */
