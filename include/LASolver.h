//
//  LASolver.hpp
//  main
//
//

#ifndef LASolver_hpp
#define LASolver_hpp

#include <memory>

#include <variant>
#include <array>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>



#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>

#include "Common/Traits.h"

#include "Common/BlockVectorWrapper.h"
#include "Common/BlockSparseMatrixWrapper.h"

#include "Common/BlockDesc.h"
#include "Common/MPIInfo.h"

#include "Common/MPIIterativeSolver.h"

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
    Direct, CG, GMRes, Bicgstab
};


enum class PrecType
{
    None,
    Jacobi,
    AMG,
    ILU,
    ICC,
    IC,
    ILUT,
    SOR,
    SSOR,
    Chebyshev,
    ParaSails,
    Shell,
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
    const std::array<std::string, 2> __names = {{"u", "d"}};
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
    
    template<typename SSolver, typename PSolver, typename TSolver>
    void __iterativeSolve(BVector & LBFGS_r_vector,
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



template <typename LATraits>
template<typename SSolver, typename PSolver, typename TSolver>
void
LASolver<LATraits>
::__iterativeSolve(BVector & LBFGS_r_vector,
                   const BVector & LBFGS_q_vector,
                   const BSMatrix& tangent_matrix)
{
    using namespace ::dealii;
    using namespace ::common;
    if constexpr (std::is_same_v<typename LATraits::TMTag, TagSerial>) {
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            SolverControl            solver_control(__tolList[ithGroup].nIters,
                                                    __tolList[ithGroup].tol);
            SSolver laSolver(solver_control);
            
            PreconditionJacobi<SparseMatrix<double>> preconditioner;
            preconditioner.initialize(tangent_matrix.block(ithGroup,
                                                           ithGroup),
                                      1.0);
            
            laSolver.solve(tangent_matrix.block(ithGroup, ithGroup),
                           LBFGS_r_vector.block(ithGroup),
                           LBFGS_q_vector.block(ithGroup),
                           preconditioner);
        }
        
        
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, TagPETSc>) {
#ifdef HAVE_PETSC
        using MatBlock   = typename LATraits::MatrixBlock;
        using LASolver = MPIIterativeSolver<MatBlock, PSolver>;
        
        
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            LASolver laSolver(__mpiInfo,
                              __names[ithGroup],
                              __tolList[ithGroup].tol,
                              __tolList[ithGroup].nIters);
            
            const auto &A = tangent_matrix.block(ithGroup, ithGroup);
            auto       &x = LBFGS_r_vector.block(ithGroup);
            const auto &b = LBFGS_q_vector.block(ithGroup);
            
            switch (__precType)
            {
                case PhaseField::PrecType::Jacobi:
                {
                    typename LATraits::PrecJacobi prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::AMG:
                {
                    typename LATraits::PrecAMG::AdditionalData amg_data;
                    amg_data.symmetric_operator = true;
                    amg_data.strong_threshold   = 0.5;
                    amg_data.aggressive_coarsening_num_levels = 1;
                    amg_data.output_details     = false;
                    amg_data.n_sweeps_coarse = 1;
                    amg_data.max_iter        = 1;
                    amg_data.w_cycle         = false;
                    
                    
                    typename LATraits::PrecAMG prec;
                    prec.initialize(A, amg_data);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ILU:
                {
                    typename LATraits::PrecILU prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ICC:
                {
                    typename LATraits::PrecICC prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ParaSails:
                {
                    typename LATraits::PrecPSails prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::SOR:
                {
                    typename LATraits::PrecSOR prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::SSOR:
                {
                    typename LATraits::PrecSSOR prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::Shell:
                {
                    typename LATraits::PrecShell prec(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::None:
                {
                    typename LATraits::PrecNone prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                default:
                    AssertThrow(false, ExcMessage("Unsupported PETSc preconditioner"));
            }
        }
#endif
        
        
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, TagTrilinos>) {
#ifdef HAVE_TRILINOS
        using MatBlock   = typename LATraits::MatrixBlock;
        using LASolver = MPIIterativeSolver<MatBlock, TSolver>;
        
        
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            LASolver laSolver(__mpiInfo,
                              __names[ithGroup],
                              __tolList[ithGroup].tol,
                              __tolList[ithGroup].nIters);
            
            const auto &A = tangent_matrix.block(ithGroup, ithGroup);
            auto       &x = LBFGS_r_vector.block(ithGroup);
            const auto &b = LBFGS_q_vector.block(ithGroup);
            
            switch (__precType)
            {
                    
                case PhaseField::PrecType::Jacobi:
                {
                    typename LATraits::PrecJacobi prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::AMG:
                {
                    typename LATraits::PrecAMG prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ILU:
                {
                    typename LATraits::PrecILU prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                    
                case PhaseField::PrecType::IC:
                {
                    typename LATraits::PrecIC prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ILUT:
                {
                    typename LATraits::PrecILUT prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                    
                case PhaseField::PrecType::SOR:
                {
                    typename LATraits::PrecSOR prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::SSOR:
                {
                    typename LATraits::PrecSSOR prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::Chebyshev:
                {
                    typename LATraits::PrecChebs prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                    
                case PhaseField::PrecType::None:
                {
                    typename LATraits::PrecI prec;
                    prec.initialize(A);
                    laSolver.solve(A, x, b, prec);
                    break;
                }
                    
                    
                default:
                    AssertThrow(false, ExcMessage("Unsupported Trilinos preconditioner"));
            }
        }
    }
#endif
}



}
#endif /* LASolver_hpp */
