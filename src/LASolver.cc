//
//  LASolver.cpp
//  main
//
//  Created by Ethan Lee on 2026-03-23.
//

#include "../include/LASolver.h"


using namespace dealii;

using namespace PhaseField;
using namespace common;


Tol::Tol(const unsigned int nIters,
         const double tol)
: nIters(nIters)
, tol(tol)
{}





template <typename LATraits>
LASolver<LATraits>
::LASolver(const SolverType&   type,
           const BlockDesc&    blockDesc,
           const MPIInfo&      mpiInfo,
           const Tol tol_u,
           const Tol tol_d)
: __type(type)
, __u_group_ID(blockDesc.ithGroupID("displacement"))
, __d_group_ID(blockDesc.ithGroupID("phase-field"))
, __tolList({tol_u, tol_d})
, __blockDesc(blockDesc)
, __mpiInfo(mpiInfo)
{}





template <typename LATraits>
void
LASolver<LATraits>::solve(BVector & LBFGS_r_vector,
                          const BVector & LBFGS_q_vector,
                          const BSMatrix& tangent_matrix)
{
    LBFGS_r_vector.initialize();
    if (__type == SolverType::Direct) {
        __directSolve(LBFGS_r_vector, LBFGS_q_vector, tangent_matrix);
    } else {
        __cgSolve(LBFGS_r_vector, LBFGS_q_vector, tangent_matrix);
    }
}



template <typename LATraits>
void
LASolver<LATraits>::__directSolve(BVector & LBFGS_r_vector,
                                  const BVector & LBFGS_q_vector,
                                  const BSMatrix& tangent_matrix)
{
    using namespace dealii;
    if constexpr (std::is_same_v<typename LATraits::TMTag, ::common::TagSerial>) {
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            SparseDirectUMFPACK A_direct;
            A_direct.initialize(tangent_matrix.block(ithGroup, ithGroup));
            A_direct.vmult(LBFGS_r_vector.block(ithGroup),
                           LBFGS_q_vector.block(ithGroup));
        }
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::common::TagPETSc>) {
        // https://dealii.org/current/doxygen/deal.II/classPETScWrappers_1_1SparseDirectMUMPS.html
        
#ifdef HAVE_PETSC
        using PrecJacobi = dealii::PETScWrappers::PreconditionBlockJacobi;
        using PrecILU    = dealii::PETScWrappers::PreconditionILU;
        using PrecICC    = dealii::PETScWrappers::PreconditionICC;
        using PrecPSails = dealii::PETScWrappers::PreconditionParaSails;
        using PrecSOR    = dealii::PETScWrappers::PreconditionSOR;
        using PrecSSOR   = dealii::PETScWrappers::PreconditionSSOR;
//        using PrecShell  = dealii::PETScWrappers::PreconditionShell;
        using PrecNone   = dealii::PETScWrappers::PreconditionNone;
        
        using PrecType = PrecJacobi;
        
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            SolverControl solver_control(__tolList[ithGroup].nIters,
                                         __tolList[ithGroup].tol);
          
            PETScWrappers::SparseDirectMUMPS solver(solver_control,
                                                    *__mpiInfo.mpiCommPtr());
            solver.set_symmetric_mode(false);
            
//            PrecType preconditioner;
//            preconditioner.initialize(tangent_matrix.block(ithGroup, ithGroup));
//            solver.initialize(preconditioner);

            solver.solve(tangent_matrix.block(ithGroup, ithGroup),
                         LBFGS_r_vector.block(ithGroup),
                         LBFGS_q_vector.block(ithGroup));


        }
#endif
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::common::TagTrilinos>) {
        // https://dealii.org/current/doxygen/deal.II/classTrilinosWrappers_1_1SolverDirect.html
        
#ifdef HAVE_TRILINOS
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            SolverControl solver_control(__tolList[ithGroup].nIters,
                                         __tolList[ithGroup].tol);
           
            
            TrilinosWrappers::SolverDirect A_direct_T(solver_control);
            A_direct_T.initialize(tangent_matrix.block(ithGroup, ithGroup));
            A_direct_T.vmult(LBFGS_r_vector.block(ithGroup),
                             LBFGS_q_vector.block(ithGroup));
        }
    }
#endif
}

template <typename LATraits>
void
LASolver<LATraits>::__cgSolve(BVector & LBFGS_r_vector,
                              const BVector & LBFGS_q_vector,
                              const BSMatrix& tangent_matrix)
{
    using namespace dealii;
    if constexpr (std::is_same_v<typename LATraits::TMTag, ::common::TagSerial>) {
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            SolverControl            solver_control(__tolList[ithGroup].nIters,
                                                    __tolList[ithGroup].tol);
            SolverCG<Vector<double>> cg(solver_control);
            
            PreconditionJacobi<SparseMatrix<double>> preconditioner;
            preconditioner.initialize(tangent_matrix.block(ithGroup, ithGroup),
                                      1.0);
            
            cg.solve(tangent_matrix.block(ithGroup, ithGroup),
                     LBFGS_r_vector.block(ithGroup),
                     LBFGS_q_vector.block(ithGroup),
                     preconditioner);
        }
        
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::common::TagPETSc>) {
#ifdef HAVE_PETSC
        using PrecJacobi = dealii::PETScWrappers::PreconditionBlockJacobi;
        using PrecILU    = dealii::PETScWrappers::PreconditionILU;
        using PrecICC    = dealii::PETScWrappers::PreconditionICC;
        using PrecPSails = dealii::PETScWrappers::PreconditionParaSails;
        using PrecSOR    = dealii::PETScWrappers::PreconditionSOR;
        using PrecSSOR   = dealii::PETScWrappers::PreconditionSSOR;
//        using PrecShell  = dealii::PETScWrappers::PreconditionShell;
        using PrecNone   = dealii::PETScWrappers::PreconditionNone;
        
        using MatBlock   = typename LATraits::MatrixBlock;
        
        
        
        using PrecType = PrecNone;
        using CGSolver = MPICGSolver<MatBlock, PETScWrappers::SolverCG>;
        
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            PrecType prec;
            prec.initialize(tangent_matrix.block(ithGroup, ithGroup));
            
            CGSolver cg(__tolList[ithGroup].tol,
                        __tolList[ithGroup].nIters);
            
            cg.solve(tangent_matrix.block(ithGroup, ithGroup),
                     LBFGS_r_vector.block(ithGroup),
                     LBFGS_q_vector.block(ithGroup),
                     prec);
        }
#endif
        
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::common::TagTrilinos>) {
        
#ifdef HAVE_TRILINOS
        using PrecJacobi = dealii::TrilinosWrappers::PreconditionBlockJacobi;
        using PrecILU    = dealii::TrilinosWrappers::PreconditionILU;
        using PrecIC     = dealii::TrilinosWrappers::PreconditionIC;
        using PrecILUT   = dealii::TrilinosWrappers::PreconditionILUT;
        using PrecSOR    = dealii::TrilinosWrappers::PreconditionSOR;
        using PrecSSOR   = dealii::TrilinosWrappers::PreconditionSSOR;
        using PrecShebs  = dealii::TrilinosWrappers::PreconditionChebyshev;
        using PrecI      = dealii::TrilinosWrappers::PreconditionIdentity;
        
        using MatBlock   = typename LATraits::MatrixBlock;
        
        
        using PrecType = PrecI;
        using CGSolver   =  MPICGSolver<MatBlock, TrilinosWrappers::SolverCG>;

        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            PrecType prec;
            prec.initialize(tangent_matrix.block(ithGroup, ithGroup));
            
            CGSolver cg(__tolList[ithGroup].tol,
                        __tolList[ithGroup].nIters);
            
            cg.solve(tangent_matrix.block(ithGroup, ithGroup),
                     LBFGS_r_vector.block(ithGroup),
                     LBFGS_q_vector.block(ithGroup),
                     prec);
        }
    }
#endif
}




template class PhaseField::LASolver<common::Traits<common::TagSerial>>;
template class PhaseField::LASolver<common::Traits<common::TagPETSc>>;
template class PhaseField::LASolver<common::Traits<common::TagTrilinos>>;
