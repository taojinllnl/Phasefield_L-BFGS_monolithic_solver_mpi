//
//  LASolver.cpp
//  main
//
//

#include "../include/LASolver.h"


using namespace ::dealii;

using namespace PhaseField;
using namespace common;





Tol::Tol(const unsigned int nIters,
         const double tol)
: nIters(nIters)
, tol(tol)
{}




template <typename LATraits>
SolverType
LASolver<LATraits>
::solverType(const std::string& typeName)
{
    if (typeName == "Direct")
        return SolverType::Direct;
    else if (typeName == "CG")
        return SolverType::CG;
    
    AssertThrow(false, ExcMessage("Unknown LA solver type: " + typeName));
}

template <typename LATraits>
PrecType
LASolver<LATraits>::precType(const std::string &typeName)
{
    if (typeName == "BlockJacobi")
        return PrecType::BlockJacobi;
    else if (typeName == "ILU")
        return PrecType::ILU;
    else if (typeName == "SOR")
        return PrecType::SOR;
    else if (typeName == "SSOR")
        return PrecType::SSOR;
    else if (typeName == "Chebyshev") // only for Trilinos
        return PrecType::Chebyshev;
    else if (typeName == "IC") // only for Trilinos
        return PrecType::IC;
    else if (typeName == "ILUT")  // only for Trilinos
        return PrecType::ILUT;
    else if (typeName == "ICC") // only for PETSc
        return PrecType::ICC;
    else if (typeName == "ParaSails") // only for PETSc
        return PrecType::ParaSails;
    else if (typeName == "None")
        return PrecType::None;
    
    AssertThrow(false, ExcMessage("Unknown preconditioner type: " + typeName));
}



template <typename LATraits>
LASolver<LATraits>
::LASolver(const SolverType&   type,
           const PrecType&     precType,
           const BlockDesc&    blockDesc,
           const MPIInfo&      mpiInfo,
           const Tol tol_u,
           const Tol tol_d)
: __type(type)
, __precType(precType)
, __u_group_ID(blockDesc.ithGroupID("displacement"))
, __d_group_ID(blockDesc.ithGroupID("phase-field"))
, __tolList({{tol_u, tol_d}})
, __blockDesc(blockDesc)
, __mpiInfo(mpiInfo)
{}


template <typename LATraits>
LASolver<LATraits>
::LASolver(const std::string&   typeName,
           const std::string&   precTypeName,
           const BlockDesc&    blockDesc,
           const MPIInfo&      mpiInfo,
           const Tol tol_u,
           const Tol tol_d)
: LASolver(solverType(typeName), precType(precTypeName), 
           blockDesc, mpiInfo, tol_u, tol_d)
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
        using PrecJacobi = PETScWrappers::PreconditionBlockJacobi;
        using PrecILU    = PETScWrappers::PreconditionILU;
        using PrecICC    = PETScWrappers::PreconditionICC;
        using PrecPSails = PETScWrappers::PreconditionParaSails;
        using PrecSOR    = PETScWrappers::PreconditionSOR;
        using PrecSSOR   = PETScWrappers::PreconditionSSOR;
        //        using PrecShell  = PETScWrappers::PreconditionShell;
        using PrecNone   = PETScWrappers::PreconditionNone;
        
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
        using PrecJacobi = PETScWrappers::PreconditionBlockJacobi;
        using PrecILU    = PETScWrappers::PreconditionILU;
        using PrecICC    = PETScWrappers::PreconditionICC;
        using PrecPSails = PETScWrappers::PreconditionParaSails;
        using PrecSOR    = PETScWrappers::PreconditionSOR;
        using PrecSSOR   = PETScWrappers::PreconditionSSOR;
        //        using PrecShell  = PETScWrappers::PreconditionShell;
        using PrecNone   = PETScWrappers::PreconditionNone;
        
        using MatBlock   = typename LATraits::MatrixBlock;
        
        
        
        using PrecType = PrecNone;
        using CGSolver = MPICGSolver<MatBlock, PETScWrappers::SolverCG>;
        
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            CGSolver cg(__tolList[ithGroup].tol,
                        __tolList[ithGroup].nIters);
            
            const auto &A = tangent_matrix.block(ithGroup, ithGroup);
            auto       &x = LBFGS_r_vector.block(ithGroup);
            const auto &b = LBFGS_q_vector.block(ithGroup);
            
            switch (__precType)
            {
                case PhaseField::PrecType::BlockJacobi:
                {
                    PETScWrappers::PreconditionBlockJacobi prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ILU:
                {
                    PETScWrappers::PreconditionILU prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ICC:
                {
                    PETScWrappers::PreconditionICC prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ParaSails:
                {
                    PETScWrappers::PreconditionParaSails prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::SOR:
                {
                    PETScWrappers::PreconditionSOR prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::SSOR:
                {
                    PETScWrappers::PreconditionSSOR prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::None:
                {
                    PETScWrappers::PreconditionNone prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                default:
                    AssertThrow(false, ExcMessage("Unsupported PETSc preconditioner"));
            }
        }
#endif
        
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::common::TagTrilinos>) {
        
#ifdef HAVE_TRILINOS
        using PrecJacobi = TrilinosWrappers::PreconditionBlockJacobi;
        using PrecILU    = TrilinosWrappers::PreconditionILU;
        using PrecIC     = TrilinosWrappers::PreconditionIC;
        using PrecILUT   = TrilinosWrappers::PreconditionILUT;
        using PrecSOR    = TrilinosWrappers::PreconditionSOR;
        using PrecSSOR   = TrilinosWrappers::PreconditionSSOR;
        using PrecShebs  = TrilinosWrappers::PreconditionChebyshev;
        using PrecI      = TrilinosWrappers::PreconditionIdentity;
        
        using MatBlock   = typename LATraits::MatrixBlock;
        
        
        using PrecType   = PrecILUT;
        using CGSolver   = MPICGSolver<MatBlock, TrilinosWrappers::SolverCG>;
        
        for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        {
            
            CGSolver cg(__tolList[ithGroup].tol,
                        __tolList[ithGroup].nIters);
            
            const auto &A = tangent_matrix.block(ithGroup, ithGroup);
            auto       &x = LBFGS_r_vector.block(ithGroup);
            const auto &b = LBFGS_q_vector.block(ithGroup);
            
            switch (__precType)
            {
                    
                case PhaseField::PrecType::BlockJacobi:
                {
                    TrilinosWrappers::PreconditionBlockJacobi prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ILU:
                {
                    TrilinosWrappers::PreconditionILU prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                    
                case PhaseField::PrecType::IC:
                {
                    TrilinosWrappers::PreconditionIC prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::ILUT:
                {
                    TrilinosWrappers::PreconditionILUT prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }

                    
                case PhaseField::PrecType::SOR:
                {
                    TrilinosWrappers::PreconditionSOR prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::SSOR:
                {
                    TrilinosWrappers::PreconditionSSOR prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                case PhaseField::PrecType::Chebyshev:
                {
                    TrilinosWrappers::PreconditionChebyshev prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                    
                    
                case PhaseField::PrecType::None:
                {
                    TrilinosWrappers::PreconditionIdentity prec;
                    prec.initialize(A);
                    cg.solve(A, x, b, prec);
                    break;
                }
                 
                
                default:
                    AssertThrow(false, ExcMessage("Unsupported Trilinos preconditioner"));
            }
        }
    }
#endif
}




template class PhaseField::LASolver<common::Traits<common::TagSerial>>;
template class PhaseField::LASolver<common::Traits<common::TagPETSc>>;
template class PhaseField::LASolver<common::Traits<common::TagTrilinos>>;
