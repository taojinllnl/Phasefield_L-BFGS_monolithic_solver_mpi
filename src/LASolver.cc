//
//  LASolver.cpp
//  main
//
//

#include "../include/LASolver.h"


using namespace ::dealii;

using namespace ::PhaseField;
using namespace ::common;



Tol::Tol(const unsigned int nIters, const double tol)
  : nIters(nIters)
  , tol(tol)
{}



template <typename LATraits>
SolverType
LASolver<LATraits>::solverType(const std::string &typeName)
{
  if (typeName == "Direct")
    return SolverType::Direct;
  else if (typeName == "CG")
    return SolverType::CG;
  else if (typeName == "GMRes")
    return SolverType::GMRes;
  else if (typeName == "Bicgstab")
    return SolverType::Bicgstab;

  AssertThrow(false, ExcMessage("Unknown LA solver type: " + typeName));
}

template <typename LATraits>
PrecType
LASolver<LATraits>::precType(const std::string &typeName)
{
  if (typeName == "Jacobi")
    return PrecType::Jacobi;
  else if (typeName == "ILU")
    return PrecType::ILU;
  else if (typeName == "AMG")
    return PrecType::AMG;
  else if (typeName == "SOR")
    return PrecType::SOR;
  else if (typeName == "SSOR")
    return PrecType::SSOR;
  else if (typeName == "Chebyshev") // only for Trilinos
    return PrecType::Chebyshev;
  else if (typeName == "IC") // only for Trilinos
    return PrecType::IC;
  else if (typeName == "ILUT") // only for Trilinos
    return PrecType::ILUT;
  else if (typeName == "ICC") // only for PETSc
    return PrecType::ICC;
  else if (typeName == "ParaSails") // only for PETSc
    return PrecType::ParaSails;
  else if (typeName == "Shell") // only for PETSc
    return PrecType::Shell;
  else if (typeName == "None")
    return PrecType::None;

  AssertThrow(false, ExcMessage("Unknown preconditioner type: " + typeName));
}



template <typename LATraits>
LASolver<LATraits>::LASolver(const SolverType &type,
                             const PrecType   &precType,
                             const BlockDesc  &blockDesc,
                             const MPIInfo    &mpiInfo,
                             const Tol         tol_u,
                             const Tol         tol_d)
  : __type(type)
  , __precType(precType)
  , __u_group_ID(blockDesc.ithGroupID("displacement"))
  , __d_group_ID(blockDesc.ithGroupID("phase-field"))
  , __tolList({{tol_u, tol_d}})
  , __blockDesc(blockDesc)
  , __mpiInfo(mpiInfo)
{}


template <typename LATraits>
LASolver<LATraits>::LASolver(const std::string &typeName,
                             const std::string &precTypeName,
                             const BlockDesc   &blockDesc,
                             const MPIInfo     &mpiInfo,
                             const Tol          tol_u,
                             const Tol          tol_d)
  : LASolver(solverType(typeName),
             precType(precTypeName),
             blockDesc,
             mpiInfo,
             tol_u,
             tol_d)
{}



template <typename LATraits>
void
LASolver<LATraits>::solve(BVector        &LBFGS_r_vector,
                          const BVector  &LBFGS_q_vector,
                          const BSMatrix &tangent_matrix)
{
  LBFGS_r_vector.initialize();
  if (__type == SolverType::Direct)
    {
      __directSolve(LBFGS_r_vector, LBFGS_q_vector, tangent_matrix);
    }
  else if (__type == SolverType::CG)
    {
      using SSolver = SolverCG<Vector<double>>;
      using PSolver = PETScWrappers::SolverCG;
      using TSolver = TrilinosWrappers::SolverCG;
      __iterativeSolve<SSolver, PSolver, TSolver>(LBFGS_r_vector,
                                                  LBFGS_q_vector,
                                                  tangent_matrix);
    }
  else if (__type == SolverType::GMRes)
    {
      using SSolver = SolverGMRES<Vector<double>>;
      using PSolver = PETScWrappers::SolverGMRES;
      using TSolver = TrilinosWrappers::SolverGMRES;
      __iterativeSolve<SSolver, PSolver, TSolver>(LBFGS_r_vector,
                                                  LBFGS_q_vector,
                                                  tangent_matrix);
    }
  else if (__type == SolverType::Bicgstab)
    {
      using SSolver = SolverBicgstab<Vector<double>>;
      using PSolver = PETScWrappers::SolverBicgstab;
      using TSolver = TrilinosWrappers::SolverBicgstab;
      __iterativeSolve<SSolver, PSolver, TSolver>(LBFGS_r_vector,
                                                  LBFGS_q_vector,
                                                  tangent_matrix);
    }
}



template <typename LATraits>
void
LASolver<LATraits>::__directSolve(BVector        &LBFGS_r_vector,
                                  const BVector  &LBFGS_q_vector,
                                  const BSMatrix &tangent_matrix)
{
  using namespace dealii;
  if constexpr (std::is_same_v<typename LATraits::TMTag, ::common::TagSerial>)
    {
      for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks();
           ++ithGroup)
        {
          SparseDirectUMFPACK A_direct;
          A_direct.initialize(tangent_matrix.block(ithGroup, ithGroup));
          A_direct.vmult(LBFGS_r_vector.block(ithGroup),
                         LBFGS_q_vector.block(ithGroup));
        }
    }
  else if constexpr (std::is_same_v<typename LATraits::TMTag,
                                    ::common::TagPETSc>)
    {
      // https://dealii.org/current/doxygen/deal.II/classPETScWrappers_1_1SparseDirectMUMPS.html

#ifdef HAVE_PETSC
      using PrecJacobi = PETScWrappers::PreconditionJacobi;
      using PrecILU    = PETScWrappers::PreconditionILU;
      using PrecICC    = PETScWrappers::PreconditionICC;
      using PrecPSails = PETScWrappers::PreconditionParaSails;
      using PrecSOR    = PETScWrappers::PreconditionSOR;
      using PrecSSOR   = PETScWrappers::PreconditionSSOR;
      //        using PrecShell  = PETScWrappers::PreconditionShell;
      using PrecNone = PETScWrappers::PreconditionNone;

      using PrecType = PrecJacobi;

      for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks();
           ++ithGroup)
        {
          SolverControl solver_control(__tolList[ithGroup].nIters,
                                       __tolList[ithGroup].tol);

          PETScWrappers::SparseDirectMUMPS solver(solver_control,
                                                  *__mpiInfo.mpiCommPtr());
          solver.set_symmetric_mode(true);

          //            PrecType preconditioner;
          //            preconditioner.initialize(tangent_matrix.block(ithGroup,
          //            ithGroup)); solver.initialize(preconditioner);

          solver.solve(tangent_matrix.block(ithGroup, ithGroup),
                       LBFGS_r_vector.block(ithGroup),
                       LBFGS_q_vector.block(ithGroup));
        }
#endif
    }
  else if constexpr (std::is_same_v<typename LATraits::TMTag,
                                    ::common::TagTrilinos>)
    {
      // https://dealii.org/current/doxygen/deal.II/classTrilinosWrappers_1_1SolverDirect.html

#ifdef HAVE_TRILINOS
      for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks();
           ++ithGroup)
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



template class PhaseField::LASolver<common::Traits<common::TagSerial>>;

#if defined(HAVE_PETSC) && HAVE_PETSC
template class PhaseField::LASolver<common::Traits<common::TagPETSc>>;
#endif

#if defined(HAVE_TRILINOS) && HAVE_TRILINOS
template class PhaseField::LASolver<common::Traits<common::TagTrilinos>>;
#endif
