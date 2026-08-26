//
//  MPIIterativeSolver.h
//  main
//
//

#ifndef MPICGSolver_h
#define MPICGSolver_h

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/mpi.h>

#include "MPIInfo.h"

/**
 *
 * This class wraps an iterative solver for MPI/distributed runs to enforce a global iteration budget.
 * In some specific versions of dealii, the MPI-version CG solver ignores the upper limit for the number of iterations given by `dealii::SolverControl` and throws an exception at 10,000 iterations even if the residual is close to the prescribed tolerance.
 * That's not suitable for strongly coupled problems.
 * Therefore, this wrapper catches such exceptions and restarts the CG solve, so that the solve only fails (i.e., propagates a NoConvergence) when the global iteration limit is reached.
 *
 */

namespace common{
template <typename MatrixType, typename CGType>
class MPIIterativeSolver
: public dealii::Subscriptor
{
public:
    using SolverControl = dealii::SolverControl;
    
private:
    const common::MPIInfo&          __mpiInfo;
    const std::string   __name;
    const double        __tol;
    const unsigned int  __maxStep;
    
    unsigned int        __failAt = 0;
    unsigned int        __maxRetryTimes = 0;
    unsigned int        __nth_try = 0;
    
    unsigned int        __total_iters = 0;
    
    std::unique_ptr<SolverControl::NoConvergence>  __noConv{};
    
    //    void __solve_again(const MatrixType& A,
    //                       VectorxType& x,
    //                       const VectorxType& b,
    //                       const PrecType& preconditioner,
    //                       oz::Logger::OFSLogger& logger,
    //                       const std::string& name,
    //                       const unsigned int  maxRetryTimes,
    //                       SolverControl::NoConvergence noConv);
    
    void __noConvLog(const SolverControl& solverControl);
    
    template <typename VectorxType, typename PrecType>
    bool __solve(const MatrixType& A,
                 VectorxType& x,
                 const VectorxType& b,
                 const PrecType& preconditioner);
    
public:
    MPIIterativeSolver(const common::MPIInfo& mpiInfo,
                       const std::string& name,
                       const double tol,
                const unsigned int maxStep);
    
    
    template <typename VectorxType, typename PrecType>
    unsigned int solve(const MatrixType& A,
                       VectorxType& x,
                       const VectorxType& b,
                       const PrecType& preconditioner);
    
    
};



template <typename MatrixType, typename CGType>
MPIIterativeSolver<MatrixType, CGType>
::MPIIterativeSolver(const common::MPIInfo&          mpiInfo,
                     const std::string& name,
                     const double tol,
              const unsigned int maxStep)
: __mpiInfo(mpiInfo)
, __name(name)
, __tol(tol)
, __maxStep(maxStep)
{}



template <typename MatrixType, typename CGType>
void MPIIterativeSolver<MatrixType, CGType>
::__noConvLog(const SolverControl& solverControl
/*, const unsigned int  maxRetryTimes*/)
{
    __failAt = solverControl.last_step();
    if (__failAt == 0)
      __failAt = 1;
    __maxRetryTimes = __maxStep / __failAt;
    
    if(__nth_try % 100 == 0 && __mpiInfo.isRankEqualsTo(0))
    {
        std::cout << "NOT CONV [" << __nth_try  << " / " << __maxRetryTimes
        << " times] in solving for [ " << __name << " ] at step "
        << solverControl.last_step() << " with residual "
        << solverControl.last_value() << " ( > "
        << solverControl.tolerance() << " [ required residual ] ) "
        << "\nMax step: " << solverControl.max_steps()
        << std::endl;
    }

    __noConv = std::make_unique<SolverControl::NoConvergence>(solverControl.last_step(),
                                                              solverControl.last_value());
}




template <typename MatrixType, typename CGType>
template <typename VectorxType, typename PrecType>
bool MPIIterativeSolver<MatrixType, CGType>
::__solve(const MatrixType &A,
          VectorxType &x,
          const VectorxType &b,
          const PrecType &preconditioner)
{
    // the ith solving by CG solver
    ++__nth_try;
    __noConv.reset();
    
    // create a SolverControl and SolverCG
    SolverControl solverControl(__maxStep, __tol);
    CGType cgSolver(solverControl);
    
    // catch the exception thrown if the iteration limit is reached
    try {
        cgSolver.solve(A, x, b, preconditioner);
    } catch (SolverControl::NoConvergence& noConv) {
        // update the total iteration number
        __total_iters += noConv.last_step;
        __noConvLog(solverControl);
        return false;
    }
    __total_iters += solverControl.last_step();
    return true;
}

template <typename MatrixType, typename CGType>
template <typename VectorxType, typename PrecType>
unsigned int MPIIterativeSolver<MatrixType, CGType>
::solve(const MatrixType& A,
        VectorxType& x,
        const VectorxType& b,
        const PrecType& preconditioner)
{
    __total_iters = 0;
    __nth_try = 0;
    __noConv.reset();
    
    // zero out the unknown vector
    x = 0.;
    
    // repeat until allowed number of iterations
    // The MPI-version CG sover may throw an execption even if the number of iterations is smaller than the given __maxStep in SolverControl
    while(__total_iters <= __maxStep)
    {
        // return the number of iterations if converge
        if(__solve(A, x, b, preconditioner))
        {
            return __total_iters;
        }
    }
    
    // not converged
    AssertThrow(false, *__noConv);
}

}

#endif /* MPICGSolver_h */
