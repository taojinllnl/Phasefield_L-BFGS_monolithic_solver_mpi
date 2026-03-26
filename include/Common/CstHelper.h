//
//  CstHelper.h
//  main
//
//

#ifndef CstHelper_h
#define CstHelper_h


#include <deal.II/lac/affine_constraints.h>


#include "BlockDesc.h"


namespace bcs {



class CstHelper
{
public:
    static void cstReinit(dealii::AffineConstraints<double>& constraints,
                          const dealii::IndexSet& locally_owned_dofs,
                          const dealii::IndexSet& locally_relevant_dofs,
                          const MPI_Comm& mpiComm);
};






}


#endif /* CstHelper_h */
