//
//  CstHelper.cpp
//  main
//
//

#include "../../include/Common/CstHelper.h"


using namespace ::dealii;
using namespace ::bcs;



CstEntry::CstEntry(const unsigned int  dofAtPnt,
                   const double        value )
: value(value)
, dofAtPnt(dofAtPnt)
, dof(numbers::invalid_dof_index)
{}


CstEntry::CstEntry(const CstEntry& view)
: value(view.value)
, dofAtPnt(view.dofAtPnt)
, dof(view.dof)
{}



::dealii::types::global_dof_index
CstEntry::getGlobalDoF() const
{
    return dof;
}

double CstEntry::getValue() const
{
    return value;
}

bool 
CstEntry::hasValidDoF() const
{
    return dof != ::dealii::numbers::invalid_dof_index;
}



void CstEntry::setDof(const ::dealii::types::global_dof_index new_dof)
{
    AssertThrow(new_dof != ::dealii::numbers::invalid_dof_index,
                    ExcMessage("CstView::setDof() received invalid_dof_index."));
    dof = new_dof;
}



void CstEntry::reset()
{
    dof = ::dealii::numbers::invalid_dof_index;
}



CstPnt
::CstPnt(const std::array<double, 3>& pntCoorinates,
         const std::vector<CstEntry> & csts,
         const double tol)
: pntCoorinates(std::make_shared<std::array<double, 3>>(pntCoorinates))
, csts(csts)
, nCsts(csts.size())
, tol(tol)
{}


CstPnt
::CstPnt(const CstPnt& cstPnt)
: pntCoorinates(cstPnt.pntCoorinates)
, csts(cstPnt.csts)
, nCsts(cstPnt.nCsts)
, tol(cstPnt.tol)
{}

const std::vector<CstEntry>& 
CstPnt
::cstEntrys() const
{
    return csts;
}




CstFunc
::CstFunc(const SelFunc& selectorFunc,
          const std::vector<CstEntry>& csts)
: nCsts(0)
, nPnts(0)
, selectorFunc(std::make_shared<SelFunc>(selectorFunc))
, valuesFunc(nullptr)
, csts(csts)
, dofsFound()
, cstsFound()
, pointsFound()
{}


CstFunc
::CstFunc(const SelFunc&               selectorFunc,
          const std::vector<CstEntry>& csts,
          const ValuesAtPntFunc&       valuesFunc)
: nCsts(0)
, nPnts(0)
, selectorFunc(std::make_shared<SelFunc>(selectorFunc))
, valuesFunc(std::make_shared<ValuesAtPntFunc>(valuesFunc))
, csts(csts)
, dofsFound()
, cstsFound()
, pointsFound()
{}

CstFunc
::CstFunc(const CstFunc& cstFunc)
: nCsts(0)
, nPnts(0)
, selectorFunc(cstFunc.selectorFunc)
, valuesFunc(cstFunc.valuesFunc)
, csts(cstFunc.csts)
, dofsFound()
, cstsFound()
, pointsFound()
{}


const std::vector<CstEntry>& 
CstFunc
::cstEntrys() const
{
    return cstsFound;
}




void CstHelper
::cstReinit(AffineConstraints<double>& constraints,
            const IndexSet& locally_owned_dofs,
            const IndexSet& locally_relevant_dofs)
{
#  if DEAL_II_VERSION_GTE(9, 6, 0)
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
#  else
    constraints.reinit(locally_relevant_dofs);
#  endif
}





void CstHelper
::addPntCst(AffineConstraints<double>&    cst,
            const ::dealii::types::global_dof_index dof,
            const double value)
{
    if(!cst.is_constrained(dof))
    {
        cst.add_line(dof);
        cst.set_inhomogeneity(dof, value);
    }
}




