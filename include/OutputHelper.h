//
//  OutputHelper.hpp
//  main
//
//

#ifndef OutputHelper_hpp
#define OutputHelper_hpp

#include "Common/Traits.h"
#include "Common/BlockVectorWrapper.h"
#include "Common/MPIInfo.h"
#include "Common/CstHelper.h"

#include <memory>
#include <array>

#include <deal.II/base/quadrature_point_data.h>



#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


namespace PhaseField {


/**
 *
 * This class is used to output the results, shortening the length of the main.cc.
 * It includes:
 * - solution: displacement, temperature, phase-field
 * - L2 projected stress
 * - L2 projected heat flux
 * - partitioning in MPI mode
 *
 */

template <typename LATraits, typename Tria, typename PointHistory>
class OutputHelper
{
public:
    static constexpr int  dim       = Tria::dimension;
    static constexpr bool is_mpi    =
        !std::is_same_v<typename LATraits::TMTag, ::common::TagSerial>;

    using BVector  = ::common::BlockVectorWrapper<LATraits>;
    
    using CellDataStorage = dealii::CellDataStorage<typename Tria::cell_iterator, PointHistory>;
    
    using DataComponentInterpretationList = std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>;
private:
    const ::common::MPIInfo&                      __mpiInfo;
                
    Tria&                               __tria;
    
    const dealii::DoFHandler<dim>&      __dof_handler;
    
    const dealii::QGauss<dim>&          __qf_cell;
    
    const std::vector<std::string>      __solution_name;
    
    const DataComponentInterpretationList __data_component_interpretation;
    
    const DataComponentInterpretationList __data_component_L2;
    
    const std::string                     __solutionName;
    const std::string                     __mpiType;
    
private:
    static std::vector<std::string> __makeSolutionName();
    
    static DataComponentInterpretationList __makeDataComponentInterpretation();
    
    static DataComponentInterpretationList __makeL2DataComponentInterpretation();
    
    
    void __stressL2(dealii::DataOut<dim>&       data_out,
                    dealii::DoFHandler<dim>&    dof_handler_L2,
                    dealii::AffineConstraints<double>&  constraints,
                    const unsigned int polyDegree,
                    const CellDataStorage& qPntHistory)   const;
    
    void __materialIDs(dealii::DataOut<dim>& data_out) const;
    
    void __solution(dealii::DataOut<dim>& data_out,
                    const BVector&     solution) const;
    
    void __partitioning(dealii::DataOut<dim>& data_out) const;
    
public:
    OutputHelper(const ::common::MPIInfo&                   mpiInfo,
                 Tria&                            tria,
                 const dealii::DoFHandler<dim>&   dof_handler,
                 const dealii::QGauss<dim>&       qf_cell,
                 const unsigned int caseID,
                 const std::string& mpiType);
    
    
    void output(const unsigned int ithTimeStep,
                const unsigned int polyDegree,
                const std::string& dir,
                const std::string& modeType,
                const std::string& laSolver,
                const BVector&     solution,
                const CellDataStorage& qPntHistory) const;
    
    
    void output(const std::string& dirName,
                const std::string& filename,
                const RTria<dim>* tria = nullptr);
};











template <typename LATraits, typename Tria, typename PointHistory>
std::vector<std::string>
OutputHelper<LATraits, Tria, PointHistory>
::__makeSolutionName()
{
    std::vector<std::string> solution_name(dim, "displacement");
    solution_name.emplace_back("phasefield");
    return solution_name;
}

template <typename LATraits, typename Tria, typename PointHistory>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
OutputHelper<LATraits, Tria, PointHistory>
::__makeDataComponentInterpretation()
{
    using namespace dealii;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    // vector for displacement
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    // scalar for phase-field
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    return data_component_interpretation;
}


template <typename LATraits, typename Tria, typename PointHistory>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
OutputHelper<LATraits, Tria, PointHistory>
::__makeL2DataComponentInterpretation()
{
    using namespace dealii;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation_L2(1,
                                     DataComponentInterpretation::component_is_scalar);
    return data_component_interpretation_L2;
}

template <typename LATraits, typename Tria, typename PointHistory>
OutputHelper<LATraits, Tria, PointHistory>
::OutputHelper(const ::common::MPIInfo&                   mpiInfo,
               Tria&                            tria,
               const dealii::DoFHandler<dim>&   dof_handler,
               const dealii::QGauss<dim>&       qf_cell,
               const unsigned int caseID,
               const std::string& mpiType)
: __mpiInfo(mpiInfo)
, __tria(tria)
, __dof_handler(dof_handler)
, __qf_cell(qf_cell)
, __solution_name(OutputHelper<LATraits, Tria, PointHistory>::__makeSolutionName())
, __data_component_interpretation(OutputHelper<LATraits, Tria, PointHistory>::__makeDataComponentInterpretation())
, __data_component_L2(OutputHelper<LATraits, Tria, PointHistory>::__makeL2DataComponentInterpretation())
, __solutionName("Case_" + std::to_string(caseID) + "-" + std::to_string(dim) + "d_")
, __mpiType(mpiType + std::to_string(__mpiInfo.nRanks()))
{}



template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__stressL2(dealii::DataOut<dim>& data_out,
             dealii::DoFHandler<dim>& dof_handler_L2,
             dealii::AffineConstraints<double>&  constraints,
             const unsigned int polyDegree,
             const CellDataStorage& qPntHistory) const
{
    using namespace dealii;
    
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = i; j < dim; ++j)
        {
            MappingQ<dim> mapping(polyDegree + 1);
            typename LATraits::VectorBlock stress_field_L2;
            typename LATraits::VectorBlock stress_field_L2_rele;
            
            if constexpr (is_mpi)
            {
                const IndexSet &locally_owned_dofs = dof_handler_L2.locally_owned_dofs();
                const IndexSet locally_relevant_dofs =
                    DoFTools::extract_locally_relevant_dofs(dof_handler_L2);
                
                stress_field_L2.reinit(locally_owned_dofs,
                                       *__mpiInfo.mpiCommPtr());
                stress_field_L2_rele.reinit(locally_owned_dofs,
                                            locally_relevant_dofs,
                                            *__mpiInfo.mpiCommPtr());
            }
            else
            {
                stress_field_L2.reinit(dof_handler_L2.n_dofs());
            }
            
            VectorTools::project(mapping,
                                 dof_handler_L2,
                                 constraints,
                                 __qf_cell,
                                 [&qPntHistory, i, j](const auto &cell, const unsigned int q) -> double {
                if constexpr (is_mpi)
                    if (!cell->is_locally_owned())
                        return 0.0;
                return qPntHistory.get_data(cell)[q]->get_cauchy_stress()[i][j];
            },
                                 stress_field_L2);
            
            if constexpr (is_mpi)
            {
                stress_field_L2.compress(dealii::VectorOperation::insert);
                stress_field_L2_rele = stress_field_L2;
                stress_field_L2_rele.update_ghost_values();
            }
            
            const std::string stress_name =
            "Cauchy_stress_" + std::to_string(i + 1) + std::to_string(j + 1) + "_L2";
            
            if constexpr (is_mpi){
                data_out.add_data_vector(dof_handler_L2,
                                         stress_field_L2_rele,
                                         stress_name,
                                         __data_component_L2);
            } else {
                data_out.add_data_vector(dof_handler_L2,
                                         stress_field_L2,
                                         stress_name,
                                         __data_component_L2);
            }
        }
}





template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__materialIDs(dealii::DataOut<dim>& data_out) const
{
    using namespace dealii;
    Vector<float> cell_material_id(__tria.n_active_cells());
    // output material ID for each cell
    
    for (auto cell = __tria.begin_active(); cell != __tria.end(); ++cell)
    {
        cell_material_id(cell->active_cell_index()) = cell->material_id();
    }
    data_out.add_data_vector(cell_material_id, "materialID");
}



template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__solution(dealii::DataOut<dim>& data_out,
             const BVector&         solution) const
{
    using namespace dealii;
    if constexpr (is_mpi) {
        // solution
        data_out.add_data_vector(solution.relevance(),
                                 __solution_name,
                                 DataOut<dim>::type_dof_data,
                                 __data_component_interpretation);
    } else {
        data_out.add_data_vector(solution.base(),
                                 __solution_name,
                                 DataOut<dim>::type_dof_data,
                                 __data_component_interpretation);
    }
}


template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__partitioning(dealii::DataOut<dim>& data_out) const
{
    using namespace dealii;
    if constexpr (is_mpi) {
        // partitioning 1
        Vector<float> subdomain(__tria.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = __tria.locally_owned_subdomain();
        data_out.add_data_vector(subdomain, "subdomain");
        
        // partitioning 2
        Vector<float> subdomain_cell(__tria.n_active_cells());
        for (auto cell = __tria.begin_active(); cell != __tria.end(); ++cell)
            subdomain_cell(cell->active_cell_index()) = cell->subdomain_id();
        data_out.add_data_vector(subdomain_cell, "partitioning");

    }
}


template <typename LATraits, typename Tria, typename PointHistory>
void OutputHelper<LATraits, Tria, PointHistory>
::output(const unsigned int ithTimeStep,
         const unsigned int polyDegree,
         const std::string& dir,
         const std::string& modeType,
         const std::string& laSolver,
         const BVector&     solution,
         const CellDataStorage& qPntHistory) const
{
    using namespace dealii;
    
    const std::string filename = __solutionName + "_" + modeType + "_" + laSolver + "_" + __mpiType + "-";
    
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler(__dof_handler);
    
    __solution(data_out, solution);
    __materialIDs(data_out);
    __partitioning(data_out);
    
    
    DoFHandler<dim> dof_handler_L2(__tria);
    FE_Q<dim>     fe_L2(polyDegree); //FE_Q element is continuous
    dof_handler_L2.distribute_dofs(fe_L2);
    
    
    AffineConstraints<double> constraints;
    constraints.clear();
    if constexpr (is_mpi)
    {
        const IndexSet &locally_owned_dofs = dof_handler_L2.locally_owned_dofs();
        const IndexSet locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler_L2);
        
        bcs::CstHelper::cstReinit(constraints, locally_owned_dofs, locally_relevant_dofs);
//        constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    }
    DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
    if constexpr (is_mpi){
        const IndexSet &locally_owned_dofs = dof_handler_L2.locally_owned_dofs();
        const IndexSet locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler_L2);
        constraints.make_consistent_in_parallel(locally_owned_dofs,
                                                locally_relevant_dofs,
                                                *__mpiInfo.mpiCommPtr());
    }
    
    constraints.close();
    
    


    __stressL2(data_out,
               dof_handler_L2,
               constraints,
               polyDegree,
               qPntHistory);

    
    
    data_out.build_patches(polyDegree);
    
    
    if constexpr (!is_mpi) {
        std::ofstream output(dir + filename +
                             Utilities::int_to_string(ithTimeStep, 4) + ".vtu");
         
        data_out.write_vtu(output);
    } else {
        data_out.write_vtu_with_pvtu_record(dir,
                                            filename,
                                            ithTimeStep,
                                            *__mpiInfo.mpiCommPtr(),
                                            4 /*n_digits*/,
                                            0 /*n_groups*/);
    }

}



template <typename LATraits, typename Tria, typename PointHistory>
void OutputHelper<LATraits, Tria, PointHistory>
::output(const std::string& dirName,
         const std::string& filename,
         const RTria<dim>* tria)
{
    using namespace ::dealii;
    if constexpr (is_mpi){
        DataOut<dim> data_out;
        data_out.attach_dof_handler(__dof_handler);
        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(dirName,
                                            filename,
                                            0,
                                            *__mpiInfo.mpiCommPtr(),
                                            1 /*n_digits*/,
                                            0 /*n_groups*/);
        
    } else {
        if(tria != nullptr)
        {
            
            std::ofstream outStream(dirName + filename);
            GridOut       grid_out;
            grid_out.write_vtu(*tria, outStream);
        }
    }
}


}
#endif /* OutputHelper_hpp */



