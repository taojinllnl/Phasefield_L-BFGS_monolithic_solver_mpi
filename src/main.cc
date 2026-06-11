/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Tao Jin
 *         University of Ottawa, Ottawa, Ontario, Canada
 *         April. 2024
 *
 * How to cite:
 *         Jin T, Li Z, Chen K. A novel phase-field monolithic scheme for
 brittle crack
 *         propagation based on the limited-memory BFGS method with adaptive
 mesh refinement.
 *         Int J Numer Methods Eng. 2024;e7572. doi: 10.1002/nme.7572
 */

/* A monolithic scheme based on the L-BFGS method to solve the phase-field crack
 * problem
 * 1. The phase-field AT-2 model is based on "A phase field model for
 * rate-independent crack propagation - Robust algorithmic implementation based
 * on operator splits" by Christian Miehe , Martina Hofacker, Fabian Welschinger
 * 2. This code implements a monolithic approach. The phase-field
 * irreversibility is enforced through the history field Phi_0^+ and the
 * viscosity parameter.
 * 3. Using TBB for stiffness assembly and Gauss point calculation.
 * 4. Using adaptive mesh refinement.
 * 5. Add the AT-1 phase-field model (Feb. 1st, 2026)
 * 6. Add the Phase-field cohesive zone model (PFCZM) (Feb. 7th 2026)
 * 7. Add a flag to differentiate between plane stress and plane strain
 *    case in 2D (Feb. 26th, 2026)
 * 8. Add the AT-1 cohesive phase-field model (April 8th, 2026)
 */



/*
 * Repartitioning mode
 *
 * If this compile-time macro is set to 0, the customized repartitioning
 * strategy is disabled. In this case, the default repartitioning behavior
 * of deal.II is used, and the parameter `Repartitioning ratio` in the .prm
 * file is ignored.
 *
 * In the default repartitioning mode, repartitioning is performed whenever
 * mesh refinement is executed. Therefore, even if only a small number of
 * cells are refined or coarsened, the mesh may still be repartitioned.
 *
 * Pros:
 *   1. The default repartitioning strategy is provided by deal.II and may be
 *      more robust across different mesh-adaptation cases.
 *   2. The solution and history variables are consistently transferred,
 *      projected, and recovered during refinement and repartitioning
 *      simultaneously.
 *
 * Cons:
 *   1. Repartitioning may be triggered even when the mesh change is very small.
 *      This can introduce unnecessary communication overhead, especially for
 *      large-scale MPI simulations.
 *   2. The user-defined `Repartitioning ratio` is ignored in this mode.
 *
 *
 * If the customized repartitioning mode is enabled, the program first evaluates
 * the workload distribution among MPI ranks. The workload imbalance is measured
 * by
 *
 *     ratio = max number of locally owned active cells among all ranks
 *           / min number of locally owned active cells among all ranks.
 *
 * Repartitioning is triggered only when this ratio exceeds the prescribed
 * threshold defined by `Repartitioning ratio` in the .prm file.
 *
 * Pros:
 *   1. Repartitioning is performed less frequently, which can reduce
 *      communication overhead.
 *   2. The repartitioning tolerance can be adjusted for different problems.
 *
 * Cons:
 *   1. When repartitioning is eventually triggered, the solution and history
 *      variables still need to be transferred, projected, and recovered one
 *      more time.
 *   2. This customized strategy may not work with some versions of deal.II.
 *      In particular, some versions may throw an exception related to uncleared
 *      refinement flags, even though these flags cannot be cleared
 * automatically or manually in the current workflow.
 */
#define ENABLE_CUSTOMIZED_REPARTITION_MODE 0

#define DEBUG_FLAG false

#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <fstream>
#include <iostream>

#include "../include/Common/BlockDesc.h"
#include "../include/Common/BlockSparseMatrixWrapper.h"
#include "../include/Common/BlockVectorWrapper.h"
#include "../include/Common/DataListReader.h"
#include "../include/Common/FileSystem.h"
#include "../include/Common/MPIInfo.h"
#include "../include/Common/TimerOutputWrapper.h"
#include "../include/Common/Traits.h"
#include "../include/Common/cst/CstMaker.h"
#include "../include/LASolver.h"
#include "../include/MaterialConstants.h"
#include "../include/OutputHelper.h"
#include "../include/SpectrumDecomposition.h"
#include "../include/Utilities.h"


namespace PhaseField
{
  using namespace dealii;
  using namespace common;
  using namespace bcs;

  // body force
  template <int dim>
  void
  right_hand_side(const std::vector<Point<dim>> &points,
                  std::vector<Tensor<1, dim>>   &values,
                  const double                   fx,
                  const double                   fy,
                  const double                   fz)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
        if (dim == 2)
          {
            values[point_n][0] = fx;
            values[point_n][1] = fy;
          }
        else
          {
            values[point_n][0] = fx;
            values[point_n][1] = fy;
            values[point_n][2] = fz;
          }
      }
  }

  // various phase-field models (AT1, AT2, PFCZM)
  double
  degradation_function(const double       d,
                       const double       p,
                       const double       a1,
                       const double       a2,
                       const double       a3,
                       const std::string &model_name)
  {
    double value = 0.0;

    if (model_name == "AT2" || model_name == "AT1")
      value = (1.0 - d) * (1.0 - d);
    else if (model_name == "PFCZM" || model_name == "AT1-Cohesive")
      {
        const double f1 = std::pow(std::fabs(1.0 - d), p);
        const double f2 = f1 + a1 * d + a1 * a2 * d * d + a1 * a3 * d * d * d;
        value           = f1 / f2;
      }
    else
      Assert(
        false,
        ExcMessage(
          "The phase-field degradation function has not been implemented!"));

    return value;
  }

  double
  degradation_function_derivative(const double       d,
                                  const double       p,
                                  const double       a1,
                                  const double       a2,
                                  const double       a3,
                                  const std::string &model_name)
  {
    double value = 0.0;

    if (model_name == "AT2" || model_name == "AT1")
      value = 2.0 * (d - 1.0);
    else if (model_name == "PFCZM" || model_name == "AT1-Cohesive")
      {
        const short  sign = (d > 1) ? -1 : 1;
        const double f1   = std::pow(std::fabs(1.0 - d), p);
        const double f2   = f1 + a1 * d + a1 * a2 * d * d + a1 * a3 * d * d * d;
        const double f1_1 = sign * (-p) * std::pow(std::fabs(1.0 - d), p - 1.0);
        const double f2_1 =
          f1_1 + a1 + 2.0 * a1 * a2 * d + 3.0 * a1 * a3 * d * d;
        value = (f1_1 * f2 - f1 * f2_1) / (f2 * f2);
      }
    else
      Assert(
        false,
        ExcMessage(
          "The phase-field degradation function has not been implemented!"));

    return value;
  }

  double
  degradation_function_2nd_order_derivative(const double       d,
                                            const double       p,
                                            const double       a1,
                                            const double       a2,
                                            const double       a3,
                                            const std::string &model_name)
  {
    double value = 0.0;

    if (model_name == "AT2" || model_name == "AT1")
      value = 2.0;
    else if (model_name == "PFCZM" || model_name == "AT1-Cohesive")
      {
        const short  sign = (d > 1) ? -1 : 1;
        const double f1   = std::pow(std::fabs(1.0 - d), p);
        const double f2   = f1 + a1 * d + a1 * a2 * d * d + a1 * a3 * d * d * d;
        const double f1_1 = sign * (-p) * std::pow(std::fabs(1.0 - d), p - 1.0);
        const double f2_1 =
          f1_1 + a1 + 2.0 * a1 * a2 * d + 3.0 * a1 * a3 * d * d;
        const double f1_2 =
          p * (p - 1.0) * std::pow(std::fabs(1.0 - d), p - 2.0);
        const double f2_2 = f1_2 + 2.0 * a1 * a2 + 6.0 * a1 * a3 * d;
        const double f3   = f1_1 * f2 - f1 * f2_1;
        const double f4   = f2 * f2;
        const double f3_1 = f1_2 * f2 - f1 * f2_2;
        const double f4_1 = 2.0 * f2 * f2_1;
        value             = (f3_1 * f4 - f3 * f4_1) / (f4 * f4);
      }
    else
      Assert(
        false,
        ExcMessage(
          "The phase-field degradation function has not been implemented!"));

    return value;
  }

  inline double
  phasefield_geometry_function(const double d, const std::string &model_name)
  {
    double value = 0.0;
    if (model_name == "AT2")
      value = d * d;
    else if (model_name == "AT1" || model_name == "AT1-Cohesive")
      value = d;
    else if (model_name == "PFCZM")
      value = 2.0 * d - d * d;
    else
      Assert(false,
             ExcMessage(
               "The phase-field geometric function has not been implemented!"));

    return value;
  }

  inline double
  phasefield_geometry_function_derivative(const double       d,
                                          const std::string &model_name)
  {
    double value = 0.0;
    if (model_name == "AT2")
      value = 2.0 * d;
    else if (model_name == "AT1" || model_name == "AT1-Cohesive")
      value = 1.0;
    else if (model_name == "PFCZM")
      value = 2.0 * (1.0 - d);
    else
      Assert(false,
             ExcMessage(
               "The phase-field geometric function has not been implemented!"));

    return value;
  }

  inline double
  phasefield_geometry_function_2nd_order_derivative(
    const double       d,
    const std::string &model_name)
  {
    (void)d;
    double value = 0.0;
    if (model_name == "AT2")
      value = 2.0;
    else if (model_name == "AT1" || model_name == "AT1-Cohesive")
      value = 0.0;
    else if (model_name == "PFCZM")
      value = -2.0;
    else
      Assert(false,
             ExcMessage(
               "The phase-field geometric function has not been implemented!"));

    return value;
  }

  inline double
  phasefield_coefficient_constant(const std::string &model_name)
  {
    double value = 0.0;
    if (model_name == "AT2")
      value = 2.0;
    else if (model_name == "AT1" || model_name == "AT1-Cohesive")
      value = 8.0 / 3.0;
    else if (model_name == "PFCZM")
      value = 4.0 * std::atan(1);
    else
      Assert(false,
             ExcMessage(
               "The phase-field geometric function has not been implemented!"));

    return value;
  }

  namespace Parameters
  {
    struct Scenario
    {
      unsigned int m_dim;
      std::string  m_mpi_type;
      std::string  m_library_dir;
      std::string  m_output_dir;
      std::string  m_mesh_file_name;
      std::string  m_extra_data_file;

      bool m_output_coarse_ori;
      bool m_output_refined_ori;
      bool m_output_step_0;


      unsigned int m_scenario;
      std::string  m_logfile_name;
      bool         m_output_iteration_history;
      std::string  m_phasefield_name;
      bool         m_plane_stress;
      std::string  m_type_nonlinear_solver;
      std::string  m_type_line_search;
      std::string  m_type_linear_solver;
      std::string  m_prec_type_linear_solver;
      std::string  m_refinement_strategy;
      double       m_repartition_ratio;
      unsigned int m_LBFGS_m;
      unsigned int m_global_refine_times;
      unsigned int m_local_prerefine_times;
      unsigned int m_max_adaptive_refine_times;
      int          m_max_allowed_refinement_level;
      double       m_phasefield_refine_threshold;
      double       m_allowed_max_h_l_ratio;
      unsigned int m_total_material_regions;
      std::string  m_material_file_name;
      int          m_reaction_force_face_id;

      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Scenario::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        prm.declare_entry("dimension",
                          "2",
                          Patterns::Integer(2),
                          "dimension of the problem");

        prm.declare_entry("mpi type",
                          "PETSc",
                          Patterns::Selection("PETSc|Trilinos|Serial"),
                          "underlying mpi type");

        prm.declare_entry("Library dir",
                          "./library/",
                          Patterns::FileName(Patterns::FileName::input),
                          "Library directory");

        prm.declare_entry("Output dir",
                          "./",
                          Patterns::FileName(Patterns::FileName::input),
                          "Output directory");

        prm.declare_entry("Mesh file name",
                          "./",
                          Patterns::FileName(Patterns::FileName::input),
                          "Mesh file name");

        prm.declare_entry("Extra data file",
                          "N/A",
                          Patterns::FileName(Patterns::FileName::input),
                          "Extra data file (default: N/A)");



        prm.declare_entry(
          "Output coarse original mesh",
          "yes",
          Patterns::Selection("yes|no"),
          "If it is yes, the .vtu file of the coarse original mesh will be created.");

        prm.declare_entry(
          "Output original mesh",
          "no",
          Patterns::Selection("yes|no"),
          "If it is yes, the .vtu file of the original mesh will be created.");

        prm.declare_entry(
          "Output timestep 0",
          "yes",
          Patterns::Selection("yes|no"),
          "If it is yes, the .vtu file of the timestep-1 will be created.");



        prm.declare_entry("Scenario number",
                          "1",
                          Patterns::Integer(0),
                          "Geometry, loading and boundary conditions scenario");

        prm.declare_entry("Log file name",
                          "Output.log",
                          Patterns::FileName(Patterns::FileName::input),
                          "Name of the file for log");

        prm.declare_entry("Output iteration history",
                          "yes",
                          Patterns::Selection("yes|no"),
                          "Shall we write iteration history to the log file?");

        prm.declare_entry("Phase-field model type",
                          "AT2",
                          Patterns::Selection("AT1|AT1-Cohesive|AT2|PFCZM"),
                          "Type of phase-field model");

        prm.declare_entry("Plane stress",
                          "no",
                          Patterns::Selection("yes|no"),
                          "If it is 2D, is it plane-stress?");

        prm.declare_entry("Nonlinear solver type",
                          "LBFGS",
                          Patterns::Selection("Newton|BFGS|LBFGS"),
                          "Type of solver used to solve the nonlinear system");

        prm.declare_entry(
          "Line search type",
          "GradientBased",
          Patterns::Selection("GradientBased|StrongWolfe"),
          "Type of line search method, the gradient-based method "
          "should be preferred since it is generally faster");

        prm.declare_entry("Linear solver type",
                          "Direct",
                          Patterns::Selection("Direct|CG|GMRes|Bicgstab"),
                          "Type of solver used to solve the linear system B0");

        prm.declare_entry(
          "Preconditioner type for iterative linear solver",
          "None",
          Patterns::Selection(
            "None|Jacobi|ILU|SOR|SSOR|Chebyshev|IC|ILUT|ICC|ParaSails|Shell|AMG"),
          "Preconditioner type for iterative linear solver: works for both Trilinos and PETSc: None|Jacobi|ILU|SOR|SSOR|AMG\nOnly works for Trilinos: Chebyshev|IC|ILUT\nOnly works for PETSc: ICC|ParaSails|Shell");

        prm.declare_entry(
          "Mesh refinement strategy",
          "adaptive-refine",
          Patterns::Selection("pre-refine|adaptive-refine"),
          "Mesh refinement strategy: pre-refine or adaptive-refine");

        prm.declare_entry("Repartitioning ratio",
                          "2.0",
                          Patterns::Double(0.0),
                          "The threshold for repartitioning");

        prm.declare_entry("LBFGS m",
                          "40",
                          Patterns::Integer(0),
                          "Number of vectors used for LBFGS");

        prm.declare_entry("Global refinement times",
                          "0",
                          Patterns::Integer(0),
                          "Global refinement times (across the entire domain)");

        prm.declare_entry(
          "Local prerefinement times",
          "0",
          Patterns::Integer(0),
          "Local pre-refinement times (assume crack path is known a priori), "
          "only refine along the crack path.");

        prm.declare_entry(
          "Max adaptive refinement times",
          "100",
          Patterns::Integer(0),
          "Maximum number of adaptive refinement times allowed in each step");

        prm.declare_entry("Max allowed refinement level",
                          "100",
                          Patterns::Integer(0),
                          "Maximum allowed cell refinement level");

        prm.declare_entry("Phasefield refine threshold",
                          "0.8",
                          Patterns::Double(),
                          "Phasefield-based refinement threshold value");

        prm.declare_entry(
          "Allowed max hl ratio",
          "0.25",
          Patterns::Double(),
          "Allowed maximum ratio between mesh size h and length scale l");

        prm.declare_entry("Material regions",
                          "1",
                          Patterns::Integer(0),
                          "Number of material regions");

        prm.declare_entry("Material data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Material data file");

        prm.declare_entry(
          "Reaction force face ID",
          "1",
          Patterns::Integer(),
          "Face id where reaction forces should be calculated "
          "(negative integer means not to calculate reaction force)");
      }
      prm.leave_subsection();
    }

    void
    Scenario::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        m_dim            = (unsigned int)prm.get_integer("dimension");
        m_mpi_type       = prm.get("mpi type");
        m_library_dir    = prm.get("Library dir");
        m_output_dir     = prm.get("Output dir");
        m_mesh_file_name = prm.get("Mesh file name");

        m_extra_data_file = prm.get("Extra data file");
        m_mesh_file_name  = m_library_dir + m_mesh_file_name;

        if (m_extra_data_file != "N/A")
          {
            m_extra_data_file = m_library_dir + m_extra_data_file;
          }

        m_output_coarse_ori  = prm.get_bool("Output coarse original mesh");
        m_output_refined_ori = prm.get_bool("Output original mesh");
        m_output_step_0      = prm.get_bool("Output timestep 0");

        m_scenario     = (unsigned int)prm.get_integer("Scenario number");
        m_logfile_name = prm.get("Log file name");
        m_output_iteration_history = prm.get_bool("Output iteration history");
        m_phasefield_name          = prm.get("Phase-field model type");
        m_plane_stress             = prm.get_bool("Plane stress");
        m_type_nonlinear_solver    = prm.get("Nonlinear solver type");
        m_type_line_search         = prm.get("Line search type");
        m_type_linear_solver       = prm.get("Linear solver type");
        m_prec_type_linear_solver =
          prm.get("Preconditioner type for iterative linear solver");

        m_refinement_strategy = prm.get("Mesh refinement strategy");
        m_repartition_ratio   = prm.get_double("Repartitioning ratio");
        m_LBFGS_m             = (unsigned int)prm.get_integer("LBFGS m");
        m_global_refine_times =
          (unsigned int)prm.get_integer("Global refinement times");
        m_local_prerefine_times =
          (unsigned int)prm.get_integer("Local prerefinement times");
        m_max_adaptive_refine_times =
          (unsigned int)prm.get_integer("Max adaptive refinement times");
        m_max_allowed_refinement_level =
          (unsigned int)prm.get_integer("Max allowed refinement level");
        m_phasefield_refine_threshold =
          prm.get_double("Phasefield refine threshold");
        m_allowed_max_h_l_ratio = prm.get_double("Allowed max hl ratio");
        m_total_material_regions =
          (unsigned int)prm.get_integer("Material regions");
        m_material_file_name = prm.get("Material data file");
        m_reaction_force_face_id =
          (unsigned int)prm.get_integer("Reaction force face ID");



        m_output_dir += m_refinement_strategy + "/" + m_phasefield_name + "/";
      }
      prm.leave_subsection();
    }

    struct FESystem
    {
      unsigned int m_poly_degree;
      unsigned int m_quad_order;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void
    FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "1",
                          Patterns::Integer(0),
                          "Phase field polynomial order");

        prm.declare_entry("Quadrature order",
                          "2",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void
    FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        m_poly_degree = (unsigned int)prm.get_integer("Polynomial degree");
        m_quad_order  = (unsigned int)prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    // body force (N/m^3)
    struct BodyForce
    {
      double m_x_component;
      double m_y_component;
      double m_z_component;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    BodyForce::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        prm.declare_entry("Body force x component",
                          "0.0",
                          Patterns::Double(),
                          "Body force x-component (N/m^3)");

        prm.declare_entry("Body force y component",
                          "0.0",
                          Patterns::Double(),
                          "Body force y-component (N/m^3)");

        prm.declare_entry("Body force z component",
                          "0.0",
                          Patterns::Double(),
                          "Body force z-component (N/m^3)");
      }
      prm.leave_subsection();
    }

    void
    BodyForce::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        m_x_component = prm.get_double("Body force x component");
        m_y_component = prm.get_double("Body force y component");
        m_z_component = prm.get_double("Body force z component");
      }
      prm.leave_subsection();
    }

    struct NonlinearSolver
    {
      unsigned int m_max_iterations_NR;
      unsigned int m_max_iterations_BFGS;
      bool         m_relative_residual;

      double m_tol_u_residual;
      double m_tol_d_residual;
      double m_tol_u_incr;
      double m_tol_d_incr;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Max iterations BFGS",
                          "20",
                          Patterns::Integer(0),
                          "Number of BFGS iterations allowed");

        prm.declare_entry("Relative residual",
                          "yes",
                          Patterns::Selection("yes|no"),
                          "Shall we use relative residual for convergence?");

        prm.declare_entry("Tolerance displacement residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Displacement residual tolerance");

        prm.declare_entry("Tolerance phasefield residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield residual tolerance");

        prm.declare_entry("Tolerance displacement increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Displacement increment tolerance");

        prm.declare_entry("Tolerance phasefield increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield increment tolerance");
      }
      prm.leave_subsection();
    }

    void
    NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        m_max_iterations_NR =
          (unsigned int)prm.get_integer("Max iterations Newton-Raphson");
        m_max_iterations_BFGS =
          (unsigned int)prm.get_integer("Max iterations BFGS");
        m_relative_residual = prm.get_bool("Relative residual");

        m_tol_u_residual = prm.get_double("Tolerance displacement residual");
        m_tol_d_residual = prm.get_double("Tolerance phasefield residual");
        m_tol_u_incr     = prm.get_double("Tolerance displacement increment");
        m_tol_d_incr     = prm.get_double("Tolerance phasefield increment");
      }
      prm.leave_subsection();
    }

    struct TimeInfo
    {
      double       m_end_time;
      std::string  m_time_file_name;
      unsigned int m_n_factors;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    TimeInfo::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Time data file");

        prm.declare_entry("n factors",
                          "1",
                          Patterns::Double(),
                          "Number of factors");
      }
      prm.leave_subsection();
    }

    void
    TimeInfo::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        m_end_time       = prm.get_double("End time");
        m_time_file_name = prm.get("Time data file");
        m_n_factors      = (unsigned int)prm.get_integer("n factors");
      }
      prm.leave_subsection();
    }

    struct AllParameters : public Scenario,
                           public FESystem,
                           public BodyForce,
                           public NonlinearSolver,
                           public TimeInfo
    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);

      // variables to store the sub-folders
      std::string subDir;
      std::string histDir;
      std::string oriDir;
      std::string resultsDir;
      std::string laSolverType;
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void
    AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Scenario::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      BodyForce::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      TimeInfo::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Scenario::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      BodyForce::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      TimeInfo::parse_parameters(prm);
    }
  } // namespace Parameters

  class Time
  {
  public:
    Time(const double time_end, const bool output_step_0)
      : m_timestep(0)
      , m_time_current(0.0)
      , m_time_end(time_end)
      , m_delta_t(0.0)
      , m_magnitude_list(1, 1.0)
      , m_need_output(output_step_0)
    {}

    virtual ~Time() = default;

    double
    current() const
    {
      return m_time_current;
    }
    double
    end() const
    {
      return m_time_end;
    }
    double
    get_delta_t() const
    {
      return m_delta_t;
    }
    double
    get_magnitude(const unsigned int ith = 0) const
    {
      return m_magnitude_list[ith];
    }
    unsigned int
    get_timestep() const
    {
      return m_timestep;
    }
    bool
    need_output() const
    {
      return m_need_output;
    }
    void
    increment(const std::vector<std::array<double, 3>> &time_table,
              const std::vector<std::vector<double>>   &factorList,
              const std::vector<unsigned int>          &intervals)
    {
      if (time_table.size() != factorList.size() ||
          time_table.size() != intervals.size())
        {
          AssertThrow(
            false,
            ExcMessage(
              "Time::increment: time_table, factorList, and intervals must have the same size."));
        }

      bool found_slot = false;

      double t_0 = 0.0;
      double t_1 = 0.0;

      unsigned int slot = 0;

      for (unsigned int i = 0; i < time_table.size(); ++i)
        {
          const auto &time_group = time_table[i];

          t_0       = time_group[0];
          t_1       = time_group[1];
          m_delta_t = time_group[2];

          if (m_delta_t <= 0.0)
            {
              AssertThrow(false,
                          ExcMessage(
                            "Time::increment: delta_t must be positive."));
            }

          const double tol = 1.0e-6 * m_delta_t;

          if (m_time_current >= t_0 - tol && m_time_current < t_1 - tol)
            {
              m_magnitude_list = factorList[i];
              slot             = i;
              found_slot       = true;
              break;
            }
        }


      m_time_current += m_delta_t;

      if (m_time_current < m_time_end && !found_slot)
        {
          AssertThrow(
            false,
            ExcMessage(
              "Time::increment: current time is not inside any time slot."));
        }

      const unsigned int step_in_slot = static_cast<unsigned int>(
        std::round((m_time_current - t_0) / m_delta_t));

      const unsigned int n_steps_in_slot =
        static_cast<unsigned int>(std::round((t_1 - t_0) / m_delta_t));

      const unsigned int interval = intervals[slot];

      if (interval == 0 || interval > n_steps_in_slot)
        {
          m_need_output = false; // turn off output at current slot
        }
      else
        {
          m_need_output = (step_in_slot % interval == 0);
        }

      ++m_timestep;
    }

  private:
    unsigned int        m_timestep;
    double              m_time_current;
    const double        m_time_end;
    double              m_delta_t;
    std::vector<double> m_magnitude_list;
    bool                m_need_output;
  };

  template <int dim>
  class LinearIsotropicElasticityAdditiveSplit
  {
  public:
    LinearIsotropicElasticityAdditiveSplit(
      const MaterialConstants<dim> &matConst)
      : matConst(matConst)
      , m_phase_field_value(0.0)
      , m_grad_phasefield(Tensor<1, dim>())
      , m_strain(SymmetricTensor<2, dim>())
      , m_stress(SymmetricTensor<2, dim>())
      , m_stress_positive(SymmetricTensor<2, dim>())
      , m_mechanical_C(SymmetricTensor<4, dim>())
      , m_strain_energy_positive(0.0)
      , m_strain_energy_negative(0.0)
      , m_strain_energy_total(0.0)
      , m_crack_energy_dissipation(0.0)
    {}

    const SymmetricTensor<4, dim> &
    get_mechanical_C() const
    {
      return m_mechanical_C;
    }

    const SymmetricTensor<2, dim> &
    get_cauchy_stress() const
    {
      return m_stress;
    }

    const SymmetricTensor<2, dim> &
    get_cauchy_stress_positive() const
    {
      return m_stress_positive;
    }

    double
    get_positive_strain_energy() const
    {
      return m_strain_energy_positive;
    }

    double
    get_negative_strain_energy() const
    {
      return m_strain_energy_negative;
    }

    double
    get_total_strain_energy() const
    {
      return m_strain_energy_total;
    }

    double
    get_crack_energy_dissipation() const
    {
      return m_crack_energy_dissipation;
    }

    double
    get_phase_field_value() const
    {
      return m_phase_field_value;
    }

    const Tensor<1, dim>
    get_phase_field_gradient() const
    {
      return m_grad_phasefield;
    }

    void
    update_material_data(const SymmetricTensor<2, dim> &strain,
                         const double                   phase_field_value,
                         const Tensor<1, dim>          &grad_phasefield,
                         const double phase_field_value_previous_step,
                         const double delta_time);

  private:
    const MaterialConstants<dim> &matConst;
    double                        m_phase_field_value;
    Tensor<1, dim>                m_grad_phasefield;
    SymmetricTensor<2, dim>       m_strain;
    SymmetricTensor<2, dim>       m_stress;
    SymmetricTensor<2, dim>       m_stress_positive;
    SymmetricTensor<4, dim>       m_mechanical_C;
    double                        m_strain_energy_positive;
    double                        m_strain_energy_negative;
    double                        m_strain_energy_total;
    double                        m_crack_energy_dissipation;
  };

  template <int dim>
  void
  LinearIsotropicElasticityAdditiveSplit<dim>::update_material_data(
    const SymmetricTensor<2, dim> &strain,
    const double                   phase_field_value,
    const Tensor<1, dim>          &grad_phasefield,
    const double                   phase_field_value_previous_step,
    const double                   delta_time)
  {
    using namespace usr_spectrum_decomposition;
    constexpr QPHNeedFlag needFlags = generate_need(flags);

    if constexpr (has_flag(needFlags, QPHNeedFlag::u_gradient))
      m_strain = strain;

    if constexpr (has_flag(needFlags, QPHNeedFlag::phasefield_value))
      m_phase_field_value = phase_field_value;

    if constexpr (has_flag(needFlags, QPHNeedFlag::phasefield_gradient))
      m_grad_phasefield = grad_phasefield;

    std::array<double, dim>         eigenvalues;
    std::array<Tensor<1, dim>, dim> eigenvectors;

    const double I_1 =
      has_flag(needFlags, QPHNeedFlag::u_gradient) ? trace(m_strain) : 0.0;

    const double positive_I1 = has_flag(needFlags, QPHNeedFlag::positive_ramp) ?
                                 positive_ramp_function(I_1) :
                                 0.0;

    const double negative_I1 = has_flag(needFlags, QPHNeedFlag::negative_ramp) ?
                                 negative_ramp_function(I_1) :
                                 0.0;

    SymmetricTensor<2, dim>                  strain_positive, strain_negative;
    std::array<SymmetricTensor<2, dim>, dim> projectors_2nd;

    if constexpr (has_flag(flags, QPHUpdateFlag::strain_decomp) ||
                  has_flag(flags, QPHUpdateFlag::stress_positive) ||
                  has_flag(flags, QPHUpdateFlag::stress) ||
                  has_flag(flags, QPHUpdateFlag::CCCC) ||
                  has_flag(flags, QPHUpdateFlag::strain_energy_pos) ||
                  has_flag(flags, QPHUpdateFlag::strain_energy_neg) ||
                  has_flag(flags, QPHUpdateFlag::strain_energy_tot))
      {
        spectrum_decomposition<dim>(m_strain, eigenvalues, eigenvectors);

        if constexpr (has_flag(flags, QPHUpdateFlag::stress_positive) ||
                      has_flag(flags, QPHUpdateFlag::CCCC) ||
                      has_flag(flags, QPHUpdateFlag::stress) ||
                      has_flag(flags, QPHUpdateFlag::strain_energy_pos) ||
                      has_flag(flags, QPHUpdateFlag::strain_energy_neg) ||
                      has_flag(flags, QPHUpdateFlag::strain_energy_tot))
          {
            for (int i = 0; i < dim; i++)
              projectors_2nd[i] =
                symmetrize(outer_product(eigenvectors[i], eigenvectors[i]));
          }

        if constexpr (has_flag(flags, QPHUpdateFlag::stress_positive) ||
                      has_flag(flags, QPHUpdateFlag::stress) ||
                      has_flag(flags, QPHUpdateFlag::strain_energy_pos) ||
                      has_flag(flags, QPHUpdateFlag::strain_energy_tot))
          {
            strain_positive = positive_tensor<dim>(eigenvalues, projectors_2nd);
          }

        if constexpr (has_flag(flags, QPHUpdateFlag::stress) ||
                      has_flag(flags, QPHUpdateFlag::strain_energy_neg) ||
                      has_flag(flags, QPHUpdateFlag::strain_energy_tot))
          strain_negative = negative_tensor<dim>(eigenvalues, projectors_2nd);
      }

    // 2D plane strain and 3D cases
    double my_lambda       = matConst.effective_lambda;
    double my_lambda_x_0_5 = matConst.effective_lambda_x_0_5;

    double degradation = 0.0;
    if constexpr (has_flag(flags, QPHUpdateFlag::stress) ||
                  has_flag(flags, QPHUpdateFlag::CCCC) ||
                  has_flag(flags, QPHUpdateFlag::strain_energy_tot))
      {
        if constexpr (has_flag(flags, QPHUpdateFlag::energy_dissipation))
          {
            pf_values.init(matConst, m_phase_field_value);
          }
        else
          pf_values.init(matConst, m_phase_field_value, true, false);

        degradation = pf_values.degradation() + matConst.residual_k;
      }

    if constexpr (has_flag(flags, QPHUpdateFlag::stress_positive) ||
                  has_flag(flags, QPHUpdateFlag::stress))
      {
        m_stress_positive = my_lambda * positive_I1 *
                              Physics::Elasticity::StandardTensors<dim>::I +
                            matConst.mu_x_2 * strain_positive;
      }

    if constexpr (has_flag(flags, QPHUpdateFlag::stress))
      {
        SymmetricTensor<2, dim> stress_negative =
          my_lambda * negative_I1 *
            Physics::Elasticity::StandardTensors<dim>::I +
          matConst.mu_x_2 * strain_negative;

        m_stress = degradation * m_stress_positive + stress_negative;
      }


    if constexpr (has_flag(flags, QPHUpdateFlag::CCCC))
      {
        SymmetricTensor<4, dim> projector_positive, projector_negative;
        positive_negative_projectors<dim>(projectors_2nd,
                                          eigenvalues,
                                          projector_positive,
                                          projector_negative);

        SymmetricTensor<4, dim> C_positive, C_negative;
        C_positive = my_lambda * heaviside_function(I_1) *
                       Physics::Elasticity::StandardTensors<dim>::IxI +
                     matConst.mu_x_2 * projector_positive;

        C_negative = my_lambda * heaviside_function(-I_1) *
                       Physics::Elasticity::StandardTensors<dim>::IxI +
                     matConst.mu_x_2 * projector_negative;

        m_mechanical_C = degradation * C_positive + C_negative;
      }

    if constexpr (has_flag(flags, QPHUpdateFlag::strain_energy_pos) ||
                  has_flag(flags, QPHUpdateFlag::strain_energy_tot))
      {
        m_strain_energy_positive =
          my_lambda_x_0_5 * positive_I1 * positive_I1 +
          matConst.mu * strain_positive * strain_positive;
      }

    if constexpr (has_flag(flags, QPHUpdateFlag::strain_energy_neg) ||
                  has_flag(flags, QPHUpdateFlag::strain_energy_tot))
      {
        m_strain_energy_negative =
          my_lambda_x_0_5 * negative_I1 * negative_I1 +
          matConst.mu * strain_negative * strain_negative;
      }

    if constexpr (has_flag(flags, QPHUpdateFlag::strain_energy_tot))
      {
        m_strain_energy_total =
          degradation * m_strain_energy_positive + m_strain_energy_negative;
      }

    if constexpr (has_flag(flags, QPHUpdateFlag::energy_dissipation))
      {
        const double phase_field_geo_value = pf_values.geometric();

        if (matConst.has_viscosity)
          {
            const double delta_pf =
              phase_field_value - phase_field_value_previous_step;

            const double viscosity_term =
              (matConst.has_viscosity) ?
                (delta_pf) * (delta_pf)*0.5 * matConst.viscosity / delta_time :
                0.0;

            m_crack_energy_dissipation =
              matConst.gc *
                ( matConst .I_over_c_alpha_l0 * phase_field_geo_value 
                 + matConst.l0_over_c_alpha * grad_phasefield * grad_phasefield)
              // the term due to viscosity regularization
              + viscosity_term;
          }
        else
          {
            m_crack_energy_dissipation =
              matConst.gc *
              ( matConst.I_over_c_alpha_l0 * phase_field_geo_value
               + matConst.l0_over_c_alpha * grad_phasefield * grad_phasefield);
          }
      }
    //(void)delta_time;
    //(void)phase_field_value_previous_step;
  }

  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
      : m_length_scale(0.0)
      , m_gc(0.0)
      , m_viscosity(0.0)
      , m_p(0.0)
      , m_a1(0.0)
      , m_a2(0.0)
      , m_a3(0.0)
      , m_history_max_positive_strain_energy(0.0)
    {}

    virtual ~PointHistory() = default;

    void
    setup_lqp(const MaterialConstants<dim> &matConst)
    {
      matConstsPtr = &matConst;

      m_history_max_positive_strain_energy = matConst.max_strain_energy;
     
      m_material.emplace(matConst);

      PFValues<0, dim> pf_values;

      constexpr QPHUpdateFlag flags = residualFlag | tangentFlag | energyFlag;
      update_field_values<flags>(
        SymmetricTensor<2, dim>(), 0.0, Tensor<1, dim>(), 0.0, 1.0, pf_values);
    }

    void
    update_field_values(const SymmetricTensor<2, dim> &strain,
                        const double                   phase_field_value,
                        const Tensor<1, dim>          &grad_phasefield,
                        const double phase_field_value_previous_step,
                        const double delta_time)
    {
      m_material->update_material_data(strain,
                                       phase_field_value,
                                       grad_phasefield,
                                       phase_field_value_previous_step,
                                       delta_time);
    }

    void
    update_history_variable()
    {
      double current_positive_strain_energy =
        m_material->get_positive_strain_energy();
      m_history_max_positive_strain_energy =
        std::fmax(m_history_max_positive_strain_energy,
                  current_positive_strain_energy);
    }

    // This is the function used to assign the history variable after remeshing
    void
    assign_history_variable(double history_variable_value)
    {
      m_history_max_positive_strain_energy = history_variable_value;
    }

    double
    get_current_positive_strain_energy() const
    {
      return m_material->get_positive_strain_energy();
    }

    const SymmetricTensor<4, dim> &
    get_mechanical_C() const
    {
      return m_material->get_mechanical_C();
    }

    const SymmetricTensor<2, dim> &
    get_cauchy_stress() const
    {
      return m_material->get_cauchy_stress();
    }

    const SymmetricTensor<2, dim> &
    get_cauchy_stress_positive() const
    {
      return m_material->get_cauchy_stress_positive();
    }

    double
    get_total_strain_energy() const
    {
      return m_material->get_total_strain_energy();
    }

    double
    get_crack_energy_dissipation() const
    {
      return m_material->get_crack_energy_dissipation();
    }

    double
    get_phase_field_value() const
    {
      return m_material->get_phase_field_value();
    }

    const Tensor<1, dim>
    get_phase_field_gradient() const
    {
      return m_material->get_phase_field_gradient();
    }

    double
    get_history_max_positive_strain_energy() const
    {
      return m_history_max_positive_strain_energy;
    }

    double
    get_length_scale() const
    {
      return m_length_scale;
    }

    double
    get_critical_energy_release_rate() const
    {
      return m_gc;
    }

    double
    get_viscosity() const
    {
      return m_viscosity;
    }

    double
    get_p() const
    {
      return m_p;
    }

    double
    get_a1() const
    {
      return m_a1;
    }

    double
    get_a2() const
    {
      return m_a2;
    }

    const MaterialConstants<dim> &
    get_material_constents() const
    {
      return *matConstsPtr;
    }

    const MaterialConstants<dim> *matConstsPtr = nullptr;
  };

  template <typename LATraits, typename Tria>
  class PhaseFieldMonolithicSolve
  {
  public:
    constexpr static int dim = Tria::dimension;

#if ENABLE_CUSTOMIZED_REPARTITION_MODE == 1
    constexpr static bool supportRepartioning = true;
#else
    constexpr static bool supportRepartioning = false;
#endif

    using BSMatrix = ::common::BlockSparseMatrixWrapper<LATraits>;
    using BVector  = ::common::BlockVectorWrapper<LATraits>;

    // variable to tell if this is class is for mpi mode
    static constexpr bool is_mpi =
      !std::is_same_v<typename LATraits::TMTag, TagSerial>;

    PhaseFieldMonolithicSolve(const Parameters::AllParameters &parameters,
                              const MPIInfo                   &mpiInfo,
                              ConditionalOStream              &logfile,
                              Tria                            &triangulation);

    virtual ~PhaseFieldMonolithicSolve() = default;
    void
    run();

  private:
    struct PerTaskData_ASM;
    struct ScratchData_ASM;

    struct PerTaskData_ASM_RHS_BFGS;
    struct ScratchData_ASM_RHS_BFGS;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    const Parameters::AllParameters &m_parameters;
    Tria                            &m_triangulation;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim>>
      m_quadrature_point_history;

    Time m_time;

    const MPIInfo                       &m_mpiInfo;
    ConditionalOStream                  &m_logfile;
    mutable TimerOutputWrapper<LATraits> m_timer;
    BlockDesc                            m_blocks_desc;

    DoFHandler<dim>                  m_dof_handler;
    FESystem<dim>                    m_fe;
    const unsigned int               m_dofs_per_cell;
    const FEValuesExtractors::Vector m_u_fe;
    const FEValuesExtractors::Scalar m_d_fe;

    static const unsigned int m_n_blocks          = 2;
    static const unsigned int m_n_components      = dim + 1;
    static const unsigned int m_first_u_component = 0;
    static const unsigned int m_d_component       = dim;

    enum
    {
      m_u_dof = 0,
      m_d_dof = 1
    };


    const QGauss<dim>     m_qf_cell;
    const QGauss<dim - 1> m_qf_face;
    const unsigned int    m_n_q_points;

    double m_vol_reference;

    AffineConstraints<double> m_constraints;


    BSMatrix m_tangent_matrix;
    BVector  m_system_rhs;
    BVector  m_solution;

    BlockVectorPool<LATraits, true>  m_vec_pool_ghosted;
    BlockVectorPool<LATraits, false> m_vec_pool;

    LASolver<LATraits> m_diag_la_solver;

    SparseDirectUMFPACK m_A_direct;

    std::map<std::string, const double>       m_extra_data;
    std::map<std::string, const unsigned int> m_bcs_id;


    std::unordered_map<unsigned int, MaterialConstants<dim>> m_material_const;

    std::vector<std::pair<double, std::vector<double>>>
      m_history_reaction_force;
    std::vector<std::pair<double, std::array<double, 3>>> m_history_energy;

    bool                m_update_dofs_for_cst;
    bcs::CstMaker<Tria> m_cst_maker;

    OutputHelper<LATraits, Tria, PointHistory<dim>> m_output;

    struct Errors
    {
      Errors()
        : m_norm(1.0)
        , m_u(1.0)
        , m_d(1.0)
      {}

      void
      reset()
      {
        m_norm = 1.0;
        m_u    = 1.0;
        m_d    = 1.0;
      }

      void
      normalize(const Errors &rhs)
      {
        if (rhs.m_norm != 0.0)
          m_norm /= rhs.m_norm;
        if (rhs.m_u != 0.0)
          m_u /= rhs.m_u;
        if (rhs.m_d != 0.0)
          m_d /= rhs.m_d;
      }

      double m_norm, m_u, m_d;
    };

    Errors m_error_residual, m_error_residual_0, m_error_residual_norm,
      m_error_update, m_error_update_0, m_error_update_norm;

    template <typename CellIter>
    bool
    setRefineFlagByCellSize(CellIter &cell, const double hlRatio);

    void
    executeRefinement(bool &isRefineFlagSet);

    void
    get_error_residual(Errors &error_residual);
    void
    get_error_update(const BVector &newton_update, Errors &error_update);

    void
    set_bcs_id();
    void
    make_grid();
    void
    make_grid_case_1();
    void
    make_grid_case_2();
    void
    make_grid_case_3();
    void
    make_grid_case_4();
    void
    make_grid_case_5();
    void
    make_grid_case_6();
    void
    make_grid_case_7();
    void
    make_grid_case_8();
    void
    make_grid_case_9();
    void
    make_grid_case_11();
    void
    make_grid_case_12();
    void
    make_grid_case_13();
    void
    make_grid_case_14();
    void
    make_grid_case_15();
    void
    make_grid_case_16();

    void
    setup_system();

    void
    determine_component_extractors();

    void
    make_constraints(const unsigned int it_nr);

    void
    assemble_system_newton(const BVector &solution_old);

    void
    assemble_system_B0(const BVector &solution_old);

    void
    assemble_system_newton_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM                                      &scratch,
      PerTaskData_ASM                                      &data) const;

    void
    assemble_system_B0_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM                                      &scratch,
      PerTaskData_ASM                                      &data) const;

    void
    assemble_system_rhs_BFGS_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_RHS_BFGS                             &scratch,
      PerTaskData_ASM_RHS_BFGS                             &data) const;

    void
    assemble_system_rhs_BFGS(const BVector &solution_old, BVector &system_rhs);

    template <bool has_body_force, bool has_surface_pressure>
    void
    assemble_system_rhs_BFGS_parallel(const BVector &solution_old,
                                      BVector       &system_rhs);

    bool
    solve_nonlinear_timestep_newton(BVector &solution_delta);

    void
    solve_nonlinear_timestep_BFGS(BVector &solution_delta);

    void
    solve_nonlinear_timestep_LBFGS(BVector &solution_delta,
                                   BVector &LBFGS_update_refine);

    double
    line_search_stepsize_strong_wolfe(const double   phi_0,
                                      const double   phi_0_prime,
                                      const BVector &BFGS_p_vector,
                                      const BVector &solution_delta,
                                      unsigned int  &num_ls);

    double
    line_search_stepsize_gradient_based(const BVector &BFGS_p_vector,
                                        const BVector &solution_delta,
                                        unsigned int  &num_ls);

    double
    line_search_zoom_strong_wolfe(double         phi_low,
                                  double         phi_low_prime,
                                  double         alpha_low,
                                  double         phi_high,
                                  double         phi_high_prime,
                                  double         alpha_high,
                                  double         phi_0,
                                  double         phi_0_prime,
                                  const BVector &BFGS_p_vector,
                                  double         c1,
                                  double         c2,
                                  unsigned int   max_iter,
                                  const BVector &solution_delta);

    double
    line_search_interpolation_cubic(const double alpha_0,
                                    const double phi_0,
                                    const double phi_0_prime,
                                    const double alpha_1,
                                    const double phi_1,
                                    const double phi_1_prime);

    std::pair<double, double>
    calculate_phi_and_phi_prime(const double   alpha,
                                const BVector &BFGS_p_vector,
                                const BVector &solution_delta);

    std::vector<double>
    solve_linear_system(BVector &newton_update);

    void
    LBFGS_B0(BVector &LBFGS_r_vector, BVector &LBFGS_q_vector);

    void
    update_history_field_step();

    void
    output_results() const;

    void
    setup_qph();

    void
    update_qph_incremental(const BVector &solution_delta,
                           const BVector &solution_old,
                           const bool     is_print);

    void
    update_qph_incremental_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_UQPH                                     &scratch,
      PerTaskData_UQPH                                     &data);

    void
    copy_local_to_global_UQPH(const PerTaskData_UQPH & /*data*/)
    {}

    BVector
    get_total_solution(const BVector &solution_delta) const;

    // Should not make this function const
    void
    read_material_data(const std::string &data_file,
                       const unsigned int total_material_regions);

    void
    read_time_data(const std::string                  &data_file,
                   std::vector<std::array<double, 3>> &time_table,
                   std::vector<std::vector<double>>   &factorList,
                   std::vector<unsigned int>          &intervals);

    void
    print_conv_header_newton();

    void
    print_conv_header_BFGS();

    void
    print_conv_header_LBFGS();

    void
    print_parameter_information();

    void
    calculate_reaction_force(unsigned int face_ID);

    void
    write_history_data();

    double
    calculate_energy_functional() const;

    std::pair<double, double>
    calculate_total_strain_energy_and_crack_energy_dissipation() const;

    bool
    local_refine_and_solution_transfer(BVector &solution_delta,
                                       BVector &LBFGS_update_refine);
    void
    repartition(BVector                              &solution_next_step,
                const typename LATraits::VectorBlock &H_vector,
                const typename LATraits::VectorBlock &H_vector_rele);
  }; // class PhaseFieldMonolithicSolve



  template <typename LATraits, typename Tria>
  template <typename CellIter>
  bool
  PhaseFieldMonolithicSolve<LATraits, Tria>::setRefineFlagByCellSize(
    CellIter    &cell,
    const double hlRatio)
  {
    bool initiation_point_refine_unfinished = false;

    const double lengthScale = m_material_data[cell->material_id()][2];

    double cellSize = 0;

    if constexpr (dim == 2)
      {
        cellSize = std::sqrt(cell->measure());
      }
    else if constexpr (dim == 3)
      {
        cellSize = std::cbrt(cell->measure());
      }
    else
      {
        AssertThrow(false,
                    ExcMessage("The dimension should be " +
                               std::to_string(dim)));
      }

    if (cellSize > lengthScale * hlRatio)
      {
        cell->set_refine_flag();
        initiation_point_refine_unfinished = true;
      }

    return initiation_point_refine_unfinished;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::executeRefinement(
    bool &isRefineFlagSet)
  {
    if constexpr (is_mpi)
      {
        // accumulate local flag over all ranks
        const unsigned int local_flag = isRefineFlagSet ? 1u : 0u;
        const unsigned int global_flag =
          Utilities::MPI::sum(local_flag, *m_mpiInfo.mpiCommPtr());
        isRefineFlagSet = (global_flag > 0u);
      }
    if (isRefineFlagSet)
      {
        m_triangulation.execute_coarsening_and_refinement();
        if constexpr (is_mpi && supportRepartioning)
          m_triangulation.repartition();
      }
  }


  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::get_error_residual(
    Errors &error_residual)
  {
    auto     vecHandle = m_vec_pool.getHandle(false);
    BVector &error_res = vecHandle.get();

    error_res.base() = m_system_rhs.base();
    m_constraints.set_zero(error_res.base());
    if constexpr (is_mpi)
      error_res.compress(VectorOperation::insert);

    error_residual.m_norm = error_res.l2_norm();
    error_residual.m_u    = error_res.block(m_u_dof).l2_norm();
    error_residual.m_d    = error_res.block(m_d_dof).l2_norm();
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::get_error_update(
    const BVector &newton_update,
    Errors        &error_update)
  {
    auto     vecHandle = m_vec_pool.getHandle(false);
    BVector &error_ud  = vecHandle.get();

    error_ud.base() = newton_update.base();
    m_constraints.set_zero(error_ud.base());
    if constexpr (is_mpi)
      error_ud.compress(VectorOperation::insert);

    error_update.m_norm = error_ud.l2_norm();
    error_update.m_u    = error_ud.block(m_u_dof).l2_norm();
    error_update.m_d    = error_ud.block(m_d_dof).l2_norm();
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::read_material_data(
    const std::string &data_file,
    const unsigned int total_material_regions)
  {
    std::ifstream myfile(data_file);

    double lame_lambda, lame_mu, length_scale, gc, viscosity, residual_k;
    // add the material tensile strength for non AT-2 models
    double tensile_strength;
    double p;
    double a2;
    double a3;
    int    material_region;
    double poisson_ratio;
    if (myfile.is_open())
      {
        m_logfile << "Reading material data file ..." << std::endl;

        while (myfile >> material_region >> lame_lambda >> lame_mu >>
               length_scale >> gc >> viscosity >> residual_k >>
               tensile_strength >> p >> a2 >> a3)
          {
            m_material_const.emplace(
              material_region,
              MaterialConstants<dim>(m_parameters.m_pf_model,
                                     lame_lambda,
                                     lame_mu,
                                     length_scale,
                                     gc,
                                     tensile_strength,
                                     viscosity,
                                     residual_k,
                                     p,
                                     a2,
                                     a3,
                                     m_parameters.m_plane_stress));

            m_logfile << "\tRegion " << material_region << " : " << std::endl;
            m_material_const.at(material_region).print(m_logfile);
          }

        if (m_material_const.size() != total_material_regions)
          {
            m_logfile << "Material data file has " << m_material_const.size()
                      << " rows. However, "
                      << "the mesh has " << total_material_regions
                      << " material regions." << std::endl;
            Assert(m_material_const.size() == total_material_regions,
                   ExcDimensionMismatch(m_material_const.size(),
                                        total_material_regions));
          }
        myfile.close();
      }
    else
      {
        m_logfile << "Material data file : " << data_file << " not exist!"
                  << std::endl;
        Assert(false, ExcMessage("Failed to read material data file"));
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::read_time_data(
    const std::string                  &data_file,
    std::vector<std::array<double, 3>> &time_table,
    std::vector<std::vector<double>>   &factorList,
    std::vector<unsigned int>          &intervals)
  {
    std::ifstream myfile(data_file);

    double t_0, t_1, delta_t;

    unsigned int interval;

    const unsigned int n_factors = m_parameters.m_n_factors;


    if (myfile.is_open())
      {
        m_logfile << "Reading time data file ..." << std::endl;

        std::string  line;
        unsigned int line_id = 0;

        while (std::getline(myfile, line))
          {
            ++line_id;

            // remove comments after '#'
            const std::size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos)
              {
                line.erase(comment_pos);
              }

            // Skip empty lines
            if (line.find_first_not_of(" \t\r\n") == std::string::npos)
              {
                continue;
              }

            std::istringstream iss(line);

            if (iss >> t_0 >> t_1 >> delta_t)
              {
                time_table.push_back({{t_0, t_1, delta_t}});
              }
            else
              {
                AssertThrow(false,
                            ExcMessage(
                              "Invalid time data line " +
                              std::to_string(line_id) +
                              ": failed to read t_0, t_1, and delta_t."));
              }

            std::vector<double> t_magnitude_list(n_factors);

            for (unsigned int i = 0; i < n_factors; ++i)
              {
                if (!(iss >> t_magnitude_list[i]))
                  {
                    AssertThrow(false,
                                ExcMessage(
                                  "Invalid time data line " +
                                  std::to_string(line_id) +
                                  ": failed to read magnitude factor " +
                                  std::to_string(i) + "."));
                  }
              }
            factorList.push_back(t_magnitude_list);

            if (iss >> interval)
              {
                // If the given interval is equal to 0, the output for this time
                // step will be turned off.
                if (interval == 0 ||
                    interval > std::round((t_1 - t_0) / delta_t))
                  {
                    m_logfile << intervals.size()
                              << "-th time slot will NOT output any .vtu files."
                              << std::endl;
                    interval = std::numeric_limits<unsigned int>::max();
                  }
                intervals.push_back(interval);
              }
            else
              {
                AssertThrow(false,
                            ExcMessage("Invalid time data line " +
                                       std::to_string(line_id) +
                                       ": failed to read output interval."));
              }

            // Check whether there are extra unexpected values
            std::string extra_token;
            if (iss >> extra_token)
              {
                AssertThrow(false,
                            ExcMessage("Invalid time data line " +
                                       std::to_string(line_id) +
                                       ": too many entries. Expected " +
                                       std::to_string(n_factors) + " values."));
              }
          }

        Assert(
          std::fabs(t_1 - m_parameters.m_end_time) < 1.0e-9,
          ExcMessage(
            "End time in time table is inconsistent with input data in parameters.prm"));

        Assert(time_table.size() > 0, ExcMessage("Time data file is empty."));
        myfile.close();
      }
    else
      {
        m_logfile << "Time data file : " << data_file << " not exist!"
                  << std::endl;
        Assert(false, ExcMessage("Failed to read time data file"));
      }

    m_logfile << "\tsteps[output interval], \t(t0, t1], ∆t \t| magnitudes"
              << std::endl;

    for (unsigned int i = 0; i < time_table.size(); ++i)
      {
        const auto &time_group = time_table[i];

        const unsigned int n_steps = static_cast<unsigned int>(
          std::round((time_group[1] - time_group[0]) / time_group[2]));

        m_logfile << "\t" << n_steps << " steps[" << intervals[i] << "], \t("
                  << time_group[0] << ", " << time_group[1] << "], "
                  << time_group[2] << " \t| ";

        for (const double factor : factorList[i])
          {
            m_logfile << factor << ", ";
          }

        m_logfile << std::endl;
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::setup_qph()
  {
    m_logfile << "\t\tSetting up quadrature point data (" << m_n_q_points
              << " points per cell)" << std::endl;

    m_quadrature_point_history.clear();
    for (auto const &cell : m_triangulation.active_cell_iterators())
      {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
          if (!cell->is_locally_owned())
            continue;
        m_quadrature_point_history.initialize(cell, m_n_q_points);
      }

    unsigned int material_id;
    double       lame_lambda      = 0.0;
    double       lame_mu          = 0.0;
    double       length_scale     = 0.0;
    double       gc               = 0.0;
    double       viscosity        = 0.0;
    double       residual_k       = 0.0;
    double       tensile_strength = 0.0;
    double       p                = 0.0;
    double       a2               = 0.0;
    double       a3               = 0.0;

    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
          if (!cell->is_locally_owned())
            continue;
        material_id = cell->material_id();
        const MaterialConstants<dim> &matConst =
          m_material_const.at(material_id);

        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(lame_lambda,
                                   lame_mu,
                                   length_scale,
                                   gc,
                                   viscosity,
                                   residual_k,
                                   tensile_strength,
                                   p,
                                   a2,
                                   a3,
                                   m_parameters.m_phasefield_name,
                                   m_parameters.m_plane_stress);
      }
  }

  template <typename LATraits, typename Tria>
  ::common::BlockVectorWrapper<LATraits>
  PhaseFieldMonolithicSolve<LATraits, Tria>::get_total_solution(
    const BVector &solution_delta) const
  {
    auto     vecHandle      = m_vec_pool_ghosted.getHandle(false);
    BVector &solution_total = vecHandle.get();

    solution_total.base() = m_solution.base();
    solution_total += solution_delta;
    solution_total.updateRelevance();
    return solution_total;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::update_qph_incremental(
    const BVector &solution_delta,
    const BVector &solution_old,
    const bool     is_print)
  {
    const std::string sectionName = "Update QPH data";
    m_timer.enter_subsection(sectionName);
    if (is_print && m_parameters.m_output_iteration_history)
      m_logfile << " UQPH " << std::flush;

    auto solution_total_handle = m_vec_pool_ghosted.getHandle(false);

    BVector &solution_total = solution_total_handle.get();
    solution_total.base()   = solution_old.base();
    solution_total.base() += solution_delta.base();
    solution_total.updateRelevance();

    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH  per_task_data_UQPH;
    ScratchData_UQPH  scratch_data_UQPH(m_fe,
                                       m_qf_cell,
                                       uf_UQPH,
                                       solution_total,
                                       solution_old,
                                       m_time.get_delta_t());

    if constexpr (!is_mpi)
      {
        // non-mpi mode
        auto worker =
          [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
                 ScratchData_UQPH                                     &scratch,
                 PerTaskData_UQPH                                     &data) {
            this->update_qph_incremental_one_cell(cell, scratch, data);
          };

        auto copier = [this](const PerTaskData_UQPH &data) {
          this->copy_local_to_global_UQPH(data);
        };

        WorkStream::run(m_dof_handler.begin_active(),
                        m_dof_handler.end(),
                        worker,
                        copier,
                        scratch_data_UQPH,
                        per_task_data_UQPH);
      }
    else
      {
        // mpi mode
        for (const auto &cell : m_dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              update_qph_incremental_one_cell(cell,
                                              scratch_data_UQPH,
                                              per_task_data_UQPH);
              copy_local_to_global_UQPH(per_task_data_UQPH);
            }
      }

    m_timer.leave_subsection(sectionName);
  }

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::PerTaskData_UQPH
  {
    void
    reset()
    {}
  };

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::ScratchData_UQPH
  {
    const BVector &m_solution_UQPH;

    std::vector<SymmetricTensor<2, dim>> m_solution_symm_grads_u_cell;
    std::vector<double>                  m_solution_values_phasefield_cell;
    std::vector<Tensor<1, dim>>          m_solution_grad_phasefield_cell;

    FEValues<dim> m_fe_values;

    const BVector      &m_solution_previous_step;
    std::vector<double> m_phasefield_previous_step_cell;

    const double m_delta_time;

    ScratchData_UQPH(const FiniteElement<dim> &fe_cell,
                     const QGauss<dim>        &qf_cell,
                     const UpdateFlags         uf_cell,
                     const BVector            &solution_total,
                     const BVector            &solution_old,
                     const double              delta_time)
      : m_solution_UQPH(solution_total)
      , m_solution_symm_grads_u_cell(qf_cell.size())
      , m_solution_values_phasefield_cell(qf_cell.size())
      , m_solution_grad_phasefield_cell(qf_cell.size())
      , m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_solution_previous_step(solution_old)
      , m_phasefield_previous_step_cell(qf_cell.size())
      , m_delta_time(delta_time)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : m_solution_UQPH(rhs.m_solution_UQPH)
      , m_solution_symm_grads_u_cell(rhs.m_solution_symm_grads_u_cell)
      , m_solution_values_phasefield_cell(rhs.m_solution_values_phasefield_cell)
      , m_solution_grad_phasefield_cell(rhs.m_solution_grad_phasefield_cell)
      , m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_solution_previous_step(rhs.m_solution_previous_step)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
      , m_delta_time(rhs.m_delta_time)
    {}

    void
    reset()
    {
      const std::size_t n_q_points = m_solution_symm_grads_u_cell.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          m_solution_symm_grads_u_cell[q]      = 0.0;
          m_solution_values_phasefield_cell[q] = 0.0;
          m_solution_grad_phasefield_cell[q]   = 0.0;
          m_phasefield_previous_step_cell[q]   = 0.0;
        }
    }
  };

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::update_qph_incremental_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_UQPH                                     &scratch,
    PerTaskData_UQPH & /*data*/)
  {
    scratch.reset();

    scratch.m_fe_values.reinit(cell);

    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const FEValuesExtractors::Vector displacement(0);

    const auto &solution_relevance = scratch.m_solution_UQPH.relevance();
    const auto &solution_previous_step_relevance =
      scratch.m_solution_previous_step.relevance();

    scratch.m_fe_values[m_u_fe].get_function_symmetric_gradients(
      solution_relevance, scratch.m_solution_symm_grads_u_cell);
    scratch.m_fe_values[m_d_fe].get_function_values(
      solution_relevance, scratch.m_solution_values_phasefield_cell);
    scratch.m_fe_values[m_d_fe].get_function_gradients(
      solution_relevance, scratch.m_solution_grad_phasefield_cell);

    scratch.m_fe_values[m_d_fe].get_function_values(
      solution_previous_step_relevance,
      scratch.m_phasefield_previous_step_cell);

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      lqph[q_point]->update_field_values(
        scratch.m_solution_symm_grads_u_cell[q_point],
        scratch.m_solution_values_phasefield_cell[q_point],
        scratch.m_solution_grad_phasefield_cell[q_point],
        scratch.m_phasefield_previous_step_cell[q_point],
        scratch.m_delta_time);
  }

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::PerTaskData_ASM
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void
    reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::PerTaskData_ASM_RHS_BFGS
  {
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM_RHS_BFGS(const unsigned int dofs_per_cell)
      : m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void
    reset()
    {
      m_cell_rhs = 0.0;
    }
  };

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::ScratchData_ASM
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<double>>
      m_Nx_phasefield; // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>
      m_grad_Nx_phasefield; // gradient of shape function values for phase field

    std::vector<std::vector<Tensor<1, dim>>>
      m_Nx_disp; // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>
      m_grad_Nx_disp; // gradient of shape function values for displacement
    std::vector<std::vector<SymmetricTensor<2, dim>>>
      m_symm_grad_Nx_disp; // symmetric gradient of shape function values for
                           // displacement

    const BVector      &m_solution_previous_step;
    std::vector<double> m_phasefield_previous_step_cell;

    ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim>        &qf_cell,
                    const UpdateFlags         uf_cell,
                    const QGauss<dim - 1>    &qf_face,
                    const UpdateFlags         uf_face,
                    const BVector            &solution_old)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_phasefield(qf_cell.size(),
                        std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_phasefield(qf_cell.size(),
                             std::vector<Tensor<1, dim>>(
                               fe_cell.n_dofs_per_cell()))
      , m_Nx_disp(qf_cell.size(),
                  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(
                              fe_cell.n_dofs_per_cell()))
      , m_solution_previous_step(solution_old)
      , m_phasefield_previous_step_cell(qf_cell.size())
    {}

    ScratchData_ASM(const ScratchData_ASM &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
                         rhs.m_fe_face_values.get_quadrature(),
                         rhs.m_fe_face_values.get_update_flags())
      , m_Nx_phasefield(rhs.m_Nx_phasefield)
      , m_grad_Nx_phasefield(rhs.m_grad_Nx_phasefield)
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
      , m_solution_previous_step(rhs.m_solution_previous_step)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
    {}

    void
    reset()
    {
      const std::size_t n_q_points      = m_Nx_phasefield.size();
      const std::size_t n_dofs_per_cell = m_Nx_phasefield[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_phasefield[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_grad_Nx_phasefield[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_phasefield_previous_step_cell[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_phasefield[q_point][k]      = 0.0;
              m_grad_Nx_phasefield[q_point][k] = 0.0;
              m_Nx_disp[q_point][k]            = 0.0;
              m_grad_Nx_disp[q_point][k]       = 0.0;
              m_symm_grad_Nx_disp[q_point][k]  = 0.0;
            }
        }
    }
  };


  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::ScratchData_ASM_RHS_BFGS
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<double>>
      m_Nx_phasefield; // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>
      m_grad_Nx_phasefield; // gradient of shape function values for phase field

    std::vector<std::vector<Tensor<1, dim>>>
      m_Nx_disp; // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>
      m_grad_Nx_disp; // gradient of shape function values for displacement
    std::vector<std::vector<SymmetricTensor<2, dim>>>
      m_symm_grad_Nx_disp; // symmetric gradient of shape function values for
                           // displacement

    const BVector      &m_solution_previous_step;
    std::vector<double> m_phasefield_previous_step_cell;

    ScratchData_ASM_RHS_BFGS(const FiniteElement<dim> &fe_cell,
                             const QGauss<dim>        &qf_cell,
                             const UpdateFlags         uf_cell,
                             const QGauss<dim - 1>    &qf_face,
                             const UpdateFlags         uf_face,
                             const BVector            &solution_old)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_phasefield(qf_cell.size(),
                        std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_phasefield(qf_cell.size(),
                             std::vector<Tensor<1, dim>>(
                               fe_cell.n_dofs_per_cell()))
      , m_Nx_disp(qf_cell.size(),
                  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(
                              fe_cell.n_dofs_per_cell()))
      , m_solution_previous_step(solution_old)
      , m_phasefield_previous_step_cell(qf_cell.size())
    {}

    ScratchData_ASM_RHS_BFGS(const ScratchData_ASM_RHS_BFGS &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
                         rhs.m_fe_face_values.get_quadrature(),
                         rhs.m_fe_face_values.get_update_flags())
      , m_Nx_phasefield(rhs.m_Nx_phasefield)
      , m_grad_Nx_phasefield(rhs.m_grad_Nx_phasefield)
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
      , m_solution_previous_step(rhs.m_solution_previous_step)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
    {}

    void
    reset()
    {
      const std::size_t n_q_points      = m_Nx_phasefield.size();
      const std::size_t n_dofs_per_cell = m_Nx_phasefield[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_phasefield[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_grad_Nx_phasefield[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_phasefield_previous_step_cell[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_phasefield[q_point][k]      = 0.0;
              m_grad_Nx_phasefield[q_point][k] = 0.0;
              m_Nx_disp[q_point][k]            = 0.0;
              m_grad_Nx_disp[q_point][k]       = 0.0;
              m_symm_grad_Nx_disp[q_point][k]  = 0.0;
            }
        }
    }
  };

  // constructor has no return type
  template <typename LATraits, typename Tria>
  PhaseFieldMonolithicSolve<LATraits, Tria>::PhaseFieldMonolithicSolve(
    const Parameters::AllParameters &parameters,
    const MPIInfo                   &mpiInfo,
    ConditionalOStream              &logfile,
    Tria                            &tria)
    : m_parameters(parameters)
    , m_triangulation(tria)
    , m_time(m_parameters.m_end_time, m_parameters.m_output_step_0)
    , m_mpiInfo(mpiInfo)
    , m_logfile(logfile)
    , m_timer(m_logfile,
              m_mpiInfo,
              TimerOutput::summary,
              TimerOutput::wall_times)
    , m_blocks_desc(m_mpiInfo, {{dim, "displacement"}, {1, "phase-field"}})
    , m_dof_handler(m_triangulation)
    , m_fe(FE_Q<dim>(m_parameters.m_poly_degree),
           dim, // displacement
           FE_Q<dim>(m_parameters.m_poly_degree),
           1) // phasefield
    , m_dofs_per_cell(m_fe.n_dofs_per_cell())
    , m_u_fe(m_first_u_component)
    , m_d_fe(m_d_component)
    , m_qf_cell(m_parameters.m_quad_order)
    , m_qf_face(m_parameters.m_quad_order)
    , m_n_q_points(m_qf_cell.size())
    , m_vol_reference(0.0)
    , m_tangent_matrix(m_mpiInfo,
                       m_blocks_desc,
                       [](unsigned int, unsigned int) {
                         return DoFTools::always;
                       })
    , m_system_rhs(m_mpiInfo, m_blocks_desc, /*relevance=*/false)
    , m_solution(m_mpiInfo, m_blocks_desc, /*relevance=*/true)
    , m_vec_pool_ghosted(m_mpiInfo, m_blocks_desc)
    , m_vec_pool(m_mpiInfo, m_blocks_desc)
    , m_diag_la_solver(m_parameters.m_type_linear_solver,
                       m_parameters.m_prec_type_linear_solver,
                       m_blocks_desc,
                       m_mpiInfo)
    , m_update_dofs_for_cst(false)
    , m_cst_maker(m_mpiInfo)
    , m_output(m_mpiInfo,
               m_triangulation,
               m_dof_handler,
               m_qf_cell,
               m_parameters.m_scenario,
               m_parameters.m_mpi_type)
  {}


  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::set_bcs_id()
  {
    if (m_parameters.m_scenario == 1)
      {
        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                if (std::fabs(face->center()[1] + 0.5) < 1.0e-9)
                  face->set_boundary_id(0);
                else if (std::fabs(face->center()[1] - 0.5) < 1.0e-9)
                  face->set_boundary_id(1);
                else
                  face->set_boundary_id(2);
              }
          }
      }
    else if (m_parameters.m_scenario == 2)
      {
        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                if (std::fabs(face->center()[1] + 0.5) < 1.0e-9)
                  face->set_boundary_id(0);
                else if (std::fabs(face->center()[1] - 0.5) < 1.0e-9)
                  face->set_boundary_id(1);
                else if ((std::fabs(face->center()[0] - 0.0) < 1.0e-9) ||
                         (std::fabs(face->center()[0] - 1.0) < 1.0e-9))
                  face->set_boundary_id(2);
                else
                  face->set_boundary_id(3);
              }
          }
      }
    else if (m_parameters.m_scenario == 3)
      {
        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                if (std::fabs(face->center()[1] - 0.0) < 1.0e-9)
                  face->set_boundary_id(0);
                else if (std::fabs(face->center()[1] - 1.0) < 1.0e-9)
                  face->set_boundary_id(1);
                else
                  face->set_boundary_id(2);
              }
          }
      }
    else if (m_parameters.m_scenario == 4)
      {
        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                if (std::fabs(face->center()[1] - 0.0) < 1.0e-9)
                  face->set_boundary_id(0);
                else if (std::fabs(face->center()[1] - 1.0) < 1.0e-9)
                  face->set_boundary_id(1);
                else if ((std::fabs(face->center()[0] - 0.0) < 1.0e-9) ||
                         (std::fabs(face->center()[0] - 1.0) < 1.0e-9))
                  face->set_boundary_id(2);
                else
                  face->set_boundary_id(3);
              }
          }
      }
    else if (m_parameters.m_scenario == 5)
      {
        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                if (std::fabs(face->center()[1] - 0.0) < 1.0e-9)
                  face->set_boundary_id(0);
                else if (std::fabs(face->center()[1] - 2.0) < 1.0e-9)
                  face->set_boundary_id(1);
                else
                  face->set_boundary_id(2);
              }
          }
      }
    else if (m_parameters.m_scenario == 6)
      {
        if constexpr (dim == 3)
          {
            for (const auto &face : m_triangulation.active_face_iterators())
              {
                if (face->at_boundary())
                  {
                    if (std::fabs(face->center()[0] - 0.0) < 1.0e-9)
                      face->set_boundary_id(0);
                    else if (std::fabs(face->center()[1] - 0.0) < 1.0e-9)
                      face->set_boundary_id(1);
                    else if (std::fabs(face->center()[2] - 0.0) < 1.0e-9)
                      face->set_boundary_id(2);
                    else if (std::fabs(face->center()[2] - 1.0) < 1.0e-9)
                      face->set_boundary_id(3);
                    else
                      face->set_boundary_id(4);
                  }
              }
          }
      }
    else if (m_parameters.m_scenario == 7)
      {
        if constexpr (dim == 3)
          {
            for (const auto &face : m_triangulation.active_face_iterators())
              {
                if (face->at_boundary())
                  {
                    if (std::fabs(face->center()[0] - 0.0) < 1.0e-9)
                      face->set_boundary_id(0);
                    else if (std::fabs(face->center()[1] - 0.0) < 1.0e-9)
                      face->set_boundary_id(1);
                    else if (std::fabs(face->center()[2] - 0.0) < 1.0e-9)
                      face->set_boundary_id(2);
                    else if (std::fabs(face->center()[2] - 1.0) < 1.0e-9)
                      face->set_boundary_id(3);
                    else
                      face->set_boundary_id(4);
                  }
              }
          }
      }
    else if (m_parameters.m_scenario == 8)
      {
        if constexpr (dim == 3)
          {
            for (const auto &face : m_triangulation.active_face_iterators())
              {
                if (face->at_boundary())
                  {
                    if (std::fabs(face->center()[0] - 0.0) < 1.0e-9)
                      face->set_boundary_id(0);
                    else if (std::fabs(face->center()[1] - 0.0) < 1.0e-9)
                      face->set_boundary_id(1);
                    else if (std::fabs(face->center()[2] - 0.0) < 1.0e-9)
                      face->set_boundary_id(2);
                    else if (std::fabs(face->center()[2] - 1.0) < 1.0e-9)
                      face->set_boundary_id(3);
                    else
                      face->set_boundary_id(4);
                  }
              }
          }
      }
    else if (m_parameters.m_scenario == 9)
      {
        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                if (std::fabs(face->center()[1] - 0.0) < 1.0e-9)
                  face->set_boundary_id(0);
                else
                  face->set_boundary_id(1);
              }
          }
      }
    else if (m_parameters.m_scenario == 11)
      {
        double const length = m_extra_data.at("length");

        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                if (std::fabs(face->center()[0] - length) < 1.0e-6)
                  face->set_boundary_id(0);
                else if (std::fabs(face->center()[0] - 0.0) < 1.0e-6)
                  face->set_boundary_id(1);
                else
                  face->set_boundary_id(2);
              }
          }
      }
    else if (m_parameters.m_scenario == 12)
      {
        double const length = 200.0;
        double const width  = 1.0;

        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                if ((std::fabs(face->center()[0] - 0.0) < 1.0e-9))
                  face->set_boundary_id(0);
                else if ((std::fabs(face->center()[0] - length) < 1.0e-9))
                  face->set_boundary_id(1);
                else if ((std::fabs(face->center()[1] - 0.0) < 1.0e-9))
                  face->set_boundary_id(2);
                else if ((std::fabs(face->center()[1] - width) < 1.0e-9))
                  face->set_boundary_id(3);
                else
                  face->set_boundary_id(4);
              }
          }
      }
    else if (m_parameters.m_scenario == 13)
      {
        AssertThrow(!m_extra_data.empty(),
                    ExcMessage("The geo data has not been read in Case 13."));

        if (m_bcs_id.empty())
          {
            m_bcs_id.emplace("top", 999);
            m_bcs_id.emplace("btm", 900);
          }

        double const H = m_extra_data.at("H");
        if constexpr (dim == 3)
          {
            for (const auto &face : m_triangulation.active_face_iterators())
              {
                if (face->at_boundary())
                  {
                    const double z = face->center()[2];

                    // bottom sureface
                    if ((std::fabs(z) < 1.0e-9))
                      face->set_boundary_id(m_bcs_id.at("btm"));
                    // top surface
                    else if ((std::fabs(z - H) < 1.0e-9))
                      face->set_boundary_id(m_bcs_id.at("top"));
                    // other surfaces
                    else
                      face->set_boundary_id(4);
                  }
              }
          }
      }
    else if (m_parameters.m_scenario == 14)
      {
        AssertThrow(!m_extra_data.empty(),
                    ExcMessage("The geo data has not been read in Case 13."));

        const double L  = m_extra_data.at("L");
        const double H  = m_extra_data.at("H");
        const double H1 = m_extra_data.at("H1");
        const double H2 = m_extra_data.at("H2");


        if (m_bcs_id.empty())
          {
            m_bcs_id.emplace("H1", 101);
            m_bcs_id.emplace("H2", 201);


            m_bcs_id.emplace("top", 999);
            m_bcs_id.emplace("btm", 900);
          }

        if constexpr (dim == 2 || dim == 3)
          {
            for (const auto &face : m_triangulation.active_face_iterators())
              {
                if (face->at_boundary())
                  {
                    const Point<dim> center = face->center();
                    const double     x      = center[0];
                    double           h      = center[1];


                    if constexpr (dim == 3)
                      {
                        h = center[2];
                      }



                    if (h > H1 && h < H)
                      {
                        if (std::fabs(x) < 1.0e-9)
                          face->set_boundary_id(m_bcs_id.at("H1"));
                      }
                    else if (h > 0 && h < H2)
                      {
                        if (std::fabs(x - L) < 1.0e-9)
                          face->set_boundary_id(m_bcs_id.at("H2"));
                      }

                    // top surface
                    else if ((std::fabs(h - H) < 1.0e-9))
                      face->set_boundary_id(m_bcs_id.at("top"));
                    // bottom sureface
                    else if ((std::fabs(h) < 1.0e-9))
                      face->set_boundary_id(m_bcs_id.at("btm"));
                    // other surfaces
                    else
                      face->set_boundary_id(4);
                  }
              }
          }
        else
          {
            AssertThrow(false,
                        ExcMessage("Dimension Error: it should be 2 or 3!"));
          }
      }
    else if (m_parameters.m_scenario == 15)
      {
        if (m_bcs_id.empty())
          {
            m_bcs_id.emplace("z_0", 900);
            m_bcs_id.emplace("other", 100);
          }

        if constexpr (dim == 3)
          for (const auto &face : m_triangulation.active_face_iterators())
            {
              if (face->at_boundary())
                {
                  if (std::fabs(face->center()[2]) < 1.0e-6)
                    face->set_boundary_id(m_bcs_id.at("z_0"));
                  else
                    face->set_boundary_id(m_bcs_id.at("other"));
                }
            }
        else
          AssertThrow(false, ExcMessage("Dimension Error: it should be 3!"));
      }
    else if (m_parameters.m_scenario == 16)
      {
        AssertThrow(!m_extra_data.empty(),
                    ExcMessage("The geo data has not been read in Case 13."));

        const double L  = m_extra_data.at("L");
        const double H  = m_extra_data.at("H");
        const double H1 = m_extra_data.at("H1") + m_extra_data.at("C1");
        const double H2 = m_extra_data.at("H2");


        if (m_bcs_id.empty())
          {
            m_bcs_id.emplace("H1", 101);
            m_bcs_id.emplace("H2", 201);


            m_bcs_id.emplace("top", 999);
            m_bcs_id.emplace("btm", 900);
          }



        for (const auto &face : m_triangulation.active_face_iterators())
          {
            if (face->at_boundary())
              {
                const Point<dim> center = face->center();
                const double     x      = center[0];
                const double     h      = center[1];


                if (h > H1 && h < H)
                  {
                    if (std::fabs(x) < 1.0e-9)
                      face->set_boundary_id(m_bcs_id.at("H1"));
                  }
                else if (h > 0 && h < H2)
                  {
                    if (std::fabs(x - L) < 1.0e-9)
                      face->set_boundary_id(m_bcs_id.at("H2"));
                  }

                // top surface
                else if ((std::fabs(h - H) < 1.0e-9))
                  {
                    face->set_boundary_id(m_bcs_id.at("top"));
                  }
                // bottom sureface
                else if ((std::fabs(h) < 1.0e-9))
                  {
                    face->set_boundary_id(m_bcs_id.at("btm"));
                  }
                // other surfaces
                else
                  face->set_boundary_id(4);
              }
          }
      }
    else
      {
        Assert(false, ExcMessage("The scenario has not been implemented!"));
      }
  }



  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid()
  {
    if (m_parameters.m_scenario == 1)
      make_grid_case_1();
    else if (m_parameters.m_scenario == 2)
      make_grid_case_2();
    else if (m_parameters.m_scenario == 3)
      make_grid_case_3();
    else if (m_parameters.m_scenario == 4)
      make_grid_case_4();
    else if (m_parameters.m_scenario == 5)
      make_grid_case_5();
    else if (m_parameters.m_scenario == 6)
      make_grid_case_6();
    else if (m_parameters.m_scenario == 7)
      make_grid_case_7();
    else if (m_parameters.m_scenario == 8)
      make_grid_case_8();
    else if (m_parameters.m_scenario == 9)
      make_grid_case_9();
    else if (m_parameters.m_scenario == 11)
      make_grid_case_11();
    else if (m_parameters.m_scenario == 12)
      make_grid_case_12();
    else if (m_parameters.m_scenario == 13)
      make_grid_case_13();
    else if (m_parameters.m_scenario == 14)
      make_grid_case_14();
    else if (m_parameters.m_scenario == 15)
      make_grid_case_15();
    else if (m_parameters.m_scenario == 16)
      make_grid_case_16();
    else
      {
        Assert(false, ExcMessage("The scenario has not been implemented!"));
      }

    unsigned int nCells    = m_triangulation.n_active_cells();
    unsigned int nVertices = m_triangulation.n_used_vertices();

    if constexpr (is_mpi)
      {
        nCells = m_triangulation.n_global_active_cells();

        nVertices = Utilities::MPI::sum(nVertices, *m_mpiInfo.mpiCommPtr());
      }

    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\tNumber of active cells: " << nCells
              << "\n\t\t\tNumber of used vertices: " << nVertices << std::endl;

    if (m_parameters.m_output_refined_ori)
      m_output.output(m_parameters.oriDir, "original_mesh", &m_triangulation);

    m_vol_reference = GridTools::volume(m_triangulation);
    m_logfile << "\t\tGrid:\n\t\t\tReference volume: " << m_vol_reference
              << std::endl;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_1()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim == 2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(m_parameters.m_mesh_file_name);
    gridin.read_msh(f);

    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);



    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
          {
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }

                if (std::fabs(cell->center()[1]) < 0.01 &&
                    cell->center()[0] > 0.495)
                  {
                    setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            m_triangulation.execute_coarsening_and_refinement();
          }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (std::fabs(cell->center()[1] - 0.0) < 0.05 &&
                    std::fabs(cell->center()[0] - 0.5) < 0.05)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }


  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_2()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim == 2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(m_parameters.m_mesh_file_name);
    gridin.read_msh(f);

    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);


    m_triangulation.refine_global(m_parameters.m_global_refine_times);
    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
          {
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if ((cell->center()[0] > 0.45) && (cell->center()[1] < 0.05))
                  {
                    setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            m_triangulation.execute_coarsening_and_refinement();
          }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (std::fabs(cell->center()[0] - 0.5) < 0.025 &&
                    cell->center()[1] < 0.0 && cell->center()[1] > -0.025)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_3()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim == 2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(m_parameters.m_mesh_file_name);
    gridin.read_msh(f);

    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    m_triangulation.refine_global(m_parameters.m_global_refine_times);
    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
          {
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if ((std::fabs(cell->center()[1] - 0.5) < 0.025) &&
                    (cell->center()[0] > 0.475))
                  {
                    setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            m_triangulation.execute_coarsening_and_refinement();
          }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (std::fabs(cell->center()[0] - 0.5) < 0.025 &&
                    std::fabs(cell->center()[1] - 0.5) < 0.025)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_4()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim == 2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(m_parameters.m_mesh_file_name);
    gridin.read_msh(f);
    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    m_triangulation.refine_global(m_parameters.m_global_refine_times);
    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
          {
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if ((cell->center()[0] > 0.475) && (cell->center()[1] < 0.525))
                  {
                    setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            m_triangulation.execute_coarsening_and_refinement();
          }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (std::fabs(cell->center()[0] - 0.5) < 0.025 &&
                    cell->center()[1] < 0.5 && cell->center()[1] > 0.475)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_5()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tThree-point bending (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim == 2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(m_parameters.m_mesh_file_name);
    gridin.read_msh(f);
    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);


    m_triangulation.refine_global(m_parameters.m_global_refine_times);
    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        for (const auto &cell : m_triangulation.active_cell_iterators())
          {
            if constexpr (is_mpi)
              {
                if (!cell->is_locally_owned())
                  continue;
              }
            if (std::fabs(cell->center()[0] - 4.0) < 0.075 &&
                cell->center()[1] < 1.6)
              {
                cell->set_refine_flag();
              }
          }
        m_triangulation.execute_coarsening_and_refinement();

        for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
          {
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (std::fabs(cell->center()[0] - 4.0) < 0.05 &&
                    cell->center()[1] < 1.6)
                  {
                    setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            m_triangulation.execute_coarsening_and_refinement();
          }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (std::fabs(cell->center()[0] - 4.0) < 0.075 &&
                    std::fabs(cell->center()[1] - 0.4) < 0.075)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_6()
  {
    AssertThrow(dim == 3, ExcMessage("The dimension has to be 3D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSphere inclusion (3D structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    Triangulation<dim> tria_inner;
    GridGenerator::hyper_ball(tria_inner, Point<dim>(), 0.5);

    Triangulation<dim> tria_outer;
    GridGenerator::hyper_shell(
      tria_outer, Point<dim>(), 0.5, std::sqrt(dim), 2 * dim);

    Triangulation<dim> tmp_triangulation;

    GridGenerator::merge_triangulations(tria_inner,
                                        tria_outer,
                                        tmp_triangulation);

    tmp_triangulation.reset_all_manifolds();
    tmp_triangulation.set_all_manifold_ids(0);

    for (const auto &cell : tmp_triangulation.cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
              {
                if (std::fabs(face->vertex(v).norm_square() - 0.25) > 1e-12)
                  face_at_sphere_boundary = false;
              }
            if (face_at_sphere_boundary)
              face->set_all_manifold_ids(1);
          }
        if (cell->center().norm_square() < 0.25)
          cell->set_material_id(1);
        else
          cell->set_material_id(0);
      }

    tmp_triangulation.set_manifold(1, SphericalManifold<dim>());

    TransfiniteInterpolationManifold<dim> transfinite_manifold;
    transfinite_manifold.initialize(tmp_triangulation);
    tmp_triangulation.set_manifold(0, transfinite_manifold);

    tmp_triangulation.refine_global(m_parameters.m_global_refine_times);

    std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;

    for (const auto &cell : tmp_triangulation.active_cell_iterators())
      {
        if (cell->center()[0] < 0.0 || cell->center()[1] < 0.0 ||
            cell->center()[2] < 0.0)
          {
            cells_to_remove.insert(cell);
          }
      }

    GridGenerator::create_triangulation_with_removed_cells(tmp_triangulation,
                                                           cells_to_remove,
                                                           m_triangulation);
    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (cell->center()[2] > 0.525 && cell->center()[2] < 0.575 &&
                    cell->center()[0] < 0.05 && cell->center()[1] < 0.05)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_7()
  {
    AssertThrow(dim == 3, ExcMessage("The dimension has to be 3D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSphere inclusion (3D structured version 2)"
              << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    Triangulation<dim> tria_inner;
    GridGenerator::hyper_ball(tria_inner, Point<dim>(), 0.49);

    Triangulation<dim> tria_outer;
    GridGenerator::hyper_shell(
      tria_outer, Point<dim>(), 0.49, std::sqrt(dim) * 0.5, 2 * dim);

    Triangulation<dim> cube1;
    GridGenerator::hyper_rectangle(cube1,
                                   Point<dim>(0, 0, 0.5),
                                   Point<dim>(1, 1, 1.5));
    Triangulation<dim> cube2;
    GridGenerator::hyper_rectangle(cube2,
                                   Point<dim>(0, 0.5, -0.5),
                                   Point<dim>(1, 1.5, 0.5));
    Triangulation<dim> cube3;
    GridGenerator::hyper_rectangle(cube3,
                                   Point<dim>(0.5, -0.5, -0.5),
                                   Point<dim>(1.5, 0.5, 0.5));

    Triangulation<dim> tmp_triangulation;
    GridGenerator::merge_triangulations(
      {&tria_inner, &tria_outer, &cube1, &cube2, &cube3}, tmp_triangulation);

    tmp_triangulation.reset_all_manifolds();
    tmp_triangulation.set_all_manifold_ids(0);

    for (const auto &cell : tmp_triangulation.cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
              {
                if (std::fabs(face->vertex(v).norm_square() - 0.49 * 0.49) >
                    1e-12)
                  face_at_sphere_boundary = false;
              }
            if (face_at_sphere_boundary)
              face->set_all_manifold_ids(1);
          }
        if (cell->center().norm_square() < 0.1)
          cell->set_material_id(1);
        else
          cell->set_material_id(0);
      }

    tmp_triangulation.set_manifold(1, SphericalManifold<dim>());

    TransfiniteInterpolationManifold<dim> transfinite_manifold;
    transfinite_manifold.initialize(tmp_triangulation);
    tmp_triangulation.set_manifold(0, transfinite_manifold);

    tmp_triangulation.refine_global(m_parameters.m_global_refine_times);

    std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;

    for (const auto &cell : tmp_triangulation.active_cell_iterators())
      {
        if (cell->center()[0] < 0.0 || cell->center()[1] < 0.0 ||
            cell->center()[2] < 0.0 || cell->center()[0] > 1.0 ||
            cell->center()[1] > 1.0 || cell->center()[2] > 1.0)
          {
            cells_to_remove.insert(cell);
          }
      }

    GridGenerator::create_triangulation_with_removed_cells(tmp_triangulation,
                                                           cells_to_remove,
                                                           m_triangulation);

    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (cell->center()[2] > 0.505 && cell->center()[2] < 0.575 &&
                    cell->center()[0] < 0.10 && cell->center()[1] < 0.10)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }


  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_8()
  {
    AssertThrow(dim == 3, ExcMessage("The dimension has to be 3D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile
      << "\t\t\t\tSphere inclusion (3D structured version 2 with barriers)"
      << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    Triangulation<dim> tria_inner;
    GridGenerator::hyper_ball(tria_inner, Point<dim>(), 0.49);

    Triangulation<dim> tria_outer;
    GridGenerator::hyper_shell(
      tria_outer, Point<dim>(), 0.49, std::sqrt(dim) * 0.5, 2 * dim);

    Triangulation<dim> cube1;
    GridGenerator::hyper_rectangle(cube1,
                                   Point<dim>(0, 0, 0.5),
                                   Point<dim>(1, 1, 1.5));
    Triangulation<dim> cube2;
    GridGenerator::hyper_rectangle(cube2,
                                   Point<dim>(0, 0.5, -0.5),
                                   Point<dim>(1, 1.5, 0.5));
    Triangulation<dim> cube3;
    GridGenerator::hyper_rectangle(cube3,
                                   Point<dim>(0.5, -0.5, -0.5),
                                   Point<dim>(1.5, 0.5, 0.5));

    Triangulation<dim> tmp_triangulation;
    GridGenerator::merge_triangulations(
      {&tria_inner, &tria_outer, &cube1, &cube2, &cube3}, tmp_triangulation);

    tmp_triangulation.reset_all_manifolds();
    tmp_triangulation.set_all_manifold_ids(0);

    for (const auto &cell : tmp_triangulation.cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
              {
                if (std::fabs(face->vertex(v).norm_square() - 0.49 * 0.49) >
                    1e-12)
                  face_at_sphere_boundary = false;
              }
            if (face_at_sphere_boundary)
              face->set_all_manifold_ids(1);
          }
        if (cell->center().norm_square() < 0.1)
          cell->set_material_id(1);
        else
          cell->set_material_id(0);
      }

    tmp_triangulation.set_manifold(1, SphericalManifold<dim>());

    TransfiniteInterpolationManifold<dim> transfinite_manifold;
    transfinite_manifold.initialize(tmp_triangulation);
    tmp_triangulation.set_manifold(0, transfinite_manifold);

    tmp_triangulation.refine_global(m_parameters.m_global_refine_times);

    // some extra barriers
    for (const auto &cell : tmp_triangulation.cell_iterators())
      {
        if (std::fabs(cell->center()[1] - 0.75) < 0.05 &&
            std::fabs(cell->center()[2] - 0.5625) < 0.05 &&
            std::fabs(cell->center()[0] - 0.0) < 0.2)
          cell->set_material_id(1);

        if (std::fabs(cell->center()[1] - 0.0) < 0.2 &&
            std::fabs(cell->center()[2] - 0.5) < 0.1 &&
            std::fabs(cell->center()[0] - 0.75) < 0.05)
          cell->set_material_id(1);
      }

    std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;

    for (const auto &cell : tmp_triangulation.active_cell_iterators())
      {
        if (cell->center()[0] < 0.0 || cell->center()[1] < 0.0 ||
            cell->center()[2] < 0.0 || cell->center()[0] > 1.0 ||
            cell->center()[1] > 1.0 || cell->center()[2] > 1.0)
          {
            cells_to_remove.insert(cell);
          }
      }

    GridGenerator::create_triangulation_with_removed_cells(tmp_triangulation,
                                                           cells_to_remove,
                                                           m_triangulation);
    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if (cell->center()[2] > 0.505 && cell->center()[2] < 0.575 &&
                    cell->center()[0] < 0.10 && cell->center()[1] < 0.10)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }


  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_9()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tL-shape bending (" << dim << "D structured)"
              << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    if (m_extra_data.empty())
      {
        const std::vector<std::string> parameter_names = {
          "X1", "X2", "Y1", "Y2", "a", "thickness", "lc"};
        common::DataListReader::read_keys_values(m_parameters.m_extra_data_file,
                                                 parameter_names,
                                                 m_extra_data);
      }

    const double X1        = m_extra_data.at("X1");
    const double Y1        = m_extra_data.at("Y1");
    const double thickness = m_extra_data.at("thickness");
    const double lc        = m_extra_data.at("lc");



    if constexpr (dim == 3)
      {
        AssertThrow(
          thickness > 0,
          ExcMessage(
            "The thickness should be negative in 3-dimensional test for L-shape bending tests."));
      }


    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(m_parameters.m_mesh_file_name);
    gridin.read_msh(f);
    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    for (const auto &cell : m_triangulation.cell_iterators())
      {
        cell->set_material_id(0);
      }

    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        const double X1_mod  = X1 + 50;
        const double Y1_mod  = Y1 - lc * 2.0;
        const double Y_upper = Y1 + 62.5;

        for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
          {
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }

                const Point<dim> &center = cell->center();
                const double      x      = center[0];
                const double      y      = center[1];

                if ((y > Y1_mod) && (y < Y_upper) && (x < X1_mod))
                  {
                    setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            m_triangulation.execute_coarsening_and_refinement();
          }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        const double bandwidth                          = 0.6 * lc;
        bool         initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }

                const Point<dim> &center = cell->center();
                const double      x      = center[0];
                const double      y      = center[1];

                if ((std::fabs(x - X1) < bandwidth) &&
                    (std::fabs(y - Y1) < bandwidth))
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_11()
  {
    AssertThrow(dim == 3, ExcMessage("The dimension has to be 3D!"));

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tBrokenshire torsion (3D structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    Triangulation<2> triangulation_2d;


    if (m_extra_data.empty())
      {
        m_extra_data.emplace("length", 200.0);
        m_extra_data.emplace("width", 50.0);
        m_extra_data.emplace("height", 50.0);
        m_extra_data.emplace("delta_L", 25.0);
      }

    double const length  = m_extra_data.at("length");
    double const width   = m_extra_data.at("width");
    double const height  = m_extra_data.at("height");
    double const delta_L = m_extra_data.at("delta_L");

    double const tan_theta = delta_L / (0.5 * width);

    std::vector<unsigned int> repetitions(2, 1);
    repetitions[0] = 20;
    repetitions[1] = 5;

    Point<2> point1(0.0, 0.0);
    Point<2> point2(length, width);

    GridGenerator::subdivided_hyper_rectangle(triangulation_2d,
                                              repetitions,
                                              point1,
                                              point2);

    typename Triangulation<2>::vertex_iterator vertex_ptr;
    vertex_ptr = triangulation_2d.begin_active_vertex();
    while (vertex_ptr != triangulation_2d.end_vertex())
      {
        Point<2> &vertex_point = vertex_ptr->vertex();

        const double delta_x = (vertex_point(1) - 0.5 * width) * tan_theta;

        if (std::fabs(vertex_point(0) - 0.5 * length) < 1.0e-6)
          {
            vertex_point(0) += delta_x;
          }
        else if (std::fabs(vertex_point(0) + length / repetitions[0] -
                           0.5 * length) < 1.0e-6)
          {
            vertex_point(0) += (delta_x + length / repetitions[0] * 0.5);
          }
        else if (std::fabs(vertex_point(0) - length / repetitions[0] -
                           0.5 * length) < 1.0e-6)
          {
            vertex_point(0) += (delta_x - length / repetitions[0] * 0.5);
          }
        else if (vertex_point(0) <
                 0.5 * length - length / repetitions[0] - 1.0e-6)
          {
            vertex_point(0) += (delta_x + length / repetitions[0] * 0.5) *
                               vertex_point(0) /
                               (0.5 * length - length / repetitions[0]);
          }
        else if (vertex_point(0) >
                 0.5 * length + length / repetitions[0] + 1.0e-6)
          {
            vertex_point(0) += (delta_x - length / repetitions[0] * 0.5) *
                               (length - vertex_point(0)) /
                               (0.5 * length - length / repetitions[0]);
          }

        ++vertex_ptr;
      }

    Triangulation<dim> tmp_triangulation;
    const unsigned int n_layer = repetitions[1] + 1;
    GridGenerator::extrude_triangulation(triangulation_2d,
                                         n_layer,
                                         height,
                                         tmp_triangulation);

    tmp_triangulation.refine_global(m_parameters.m_global_refine_times);

    std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;

    for (const auto &cell : tmp_triangulation.active_cell_iterators())
      {
        if ((std::fabs(cell->center()[0] -
                       (cell->center()[1] - 0.5 * width) * tan_theta -
                       0.5 * length) < 2.5) &&
            cell->center()[2] > 0.5 * height)
          {
            cells_to_remove.insert(cell);
          }
      }

    GridGenerator::create_triangulation_with_removed_cells(tmp_triangulation,
                                                           cells_to_remove,
                                                           m_triangulation);

    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if ((std::fabs(cell->center()[0] -
                               (cell->center()[1] - 0.5 * width) * tan_theta -
                               0.5 * length) < 5.0) &&
                    cell->center()[2] <= 0.5 * height &&
                    cell->center()[2] > 0.5 * height - 5.0)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        bool initiation_point_refine_unfinished = true;

        const double halfWidth  = 0.5 * width;
        const double halfLength = 0.5 * length;
        const double halfHeight = 0.5 * height;

        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                const double offset =
                  cell->center()[0] -
                  (cell->center()[1] - halfWidth) * tan_theta - halfLength;
                if (((offset < 55.0 && offset > 0.0) &&
                     cell->center()[2] <= halfHeight &&
                     cell->center()[2] > 0) ||
                    ((offset < 0.0 && offset > -25.0) &&
                     cell->center()[2] <= halfHeight &&
                     cell->center()[2] > 0) ||
                    ((offset > -10.0 && offset < 0.0) &&
                     cell->center()[1] < 10.0 &&
                     cell->center()[2] >= halfHeight &&
                     cell->center()[2] < halfHeight + 6.0) ||
                    ((offset < 10.0 && offset > 0.0) &&
                     cell->center()[1] > width - 10.0 &&
                     cell->center()[2] >= halfHeight &&
                     cell->center()[2] < halfHeight + 6.0))
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_12()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t1-D bar (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim == 2, ExcMessage("The dimension has to be 2D!"));

    double const length = 200.0;
    double const width  = 1.0;
    double const h_size = 0.1;

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = (unsigned int)(length / h_size);
    repetitions[1] = (unsigned int)(width / h_size);

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
                                              repetitions,
                                              Point<dim>(0.0, 0.0),
                                              Point<dim>(length, width));
    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_13()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile
      << "\t\t\t3D tensile test with 2 preexisting cracks on the edges (structured)"
      << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    if (m_extra_data.empty())
      {
        const std::vector<std::string> parameter_names = {
          "L", "W", "H", "L1", "L2", "W1", "W2", "H1", "H2", "C1", "C2", "lc"};
        common::DataListReader::read_keys_values(m_parameters.m_extra_data_file,
                                                 parameter_names,
                                                 m_extra_data);
      }

    if constexpr (dim == 3)
      {
        GridIn<dim> gridin;
        gridin.attach_triangulation(m_triangulation);
        std::ifstream f(m_parameters.m_mesh_file_name);
        gridin.read_msh(f);
        if (m_parameters.m_output_coarse_ori)
          m_output.output(m_parameters.oriDir,
                          "coarse_original_mesh",
                          &m_triangulation);

        for (const auto &cell : m_triangulation.cell_iterators())
          {
            cell->set_material_id(0);
          }

        m_triangulation.refine_global(m_parameters.m_global_refine_times);

        const double bw_z  = 0.030;
        const double bw_xy = 0.030;

        // global dimensions
        const double L = m_extra_data.at("L"); // total length  (x direction)
        const double W = m_extra_data.at("W"); // total width   (y direction)

        // crack-plan sizes in top view
        const double L1 =
          m_extra_data.at("L1"); // x-size of C1 patch from left boundary
        const double L2 =
          m_extra_data.at("L2"); // x-size of C2 patch from right boundary
        const double W1 =
          m_extra_data.at("W1"); // y-size of C1 patch from top boundary
        const double W2 =
          m_extra_data.at("W2"); // y-size of C2 patch from bottom boundary

        // crack heights in front view
        const double H1 = m_extra_data.at("H1"); // height of C1 crack
        const double H2 = m_extra_data.at("H2"); // height of C2 crack



        // x partition: [0, L1] [middle] [L-L2, L]
        const double x1 = L1;
        const double x2 = L - L2;

        // y partition: [0, W2] [middle] [W-W1, W]
        const double y1 = W2;
        const double y2 = W - W1;

        // z partition: [0, H2] [H2, H1] [H1, H]
        const double z1 = H2;
        const double z2 = H1;

        const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;
        if (m_parameters.m_refinement_strategy == "adaptive-refine")
          {
            bool initiation_point_refine_unfinished = true;
            while (initiation_point_refine_unfinished)
              {
                initiation_point_refine_unfinished = false;
                for (const auto &cell : m_triangulation.active_cell_iterators())
                  {
                    if constexpr (is_mpi)
                      {
                        if (!cell->is_locally_owned())
                          continue;
                      }

                    const double x = cell->center()[0];
                    const double y = cell->center()[1];
                    const double z = cell->center()[2];

                    if ((std::fabs(z - z1) < bw_z &&
                         (((std::fabs(x - x2) < bw_xy) && (y < y1)) ||
                          ((std::fabs(y - y1) < bw_xy) && (x > x2)))) ||
                        (std::fabs(z - z2) < bw_z &&
                         (((std::fabs(x - x1) < bw_xy) && (y > y2)) ||
                          ((std::fabs(y - y2) < bw_xy) && (x < x1)))))
                      {
                        initiation_point_refine_unfinished |=
                          setRefineFlagByCellSize(cell, hlRatio);
                      }
                  }
                executeRefinement(initiation_point_refine_unfinished);
              }
          }
        else if (m_parameters.m_refinement_strategy == "pre-refine")
          {
            bool initiation_point_refine_unfinished = true;


            const double upper = H1 + 0.05;
            const double lower = H2 - 0.05;

            const double x1_mod = x1 - bw_xy;
            const double x2_mod = x2 + bw_xy;
            const double y1_mod = y1 - bw_xy;
            const double y2_mod = y2 + bw_xy;

            while (initiation_point_refine_unfinished)
              {
                initiation_point_refine_unfinished = false;
                for (const auto &cell : m_triangulation.active_cell_iterators())
                  {
                    if constexpr (is_mpi)
                      {
                        if (!cell->is_locally_owned())
                          continue;
                      }
                    const double z = cell->center()[2];

                    if (!(z <= upper && z >= lower))
                      continue;
                    ;

                    const double x = cell->center()[0];
                    const double y = cell->center()[1];
                    if ((x <= x1_mod && y <= y2_mod) ||
                        (x >= x1_mod && x <= x2_mod) ||
                        (x >= x2_mod && y >= y1_mod))
                      {
                        initiation_point_refine_unfinished |=
                          setRefineFlagByCellSize(cell, hlRatio);
                      }
                  }
                executeRefinement(initiation_point_refine_unfinished);
              }
          }
        else
          {
            AssertThrow(
              false,
              ExcMessage("Selected mesh refinement strategy not implemented!"));
          }
      }
    else
      {
        AssertThrow(false, ExcMessage("The dimension has to be 2D!"));
      }
  }


  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_14()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tNooru-Mohamed test " << dim << "d (structured)"
              << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;


    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(m_parameters.m_mesh_file_name);
    gridin.read_msh(f);
    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    for (const auto &cell : m_triangulation.cell_iterators())
      {
        cell->set_material_id(0);
      }

    m_triangulation.refine_global(m_parameters.m_global_refine_times);


    if (m_extra_data.empty())
      {
        const std::vector<std::string> parameter_names = {
          "L",
          "W",
          "H",
          "L1",
          "L2",
          "W1",
          "W2",
          "H1",
          "H2",
          "C1",
          "C2",
          "lc",
        };
        common::DataListReader::read_keys_values(m_parameters.m_extra_data_file,
                                                 parameter_names,
                                                 m_extra_data);
      }



    // global dimensions
    const double L = m_extra_data.at("L"); // total length  (x direction)
    const double W = m_extra_data.at("W"); // total width   (y direction)

    // crack-plan sizes in top view
    const double L1 =
      m_extra_data.at("L1"); // x-size of C1 patch from left boundary
    const double L2 =
      m_extra_data.at("L2"); // x-size of C2 patch from right boundary
    const double W1 =
      m_extra_data.at("W1"); // y-size of C1 patch from top boundary
    const double W2 =
      m_extra_data.at("W2"); // y-size of C2 patch from bottom boundary

    // crack heights in front view
    const double H1 = m_extra_data.at("H1"); // height of C1 crack
    const double H2 = m_extra_data.at("H2"); // height of C2 crack

    const double lc = m_extra_data.at("lc");


    const double bw_z  = 0.60 * lc;
    const double bw_xy = 0.60 * lc;

    // x partition: [0, L1] [middle] [L-L2, L]
    double x1 = L1;
    double x2 = L - L2;

    // y partition: [0, W2] [middle] [W-W1, W]
    double y1 = W2;
    double y2 = W - W1;

    // z partition: [0, H2] [H2, H1] [H1, H]
    double z1 = H2;
    double z2 = H1;

    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }

                const double x = cell->center()[0];
                const double y = cell->center()[1];

                bool willBeRefined = false;

                if constexpr (dim == 2)
                  {
                    willBeRefined =
                      (std::fabs(y - H1) < bw_z && std::fabs(x - x1) < bw_xy) ||
                      (std::fabs(y - H2) < bw_z && std::fabs(x - x2) < bw_xy);
                  }
                else if constexpr (dim == 3)
                  {
                    const double z = cell->center()[2];

                    willBeRefined = (std::fabs(z - z1) < bw_z &&
                                     (((std::fabs(x - x2) < bw_xy)))) ||
                                    (std::fabs(z - z2) < bw_z &&
                                     ((std::fabs(x - x1) < bw_xy)));
                  }


                if (willBeRefined)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        bool initiation_point_refine_unfinished = true;


        double offset1 = 0.5 * lc;
        double offset2 = 0.5 * lc;

        const double upper = H1 + offset1;
        const double lower = H2 - offset2;

        const double x1_mod = x1 - bw_xy;
        const double x2_mod = x2 + bw_xy;
        const double y1_mod = y1 - bw_xy;
        const double y2_mod = y2 + bw_xy;

        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if constexpr (dim == 3)
                  {
                    const double z = cell->center()[2];

                    if (!(z <= upper && z >= lower))
                      continue;
                    ;
                  }

                const double x = cell->center()[0];
                const double y = cell->center()[1];
                if ((x <= x1_mod && y <= y2_mod) ||
                    (x >= x1_mod && x <= x2_mod) ||
                    (x >= x2_mod && y >= y1_mod))
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }



  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_15()
  {
    if (m_extra_data.empty())
      {
        const std::vector<std::string> parameter_names = {"length",
                                                          "thickness",
                                                          "height",
                                                          "ex1",
                                                          "ex2",
                                                          "ex3",
                                                          "ex4",
                                                          "crack_depth",
                                                          "theta",
                                                          "x_crack_factor",
                                                          "notch_width_normal"};
        common::DataListReader::read_keys_values(m_parameters.m_extra_data_file,
                                                 parameter_names,
                                                 m_extra_data);
      }


    const double ex3               = m_extra_data.at("ex3");
    const double ex4               = m_extra_data.at("ex4");
    const bool   isThreePntBending = (ex3 < 0.0 && ex4 < 0.0);
    const bool   isFourPntBending  = (ex3 > 0.0 && ex4 > 0.0);

    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile
      << "\n\t\t\t3D notched beam bending with slanted notch (structured)\n\t\t\ttheta: "
      << std::round(m_extra_data.at("theta") * 180.0 / std::acos(-1.0) * 1e2) /
           1e2
      << "º\n\t\t\t";
    if (isThreePntBending)
      {
        m_logfile << "3-point bending" << std::endl;
      }
    else if (isFourPntBending)
      {
        m_logfile << "4-point bending [ loading at ";
        if (m_time.get_magnitude() > 0)
          {
            m_logfile << "notched end (z = " << m_extra_data.at("height")
                      << ")]" << std::endl;
          }
        else
          {
            m_logfile << "intact end (z = 0) ]" << std::endl;
          }
      }
    else
      {
        m_logfile
          << "Invalid parameters: ex3 and ex4 must both be negative for three-point bending or both positive for four-point bending."
          << std::endl;

        AssertThrow(
          isThreePntBending || isFourPntBending,
          ExcMessage(
            "Invalid parameters: ex3 and ex4 must both be negative for three-point bending or both positive for four-point bending."));
      }
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;


    if constexpr (dim == 3)
      {
        GridIn<dim> gridin;
        gridin.attach_triangulation(m_triangulation);
        std::ifstream f(m_parameters.m_mesh_file_name);
        gridin.read_msh(f);
        if (m_parameters.m_output_coarse_ori)
          m_output.output(m_parameters.oriDir,
                          "coarse_original_mesh",
                          &m_triangulation);

        for (const auto &cell : m_triangulation.cell_iterators())
          {
            cell->set_material_id(0);
          }

        m_triangulation.refine_global(m_parameters.m_global_refine_times);

        const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;

        // global dimensions
        const double x_crack_factor = m_extra_data.at("x_crack_factor");
        const double halfLength    = m_extra_data.at("length") * x_crack_factor;
        const double halfThickness = m_extra_data.at("thickness") * 0.5;
        // crack heights
        const double Hc = m_extra_data.at("height") -
                          m_extra_data.at("crack_depth"); // height of C1 crack
        const double tan_theta = std::tan(m_extra_data.at("theta"));



        double       bandWidth_x          = 5.0;
        double       offset_z_under_crack = 0.0;
        const double offset_z_above_crack = 1.0;

        if (m_parameters.m_refinement_strategy == "adaptive-refine")
          {
            offset_z_under_crack = 4.0;
          }
        else if (m_parameters.m_refinement_strategy == "pre-refine")
          {
            bandWidth_x          = halfThickness * tan_theta * 1.35;
            offset_z_under_crack = Hc;
          }
        else
          {
            AssertThrow(
              false,
              ExcMessage("Selected mesh refinement strategy not implemented!"));
          }


        bool initiation_point_refine_unfinished = true;

        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }

                const Point<3> center = cell->center();
                const double   x      = center[0];
                const double   y      = center[1];
                const double   z      = center[2];

                if (z <= (Hc + offset_z_above_crack) &&
                    z > (Hc - offset_z_under_crack) &&
                    (std::fabs(x - halfLength -
                               (y - halfThickness) * tan_theta) < bandWidth_x))
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }

            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false, ExcMessage("The dimension has to be 2D!"));
      }
  }



  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_16()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tNooru-Mohamed test " << dim << "d (structured)"
              << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;


    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f(m_parameters.m_mesh_file_name);
    gridin.read_msh(f);
    if (m_parameters.m_output_coarse_ori)
      m_output.output(m_parameters.oriDir,
                      "coarse_original_mesh",
                      &m_triangulation);

    for (const auto &cell : m_triangulation.cell_iterators())
      {
        cell->set_material_id(0);
      }

    m_triangulation.refine_global(m_parameters.m_global_refine_times);


    if (m_extra_data.empty())
      {
        const std::vector<std::string> parameter_names = {
          "L",
          "W",
          "H",
          "L1",
          "L2",
          "W1",
          "W2",
          "H1",
          "H2",
          "C1",
          "C2",
          "lc",
        };
        ::common::DataListReader::read_keys_values(
          m_parameters.m_extra_data_file, parameter_names, m_extra_data);
      }



    // global dimensions
    const double L = m_extra_data.at("L"); // total length  (x direction)
    const double W = m_extra_data.at("W"); // total width   (y direction)

    // crack-plan sizes in top view
    const double L1 =
      m_extra_data.at("L1"); // x-size of C1 patch from left boundary
    const double L2 =
      m_extra_data.at("L2"); // x-size of C2 patch from right boundary
    const double W1 =
      m_extra_data.at("W1"); // y-size of C1 patch from top boundary
    const double W2 =
      m_extra_data.at("W2"); // y-size of C2 patch from bottom boundary


    const double C1 = m_extra_data.at("C1"); // height of C1 crack
    const double C2 = m_extra_data.at("C2"); // height of C2 crack


    // crack heights in front view
    const double H1 = m_extra_data.at("H1") + C1 * 0.5; // height of C1 crack
    const double H2 = m_extra_data.at("H2") + C2 * 0.5; // height of C2 crack


    const double lc = m_extra_data.at("lc");


    const double bw_z1 = 0.60 * lc + C1 * 0.5;
    const double bw_z2 = 0.60 * lc + C2 * 0.5;
    const double bw_xy = 0.60 * lc;

    // x partition: [0, L1] [middle] [L-L2, L]
    double x1 = L1;
    double x2 = L - L2;



    const double hlRatio = m_parameters.m_allowed_max_h_l_ratio;

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        bool initiation_point_refine_unfinished = true;
        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }

                const double x = cell->center()[0];
                const double y = cell->center()[1];

                bool willBeRefined =
                  (std::fabs(y - H1) < bw_z1 && std::fabs(x - x1) < bw_xy) ||
                  (std::fabs(y - H2) < bw_z2 && std::fabs(x - x2) < bw_xy);

                if (willBeRefined)
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else if (m_parameters.m_refinement_strategy == "pre-refine")
      {
        bool initiation_point_refine_unfinished = true;

        // y partition: [0, W2] [middle] [W-W1, W]
        double y1 = W2;
        double y2 = W - W1;

        double offset1 = 0.5 * lc;
        double offset2 = 0.5 * lc;

        const double upper = H1 + offset1;
        const double lower = H2 - offset2;

        const double x1_mod = x1 - bw_xy;
        const double x2_mod = x2 + bw_xy;
        const double y1_mod = y1 - bw_xy;
        const double y2_mod = y2 + bw_xy;

        while (initiation_point_refine_unfinished)
          {
            initiation_point_refine_unfinished = false;
            for (const auto &cell : m_triangulation.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                if constexpr (dim == 3)
                  {
                    const double z = cell->center()[2];

                    if (!(z <= upper && z >= lower))
                      continue;
                    ;
                  }

                const double x = cell->center()[0];
                const double y = cell->center()[1];
                if ((x <= x1_mod && y <= y2_mod) ||
                    (x >= x1_mod && x <= x2_mod) ||
                    (x >= x2_mod && y >= y1_mod))
                  {
                    initiation_point_refine_unfinished |=
                      setRefineFlagByCellSize(cell, hlRatio);
                  }
              }
            executeRefinement(initiation_point_refine_unfinished);
          }
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Selected mesh refinement strategy not implemented!"));
      }
  }


  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::setup_system()
  {
    const std::string sectionName = "Setup system";
    m_timer.enter_subsection(sectionName);


    m_dof_handler.distribute_dofs(m_fe);
    DoFRenumbering::Cuthill_McKee(m_dof_handler);
    DoFRenumbering::component_wise(m_dof_handler, m_blocks_desc.groupIDs());

    m_blocks_desc.updateDoFsInfo(m_dof_handler);


    m_constraints.clear();
    if constexpr (is_mpi)
      {
        m_cst_maker.cstReinit(m_constraints,
                              m_dof_handler.locally_owned_dofs(),
                              *m_blocks_desc.localRelevantPartition());
      }
    DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
    if constexpr (is_mpi)
      {
        m_constraints.make_consistent_in_parallel(
          m_dof_handler.locally_owned_dofs(),
          *m_blocks_desc.localRelevantPartition(),
          *m_mpiInfo.mpiCommPtr());
      }
    m_constraints.close();


    unsigned int nCells    = m_triangulation.n_active_cells();
    unsigned int nVertices = m_triangulation.n_used_vertices();
    unsigned int nLines    = m_triangulation.n_active_lines();
    unsigned int nFaces    = m_triangulation.n_active_faces();

    if constexpr (is_mpi)
      {
        nCells    = m_triangulation.n_global_active_cells();
        nVertices = Utilities::MPI::sum(nVertices, *m_mpiInfo.mpiCommPtr());
        nLines    = Utilities::MPI::sum(nLines, *m_mpiInfo.mpiCommPtr());
        nFaces    = Utilities::MPI::sum(nFaces, *m_mpiInfo.mpiCommPtr());
      }


    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\t Number of active cells: " << nCells
              << "\n\t\t\t Number of used vertices: " << nVertices
              << "\n\t\t\t Number of active edges: " << nLines
              << "\n\t\t\t Number of active faces: " << nFaces
              << "\n\t\t\t Number of degrees of freedom (total): "
              << m_dof_handler.n_dofs()
              << "\n\t\t\t Number of degrees of freedom (disp): "
              << (*m_blocks_desc.dofsPerBlockPtr())[m_u_dof]
              << "\n\t\t\t Number of degrees of freedom (phasefield): "
              << (*m_blocks_desc.dofsPerBlockPtr())[m_d_dof] << std::endl;


    m_tangent_matrix.initalize(m_dof_handler, m_constraints, false);

    m_system_rhs.initialize();
    m_solution.initialize();

    m_vec_pool.initialize();
    m_vec_pool_ghosted.initialize();

    setup_qph();

    m_update_dofs_for_cst = true;
    m_timer.leave_subsection(sectionName);
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::make_constraints(
    const unsigned int it_nr)
  {
    using namespace ::bcs;

    const bool apply_dirichlet_bc = (it_nr == 0);

    if (it_nr > 1)
      {
        if (m_parameters.m_output_iteration_history)
          m_logfile << " --- " << std::flush;
        return;
      }

    if (m_parameters.m_output_iteration_history)
      {
        if (!m_update_dofs_for_cst)
          {
            m_logfile << " CST " << std::flush;
          }
        else
          {
            m_logfile << "[CST]" << std::flush;
          }
      }
    if (apply_dirichlet_bc)
      {
        set_bcs_id();

        m_constraints.clear();
        if constexpr (is_mpi)
          {
            m_cst_maker.cstReinit(m_constraints,
                                  m_dof_handler.locally_owned_dofs(),
                                  DoFTools::extract_locally_relevant_dofs(
                                    m_dof_handler));
          }
        DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);

        const FEValuesExtractors::Scalar x_displacement(0);
        const FEValuesExtractors::Scalar y_displacement(1);
        const FEValuesExtractors::Scalar z_displacement(2);

        const FEValuesExtractors::Vector displacements(0);
        const FEValuesExtractors::Scalar phasefield(dim);

        if (m_parameters.m_scenario == 1 || m_parameters.m_scenario == 3)
          {
            // Dirichlet B,C. bottom surface
            const int boundary_id_bottom_surface = 0;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_bottom_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(y_displacement));


            const int boundary_id_top_surface = 1;
            /*
             VectorTools::interpolate_boundary_values(m_dof_handler,
             boundary_id_top_surface,
             Functions::ZeroFunction<dim>(m_n_components),
             m_constraints,
             m_fe.component_mask(x_displacement));
             */
            const double time_inc       = m_time.get_delta_t();
            double       disp_magnitude = m_time.get_magnitude();
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_top_surface,
              Functions::ConstantFunction<dim>(disp_magnitude * time_inc,
                                               m_n_components),
              m_constraints,
              m_fe.component_mask(y_displacement));

            if (!m_cst_maker.isAddedSelectors())
              {
                const CstPnt<Tria> fixedXY({{0.0, 0.0, 0.0}},
                                           {CstEntry<Tria>(0),
                                            CstEntry<Tria>(1)});

                m_cst_maker.addCstSelector(fixedXY);
              }

            m_cst_maker.makeCstIfPrepareNeeded(m_update_dofs_for_cst,
                                               m_constraints,
                                               m_triangulation,
                                               m_dof_handler);
          }
        else if (m_parameters.m_scenario == 2 || m_parameters.m_scenario == 4)
          {
            // Dirichlet B,C. bottom surface
            const int boundary_id_bottom_surface = 0;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_bottom_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(displacements));

            const int boundary_id_top_surface = 1;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_top_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(y_displacement));

            const double time_inc       = m_time.get_delta_t();
            double       disp_magnitude = m_time.get_magnitude();
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_top_surface,
              Functions::ConstantFunction<dim>(disp_magnitude * time_inc,
                                               m_n_components),
              m_constraints,
              m_fe.component_mask(x_displacement));

            const int boundary_id_side_surfaces = 2;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_side_surfaces,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(y_displacement));
          }
        else if (m_parameters.m_scenario == 5)
          {
            if (!m_cst_maker.isAddedSelectors())
              {
                const unsigned int xDoF = 0;
                const unsigned int yDoF = 1;
                // top-center node applied with y-displacement
                const double time_inc       = m_time.get_delta_t();
                const double disp_magnitude = m_time.get_magnitude();
                const double du_y           = time_inc * disp_magnitude;

                const CstEntry<Tria> loadingCst(yDoF, du_y);
                const CstEntry<Tria> fixedXCst(xDoF);
                const CstEntry<Tria> fixedYCst(yDoF);

                const CstPnt<Tria> fixedXY({{0.0, 0.0, 0.0}},
                                           {fixedXCst, fixedYCst});

                const CstPnt<Tria> fixedY({{0.0, 8.0, 0.0}}, {fixedYCst});

                const CstPnt<Tria> loadingY({{4.0, 2.0, 0.0}}, {loadingCst});


                m_cst_maker.addCstSelector(fixedXY)
                  .addCstSelector(fixedY)
                  .addCstSelector(loadingY);
              }

            m_cst_maker.makeCstIfPrepareNeeded(m_update_dofs_for_cst,
                                               m_constraints,
                                               m_triangulation,
                                               m_dof_handler);
          }
        else if (m_parameters.m_scenario == 6 || m_parameters.m_scenario == 7 ||
                 m_parameters.m_scenario == 8)
          {
            const int x0_surface = 0;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              x0_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(x_displacement));
            const int y0_surface = 1;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              y0_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(y_displacement));
            const int z0_surface = 2;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              z0_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(z_displacement));

            const int    z1_surface     = 3;
            const double time_inc       = m_time.get_delta_t();
            double       disp_magnitude = m_time.get_magnitude();
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              z1_surface,
              Functions::ConstantFunction<dim>(disp_magnitude * time_inc,
                                               m_n_components),
              m_constraints,
              m_fe.component_mask(z_displacement));
          }
        else if (m_parameters.m_scenario == 9)
          {
            const double a  = m_extra_data.at("a");
            const double Y1 = m_extra_data.at("Y1");
            const double X1 = m_extra_data.at("X1");
            const double X2 = m_extra_data.at("X2");

            const double x_loading = X1 + X2 - a;


            const double time_inc       = m_time.get_delta_t();
            const double disp_magnitude = m_time.get_magnitude();
            const double du_y           = disp_magnitude * time_inc;

            // Dirichlet B,C. bottom surface
            const int boundary_id_bottom_surface = 0;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_bottom_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(displacements));

            if (!m_cst_maker.isAddedSelectors())
              {
                const unsigned int yDoF = 1;
                CstEntry<Tria>     loadingCst(yDoF, du_y);

                const auto pntSelFunc =
                  [x_loading, Y1](double x, double y, double) -> bool {
                  return (std::fabs(x - x_loading) < 1.0e-9) &&
                         (std::fabs(y - Y1) < 1.0e-9);
                };

                const CstFunc<Tria> singlePntLoadingCst(pntSelFunc,
                                                        {loadingCst});

                m_cst_maker.addCstSelector(singlePntLoadingCst);
              }

            m_cst_maker.makeCstIfPrepareNeeded(m_update_dofs_for_cst,
                                               m_constraints,
                                               m_triangulation,
                                               m_dof_handler);
          }
        else if (m_parameters.m_scenario == 11)
          {
            // Dirichlet B,C. right surface
            const int boundary_id_right_surface = 0;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_right_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(displacements));

            // Dirichlet B,C. left surface
            const int boundary_id_left_surface = 1;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_left_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(x_displacement));


            if (!m_cst_maker.isAddedSelectors())
              {
                const double dt          = m_time.get_delta_t();
                const double magnitude   = m_time.get_magnitude();
                const double angle_theta = dt * magnitude;

                const auto disp = [angle_theta](const double,
                                                const double         y,
                                                const double         z,
                                                std::vector<double> &results) {
                  double disp_y = 0.0;
                  double disp_z = 0.0;

                  const double node_dist = std::sqrt(y * y + z * z);

                  if (node_dist > 0)
                    {
                      const double disp_mag = node_dist * std::tan(angle_theta);
                      const double factor   = disp_mag / node_dist;
                      disp_y                = z * factor;
                      disp_z                = -y * factor;
                    }
                  results[0] = 0;
                  results[1] = disp_y;
                  results[2] = disp_z;
                };


                const auto pntSelFunc = [](double x, double, double) -> bool {
                  return std::fabs(x - 0.0) < 1.0e-9;
                };

                const CstFunc<Tria> rotationCst(
                  pntSelFunc, {CstEntry<Tria>(1), CstEntry<Tria>(2)}, disp);
                m_cst_maker.addCstSelector(rotationCst);
              }

            m_cst_maker.makeCstIfPrepareNeeded(m_update_dofs_for_cst,
                                               m_constraints,
                                               m_triangulation,
                                               m_dof_handler);
          }
        else if (m_parameters.m_scenario == 12)
          {
            // Dirichlet B.C. left surface (x = 0)
            const int boundary_id_left_surface = 0;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_left_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(displacements));
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_left_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(phasefield));

            const int boundary_id_right_surface = 1;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_right_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(y_displacement));
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_right_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(phasefield));

            const double time_inc       = m_time.get_delta_t();
            double       disp_magnitude = m_time.get_magnitude();
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_right_surface,
              Functions::ConstantFunction<dim>(disp_magnitude * time_inc,
                                               m_n_components),
              m_constraints,
              m_fe.component_mask(x_displacement));

            const int boundary_id_bottom_surfaces = 2;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_bottom_surfaces,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(y_displacement));

            const int boundary_id_top_surfaces = 3;
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_top_surfaces,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(y_displacement));
          }
        else if (m_parameters.m_scenario == 13)
          {
            // Dirichlet B.C. bottom surface (z = 0)
            const int boundary_id_btm_surface = m_bcs_id.at("btm");
            VectorTools::interpolate_boundary_values(
              m_dof_handler,
              boundary_id_btm_surface,
              Functions::ZeroFunction<dim>(m_n_components),
              m_constraints,
              m_fe.component_mask(displacements));



            const double time_inc       = m_time.get_delta_t();
            double       disp_magnitude = m_time.get_magnitude();
            const double displacement   = disp_magnitude * time_inc;
            const auto   du =
              Functions::ConstantFunction<dim>(displacement, m_n_components);
            const int boundary_id_top_surfaces = m_bcs_id.at("top");
            VectorTools::interpolate_boundary_values(m_dof_handler,
                                                     boundary_id_top_surfaces,
                                                     du,
                                                     m_constraints,
                                                     m_fe.component_mask(
                                                       z_displacement));
          }
        else if (m_parameters.m_scenario == 14)
          {
            // Dirichlet B.C. bottom surface (z = 0)

            const double time_inc         = m_time.get_delta_t();
            const double disp_magnitude_x = m_time.get_magnitude(0);
            const double disp_magnitude_z = m_time.get_magnitude(1);
            const double displacement_x   = disp_magnitude_x * time_inc;
            const double displacement_z   = disp_magnitude_z * time_inc;

            const auto dx =
              Functions::ConstantFunction<dim>(displacement_x, m_n_components);
            const auto dx_ =
              Functions::ConstantFunction<dim>(-displacement_x, m_n_components);

            const auto dz =
              Functions::ConstantFunction<dim>(displacement_z, m_n_components);
            const auto dz_ =
              Functions::ConstantFunction<dim>(-displacement_z, m_n_components);


            const auto shear   = m_fe.component_mask(x_displacement);
            auto       tensile = m_fe.component_mask(y_displacement);
            if constexpr (dim == 3)
              {
                tensile = m_fe.component_mask(z_displacement);
              }

            VectorTools::interpolate_boundary_values(
              m_dof_handler, m_bcs_id.at("btm"), dz_, m_constraints, tensile);

            VectorTools::interpolate_boundary_values(
              m_dof_handler, m_bcs_id.at("top"), dz, m_constraints, tensile);


            VectorTools::interpolate_boundary_values(
              m_dof_handler, m_bcs_id.at("H1"), dx, m_constraints, shear);


            VectorTools::interpolate_boundary_values(
              m_dof_handler, m_bcs_id.at("H2"), dx_, m_constraints, shear);
          }
        else if (m_parameters.m_scenario == 15)
          {
            /**
            Three-point bending
                ex3 < 0
                ex4 < 0
                dz  > 0
                                  load = dz > 0
                                    ↓  x = halfLength
            + ---|------------------|------------------|---  ---> x
            | ---|                                     |---
            | ---|------------------^------------------|---
            |    ▲ support                             ▲ support
            |    ex1 fixed x,y,z                       ex2 fixed y,z
            |
            v z


            Four-point bending (1)
                ex3 > 0
                ex4 > 0
                dz  > 0
                       load = dz > 0     load = dz > 0
                              ↓ x = ex3   ↓ x = ex4
            + ---|------------|-----------|------------|--- ---> x
            | ---|                                     |---
            | ---|------------|-----^-----|------------|---
            |    ▲ support                             ▲ support
            |    ex1 fixed x,y,z                      ex2 fixed y,z
            |
            v z


            Four-point bending (2)
                ex3 > 0
                ex4 > 0
                dz  < 0
            ^ z
            |   load = dz < 0                            load = dz < 0
            |    ↓ ex1                                     ↓ ex2
            | ---|-------------|------v-------|------------|---
            | ---|                                         |---
            + ---|-------------|--------------|------------|--- ---> x
                               ▲ support      ▲ support
                               ex3            ex4
                               fixed x,y,z    fixed y,z
             */

            if constexpr (dim == 3)
              {
                if (!m_cst_maker.isAddedSelectors())
                  {
                    const double ex1 = m_extra_data.at("ex1");
                    const double ex2 = m_extra_data.at("ex2");
                    const double ex3 = m_extra_data.at("ex3");
                    const double ex4 = m_extra_data.at("ex4");

                    const bool isFourPntBending = (ex3 > 0.0 && ex4 > 0.0);


                    const double length = m_extra_data.at("length");
                    const double x_crack_factor =
                      m_extra_data.at("x_crack_factor");

                    const double notch_x = length * x_crack_factor;
                    const double ex1_x   = ex1;
                    const double ex2_x   = length - ex2;

                    const double ex3_x = ex3;
                    const double ex4_x = length - ex4;

                    if (isFourPntBending)
                      {
                        AssertThrow(
                          ex1_x < ex3_x && ex3_x < ex4_x && ex4_x < ex2_x,
                          ExcMessage(
                            "Invalid four-point bending geometry: expected ex1_x < ex3_x < ex4_x < ex2_x."));
                      }


                    const double intactZ  = 0.0;
                    const double notchedZ = m_extra_data.at("height");

                    // top-center node applied with y-displacement
                    const double time_inc       = m_time.get_delta_t();
                    const double disp_magnitude = m_time.get_magnitude();
                    double       dz             = time_inc * disp_magnitude;

                    if (!isFourPntBending)
                      {
                        // guarantee the dz is positive in the three-point
                        // bending
                        if (dz < 0)
                          dz = -dz;
                      }

                    const bool is1stFourPntBending =
                      isFourPntBending && (dz > 0 ? true : false);


                    const auto ex1_notchedZ =
                      [ex1_x, notchedZ](double x, double, double z) -> bool {
                      return (std::fabs(x - ex1_x) < 1.0e-9) &&
                             (std::fabs(z - notchedZ) < 1.0e-9);
                    };

                    const auto ex2_notchedZ =
                      [ex2_x, notchedZ](double x, double, double z) -> bool {
                      return (std::fabs(x - ex2_x) < 1.0e-9) &&
                             (std::fabs(z - notchedZ) < 1.0e-9);
                    };

                    const auto notchedX_intactZ =
                      [notch_x, intactZ](double x, double, double z) -> bool {
                      return (std::fabs(x - notch_x) < 1.0e-9) &&
                             (std::fabs(z - intactZ) < 1.0e-9);
                    };

                    const auto ex3_intactZ =
                      [ex3_x, intactZ](double x, double, double z) -> bool {
                      return (std::fabs(x - ex3_x) < 1.0e-9) &&
                             (std::fabs(z - intactZ) < 1.0e-9);
                    };

                    const auto ex4_intactZ =
                      [ex4_x, intactZ](double x, double, double z) -> bool {
                      return (std::fabs(x - ex4_x) < 1.0e-9) &&
                             (std::fabs(z - intactZ) < 1.0e-9);
                    };

                    const unsigned int xDoF = 0;
                    const unsigned int yDoF = 1;
                    const unsigned int zDoF = 2;

                    const CstEntry<Tria> fixedX(xDoF);
                    const CstEntry<Tria> fixedY(yDoF);
                    const CstEntry<Tria> fixedZ(zDoF);

                    const CstEntry<Tria> loadingZ(zDoF, dz);

                    if (isFourPntBending)
                      {
                        if (is1stFourPntBending)
                          {
                            // Four-point bending (1)
                            const CstFunc<Tria> fixedXYZ(ex1_notchedZ,
                                                         {loadingZ});

                            const CstFunc<Tria> fixedYZ(ex2_notchedZ,
                                                        {loadingZ});

                            const CstFunc<Tria> loadingPnt1(
                              ex3_intactZ, {fixedX, fixedY, fixedZ});

                            const CstFunc<Tria> loadingPnt2(ex4_intactZ,
                                                            {fixedY, fixedZ});


                            m_cst_maker.addCstSelector(fixedXYZ)
                              .addCstSelector(fixedYZ)
                              .addCstSelector(loadingPnt1)
                              .addCstSelector(loadingPnt2);
                          }
                        else
                          {
                            // Four-point bending (2)
                            const CstFunc<Tria> fixedXYZ(
                              ex3_intactZ, {fixedX, fixedY, fixedZ});

                            const CstFunc<Tria> fixedYZ(ex4_intactZ,
                                                        {fixedY, fixedZ});

                            const CstFunc<Tria> loadingPnt1(ex1_notchedZ,
                                                            {loadingZ});

                            const CstFunc<Tria> loadingPnt2(ex2_notchedZ,
                                                            {loadingZ});

                            m_cst_maker.addCstSelector(fixedXYZ)
                              .addCstSelector(fixedYZ)
                              .addCstSelector(loadingPnt1)
                              .addCstSelector(loadingPnt2);
                          }
                      }
                    else
                      {
                        // Three-point bending
                        const CstFunc<Tria> fixedXYZ(ex1_notchedZ,
                                                     {fixedX, fixedY, fixedZ});

                        const CstFunc<Tria> fixedYZ(ex2_notchedZ,
                                                    {fixedY, fixedZ});

                        const CstFunc<Tria> loadingPnt(notchedX_intactZ,
                                                       {loadingZ});

                        m_cst_maker.addCstSelector(fixedXYZ)
                          .addCstSelector(fixedYZ)
                          .addCstSelector(loadingPnt);
                      }
                  }

                m_cst_maker.makeCstIfPrepareNeeded(m_update_dofs_for_cst,
                                                   m_constraints,
                                                   m_triangulation,
                                                   m_dof_handler);
              }
          }
        else if (m_parameters.m_scenario == 16)
          {
            // Dirichlet B.C. bottom surface (z = 0)

            const double time_inc         = m_time.get_delta_t();
            const double disp_magnitude_x = m_time.get_magnitude(0);
            const double disp_magnitude_y = m_time.get_magnitude(1);
            const double displacement_x   = disp_magnitude_x * time_inc;
            const double displacement_y   = disp_magnitude_y * time_inc;


            const auto dx =
              Functions::ConstantFunction<dim>(displacement_x, m_n_components);
            const auto dx_ =
              Functions::ConstantFunction<dim>(-displacement_x, m_n_components);

            const auto dy =
              Functions::ConstantFunction<dim>(displacement_y, m_n_components);
            const auto dy_ =
              Functions::ConstantFunction<dim>(-displacement_y, m_n_components);


            const auto shear   = m_fe.component_mask(x_displacement);
            const auto tensile = m_fe.component_mask(y_displacement);


            VectorTools::interpolate_boundary_values(
              m_dof_handler, m_bcs_id.at("btm"), dy_, m_constraints, tensile);

            VectorTools::interpolate_boundary_values(
              m_dof_handler, m_bcs_id.at("top"), dy, m_constraints, tensile);


            VectorTools::interpolate_boundary_values(
              m_dof_handler, m_bcs_id.at("H1"), dx_, m_constraints, shear);


            VectorTools::interpolate_boundary_values(
              m_dof_handler, m_bcs_id.at("H2"), dx, m_constraints, shear);
          }
        else
          Assert(false, ExcMessage("The scenario has not been implemented!"));

        if constexpr (is_mpi)
          {
            m_constraints.make_consistent_in_parallel(
              m_dof_handler.locally_owned_dofs(),
              *m_blocks_desc.localRelevantPartition(),
              *m_mpiInfo.mpiCommPtr());
          }
      }
    else // inhomogeneous constraints
      {
        bool has_inhomo = m_constraints.has_inhomogeneities();

        if constexpr (is_mpi)
          {
            // accumulate local flag over all ranks
            const unsigned int local_flag = has_inhomo ? 1u : 0u;
            const unsigned int global_flag =
              Utilities::MPI::sum(local_flag, *m_mpiInfo.mpiCommPtr());
            has_inhomo = (global_flag > 0u);
          }

        if (has_inhomo)
          {
            AffineConstraints<double> homoCst(m_constraints);
            if constexpr (is_mpi)
              {
                std::vector<IndexSet::size_type> indices;
#if DEAL_II_VERSION_GTE(9, 4, 0)
                indices = m_dof_handler.locally_owned_dofs().get_index_vector();
#else
                m_dof_handler.locally_owned_dofs().fill_index_vector(indices);
#endif

                for (unsigned int dof : indices)
                  if (homoCst.is_inhomogeneously_constrained(dof))
                    homoCst.set_inhomogeneity(dof, 0.0);
              }
            else
              {
                for (unsigned int dof = 0; dof != m_dof_handler.n_dofs(); ++dof)
                  if (homoCst.is_inhomogeneously_constrained(dof))
                    homoCst.set_inhomogeneity(dof, 0.0);
              }

            homoCst.close();
            m_constraints.clear();

            if constexpr (is_mpi)
              {
                m_cst_maker.cstReinit(m_constraints,
                                      m_dof_handler.locally_owned_dofs(),
                                      DoFTools::extract_locally_relevant_dofs(
                                        m_dof_handler));
              }

            m_constraints.copy_from(homoCst);
          }
      }
    m_constraints.close();
    m_update_dofs_for_cst = false;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_newton(
    const BVector & /*solution_old*/)
  {
    const std::string sectionName = "Assemble system";
    m_timer.enter_subsection(sectionName);
    /*
     if (m_parameters.m_output_iteration_history)
     m_logfile << " ASM_SYS " << std::flush;

     m_tangent_matrix = 0.0;
     m_system_rhs    = 0.0;

     const UpdateFlags uf_cell(update_values | update_gradients |
     update_quadrature_points | update_JxW_values);
     const UpdateFlags uf_face(update_values | update_normal_vectors |
     update_JxW_values);

     PerTaskData_ASM_RHS_BFGS per_task_data(m_fe.n_dofs_per_cell());
     ScratchData_ASM_RHS_BFGS scratch_data(m_fe, m_qf_cell, uf_cell, m_qf_face,
     uf_face, solution_old);

     auto worker =
     [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
     ScratchData_ASM_RHS_BFGS & scratch,
     PerTaskData_ASM_RHS_BFGS & data)
     {
     this->assemble_system_rhs_BFGS_one_cell(cell, scratch, data);
     };

     auto copier = [this, &system_rhs](const PerTaskData_ASM_RHS_BFGS &data)
     {
     this->m_constraints.distribute_local_to_global(data.m_cell_rhs,
     data.m_local_dof_indices,
     system_rhs);
     };

     WorkStream::run(
     m_dof_handler.active_cell_iterators(),
     worker,
     copier,
     scratch_data,
     per_task_data);
     */
    m_timer.leave_subsection(sectionName);
  }


  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_B0(
    const BVector &solution_old)
  {
    const std::string sectionName = "Assemble B0";
    m_timer.enter_subsection(sectionName);

    m_tangent_matrix = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM per_task_data(m_fe.n_dofs_per_cell());
    ScratchData_ASM scratch_data(
      m_fe, m_qf_cell, uf_cell, m_qf_face, uf_face, solution_old);

    if constexpr (!is_mpi)
      {
        // non-mpi mode
        auto worker =
          [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
                 ScratchData_ASM                                      &scratch,
                 PerTaskData_ASM                                      &data) {
            this->assemble_system_B0_one_cell(cell, scratch, data);
          };

        auto copier = [this](const PerTaskData_ASM &data) {
          this->m_constraints.distribute_local_to_global(
            data.m_cell_matrix,
            data.m_local_dof_indices,
            m_tangent_matrix.base());
        };

        WorkStream::run(m_dof_handler.active_cell_iterators(),
                        worker,
                        copier,
                        scratch_data,
                        per_task_data);
      }
    else
      {
        // mpi mode
        for (const auto &cell : m_dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              assemble_system_B0_one_cell(cell, scratch_data, per_task_data);

              m_constraints.distribute_local_to_global(
                per_task_data.m_cell_matrix,
                per_task_data.m_local_dof_indices,
                m_tangent_matrix.base());
            }

        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        m_tangent_matrix.compress(VectorOperation::add);
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
      }

    m_timer.leave_subsection(sectionName);
  }

  template <typename LATraits, typename Tria>
  template <bool has_body_force, bool has_surface_pressure>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_rhs_BFGS_parallel(
    const BVector &solution_old,
    BVector       &system_rhs)
  {
    const std::string sectionName = "Assemble RHS";
    m_timer.enter_subsection(sectionName);

    // m_logfile << " A_RHS " << std::flush;

    system_rhs = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM_RHS_BFGS per_task_data(m_fe.n_dofs_per_cell());
    ScratchData_ASM_RHS_BFGS scratch_data(
      m_fe, m_qf_cell, uf_cell, m_qf_face, uf_face, solution_old);

    if constexpr (!is_mpi)
      {
        // non-mpi mode
        auto worker =
          [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
                 ScratchData_ASM_RHS_BFGS                             &scratch,
                 PerTaskData_ASM_RHS_BFGS                             &data) {
            this->template assemble_system_rhs_BFGS_one_cell<
              has_body_force,
              has_surface_pressure>(cell, scratch, data);
          };

        auto copier = [this,
                       &system_rhs](const PerTaskData_ASM_RHS_BFGS &data) {
          this->m_constraints.distribute_local_to_global(
            data.m_cell_rhs, data.m_local_dof_indices, system_rhs.base());
        };

        WorkStream::run(m_dof_handler.active_cell_iterators(),
                        worker,
                        copier,
                        scratch_data,
                        per_task_data);
      }
    else
      {
        // mpi mode

        for (const auto &cell : m_dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              assemble_system_rhs_BFGS_one_cell<has_body_force,
                                                has_surface_pressure>(
                cell, scratch_data, per_task_data);

              m_constraints.distribute_local_to_global(
                per_task_data.m_cell_rhs,
                per_task_data.m_local_dof_indices,
                system_rhs.base());
            }


        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        system_rhs.compress(VectorOperation::add);
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
      }

    m_timer.leave_subsection(sectionName);
  }

  template <typename LATraits, typename Tria>
  template <bool has_body_force, bool has_surface_pressure>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_rhs_BFGS_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM_RHS_BFGS                             &scratch,
    PerTaskData_ASM_RHS_BFGS                             &data) const
  {
    data.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_previous_step.relevance(),
      scratch.m_phasefield_previous_step_cell);

    const auto &lqph = m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());



    if constexpr (has_body_force)
      {
        right_hand_side(
          m_n_q_points,
          scratch.rhs_values,
          m_parameters.m_x_component * 1.0,
          m_parameters.m_y_component * 1.0,
          m_parameters.m_z_component * 1.0);
      }
    const double inv_delta_time = m_time.get_inv_delta_t();


    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        const MaterialConstants<dim> &matConst =
          lqph[q_point]->get_material_constents();

        const double history_strain_energy =
          lqph[q_point]->get_history_max_positive_strain_energy();

        const double current_positive_strain_energy =
          lqph[q_point]->get_current_positive_strain_energy();


        double history_value = history_strain_energy;
        if (current_positive_strain_energy > history_strain_energy)
          history_value = current_positive_strain_energy;

        const double phasefield_value = lqph[q_point]->get_phase_field_value();
        const Tensor<1, dim> &phasefield_grad =
          lqph[q_point]->get_phase_field_gradient();

        
        const SymmetricTensor<2, dim> &cauchy_stress =
          lqph[q_point]->get_cauchy_stress();

        
        const double JxW = scratch.m_fe_values.JxW(q_point);

        scratch.m_d_pf_values.init(matConst, phasefield_value);

        const double phasefield_geo_derivative =
          scratch.m_d_pf_values.geometric();
        

        const double diff_degradation = scratch.m_d_pf_values.degradation();
        

        const Tensor<1, dim> coefGradN =
          matConst.two_gc_l0_over_c_alpha * phasefield_grad;



        const double viscosity_term =
          (matConst.has_viscosity) ?
            (matConst.viscosity * inv_delta_time *
             (phasefield_value -
              scratch.m_phasefield_previous_step_cell[q_point])) :
            0.0;

        const double coefN =
          viscosity_term +
          diff_degradation * history_value
          + matConst.gc_over_c_alpha_l0 * phasefield_geo_derivative;

        for (const unsigned int i : m_u_local_dofs)
          {
            const SymmetricTensor<2, dim> symm_grad_N_i =
              symmetrize(scratch.m_fe_values[m_u_fe].gradient(i, q_point));

            data.m_cell_rhs(i) += symm_grad_N_i * cauchy_stress * JxW;


            // contributions from the body force to right-hand side
            if constexpr (has_body_force)
              {
                const Tensor<1, dim> N_i =
                  scratch.m_fe_values[m_u_fe].value(i, q_point);

                data.m_cell_rhs(i) -= N_i * scratch.rhs_values[q_point] * JxW;
              }
          }

        for (const unsigned int i : m_d_local_dofs)
          {
            const double N_i = scratch.m_fe_values[m_d_fe].value(i, q_point);

            const Tensor<1, dim> grad_N_i =
              scratch.m_fe_values[m_d_fe].gradient(i, q_point);

            data.m_cell_rhs(i) += (coefGradN * grad_N_i + coefN * N_i) * JxW;
          }
      } // q_point

    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration


    if constexpr (has_surface_pressure)
      {
        const double       time_ramp        = m_time.timeRamp();
        const double       p0               = m_parameters.m_pressure;
        const unsigned int face_pressure_id = m_parameters.m_face_id;
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() && face->boundary_id() == face_pressure_id)
            {
              scratch.m_fe_face_values.reinit(cell, face);

              for (const unsigned int f_q_point :
                   scratch.m_fe_face_values.quadrature_point_indices())
                {
                  const Tensor<1, dim> &N =
                    scratch.m_fe_face_values.normal_vector(f_q_point);

                  const double          pressure = p0 * time_ramp;
                  const Tensor<1, dim> &traction = pressure * N;

                  for (const unsigned int i : scratch.m_fe_values.dof_indices())
                    {
                      const unsigned int i_group =
                        m_fe.system_to_base_index(i).first.first;

                      if (i_group == m_u_dof)
                        {
                          const unsigned int component_i =
                            m_fe.system_to_component_index(i).first;
                          const double Ni =
                            scratch.m_fe_face_values.shape_value(i, f_q_point);
                          const double JxW =
                            scratch.m_fe_face_values.JxW(f_q_point);
                          data.m_cell_rhs(i) -=
                            (Ni * traction[component_i]) * JxW;
                        }
                    }
                }
            }
      }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_newton_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM                                      &scratch,
    PerTaskData_ASM                                      &data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_previous_step.relevance(),
      scratch.m_phasefield_previous_step_cell);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double                time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);

    right_hand_side(scratch.m_fe_values.get_quadrature_points(),
                    scratch.rhs_values,
                    m_parameters.m_x_component * 1.0,
                    m_parameters.m_y_component * 1.0,
                    m_parameters.m_z_component * 1.0);

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
            const unsigned int k_group =
              m_fe.system_to_base_index(k).first.first;

            if (k_group == m_u_dof)
              {
                scratch.m_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].value(k, q_point);
                scratch.m_grad_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].gradient(k, q_point);
                scratch.m_symm_grad_Nx_disp[q_point][k] =
                  symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
              }
            else if (k_group == m_d_dof)
              {
                scratch.m_Nx_phasefield[q_point][k] =
                  scratch.m_fe_values[m_d_fe].value(k, q_point);
                scratch.m_grad_Nx_phasefield[q_point][k] =
                  scratch.m_fe_values[m_d_fe].gradient(k, q_point);
              }
            else
              Assert(k_group <= m_d_dof, ExcInternalError());
          }
      }

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        const double length_scale = lqph[q_point]->get_length_scale();
        const double gc  = lqph[q_point]->get_critical_energy_release_rate();
        const double eta = lqph[q_point]->get_viscosity();
        const double history_strain_energy =
          lqph[q_point]->get_history_max_positive_strain_energy();
        const double current_positive_strain_energy =
          lqph[q_point]->get_current_positive_strain_energy();
        const double p  = lqph[q_point]->get_p();
        const double a1 = lqph[q_point]->get_a1();
        const double a2 = lqph[q_point]->get_a2();
        const double a3 = lqph[q_point]->get_a3();

        double history_value = history_strain_energy;
        if (current_positive_strain_energy > history_strain_energy)
          history_value = current_positive_strain_energy;

        const double phasefield_value = lqph[q_point]->get_phase_field_value();
        const Tensor<1, dim> phasefield_grad =
          lqph[q_point]->get_phase_field_gradient();

        const std::vector<double> &N_phasefield =
          scratch.m_Nx_phasefield[q_point];
        const std::vector<Tensor<1, dim>> &grad_N_phasefield =
          scratch.m_grad_Nx_phasefield[q_point];
        const double old_phasefield =
          scratch.m_phasefield_previous_step_cell[q_point];

        const SymmetricTensor<2, dim> &cauchy_stress =
          lqph[q_point]->get_cauchy_stress();
        const SymmetricTensor<2, dim> &cauchy_stress_positive =
          lqph[q_point]->get_cauchy_stress_positive();
        const SymmetricTensor<4, dim> &mechanical_C =
          lqph[q_point]->get_mechanical_C();

        const std::vector<Tensor<1, dim>> &N_disp = scratch.m_Nx_disp[q_point];
        const std::vector<SymmetricTensor<2, dim>> &symm_grad_N_disp =
          scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;

        const double phasefield_geo_derivative =
          phasefield_geometry_function_derivative(
            phasefield_value, m_parameters.m_phasefield_name);

        const double phasefield_geo_2nd_order_derivative =
          phasefield_geometry_function_2nd_order_derivative(
            phasefield_value, m_parameters.m_phasefield_name);

        const double phasefield_coeff_const =
          phasefield_coefficient_constant(m_parameters.m_phasefield_name);

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            const unsigned int i_group =
              m_fe.system_to_base_index(i).first.first;

            if (i_group == m_u_dof)
              {
                data.m_cell_rhs(i) -=
                  (symm_grad_N_disp[i] * cauchy_stress) * JxW;

                // contributions from the body force to right-hand side
                data.m_cell_rhs(i) += N_disp[i] * rhs_values[q_point] * JxW;
              }
            else if (i_group == m_d_dof)
              {
                data.m_cell_rhs(i) -=
                  (2.0 / phasefield_coeff_const * gc * length_scale *
                     grad_N_phasefield[i] * phasefield_grad +
                   (gc / length_scale / phasefield_coeff_const *
                      phasefield_geo_derivative +
                    eta / delta_time * (phasefield_value - old_phasefield) +
                    degradation_function_derivative(
                      phasefield_value,
                      p,
                      a1,
                      a2,
                      a3,
                      m_parameters.m_phasefield_name) *
                      history_value) *
                     N_phasefield[i]) *
                  JxW;
              }
            else
              Assert(i_group <= m_d_dof, ExcInternalError());

            if (i_group == m_u_dof)
              {
                symm_grad_Nx_i_x_C = symm_grad_N_disp[i] * mechanical_C;
              }

            for (const unsigned int j : scratch.m_fe_values.dof_indices())
              {
                const unsigned int j_group =
                  m_fe.system_to_base_index(j).first.first;

                if ((i_group == j_group) && (i_group == m_u_dof))
                  {
                    data.m_cell_matrix(i, j) +=
                      symm_grad_Nx_i_x_C * symm_grad_N_disp[j] * JxW;
                  }
                else if ((i_group == j_group) && (i_group == m_d_dof))
                  {
                    data.m_cell_matrix(i, j) +=
                      ((gc / length_scale / phasefield_coeff_const *
                          phasefield_geo_2nd_order_derivative +
                        eta / delta_time +
                        degradation_function_2nd_order_derivative(
                          phasefield_value,
                          p,
                          a1,
                          a2,
                          a3,
                          m_parameters.m_phasefield_name) *
                          history_value) *
                         N_phasefield[i] * N_phasefield[j] +
                       2.0 / phasefield_coeff_const * gc * length_scale *
                         grad_N_phasefield[i] * grad_N_phasefield[j]) *
                      JxW;
                  }
                else if ((i_group == m_u_dof) && (j_group == m_d_dof))
                  {
                    data.m_cell_matrix(i, j) +=
                      symm_grad_N_disp[i] * cauchy_stress_positive *
                      degradation_function_derivative(
                        phasefield_value,
                        p,
                        a1,
                        a2,
                        a3,
                        m_parameters.m_phasefield_name) *
                      N_phasefield[j] * JxW;
                  }
                else if ((i_group == m_d_dof) && (j_group == m_u_dof))
                  {
                    if (current_positive_strain_energy > history_strain_energy)
                      data.m_cell_matrix(i, j) +=
                        N_phasefield[i] *
                        degradation_function_derivative(
                          phasefield_value,
                          p,
                          a1,
                          a2,
                          a3,
                          m_parameters.m_phasefield_name) *
                        cauchy_stress_positive * symm_grad_N_disp[j] * JxW;
                    else
                      data.m_cell_matrix(i, j) += 0.0;
                  }
                else
                  Assert((i_group <= m_d_dof) && (j_group <= m_d_dof),
                         ExcInternalError());
              } // j
          }     // i
      }         // q_point

    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration
    const unsigned int face_pressure_id = 100;
    const double       p0               = 0.0;

    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_pressure_id)
        {
          scratch.m_fe_face_values.reinit(cell, face);

          for (const unsigned int f_q_point :
               scratch.m_fe_face_values.quadrature_point_indices())
            {
              const Tensor<1, dim> &N =
                scratch.m_fe_face_values.normal_vector(f_q_point);

              const double         pressure = p0 * time_ramp;
              const Tensor<1, dim> traction = pressure * N;

              for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
                  const unsigned int i_group =
                    m_fe.system_to_base_index(i).first.first;

                  if (i_group == m_u_dof)
                    {
                      const unsigned int component_i =
                        m_fe.system_to_component_index(i).first;
                      const double Ni =
                        scratch.m_fe_face_values.shape_value(i, f_q_point);
                      const double JxW =
                        scratch.m_fe_face_values.JxW(f_q_point);
                      data.m_cell_rhs(i) += (Ni * traction[component_i]) * JxW;
                    }
                }
            }
        }
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_B0_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM                                      &scratch,
    PerTaskData_ASM                                      &data) const
  {
    data.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    const auto &lqph = m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());


    const double inv_delta_time = m_time.get_inv_delta_t();

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : m_u_local_dofs)
          {
            scratch.m_symm_grad_Nx_disp[q_point][k] =
              symmetrize(scratch.m_fe_values[m_u_fe].gradient(k, q_point));
          }

        for (const unsigned int k : m_d_local_dofs)
          {
            scratch.m_Nx_phasefield[q_point][k] =
              scratch.m_fe_values[m_d_fe].value(k, q_point);

            scratch.m_grad_Nx_phasefield[q_point][k] =
              scratch.m_fe_values[m_d_fe].gradient(k, q_point);
          }
      }

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        const MaterialConstants<dim> &matConst =
          lqph[q_point]->get_material_constents();
        
        const double history_strain_energy =
          lqph[q_point]->get_history_max_positive_strain_energy();
        const double current_positive_strain_energy =
          lqph[q_point]->get_current_positive_strain_energy();


        double history_value = history_strain_energy;
        if (current_positive_strain_energy > history_strain_energy)
          history_value = current_positive_strain_energy;

        const double phasefield_value = lqph[q_point]->get_phase_field_value();

        const std::vector<double> &N_phasefield =
          scratch.m_Nx_phasefield[q_point];
        const std::vector<Tensor<1, dim>> &grad_N_phasefield =
          scratch.m_grad_Nx_phasefield[q_point];

        // const SymmetricTensor<2, dim> & cauchy_stress_positive =
        // lqph[q_point]->get_cauchy_stress_positive();
        const SymmetricTensor<4, dim> &mechanical_C =
          lqph[q_point]->get_mechanical_C();

        const std::vector<SymmetricTensor<2, dim>> &symm_grad_N_disp =
          scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;


        scratch.m_dd_pf_values.init(matConst, phasefield_value);


        const double phasefield_geo_2nd_order_derivative =
          scratch.m_dd_pf_values.geometric();
     

        const double ddiff_degradation = scratch.m_dd_pf_values.degradation();
     
        const double viscosity_term = (matConst.has_viscosity) ?
                                        (matConst.viscosity * inv_delta_time) :
                                        0.0;
        const double coefN =
          matConst.gc_over_c_alpha_l0 * phasefield_geo_2nd_order_derivative +
          viscosity_term + ddiff_degradation * history_value;


        const double coefGradN = matConst.two_gc_l0_over_c_alpha;
        

        const std::size_t n_u = m_u_local_dofs.size();
        for (unsigned int ii = 0; ii < n_u; ++ii)
          {
            const unsigned int i = m_u_local_dofs[ii];
            symm_grad_Nx_i_x_C   = symm_grad_N_disp[i] * mechanical_C;

            for (unsigned int jj = ii; jj < n_u; ++jj)
              {
                const unsigned int j = m_u_local_dofs[jj];

                const double value =
                  symm_grad_Nx_i_x_C * symm_grad_N_disp[j] * JxW;
                ;

                data.m_cell_matrix(i, j) += value;

                if (j != i)
                  data.m_cell_matrix(j, i) += value;
              }
          }

        const std::size_t n_d = m_d_local_dofs.size();

        for (unsigned int ii = 0; ii < n_d; ++ii)
          {
            const unsigned int i = m_d_local_dofs[ii];

            for (unsigned int jj = ii; jj < n_d; ++jj)
              {
                const unsigned int j = m_d_local_dofs[jj];

                const double value =
                  (coefN * N_phasefield[i] * N_phasefield[j] +
                   coefGradN * grad_N_phasefield[i] * grad_N_phasefield[j]) *
                  JxW;

                data.m_cell_matrix(i, j) += value;

                if (j != i)
                  data.m_cell_matrix(j, i) += value;
              }
          }
      } // q_point
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_rhs_BFGS(
    const BVector &solution_old,
    BVector       &system_rhs)
  {
    const std::string sectionName = "Assemble RHS";
    m_timer.enter_subsection(sectionName);

    // m_logfile << " A_RHS " << std::flush;

    system_rhs = 0.0;

    Vector<double>                       cell_rhs(m_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(m_dofs_per_cell);

    const double time_ramp  = (m_time.current() / m_time.end());
    const double delta_time = m_time.get_delta_t();

    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);
    const UpdateFlags           uf_cell(update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    const UpdateFlags           uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    FEValues<dim>     fe_values(m_fe, m_qf_cell, uf_cell);
    FEFaceValues<dim> fe_face_values(m_fe, m_qf_face, uf_face);

    // shape function values for displacement field
    std::vector<std::vector<Tensor<1, dim>>> Nx_disp(
      m_qf_cell.size(), std::vector<Tensor<1, dim>>(m_dofs_per_cell));
    std::vector<std::vector<Tensor<2, dim>>> grad_Nx_disp(
      m_qf_cell.size(), std::vector<Tensor<2, dim>>(m_dofs_per_cell));
    std::vector<std::vector<SymmetricTensor<2, dim>>> symm_grad_Nx_disp(
      m_qf_cell.size(), std::vector<SymmetricTensor<2, dim>>(m_dofs_per_cell));

    // shape function values for phase field
    std::vector<std::vector<double>> Nx_phasefield(
      m_qf_cell.size(), std::vector<double>(m_dofs_per_cell));
    std::vector<std::vector<Tensor<1, dim>>> grad_Nx_phasefield(
      m_qf_cell.size(), std::vector<Tensor<1, dim>>(m_dofs_per_cell));

    std::vector<double> phasefield_previous_step_cell(m_qf_cell.size());

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
          if (!cell->is_locally_owned())
            continue;
        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        cell_rhs = 0.0;
        fe_values.reinit(cell);
        right_hand_side(fe_values.get_quadrature_points(),
                        rhs_values,
                        m_parameters.m_x_component * time_ramp,
                        m_parameters.m_y_component * time_ramp,
                        m_parameters.m_z_component * time_ramp);

        fe_values[m_d_fe].get_function_values(solution_old.relevance(),
                                              phasefield_previous_step_cell);

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            for (const unsigned int k : fe_values.dof_indices())
              {
                const unsigned int k_group =
                  m_fe.system_to_base_index(k).first.first;

                if (k_group == m_u_dof)
                  {
                    Nx_disp[q_point][k] = fe_values[m_u_fe].value(k, q_point);
                    grad_Nx_disp[q_point][k] =
                      fe_values[m_u_fe].gradient(k, q_point);
                    symm_grad_Nx_disp[q_point][k] =
                      symmetrize(grad_Nx_disp[q_point][k]);
                  }
                else if (k_group == m_d_dof)
                  {
                    Nx_phasefield[q_point][k] =
                      fe_values[m_d_fe].value(k, q_point);
                    grad_Nx_phasefield[q_point][k] =
                      fe_values[m_d_fe].gradient(k, q_point);
                  }
                else
                  Assert(k_group <= m_d_dof, ExcInternalError());
              }
          }

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            const MaterialConstants<dim> &matConst =
              lqph[q_point]->get_material_constents();

            const double length_scale = lqph[q_point]->get_length_scale();
            const double gc = lqph[q_point]->get_critical_energy_release_rate();
            const double eta = lqph[q_point]->get_viscosity();
            const double history_strain_energy =
              lqph[q_point]->get_history_max_positive_strain_energy();
            const double current_positive_strain_energy =
              lqph[q_point]->get_current_positive_strain_energy();
            const double p  = lqph[q_point]->get_p();
            const double a1 = lqph[q_point]->get_a1();
            const double a2 = lqph[q_point]->get_a2();
            const double a3 = lqph[q_point]->get_a3();

            double history_value = history_strain_energy;
            if (current_positive_strain_energy > history_strain_energy)
              history_value = current_positive_strain_energy;

            const double phasefield_value =
              lqph[q_point]->get_phase_field_value();
            const Tensor<1, dim> phasefield_grad =
              lqph[q_point]->get_phase_field_gradient();

            const std::vector<double> &N_phasefield = Nx_phasefield[q_point];
            const std::vector<Tensor<1, dim>> &grad_N_phasefield =
              grad_Nx_phasefield[q_point];
            const double old_phasefield =
              phasefield_previous_step_cell[q_point];

            const SymmetricTensor<2, dim> &cauchy_stress =
              lqph[q_point]->get_cauchy_stress();

            const std::vector<Tensor<1, dim>>          &N = Nx_disp[q_point];
            const std::vector<SymmetricTensor<2, dim>> &symm_grad_N =
              symm_grad_Nx_disp[q_point];
            const double JxW = fe_values.JxW(q_point);

            const double phasefield_geo_derivative =
              phasefield_geometry_function_derivative(
                phasefield_value, m_parameters.m_phasefield_name);

            const double phasefield_coeff_const =
              phasefield_coefficient_constant(m_parameters.m_phasefield_name);

            for (const unsigned int i : fe_values.dof_indices())
              {
                const unsigned int i_group =
                  m_fe.system_to_base_index(i).first.first;

                if (i_group == m_u_dof)
                  {
                    cell_rhs(i) += (symm_grad_N[i] * cauchy_stress) * JxW;
                    // contributions from the body force to right-hand side
                    cell_rhs(i) -= N[i] * rhs_values[q_point] * JxW;
                  }
                else if (i_group == m_d_dof)
                  {
                    cell_rhs(i) +=
                      (2.0 / phasefield_coeff_const * gc * length_scale *
                         grad_N_phasefield[i] * phasefield_grad +
                       (gc / length_scale / phasefield_coeff_const *
                          phasefield_geo_derivative +
                        eta / delta_time * (phasefield_value - old_phasefield) +
                        degradation_function_derivative(
                          phasefield_value,
                          p,
                          a1,
                          a2,
                          a3,
                          m_parameters.m_phasefield_name) *
                          history_value) *
                         N_phasefield[i]) *
                      JxW;
                  }
                else
                  Assert(i_group <= m_d_dof, ExcInternalError());
              }
          }

        // if there is surface pressure, this surface pressure always applied to
        // the reference configuration
        const unsigned int face_pressure_id = 100;
        const double       p0               = 0.0;

        for (const auto &face : cell->face_iterators())
          {
            if (face->at_boundary() && face->boundary_id() == face_pressure_id)
              {
                fe_face_values.reinit(cell, face);

                for (const unsigned int f_q_point :
                     fe_face_values.quadrature_point_indices())
                  {
                    const Tensor<1, dim> &N =
                      fe_face_values.normal_vector(f_q_point);

                    const double         pressure = p0 * time_ramp;
                    const Tensor<1, dim> traction = pressure * N;

                    for (const unsigned int i : fe_values.dof_indices())
                      {
                        const unsigned int i_group =
                          m_fe.system_to_base_index(i).first.first;

                        if (i_group == m_u_dof)
                          {
                            const unsigned int component_i =
                              m_fe.system_to_component_index(i).first;
                            const double Ni =
                              fe_face_values.shape_value(i, f_q_point);
                            const double JxW = fe_face_values.JxW(f_q_point);
                            cell_rhs(i) -= (Ni * traction[component_i]) * JxW;
                          }
                      }
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
      } // for (const auto &cell : m_dof_handler.active_cell_iterators())

    m_timer.leave_subsection(sectionName);
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::update_history_field_step()
  {
    m_logfile << "\t\tUpdate history variable" << std::endl;

    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
          if (!cell->is_locally_owned())
            continue;
        std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            lqph[q_point]->update_history_variable();
          }
      }
  }

  template <typename LATraits, typename Tria>
  double
  PhaseFieldMonolithicSolve<LATraits, Tria>::
    line_search_stepsize_gradient_based(const BVector &BFGS_p_vector,
                                        const BVector &solution_delta,
                                        unsigned int  &num_ls)
  {
    auto     g_old_handle = m_vec_pool.getHandle(false);
    BVector &g_old        = g_old_handle.get();
    g_old.base()          = m_system_rhs.base();

    // BFGS_p_vector is the search direction
    auto     solution_delta_trial_handle = m_vec_pool_ghosted.getHandle(false);
    BVector &solution_delta_trial        = solution_delta_trial_handle.get();
    solution_delta_trial.base()          = solution_delta.base();        
    // take a full step size 1.0
    solution_delta_trial.add(1.0, BFGS_p_vector);
    solution_delta_trial.updateRelevance();


 
    auto     g_new_handle = m_vec_pool.getHandle(false);
    BVector &g_new        = g_new_handle.get();

    if (m_parameters.has_body_force && m_parameters.has_surface_pressure)
      assemble_system_rhs_BFGS_parallel<true, true>(m_solution, g_new);
    else if (m_parameters.has_body_force)
      assemble_system_rhs_BFGS_parallel<true, false>(m_solution, g_new);
    else if (m_parameters.has_surface_pressure)
      assemble_system_rhs_BFGS_parallel<false, true>(m_solution, g_new);
    else
      assemble_system_rhs_BFGS_parallel<false, false>(m_solution, g_new);



    auto     y_old_handle = m_vec_pool.getHandle(false);
    BVector &y_old        = y_old_handle.get();

    y_old.base() = g_new.base() - g_old.base();

    double alpha = 1.0;

    double alpha_old = 0.0;

    double delta_alpha_old = alpha - alpha_old;

    double delta_alpha_new;

    unsigned int ls_max = 10;

    unsigned int i = 1;

    for (; i <= ls_max; ++i)
      {
        delta_alpha_new =
          -delta_alpha_old * (g_new * BFGS_p_vector) / (y_old * BFGS_p_vector);
        alpha += delta_alpha_new;

        if (std::fabs(delta_alpha_new) < 1.0e-5)
          break;

        if (i == ls_max)
          {
            // alpha = 1.0;
            break;
          }

        g_old = g_new;

        // BFGS_p_vector is the search direction
        solution_delta_trial = solution_delta;
        solution_delta_trial.add(alpha, BFGS_p_vector);
        solution_delta_trial.updateRelevance();


        if (m_parameters.has_body_force && m_parameters.has_surface_pressure)
          assemble_system_rhs_BFGS_parallel<true, true>(m_solution, g_new);
        else if (m_parameters.has_body_force)
          assemble_system_rhs_BFGS_parallel<true, false>(m_solution, g_new);
        else if (m_parameters.has_surface_pressure)
          assemble_system_rhs_BFGS_parallel<false, true>(m_solution, g_new);
        else
          assemble_system_rhs_BFGS_parallel<false, false>(m_solution, g_new);


        y_old.base() = g_new.base() - g_old.base();

        delta_alpha_old = delta_alpha_new;
      }

    // if (alpha < 1.0e-3)
    //   alpha = 1.0;

    num_ls = i;
    return alpha;
  }

  template <typename LATraits, typename Tria>
  double
  PhaseFieldMonolithicSolve<LATraits, Tria>::line_search_stepsize_strong_wolfe(
    const double   phi_0,
    const double   phi_0_prime,
    const BVector &BFGS_p_vector,
    const BVector &solution_delta,
    unsigned int  &num_ls)
  {
    // AssertThrow(phi_0_prime < 0,
    //             ExcMessage("The derivative of phi at alpha = 0 should be
    //             negative!"));

    // Some line search parameters
    const double       c1        = 0.0001;
    const double       c2        = 0.9;
    const double       alpha_max = 100.0;
    const unsigned int max_iter  = 20;
    double             alpha     = 1.0;

    double phi_old       = phi_0;
    double phi_prime_old = phi_0_prime;
    double alpha_old     = 0.0;

    double phi, phi_prime;

    std::pair<double, double> current_phi_phi_prime;

    unsigned int i = 1;
    for (; i <= max_iter; ++i)
      {
        current_phi_phi_prime =
          calculate_phi_and_phi_prime(alpha, BFGS_p_vector, solution_delta);
        phi       = current_phi_phi_prime.first;
        phi_prime = current_phi_phi_prime.second;

        if ((phi > (phi_0 + c1 * alpha * phi_0_prime)) ||
            (i > 0 && phi > phi_old))
          {
            num_ls = i;
            return line_search_zoom_strong_wolfe(phi_old,
                                                 phi_prime_old,
                                                 alpha_old,
                                                 phi,
                                                 phi_prime,
                                                 alpha,
                                                 phi_0,
                                                 phi_0_prime,
                                                 BFGS_p_vector,
                                                 c1,
                                                 c2,
                                                 max_iter,
                                                 solution_delta);
          }

        if (std::fabs(phi_prime) <= c2 * std::fabs(phi_0_prime))
          {
            num_ls = i;
            return alpha;
          }

        if (phi_prime >= 0)
          {
            num_ls = i;
            return line_search_zoom_strong_wolfe(phi,
                                                 phi_prime,
                                                 alpha,
                                                 phi_old,
                                                 phi_prime_old,
                                                 alpha_old,
                                                 phi_0,
                                                 phi_0_prime,
                                                 BFGS_p_vector,
                                                 c1,
                                                 c2,
                                                 max_iter,
                                                 solution_delta);
          }

        phi_old       = phi;
        phi_prime_old = phi_prime;
        alpha_old     = alpha;

        alpha = std::min(2.0 * alpha, alpha_max);

        // AssertThrow(alpha < alpha_max,
        //	    ExcMessage("alpha is bigger than alpha_max, line search
        // failed!"));
      }

    // AssertThrow(i < max_iter,
    //             ExcMessage("max number attempts arrived, line search
    //             failed!"));
    //  Instead of terminating the program, we can just take a full step.
    if (i == max_iter)
      alpha = 1.0;

    num_ls = i;
    return alpha;
  }

  template <typename LATraits, typename Tria>
  double
  PhaseFieldMonolithicSolve<LATraits, Tria>::line_search_zoom_strong_wolfe(
    double         phi_low,
    double         phi_low_prime,
    double         alpha_low,
    double         phi_high,
    double         phi_high_prime,
    double         alpha_high,
    double         phi_0,
    double         phi_0_prime,
    const BVector &BFGS_p_vector,
    double         c1,
    double         c2,
    unsigned int   max_iter,
    const BVector &solution_delta)
  {
    double                    alpha = 0;
    std::pair<double, double> current_phi_phi_prime;
    double                    phi, phi_prime;

    unsigned int i = 0;
    for (; i < max_iter; ++i)
      {
        // a simple bisection is faster than cubic interpolation
        alpha = 0.5 * (alpha_low + alpha_high);
        // alpha = line_search_interpolation_cubic(alpha_low, phi_low,
        // phi_low_prime, 					alpha_high, phi_high, phi_high_prime);
        current_phi_phi_prime =
          calculate_phi_and_phi_prime(alpha, BFGS_p_vector, solution_delta);
        phi       = current_phi_phi_prime.first;
        phi_prime = current_phi_phi_prime.second;

        if ((phi > phi_0 + c1 * alpha * phi_0_prime) || (phi > phi_low))
          {
            alpha_high     = alpha;
            phi_high       = phi;
            phi_high_prime = phi_prime;
          }
        else
          {
            if (std::fabs(phi_prime) <= c2 * std::fabs(phi_0_prime))
              {
                // if (alpha < 1.0e-3)
                //   alpha = 1.0e-3;
                return alpha;
              }

            if (phi_prime * (alpha_high - alpha_low) >= 0.0)
              {
                alpha_high     = alpha_low;
                phi_high_prime = phi_low_prime;
                phi_high       = phi_low;
              }

            alpha_low     = alpha;
            phi_low_prime = phi_prime;
            phi_low       = phi;
          }
      }

    if (alpha < 1.0e-3)
      alpha = 1.0;

    // avoid unused variable warnings from compiler
    (void)phi_high;
    (void)phi_high_prime;
    return alpha;
  }

  template <typename LATraits, typename Tria>
  double
  PhaseFieldMonolithicSolve<LATraits, Tria>::line_search_interpolation_cubic(
    const double alpha_0,
    const double phi_0,
    const double phi_0_prime,
    const double alpha_1,
    const double phi_1,
    const double phi_1_prime)
  {
    const double d1 =
      phi_0_prime + phi_1_prime - 3.0 * (phi_0 - phi_1) / (alpha_0 - alpha_1);

    const double temp = d1 * d1 - phi_0_prime * phi_1_prime;

    if (temp < 0.0)
      return 0.5 * (alpha_0 + alpha_1);

    int sign;
    if (alpha_1 > alpha_0)
      sign = 1;
    else
      sign = -1;

    const double d2 = sign * std::sqrt(temp);

    const double alpha = alpha_1 - (alpha_1 - alpha_0) *
                                     (phi_1_prime + d2 - d1) /
                                     (phi_1_prime - phi_0_prime + 2 * d2);

    if ((alpha_1 > alpha_0) && (alpha > alpha_1 || alpha < alpha_0))
      return 0.5 * (alpha_0 + alpha_1);

    if ((alpha_0 > alpha_1) && (alpha > alpha_0 || alpha < alpha_1))
      return 0.5 * (alpha_0 + alpha_1);

    return alpha;
  }

  template <typename LATraits, typename Tria>
  std::pair<double, double>
  PhaseFieldMonolithicSolve<LATraits, Tria>::calculate_phi_and_phi_prime(
    const double   alpha,
    const BVector &BFGS_p_vector,
    const BVector &solution_delta)
  {
    // the first component is phi(alpha), the second component is
    // phi_prime(alpha),
    std::pair<double, double> phi_values;

    auto     solution_delta_trial_handle = m_vec_pool_ghosted.getHandle(false);
    BVector &solution_delta_trial        = solution_delta_trial_handle.get();
    solution_delta_trial.base()          = solution_delta.base();
    solution_delta_trial.add(alpha, BFGS_p_vector);

    solution_delta_trial.updateRelevance();


    auto     system_rhs_handle = m_vec_pool.getHandle(false);
    BVector &system_rhs        = system_rhs_handle.get();


    if (m_parameters.has_body_force && m_parameters.has_surface_pressure)
      assemble_system_rhs_BFGS_parallel<true, true>(m_solution, system_rhs);
    else if (m_parameters.has_body_force)
      assemble_system_rhs_BFGS_parallel<true, false>(m_solution, system_rhs);
    else if (m_parameters.has_surface_pressure)
      assemble_system_rhs_BFGS_parallel<false, true>(m_solution, system_rhs);
    else
      assemble_system_rhs_BFGS_parallel<false, false>(m_solution, system_rhs);


    phi_values.first  = calculate_energy_functional();
    phi_values.second = system_rhs * BFGS_p_vector;
    return phi_values;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::LBFGS_B0(BVector &LBFGS_r_vector,
                                                      BVector &LBFGS_q_vector)
  {
    const std::string sectionName = "Solve B0";
    m_timer.enter_subsection(sectionName);

    LBFGS_r_vector = 0.0;
    assemble_system_B0(m_solution);

    m_diag_la_solver.solve(LBFGS_r_vector, LBFGS_q_vector, m_tangent_matrix);
    LBFGS_r_vector.updateRelevance();


    m_timer.leave_subsection(sectionName);
  }

  template <typename LATraits, typename Tria>
  std::vector<double>
  PhaseFieldMonolithicSolve<LATraits, Tria>::solve_linear_system(
    BVector & /*newton_update*/)
  {
    const std::string sectionName = "Solve coupled linear system";
    m_timer.enter_subsection(sectionName);

    if (m_parameters.m_output_iteration_history)
      m_logfile << " SLV " << std::flush;

    std::vector<double> linear_solver_parameters(3);
    /*
     {
     SolverControl            solver_control(1e6, 1e-9);
     SolverCG<Vector<double>> cg(solver_control);
     cg.connect_condition_number_slot(
     [&] (double condition_number)
     {
     linear_solver_parameters[0] = condition_number;
     //m_logfile << "   Estimated condition number = "<< condition_number <<
     std::endl;
     },
     false);

     PreconditionSSOR<SparseMatrix<double>> preconditioner;
     preconditioner.initialize(m_system_matrix_displacement, 1.2);

     cg.solve(m_system_matrix_displacement,
     newton_update,
     m_system_rhs_displacement,
     preconditioner);

     //m_logfile << "   " << solver_control.last_step()
     //<< " CG iterations needed to obtain convergence." << std::endl;
     linear_solver_parameters[1] = solver_control.last_step();
     linear_solver_parameters[2] = solver_control.last_value();
     }

     SparseDirectUMFPACK A_direct;
     A_direct.initialize(m_tangent_matrix);
     A_direct.vmult(newton_update,
     m_system_rhs);

     m_constraints.distribute(newton_update);
     */
    m_timer.leave_subsection(sectionName);
    return linear_solver_parameters;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::print_conv_header_newton()
  {
    static const unsigned int l_width = 135;
    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "                  SOLVER STEP (Newton)      "
              << " |        Cond No.   Lin_Iter   Lin_Res    Res_Norm  "
              << " Res_u      Res_d      Inc_Norm  "
              << " Inc_u      Inc_d" << std::endl;

    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::print_conv_header_BFGS()
  {
    static const unsigned int l_width = 125;
    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "                  SOLVER STEP (BFGS)   "
              << " |    Line Search alpha    Energy    Res_Norm    "
              << " Res_u      Res_d    Inc_Norm    "
              << " Inc_u      Inc_d" << std::endl;

    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::print_conv_header_LBFGS()
  {
    static const unsigned int l_width = 128;
    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "          SOLVER STEP (LBFGS)  "
              << "   |  LS-alpha    LS-N    Energy      Res_Norm    "
              << " Res_u      Res_d    Inc_Norm   "
              << " Inc_u      Inc_d" << std::endl;

    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <typename LATraits, typename Tria>
  bool
  PhaseFieldMonolithicSolve<LATraits, Tria>::solve_nonlinear_timestep_newton(
    BVector & /*solution_delta*/)
  {
    /*
     BVector newton_update(m_dofs_per_block);

     m_error_residual.reset();
     m_error_residual_0.reset();
     m_error_residual_norm.reset();
     m_error_update.reset();
     m_error_update_0.reset();
     m_error_update_norm.reset();

     if (m_parameters.m_output_iteration_history)
     print_conv_header_newton();

     unsigned int newton_iteration = 0;
     for (; newton_iteration < m_parameters.m_max_iterations_NR;
     ++newton_iteration)
     {
     if (m_parameters.m_output_iteration_history)
     m_logfile << '\t' << '\t' << std::setw(2) << newton_iteration << ' '
     << std::flush;

     make_constraints(newton_iteration);
     assemble_system_newton(m_solution);

     get_error_residual(m_error_residual);
     if (newton_iteration == 0)
     m_error_residual_0 = m_error_residual;

     m_error_residual_norm = m_error_residual;
     m_error_residual_norm.normalize(m_error_residual_0);

     if (newton_iteration > 0 && m_error_update_norm.m_u <=
     m_parameters.m_tol_u_incr
     && m_error_residual_norm.m_u <= m_parameters.m_tol_u_residual
     && m_error_update_norm.m_d <= m_parameters.m_tol_d_incr
     && m_error_residual_norm.m_d <= m_parameters.m_tol_d_residual)
     {
     if (m_parameters.m_output_iteration_history)
     {
     m_logfile << " CONVERGED!";
     m_logfile << " | " << std::fixed << std::setprecision(3) << std::setw(7)
     << std::scientific
     << "  " << "  ----   "
     << "  " << "  ----   "
     << "  " << "  ----   "
     << "  " << m_error_residual_norm.m_norm
     << "  " << m_error_residual_norm.m_u
     << "  " << m_error_residual_norm.m_d
     << "  " << m_error_update_norm.m_norm
     << "  " << m_error_update_norm.m_u
     << "  " << m_error_update_norm.m_d
     << "  " << std::endl;

     m_logfile << '\t' << '\t';
     for (unsigned int i = 0; i < 135; ++i)
     m_logfile << '_';
     m_logfile << std::endl;
     }

     m_logfile << "\t\tConvergence is reached after "
     << newton_iteration << " Newton iterations."<< std::endl;

     m_logfile << "\t\tResidual information of convergence:" << std::endl;

     m_logfile << "\t\t\tRelative residual of disp. equation: "
     << m_error_residual_norm.m_u << std::endl;

     m_logfile << "\t\t\tAbsolute residual of disp. equation: "
     << m_error_residual_norm.m_u * m_error_residual_0.m_u << std::endl;

     m_logfile << "\t\t\tRelative residual of phasefield equation: "
     << m_error_residual_norm.m_d << std::endl;

     m_logfile << "\t\t\tAbsolute residual of phasefield equation: "
     << m_error_residual_norm.m_d * m_error_residual_0.m_d << std::endl;

     m_logfile << "\t\t\tRelative increment of disp.: "
     << m_error_update_norm.m_u << std::endl;

     m_logfile << "\t\t\tAbsolute increment of disp.: "
     << m_error_update_norm.m_u * m_error_update_0.m_u << std::endl;

     m_logfile << "\t\t\tRelative increment of phasefield: "
     << m_error_update_norm.m_d << std::endl;

     m_logfile << "\t\t\tAbsolute increment of phasefield: "
     << m_error_update_norm.m_d * m_error_update_0.m_d << std::endl;

     //break;
     return true;
     }

     std::vector<double> linear_solver_parameters(3);

     linear_solver_parameters = solve_linear_system(newton_update);

     get_error_update(newton_update, m_error_update);
     if (newton_iteration == 0)
     m_error_update_0 = m_error_update;

     m_error_update_norm = m_error_update;
     m_error_update_norm.normalize(m_error_update_0);

     solution_delta += newton_update;
     update_qph_incremental(solution_delta, m_solution, true);

     if (m_parameters.m_output_iteration_history)
     {
     m_logfile << " | " << std::fixed << std::setprecision(3) << std::setw(7)
     << std::scientific
     << "  " << linear_solver_parameters[0]
     << "  " << linear_solver_parameters[1]
     << "  " << linear_solver_parameters[2]
     << "  " << m_error_residual_norm.m_norm
     << "  " << m_error_residual_norm.m_u
     << "  " << m_error_residual_norm.m_d
     << "  " << m_error_update_norm.m_norm
     << "  " << m_error_update_norm.m_u
     << "  " << m_error_update_norm.m_d
     << "  " << std::endl;
     }
     }

     //AssertThrow(newton_iteration < m_parameters.m_max_iterations_NR,
     //            ExcMessage("No convergence in Newton-Raphson nonlinear
     solver!"));
     */
    return false;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::solve_nonlinear_timestep_BFGS(
    BVector & /*solution_delta*/)
  {
    AssertThrow(
      false, ExcMessage("BFGS requires too much memory. Please use L-BFGS!"));
    /*
     BVector BFGS_update(m_dofs_per_block);

     m_error_residual.reset();
     m_error_residual_0.reset();
     m_error_residual_norm.reset();
     m_error_update.reset();
     m_error_update_0.reset();
     m_error_update_norm.reset();

     print_conv_header_BFGS();

     unsigned int BFGS_iteration = 0;

     // Initial guess B_0, which is a full matrix and takes a lot of memory
     FullMatrix<double> BFGS_matrix = IdentityMatrix(m_dof_handler.n_dofs());
     Vector<double> BFGS_r_vector(m_dof_handler.n_dofs());
     Vector<double> BFGS_p_vector(m_dof_handler.n_dofs());
     Vector<double> BFGS_y_vector(m_dof_handler.n_dofs());
     Vector<double> BFGS_temp_vector(m_dof_handler.n_dofs());

     double line_search_parameter, rho;

     // Most likely, we will not be able to create a second full matrix since
     // we will run out of memory on a laptop workstation
     FullMatrix<double> temp_matrix_1(m_dof_handler.n_dofs());
     FullMatrix<double> temp_matrix_2(m_dof_handler.n_dofs());

     for (; BFGS_iteration < m_parameters.m_max_iterations_BFGS;
     ++BFGS_iteration)
     {
     m_logfile << '\t' << '\t' << std::setw(2) << BFGS_iteration << ' '
     << std::flush;

     make_constraints(BFGS_iteration);

     // At the first step, we simply distribute the inhomogeneous part of
     // the constraints
     if (BFGS_iteration == 0)
     {
     m_constraints.distribute(BFGS_update);
     solution_delta += BFGS_update;
     m_logfile << " --- " << std::flush;
     m_logfile << " --- " << std::flush;
     update_qph_incremental(solution_delta, m_solution, false);
     m_logfile << " ---  |" << std::flush;
     m_logfile << std::endl;
     continue;
     }
     else if (BFGS_iteration == 1)
     {
     // Calculate the residual vector r. NOTICE that in the context of
     // BFGS, this r is the gradient of the energy functional (objective
     function),
     // NOT the negative gradient of the energy functional
     assemble_system_rhs_BFGS(m_solution, m_system_rhs);

     // We cannot simply zero out the dofs that are constrained, since we might
     // have hanging node constraints. In this case, we need to modify the RHS
     // as C^T * b, which C contains entries of 0.5 (x_3 = 0.5*x_1 + 0.5*x_2)
     //for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
     //if (m_constraints.is_constrained(i))
     //m_system_rhs(i) = 0.0;

     // if m_constraints has inhomogeneity, we cannot call
     m_constraints.condense(m_system_rhs),
     // since the m_system_matrix needs to be provided to modify the RHS
     properly. However, this
     // error will not be detected in the release mode and only will be detected
     on the debug mode m_constraints.condense(m_system_rhs);
     }

     m_logfile << " --- " << std::flush;
     m_logfile << " --- " << std::flush;
     m_logfile << " --- " << std::flush;

     get_error_residual(m_error_residual);
     if (BFGS_iteration == 1)
     m_error_residual_0 = m_error_residual;

     m_error_residual_norm = m_error_residual;
     m_error_residual_norm.normalize(m_error_residual_0);

     if (BFGS_iteration > 1 && m_error_update_norm.m_u <=
     m_parameters.m_tol_u_incr
     && m_error_residual_norm.m_u <= m_parameters.m_tol_u_residual
     && m_error_update_norm.m_d <= m_parameters.m_tol_d_incr
     && m_error_residual_norm.m_d <= m_parameters.m_tol_d_residual)
     {
     m_logfile << " CONVERGED!";
     m_logfile << "| " << std::fixed << std::setprecision(3) << std::setw(7)
     << std::scientific
     << "  " << "  ----   "
     << "  " << "  ----   "
     << "  " << "  ----   "
     << "  " << m_error_residual_norm.m_norm
     << "  " << m_error_residual_norm.m_u
     << "  " << m_error_residual_norm.m_d
     << "  " << m_error_update_norm.m_norm
     << "  " << m_error_update_norm.m_u
     << "  " << m_error_update_norm.m_d
     << "  " << std::endl;

     m_logfile << '\t' << '\t';
     for (unsigned int i = 0; i < 135; ++i)
     m_logfile << '_';
     m_logfile << std::endl;

     break;
     }

     // BFGS algorithm
     BFGS_r_vector = m_system_rhs;
     BFGS_matrix.vmult(BFGS_p_vector, BFGS_r_vector);
     BFGS_p_vector *= -1.0;
     m_constraints.distribute(BFGS_p_vector);

     // We need a line search algorithm to decide line_search_parameter
     const double phi_0 = calculate_energy_functional();
     const double phi_0_prime = BFGS_r_vector * BFGS_p_vector;

     BlockVector<double> BFGS_p_vector_block(m_dofs_per_block);
     BFGS_p_vector_block = BFGS_p_vector;
     unsigned int num_line_search = 0;
     line_search_parameter = line_search_stepsize_strong_wolfe(phi_0,
     phi_0_prime,
     BFGS_p_vector_block,
     solution_delta,
     num_line_search);

     BFGS_p_vector *= line_search_parameter;
     BFGS_update = BFGS_p_vector;

     get_error_update(BFGS_update, m_error_update);
     if (BFGS_iteration == 1)
     m_error_update_0 = m_error_update;

     m_error_update_norm = m_error_update;
     m_error_update_norm.normalize(m_error_update_0);

     solution_delta += BFGS_update;
     update_qph_incremental(solution_delta, m_solution, false);

     BFGS_y_vector = m_system_rhs;
     BFGS_y_vector *= -1.0;
     assemble_system_rhs_BFGS(m_solution, m_system_rhs);
     m_constraints.condense(m_system_rhs);
     BFGS_temp_vector = m_system_rhs;
     BFGS_y_vector += BFGS_temp_vector;

     // rho should be positive with the proper line search
     rho = BFGS_y_vector * BFGS_p_vector;
     rho = 1.0/rho;

     if (rho < 0)
     m_logfile << "Rho is negative!" << std::endl;

     // In the first step, we scale the identity matrix as
     // the BFGS matrix
     if (BFGS_iteration == 1)
     {
     double scale_parameter = (BFGS_y_vector * BFGS_p_vector) /
     (BFGS_y_vector.norm_sqr()); BFGS_matrix *= scale_parameter;
     }

     temp_matrix_1.outer_product(BFGS_p_vector, BFGS_y_vector);
     temp_matrix_2 = IdentityMatrix(m_dof_handler.n_dofs());
     temp_matrix_2.add(-rho, temp_matrix_1);

     temp_matrix_2.mmult(temp_matrix_1, BFGS_matrix);
     temp_matrix_1.mTmult(BFGS_matrix, temp_matrix_2);

     temp_matrix_1.outer_product(BFGS_p_vector, BFGS_p_vector);

     BFGS_matrix.add(rho, temp_matrix_1);

     const double energy_functional = calculate_energy_functional();

     m_logfile << " | " << std::fixed << std::setprecision(3) << std::setw(7)
     << std::scientific
     << "  " << line_search_parameter
     << "       " << energy_functional
     << "  " << m_error_residual_norm.m_norm
     << "  " << m_error_residual_norm.m_u
     << "  " << m_error_residual_norm.m_d
     << "  " << m_error_update_norm.m_norm
     << "  " << m_error_update_norm.m_u
     << "  " << m_error_update_norm.m_d
     << "  " << std::endl;
     }

     AssertThrow(BFGS_iteration < m_parameters.m_max_iterations_BFGS,
     ExcMessage("No convergence in BFGS nonlinear solver!"));
     */
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::solve_nonlinear_timestep_LBFGS(
    BVector &solution_delta,
    BVector &LBFGS_update_refine)
  {
    // Define an index to track how many times the line search parameter
    // is smaller than a threshold (1.0e-3) CONSECUTIVELY. If the line
    // search parameter is too small several times in a row, we set it to
    // 1.0 to jump out of the local trap.
    int                line_search_tracker                = 0;
    const double       line_search_parameter_lower_limit  = 1.0e-3;
    const unsigned int small_line_search_max_allowed_time = 5;

    auto     LBFGS_update_handle = m_vec_pool_ghosted.getHandle(false);
    BVector &LBFGS_update        = LBFGS_update_handle.get();

    m_error_residual.reset();
    m_error_residual_0.reset();
    m_error_residual_norm.reset();
    m_error_update.reset();
    m_error_update_0.reset();
    m_error_update_norm.reset();

    if (m_parameters.m_output_iteration_history)
      print_conv_header_LBFGS();

    unsigned int LBFGS_iteration = 0;

    auto     LBFGS_r_vector_handle = m_vec_pool_ghosted.getHandle(false);
    BVector &LBFGS_r_vector        = LBFGS_r_vector_handle.get();

    auto     LBFGS_y_vector_handle = m_vec_pool.getHandle(false);
    BVector &LBFGS_y_vector        = LBFGS_y_vector_handle.get();

    auto     LBFGS_q_vector_handle = m_vec_pool.getHandle(false);
    BVector &LBFGS_q_vector        = LBFGS_q_vector_handle.get();
    
      
    std::list<std::pair<std::pair<BVector, BVector>, double>> LBFGS_vector_list;

    const unsigned int LBFGS_m = m_parameters.m_LBFGS_m;
    std::list<double>  LBFGS_alpha_list;

    double line_search_parameter = 0.0;
    double LBFGS_beta            = 0.0;
    double rho                   = 0.0;

    for (; LBFGS_iteration < m_parameters.m_max_iterations_BFGS;
         ++LBFGS_iteration)
      {
        if (m_parameters.m_output_iteration_history)
          m_logfile << '\t' << '\t' << std::setw(4) << LBFGS_iteration << ' '
                    << std::flush;

        make_constraints(LBFGS_iteration);

        // At the first step, we simply distribute the inhomogeneous part of
        // the constraints
        if (LBFGS_iteration == 0)
          {
            // use the solution from the previous solve on the
            // refined mesh as initial guess
            LBFGS_update = LBFGS_update_refine;

            //            m_constraints.distribute(LBFGS_update);
            LBFGS_update.distributeCst(m_constraints);
            solution_delta += LBFGS_update;
            solution_delta.updateRelevance();

            if (m_parameters.m_output_iteration_history)
              {
                m_logfile << " --- " << std::flush;
                m_logfile << " --- " << std::flush;
              }
            m_solution.updateRelevance();

            update_qph_incremental(solution_delta, m_solution, false);

            if (m_parameters.m_output_iteration_history)
              {
                m_logfile << " ---  |" << std::flush;
                m_logfile << std::endl;
              }
            continue;
          }
        else if (LBFGS_iteration == 1)
          {
            // Calculate the residual vector r. NOTICE that in the context of
            // BFGS, this r is the gradient of the energy functional (objective
            // function), NOT the negative gradient of the energy functional
            if (m_parameters.has_body_force &&
                m_parameters.has_surface_pressure)
              assemble_system_rhs_BFGS_parallel<true, true>(m_solution,
                                                            m_system_rhs);
            else if (m_parameters.has_body_force)
              assemble_system_rhs_BFGS_parallel<true, false>(m_solution,
                                                             m_system_rhs);
            else if (m_parameters.has_surface_pressure)
              assemble_system_rhs_BFGS_parallel<false, true>(m_solution,
                                                             m_system_rhs);
            else
              assemble_system_rhs_BFGS_parallel<false, false>(m_solution,
                                                              m_system_rhs);
 

            // We cannot simply zero out the dofs that are constrained, since we
            // might have hanging node constraints. In this case, we need to
            // modify the RHS as C^T * b, which C contains entries of 0.5 (x_3 =
            // 0.5*x_1 + 0.5*x_2)
            // for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
            // if (m_constraints.is_constrained(i))
            // m_system_rhs(i) = 0.0;

            // if m_constraints has inhomogeneity, we cannot call
            // m_constraints.condense(m_system_rhs), since the m_system_matrix
            // needs to be provided to modify the RHS properly. However, this
            // error will not be detected in the release mode and only will be
            // detected on the debug mode if we use
            // assemble_system_rhs_BFGS_parallel, then condense() is not
            // necessary
            // m_constraints.condense(m_system_rhs);
          }
        if (m_parameters.m_output_iteration_history)
          {
            m_logfile << " --- " << std::flush;
            m_logfile << " --- " << std::flush;
            m_logfile << " --- " << std::flush;
          }


        get_error_residual(m_error_residual);
        if (LBFGS_iteration == 1)
          m_error_residual_0 = m_error_residual;


        m_error_residual_norm = m_error_residual;
        // For three-point bending problem and 3D problem, we use absolute
        // residual for convergence test
        if (m_parameters.m_relative_residual)
          m_error_residual_norm.normalize(m_error_residual_0);

        //        const double sqrtNu =
        //        (*m_blocks_desc.sqrtDoFsPerBlockPtr())[m_u_dof]; const double
        //        sqrtNd = (*m_blocks_desc.sqrtDoFsPerBlockPtr())[m_u_dof];

        if (LBFGS_iteration > 1 &&
            m_error_residual_norm.m_u <= m_parameters.m_tol_u_residual &&
            m_error_residual_norm.m_d <= m_parameters.m_tol_d_residual &&
            (m_error_update_norm.m_u /*/ sqrtNu*/) <=
              m_parameters.m_tol_u_incr &&
            (m_error_update_norm.m_d /*/ sqrtNd*/) <= m_parameters.m_tol_d_incr)
          {
            if (m_parameters.m_output_iteration_history)
              {
                m_logfile << " | ";
                m_logfile
                  << " CONVERGED! " << std::fixed << std::setprecision(3)
                  << std::setw(7) << std::scientific << "        ----       "
                  << "  " << m_error_residual_norm.m_norm << "  "
                  << m_error_residual_norm.m_u << "  "
                  << m_error_residual_norm.m_d << "  "
                  << (m_error_update_norm.m_norm /*/ (sqrtNu + sqrtNd)*/)
                  << "  " << (m_error_update_norm.m_u /*/ sqrtNu*/) << "  "
                  << (m_error_update_norm.m_d /*/ sqrtNd*/) << "  "
                  << std::endl;

                m_logfile << '\t' << '\t';
                for (unsigned int i = 0; i < 128; ++i)
                  m_logfile << '_';
                m_logfile << std::endl;
              }

            m_logfile << "\t\tConvergence is reached after " << LBFGS_iteration
                      << " L-BFGS iterations." << std::endl;

            m_logfile << "\t\tResidual information of convergence:"
                      << std::endl;

            if (m_parameters.m_relative_residual)
              {
                m_logfile << "\t\t\tRelative residual of disp. equation: "
                          << m_error_residual_norm.m_u << std::endl;

                m_logfile << "\t\t\tAbsolute residual of disp. equation: "
                          << m_error_residual_norm.m_u * m_error_residual_0.m_u
                          << std::endl;

                m_logfile << "\t\t\tRelative residual of phasefield equation: "
                          << m_error_residual_norm.m_d << std::endl;

                m_logfile << "\t\t\tAbsolute residual of phasefield equation: "
                          << m_error_residual_norm.m_d * m_error_residual_0.m_d
                          << std::endl;

                m_logfile << "\t\t\tRelative increment of disp.: "
                          << m_error_update_norm.m_u << std::endl;

                m_logfile << "\t\t\tAbsolute increment of disp.: "
                          << m_error_update_norm.m_u * m_error_update_0.m_u
                          << std::endl;

                m_logfile << "\t\t\tRelative increment of phasefield: "
                          << m_error_update_norm.m_d << std::endl;

                m_logfile << "\t\t\tAbsolute increment of phasefield: "
                          << m_error_update_norm.m_d * m_error_update_0.m_d
                          << std::endl;
              }
            else
              {
                m_logfile << "\t\t\tAbsolute residual of disp. equation: "
                          << m_error_residual_norm.m_u << std::endl;

                m_logfile << "\t\t\tAbsolute residual of phasefield equation: "
                          << m_error_residual_norm.m_d << std::endl;

                m_logfile << "\t\t\tAbsolute increment of disp.: "
                          << m_error_update_norm.m_u << std::endl;

                m_logfile << "\t\t\tAbsolute increment of phasefield: "
                          << m_error_update_norm.m_d << std::endl;
              }

            break;
          }

        // LBFGS algorithm
        LBFGS_q_vector = m_system_rhs;

        LBFGS_alpha_list.clear();
        for (auto itr = LBFGS_vector_list.begin();
             itr != LBFGS_vector_list.end();
             ++itr)
          {
            LBFGS_s_vector = (itr->first).first;
            LBFGS_y_vector = (itr->first).second;
            rho            = itr->second;

            const double alpha = rho * (LBFGS_s_vector * LBFGS_q_vector);
            LBFGS_alpha_list.push_back(alpha);

            LBFGS_q_vector.add(-alpha, LBFGS_y_vector);
          }
        /*
         double scale_gamma = 0.0;
         if (LBFGS_iteration == 1)
         {
         scale_gamma = 1.0;
         }
         else
         {
         LBFGS_s_vector = LBFGS_vector_list.front().first.first;
         LBFGS_y_vector = LBFGS_vector_list.front().first.second;
         scale_gamma = (LBFGS_s_vector * LBFGS_y_vector)/(LBFGS_y_vector *
         LBFGS_y_vector);
         }

         LBFGS_q_vector *= scale_gamma;
         LBFGS_r_vector = LBFGS_q_vector;
         */

        LBFGS_B0(LBFGS_r_vector, LBFGS_q_vector);

        for (auto itr = LBFGS_vector_list.rbegin();
             itr != LBFGS_vector_list.rend();
             ++itr)
          {
            LBFGS_s_vector = (itr->first).first;
            LBFGS_y_vector = (itr->first).second;
            rho            = itr->second;

            LBFGS_beta = rho * (LBFGS_y_vector * LBFGS_r_vector);

            const double alpha = LBFGS_alpha_list.back();
            LBFGS_alpha_list.pop_back();

            LBFGS_r_vector.add(alpha - LBFGS_beta, LBFGS_s_vector);
          }


        LBFGS_r_vector *= -1.0; // this is the p_vector (search direction)

        //        m_constraints.distribute(LBFGS_r_vector);
        LBFGS_r_vector.distributeCst(m_constraints);



        // We need a line search algorithm to decide line_search_parameter
        unsigned int num_line_search = 0;
        if (m_parameters.m_type_line_search == "StrongWolfe")
          {
            const double phi_0       = calculate_energy_functional();
            const double phi_0_prime = m_system_rhs * LBFGS_r_vector;

            line_search_parameter =
              line_search_stepsize_strong_wolfe(phi_0,
                                                phi_0_prime,
                                                LBFGS_r_vector,
                                                solution_delta,
                                                num_line_search);
          }
        else if (m_parameters.m_type_line_search == "GradientBased")
          {
            // LBFGS_r_vector is the search direction
            line_search_parameter =
              line_search_stepsize_gradient_based(LBFGS_r_vector,
                                                  solution_delta,
                                                  num_line_search);
          }
        else
          {
            Assert(false,
                   ExcMessage("An unknown line search method is called!"));
          }

        if (line_search_parameter < line_search_parameter_lower_limit)
          ++line_search_tracker;
        else
          // If the line search parameter is larger than the prescribed limit
          // we reset the tracker
          line_search_tracker = 0;

        // If line search parameter is smaller the limit several times in a row
        // we set line search parameter to 1.0 to jump out of the local trap
        if (line_search_tracker == small_line_search_max_allowed_time)
          {
            line_search_parameter = 1.0;
            line_search_tracker   = 0;
          }


        // Note: apply the length step on rank 1 for others
        if constexpr (is_mpi)
          {
            line_search_parameter =
              Utilities::MPI::broadcast(*m_mpiInfo.mpiCommPtr(),
                                        line_search_parameter,
                                        /*root=*/0);
          }

        LBFGS_r_vector *= line_search_parameter;
        LBFGS_r_vector.updateRelevance();
        LBFGS_update = LBFGS_r_vector;
        LBFGS_update.updateRelevance();

        get_error_update(LBFGS_update, m_error_update);
        if (LBFGS_iteration == 1)
          m_error_update_0 = m_error_update;

        m_error_update_norm = m_error_update;
        // For three-point bending problem and the sphere inclusion problem,
        // we use absolute residual for convergence test
        if (m_parameters.m_relative_residual)
          m_error_update_norm.normalize(m_error_update_0);


        solution_delta += LBFGS_update;
        solution_delta.updateRelevance();
        update_qph_incremental(solution_delta, m_solution, false);

        LBFGS_y_vector = m_system_rhs;
        LBFGS_y_vector *= -1.0;

        if (m_parameters.has_body_force && m_parameters.has_surface_pressure)
          assemble_system_rhs_BFGS_parallel<true, true>(m_solution,
                                                        m_system_rhs);
        else if (m_parameters.has_body_force)
          assemble_system_rhs_BFGS_parallel<true, false>(m_solution,
                                                         m_system_rhs);
        else if (m_parameters.has_surface_pressure)
          assemble_system_rhs_BFGS_parallel<false, true>(m_solution,
                                                         m_system_rhs);
        else
          assemble_system_rhs_BFGS_parallel<false, false>(m_solution,
                                                          m_system_rhs);

        // if we use assemble_system_rhs_BFGS_parallel, then condense() is not
        // necessary
        // m_constraints.condense(m_system_rhs);
        LBFGS_y_vector += m_system_rhs;

        LBFGS_s_vector = LBFGS_update;

        const double g_norm = m_system_rhs.l2_norm();

        const double yxs = LBFGS_y_vector * LBFGS_s_vector;

        const double sxs = LBFGS_s_vector * LBFGS_s_vector;

        if (yxs / sxs >= 1.0e-6 * g_norm)
          {
            if (LBFGS_iteration > LBFGS_m)
              LBFGS_vector_list.pop_back();

            rho = 1.0 / yxs;

            LBFGS_vector_list.push_front(
              std::make_pair(std::make_pair(LBFGS_s_vector, LBFGS_y_vector),
                             rho));
          }

        if (m_parameters.m_output_iteration_history)
          {
            const double energy_functional = calculate_energy_functional();

            m_logfile << " | " << std::fixed << std::setprecision(3)
                      << std::setw(1) << std::scientific << ""
                      << line_search_parameter << "  " << std::setw(3)
                      << num_line_search << "  " << std::fixed
                      << std::setprecision(6) << std::setw(1) << std::scientific
                      << "  " << energy_functional << std::fixed
                      << std::setprecision(3) << std::setw(1) << std::scientific
                      << "  " << m_error_residual_norm.m_norm << "  "
                      << m_error_residual_norm.m_u << "  "
                      << m_error_residual_norm.m_d << "  "
                      << (m_error_update_norm.m_norm /*/ (sqrtNu + sqrtNd)*/)
                      << "  " << (m_error_update_norm.m_u /*/ sqrtNu*/) << "  "
                      << (m_error_update_norm.m_d /*/ sqrtNd*/) << "  "
                      << std::endl;
          }
      }



    AssertThrow(LBFGS_iteration < m_parameters.m_max_iterations_BFGS,
                ExcMessage("No convergence in L-BFGS nonlinear solver!"));
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::output_results() const
  {
    const std::string sectionName = "Output results";
    m_timer.enter_subsection(sectionName);


    if (m_time.need_output())
      m_output.output(m_time.get_timestep(),
                      m_parameters.m_poly_degree,
                      m_parameters.resultsDir,
                      m_parameters.m_phasefield_name,
                      m_parameters.laSolverType,
                      m_solution,
                      m_quadrature_point_history);

    m_timer.leave_subsection(sectionName);
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::calculate_reaction_force(
    unsigned int face_ID)
  {
    const std::string sectionName = "Calculate reaction force";
    m_timer.enter_subsection(sectionName);

    auto     system_rhs_handle = m_vec_pool.getHandle(false);
    BVector &system_rhs        = system_rhs_handle.get();
    //    BVector system_rhs(m_mpiInfo, m_blocks_desc, /*relevance=*/false);
    //    system_rhs.initialize();

    Vector<double>                       cell_rhs(m_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(m_dofs_per_cell);

    const double                time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);
    const UpdateFlags           uf_cell(update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    const UpdateFlags           uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    FEValues<dim>     fe_values(m_fe, m_qf_cell, uf_cell);
    FEFaceValues<dim> fe_face_values(m_fe, m_qf_face, uf_face);

    // shape function values for displacement field
    std::vector<std::vector<Tensor<1, dim>>> Nx(
      m_qf_cell.size(), std::vector<Tensor<1, dim>>(m_dofs_per_cell));
    std::vector<std::vector<Tensor<2, dim>>> grad_Nx(
      m_qf_cell.size(), std::vector<Tensor<2, dim>>(m_dofs_per_cell));
    std::vector<std::vector<SymmetricTensor<2, dim>>> symm_grad_Nx(
      m_qf_cell.size(), std::vector<SymmetricTensor<2, dim>>(m_dofs_per_cell));

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
          if (!cell->is_locally_owned())
            continue;
        // if calculate_reaction_force() is defined as const, then
        // we also need to put a const in std::shared_ptr,
        // that is, std::shared_ptr<const PointHistory<dim>>
        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());
        cell_rhs = 0.0;
        fe_values.reinit(cell);
        right_hand_side(fe_values.get_quadrature_points(),
                        rhs_values,
                        m_parameters.m_x_component * time_ramp,
                        m_parameters.m_y_component * time_ramp,
                        m_parameters.m_z_component * time_ramp);

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            for (const unsigned int k : fe_values.dof_indices())
              {
                const unsigned int k_group =
                  m_fe.system_to_base_index(k).first.first;

                if (k_group == m_u_dof)
                  {
                    Nx[q_point][k] = fe_values[m_u_fe].value(k, q_point);
                    grad_Nx[q_point][k] =
                      fe_values[m_u_fe].gradient(k, q_point);
                    symm_grad_Nx[q_point][k] = symmetrize(grad_Nx[q_point][k]);
                  }
              }
          }

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            const SymmetricTensor<2, dim> &cauchy_stress =
              lqph[q_point]->get_cauchy_stress();

            const std::vector<Tensor<1, dim>>          &N = Nx[q_point];
            const std::vector<SymmetricTensor<2, dim>> &symm_grad_N =
              symm_grad_Nx[q_point];
            const double JxW = fe_values.JxW(q_point);

            for (const unsigned int i : fe_values.dof_indices())
              {
                const unsigned int i_group =
                  m_fe.system_to_base_index(i).first.first;

                if (i_group == m_u_dof)
                  {
                    cell_rhs(i) -= (symm_grad_N[i] * cauchy_stress) * JxW;
                    // contributions from the body force to right-hand side
                    cell_rhs(i) += N[i] * rhs_values[q_point] * JxW;
                  }
              }
          }

        // if there is surface pressure, this surface pressure always applied to
        // the reference configuration
        const unsigned int face_pressure_id = 100;
        const double       p0               = 0.0;

        for (const auto &face : cell->face_iterators())
          {
            if (face->at_boundary() && face->boundary_id() == face_pressure_id)
              {
                fe_face_values.reinit(cell, face);

                for (const unsigned int f_q_point :
                     fe_face_values.quadrature_point_indices())
                  {
                    const Tensor<1, dim> &N =
                      fe_face_values.normal_vector(f_q_point);

                    const double         pressure = p0 * time_ramp;
                    const Tensor<1, dim> traction = pressure * N;

                    for (const unsigned int i : fe_values.dof_indices())
                      {
                        const unsigned int i_group =
                          m_fe.system_to_base_index(i).first.first;

                        if (i_group == m_u_dof)
                          {
                            const unsigned int component_i =
                              m_fe.system_to_component_index(i).first;
                            const double Ni =
                              fe_face_values.shape_value(i, f_q_point);
                            const double JxW = fe_face_values.JxW(f_q_point);
                            cell_rhs(i) += (Ni * traction[component_i]) * JxW;
                          }
                      }
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
      } // for (const auto &cell : m_dof_handler.active_cell_iterators())

    // The difference between the above assembled system_rhs and m_system_rhs
    // is that m_system_rhs is condensed by the m_constraints, which zero out
    // the rhs values associated with the constrained DOFs and modify the rhs
    // values associated with the unconstrained DOFs.

    std::vector<types::global_dof_index> mapping;
    std::set<types::boundary_id>         boundary_ids;
    boundary_ids.insert(face_ID);

    std::vector<double> reaction_force(dim, 0.0);

    if constexpr (!is_mpi)
      {
        DoFTools::map_dof_to_boundary_indices(m_dof_handler,
                                              boundary_ids,
                                              mapping);
        const std::size_t nDoFsOnDisp =
          (*m_blocks_desc.dofsPerBlockPtr())[m_u_dof];
        for (unsigned int i = 0; i < nDoFsOnDisp; ++i)
          {
            if (mapping[i] != numbers::invalid_dof_index)
              {
                reaction_force[i % dim] += system_rhs.block(m_u_dof)(i);
              }
          }
      }
    else
      {
        // finalize distributed assembly (accumulate contributions to owners)
        system_rhs.compress(VectorOperation::add);

        // only loop over locally owned dofs
        const IndexSet &owned = m_dof_handler.locally_owned_dofs();

        for (unsigned int d = 0; d < dim; ++d)
          {
            ComponentMask comp_mask(m_fe.n_components(), false);
            comp_mask.set(m_u_fe.first_vector_component + d, true);

            const IndexSet boundary_comp =
              DoFTools::extract_boundary_dofs(m_dof_handler,
                                              comp_mask,
                                              boundary_ids);

            // owned ∩ boundary_comp
            const IndexSet owned_boundary = owned & boundary_comp;

            double reaction_force_comp = 0.0;
            for (auto i = owned_boundary.begin(); i != owned_boundary.end();
                 ++i)
              {
                reaction_force_comp += system_rhs(*i);
              }

            // sychronize results
            reaction_force[d] =
              Utilities::MPI::sum(reaction_force_comp, *m_mpiInfo.mpiCommPtr());
          }
      }

    for (unsigned int i = 0; i < dim; i++)
      m_logfile << "\t\tReaction force in direction " << i << " on boundary ID "
                << face_ID << " = " << std::fixed << std::setprecision(3)
                << std::setw(1) << std::scientific << reaction_force[i]
                << std::endl;

    std::pair<double, std::vector<double>> time_force;
    time_force.first  = m_time.current();
    time_force.second = reaction_force;
    m_history_reaction_force.push_back(time_force);

    m_timer.leave_subsection(sectionName);
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::write_history_data()
  {
    if constexpr (is_mpi)
      {
        if (!m_mpiInfo.isRankEqualsTo(0))
          return;
      }
    m_logfile << "\t\tWrite history data ... \n" << std::endl;

    std::ofstream myfile_reaction_force(m_parameters.histDir +
                                        "Reaction_force.hist");
    if (myfile_reaction_force.is_open())
      {
        myfile_reaction_force << 0.0 << "\t";
        if (dim == 2)
          myfile_reaction_force << 0.0 << "\t" << 0.0 << std::endl;
        if (dim == 3)
          myfile_reaction_force << 0.0 << "\t" << 0.0 << "\t" << 0.0
                                << std::endl;

        for (auto const &time_force : m_history_reaction_force)
          {
            myfile_reaction_force << time_force.first << "\t";
            if (dim == 2)
              myfile_reaction_force << time_force.second[0] << "\t"
                                    << time_force.second[1] << std::endl;
            if (dim == 3)
              myfile_reaction_force << time_force.second[0] << "\t"
                                    << time_force.second[1] << "\t"
                                    << time_force.second[2] << std::endl;
          }
        myfile_reaction_force.close();
      }
    else
      m_logfile << "Unable to open file";

    std::ofstream myfile_energy(m_parameters.histDir + "Energy.hist");
    if (myfile_energy.is_open())
      {
        myfile_energy << std::fixed << std::setprecision(10) << std::scientific
                      << 0.0 << "\t" << 0.0 << "\t" << 0.0 << "\t" << 0.0
                      << std::endl;

        for (auto const &time_energy : m_history_energy)
          {
            myfile_energy << std::fixed << std::setprecision(10)
                          << std::scientific << time_energy.first << "\t"
                          << time_energy.second[0] << "\t"
                          << time_energy.second[1] << "\t"
                          << time_energy.second[2] << std::endl;
          }
        myfile_energy.close();
      }
    else
      m_logfile << "Unable to open file";
  }

  template <typename LATraits, typename Tria>
  double
  PhaseFieldMonolithicSolve<LATraits, Tria>::calculate_energy_functional() const
  {
    double energy_functional = 0.0;

    FEValues<dim> fe_values(m_fe, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
          if (!cell->is_locally_owned())
            continue;
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            energy_functional += lqph[q_point]->get_total_strain_energy() * JxW;
            energy_functional +=
              lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

    // sync energy_functional with all ranks
    if constexpr (is_mpi)
      energy_functional =
        Utilities::MPI::sum(energy_functional, *m_mpiInfo.mpiCommPtr());

    return energy_functional;
  }

  template <typename LATraits, typename Tria>
  std::pair<double, double>
  PhaseFieldMonolithicSolve<LATraits, Tria>::
    calculate_total_strain_energy_and_crack_energy_dissipation() const
  {
    double total_strain_energy      = 0.0;
    double crack_energy_dissipation = 0.0;

    FEValues<dim> fe_values(m_fe, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
          if (!cell->is_locally_owned())
            continue;
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            total_strain_energy +=
              lqph[q_point]->get_total_strain_energy() * JxW;
            crack_energy_dissipation +=
              lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

    // sync total_strain_energy and crack_energy_dissipation with all ranks
    if constexpr (is_mpi)
      {
        total_strain_energy =
          Utilities::MPI::sum(total_strain_energy, *m_mpiInfo.mpiCommPtr());
        crack_energy_dissipation = Utilities::MPI::sum(crack_energy_dissipation,
                                                       *m_mpiInfo.mpiCommPtr());
      }
    return std::make_pair(total_strain_energy, crack_energy_dissipation);
  }

#if ENABLE_CUSTOMIZED_REPARTITION_MODE == 0
  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::repartition(
    BVector & /*solution_next_step*/,
    const typename LATraits::VectorBlock & /*old_history_variable_field_L2*/,
    const typename LATraits::VectorBlock
      & /*old_history_variable_field_L2_rele*/)
  {}
#else
  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::repartition(
    BVector                              &solution_next_step,
    const typename LATraits::VectorBlock &old_history_variable_field_L2,
    const typename LATraits::VectorBlock &old_history_variable_field_L2_rele)
  {
    if constexpr (std::is_same_v<Tria, DTria<2>> ||
                  std::is_same_v<Tria, DTria<3>>)
      {
        const std::string sectionName = "Repartition";
        m_timer.enter_subsection(sectionName);

        const unsigned int nOwnedCells =
          m_triangulation.n_locally_owned_active_cells();
        const unsigned int max =
          Utilities::MPI::max(nOwnedCells, *m_mpiInfo.mpiCommPtr());
        const unsigned int min =
          Utilities::MPI::min(nOwnedCells, *m_mpiInfo.mpiCommPtr());

        const double ratio = (double)max / min;
        const bool   will_repartition =
          (ratio > m_parameters.m_repartition_ratio);


        m_logfile << "\t\twill repartition: " << will_repartition << " [ "
                  << ratio << ", " << m_parameters.m_repartition_ratio << " ] "
                  << std::endl;
        m_logfile << "\t\t\tmax/min: " << ((double)max / min) << std::endl;
        m_logfile << "\t\t\tmax n owned cells: " << max << std::endl;
        m_logfile << "\t\t\tmin n owned cells: " << min << std::endl
                  << std::endl;


        if (!will_repartition)
          {
            m_timer.leave_subsection(sectionName);
            return;
          }



        for (const auto &cell : m_triangulation.active_cell_iterators())
          {
            if constexpr (is_mpi)
              {
                if (!cell->is_locally_owned())
                  continue;
              }
            cell->clear_refine_flag();
          }



        using VecType  = typename BVector::VecType;
        using VecBType = typename LATraits::VectorBlock;


        std::vector<VecType> old_solutions;
        std::vector<VecType> old_solutions_rele;
        old_solutions.reserve(2);

        old_solutions.emplace_back(solution_next_step.base());
        old_solutions.emplace_back(m_solution.base());

        // history variable field L2 projection
        DoFHandler<dim> dof_handler_L2(m_triangulation);
        FE_DGQ<dim> fe_L2(m_parameters.m_poly_degree); // Discontinuous Galerkin
        dof_handler_L2.distribute_dofs(fe_L2);
        AffineConstraints<double> constraints;
        constraints.clear();
        if constexpr (is_mpi)
          {
            const IndexSet &owned_L2 = dof_handler_L2.locally_owned_dofs();
            const IndexSet  relevant_L2 =
              DoFTools::extract_locally_relevant_dofs(dof_handler_L2);

            m_cst_maker.cstReinit(constraints, owned_L2, relevant_L2);

            DoFTools::make_hanging_node_constraints(dof_handler_L2,
                                                    constraints);

            constraints.make_consistent_in_parallel(owned_L2,
                                                    relevant_L2,
                                                    *m_mpiInfo.mpiCommPtr());
          }
        else
          {
            DoFTools::make_hanging_node_constraints(dof_handler_L2,
                                                    constraints);
          }
        constraints.close();


        if constexpr (is_mpi)
          {
            old_solutions_rele.reserve(2);
            old_solutions_rele.emplace_back();
            old_solutions_rele.emplace_back();
            old_solutions_rele[0].reinit(*m_blocks_desc.ownedPartitionPtr(),
                                         *m_blocks_desc.relevantPartitionPtr(),
                                         *m_mpiInfo.mpiCommPtr());
            old_solutions_rele[1].reinit(*m_blocks_desc.ownedPartitionPtr(),
                                         *m_blocks_desc.relevantPartitionPtr(),
                                         *m_mpiInfo.mpiCommPtr());

            old_solutions_rele[0] = old_solutions[0];
            old_solutions_rele[1] = old_solutions[1];


            old_solutions_rele[0].update_ghost_values();
            old_solutions_rele[1].update_ghost_values();
          }

        m_triangulation.prepare_coarsening_and_refinement();


        using SolTransBlockVector =
          typename SolutionTransferSelector<dim, VecType, is_mpi>::type;
        using SolTransVector =
          typename SolutionTransferSelector<dim, VecBType, is_mpi>::type;

        SolTransBlockVector solution_transfer(m_dof_handler);
        SolTransVector      solution_transfer_history_variable(dof_handler_L2);

        if constexpr (is_mpi)
          {
#  if DEAL_II_VERSION_GTE(9, 7, 0)
            solution_transfer.prepare_for_coarsening_and_refinement(
              old_solutions_rele);

            solution_transfer_history_variable
              .prepare_for_coarsening_and_refinement(
                old_history_variable_field_L2_rele);
#  else

            std::vector<const VecType *> old_solutions_ptrs = {
              &old_solutions_rele[0], &old_solutions_rele[1]};

            solution_transfer.prepare_for_coarsening_and_refinement(
              old_solutions_ptrs);
            solution_transfer_history_variable
              .prepare_for_coarsening_and_refinement(
                old_history_variable_field_L2_rele);
#  endif
          }
        else
          {
            solution_transfer.prepare_for_coarsening_and_refinement(
              old_solutions);

            solution_transfer_history_variable
              .prepare_for_coarsening_and_refinement(
                old_history_variable_field_L2);
          }



        m_logfile << "\t\trepartitioning...." << std::endl;
        m_triangulation.repartition();
        m_logfile << "\t\trepartitioninged, data transferring...." << std::endl;



        setup_system();

        m_logfile << "\t\tset up system" << std::endl;



        dof_handler_L2.distribute_dofs(fe_L2);
        constraints.clear();
        if constexpr (is_mpi)
          {
            const IndexSet &owned_L2 = dof_handler_L2.locally_owned_dofs();
            const IndexSet  relevant_L2 =
              DoFTools::extract_locally_relevant_dofs(dof_handler_L2);

            m_cst_maker.cstReinit(constraints, owned_L2, relevant_L2);

            DoFTools::make_hanging_node_constraints(dof_handler_L2,
                                                    constraints);

            constraints.make_consistent_in_parallel(owned_L2,
                                                    relevant_L2,
                                                    *m_mpiInfo.mpiCommPtr());
          }
        else
          {
            DoFTools::make_hanging_node_constraints(dof_handler_L2,
                                                    constraints);
          }
        constraints.close();


        std::vector<VecType> tmp_solutions(2);
        if constexpr (is_mpi)
          {
            // target vectors should have info about ghost cells
            tmp_solutions[0].reinit(*m_blocks_desc.ownedPartitionPtr(),
                                    *m_mpiInfo.mpiCommPtr());
            tmp_solutions[1].reinit(*m_blocks_desc.ownedPartitionPtr(),
                                    *m_mpiInfo.mpiCommPtr());
          }
        else
          {
            tmp_solutions[0].reinit(*m_blocks_desc.dofsPerBlockPtr());
            tmp_solutions[1].reinit(*m_blocks_desc.dofsPerBlockPtr());
          }

        solution_next_step.initialize();

        VecBType new_history_variable_field_L2;
        VecBType new_history_variable_field_L2_rele;
        if constexpr (is_mpi)
          {
            const IndexSet relevant_dofs =
              DoFTools::extract_locally_relevant_dofs(dof_handler_L2);

            new_history_variable_field_L2.reinit(
              dof_handler_L2.locally_owned_dofs(), *m_mpiInfo.mpiCommPtr());
            new_history_variable_field_L2_rele.reinit(
              dof_handler_L2.locally_owned_dofs(),
              relevant_dofs,
              *m_mpiInfo.mpiCommPtr());
          }
        else
          {
            new_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());
          }

        m_logfile << "\t\ttransferring solutions" << std::endl;
#  if DEAL_II_VERSION_GTE(9, 7, 0)
        solution_transfer.interpolate(tmp_solutions);
#  else
        // If an older version of dealII is used, for example, 9.4.0,
        // interpolate() needs to use the following interface.
        if constexpr (is_mpi)
          {
            std::vector<VecType *> tmp_solutions_ptrs = {&tmp_solutions[0],
                                                         &tmp_solutions[1]};
            solution_transfer.interpolate(tmp_solutions_ptrs);
          }
        else
          solution_transfer.interpolate(old_solutions, tmp_solutions);
#  endif
        m_logfile << "\t\tsolutions transferred" << std::endl;

        m_logfile << "\t\ttransferring H" << std::endl;
#  if DEAL_II_VERSION_GTE(9, 7, 0)
        solution_transfer_history_variable.interpolate(
          new_history_variable_field_L2);
#  else
        // If an older version of dealII is used, for example, 9.4.0,
        // interpolate() needs to use the following interface.
        if constexpr (is_mpi)
          {
            solution_transfer_history_variable.interpolate(
              new_history_variable_field_L2);
          }
        else
          solution_transfer_history_variable.interpolate(
            old_history_variable_field_L2, new_history_variable_field_L2);
#  endif
        m_logfile << "\t\tH transferred" << std::endl;



        solution_next_step.base() = tmp_solutions[0];
        m_solution.base()         = tmp_solutions[1];



        // make sure the projected solutions still satisfy
        // hanging node constraints

        // distribute and update relevance
        solution_next_step.distributeCst(m_constraints); // ghost cells updated
        m_solution.distributeCst(m_constraints);         // ghost cells updated
        constraints.distribute(new_history_variable_field_L2);


        if constexpr (is_mpi)
          {
            new_history_variable_field_L2_rele = new_history_variable_field_L2;
            new_history_variable_field_L2_rele.update_ghost_values();
          }

        m_logfile << "\t\tupdate H into QPnts" << std::endl;
        // new_history_variable_field_L2 contains the history variable projected
        // onto the newly refined mesh
        FEValues<dim> fe_values(fe_L2,
                                m_qf_cell,
                                update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);

        for (const auto &cell : dof_handler_L2.active_cell_iterators())
          {
            if constexpr (is_mpi)
              {
                if (!cell->is_locally_owned())
                  continue;
              }
            fe_values.reinit(cell);

            const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
              m_quadrature_point_history.get_data(cell);

            std::vector<double> history_variable_values_cell(m_n_q_points);

            fe_values.get_function_values(new_history_variable_field_L2_rele,
                                          history_variable_values_cell);

            for (unsigned int q_point : fe_values.quadrature_point_indices())
              {
                lqph[q_point]->assign_history_variable(
                  history_variable_values_cell[q_point]);
              }
          }

        {
          const unsigned int nOwnedCells =
            m_triangulation.n_locally_owned_active_cells();
          const unsigned int max =
            Utilities::MPI::max(nOwnedCells, *m_mpiInfo.mpiCommPtr());
          const unsigned int min =
            Utilities::MPI::min(nOwnedCells, *m_mpiInfo.mpiCommPtr());
          m_logfile << "\t\trepartitioned: " << std::endl;
          m_logfile << "\t\t\tmax n owned cells: " << max << std::endl;
          m_logfile << "\t\t\tmin n owned cells: " << min << std::endl
                    << std::endl;
        }

        m_timer.leave_subsection(sectionName);
      }
  }
#endif



  template <typename LATraits, typename Tria>
  bool
  PhaseFieldMonolithicSolve<LATraits, Tria>::local_refine_and_solution_transfer(
    BVector &solution_delta,
    BVector &LBFGS_update_refine)
  {
    // This is the solution at (n+1) obtained from the old (coarse) mesh
    auto     solution_next_step_handle = m_vec_pool_ghosted.getHandle(false);
    BVector &solution_next_step        = solution_next_step_handle.get();

    
    solution_next_step.base() = m_solution.base() + solution_delta.base();
    solution_next_step.updateRelevance();


    bool mesh_is_same     = true;
    bool cell_refine_flag = true;

    unsigned int material_id;
    double       length_scale;
    double       cell_length;


    // target H-vectors to avoid recalculating
    using VecType  = typename BVector::VecType;
    using VecBType = typename LATraits::VectorBlock;

    VecBType new_history_variable_field_L2;
    VecBType new_history_variable_field_L2_rele;

    while (cell_refine_flag)
      {
        cell_refine_flag = false;

        std::vector<types::global_dof_index> local_dof_indices(
          m_fe.dofs_per_cell);
        for (const auto &cell : m_dof_handler.active_cell_iterators())
          {
            if constexpr (is_mpi)
              {
                if (!cell->is_locally_owned())
                  continue;
              }

            cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < m_fe.dofs_per_cell; ++i)
              {
                const unsigned int comp_i =
                  m_fe.system_to_component_index(i).first;
                if (comp_i == m_d_component) // phasefield component
                  {
                    if (solution_next_step.relevance()(local_dof_indices[i]) >
                        m_parameters.m_phasefield_refine_threshold)
                      {
                        material_id  = cell->material_id();
                        length_scale = m_material_data[material_id][2];
                        if (dim == 2)
                          cell_length = std::sqrt(cell->measure());
                        else
                          cell_length = std::cbrt(cell->measure());
                        if (cell_length >
                            length_scale * m_parameters.m_allowed_max_h_l_ratio)
                          {
                            if (cell->level() <
                                m_parameters.m_max_allowed_refinement_level)
                              {
                                cell->set_refine_flag();
                                break;
                              }
                          }
                      }
                  }
              }
          }

        for (const auto &cell : m_dof_handler.active_cell_iterators())
          {
            if constexpr (is_mpi)
              {
                if (!cell->is_locally_owned())
                  continue;
                ;
              }

            if (cell->refine_flag_set())
              {
                cell_refine_flag = true;
                break;
              }
          }


        if constexpr (is_mpi)
          {
            // accumulate local flag over all ranks
            const unsigned int local_flag = cell_refine_flag ? 1u : 0u;
            const unsigned int global_flag =
              Utilities::MPI::sum(local_flag, *m_mpiInfo.mpiCommPtr());
            cell_refine_flag = (global_flag > 0u);
          }

        // if any cell is refined, we need to project the solution
        // to the newly refined mesh
        if (cell_refine_flag)
          {
            mesh_is_same = false;

            std::vector<VecType> old_solutions;
            std::vector<VecType> old_solutions_rele;
            old_solutions.reserve(2);

            old_solutions.emplace_back(solution_next_step.base());
            old_solutions.emplace_back(m_solution.base());

            // history variable field L2 projection
            DoFHandler<dim> dof_handler_L2(m_triangulation);
            FE_DGQ<dim>     fe_L2(
              m_parameters.m_poly_degree); // Discontinuous Galerkin
            dof_handler_L2.distribute_dofs(fe_L2);
            AffineConstraints<double> constraints;
            constraints.clear();
            // Since we use discontinuous Lagrange polynomials as shape
            // functions we don't need to worry about enforcing continuity of
            // the history variable at hanging nodes.
            // DoFTools::make_hanging_node_constraints(dof_handler_L2,
            // constraints);
            constraints.close();

            VecBType old_history_variable_field_L2;
            VecBType old_history_variable_field_L2_rele;

            if constexpr (is_mpi)
              {
                old_history_variable_field_L2.reinit(
                  dof_handler_L2.locally_owned_dofs(), *m_mpiInfo.mpiCommPtr());
              }
            else
              {
                old_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());
              }
            old_history_variable_field_L2 = 0.0;


            MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
            VectorTools::project(
              mapping,
              dof_handler_L2,
              constraints,
              m_qf_cell,
              [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                  const unsigned int q) -> double {
                return m_quadrature_point_history.get_data(cell)[q]
                  ->get_history_max_positive_strain_energy();
              },
              old_history_variable_field_L2);



            if constexpr (is_mpi)
              {
                // prepare relevant data for H and x_{n} x_{n+1}
                old_history_variable_field_L2.compress(
                  dealii::VectorOperation::insert);
                old_solutions_rele.reserve(2);
                old_solutions_rele.emplace_back();
                old_solutions_rele.emplace_back();
                old_solutions_rele[0].reinit(
                  *m_blocks_desc.ownedPartitionPtr(),
                  *m_blocks_desc.relevantPartitionPtr(),
                  *m_mpiInfo.mpiCommPtr());
                old_solutions_rele[1].reinit(
                  *m_blocks_desc.ownedPartitionPtr(),
                  *m_blocks_desc.relevantPartitionPtr(),
                  *m_mpiInfo.mpiCommPtr());

                old_solutions_rele[0] = old_solutions[0];
                old_solutions_rele[1] = old_solutions[1];


                old_history_variable_field_L2_rele.reinit(
                  dof_handler_L2.locally_owned_dofs(),
                  DoFTools::extract_locally_relevant_dofs(dof_handler_L2),
                  *m_mpiInfo.mpiCommPtr());
                old_history_variable_field_L2_rele =
                  old_history_variable_field_L2;

                old_solutions_rele[0].update_ghost_values();
                old_solutions_rele[1].update_ghost_values();

                old_history_variable_field_L2_rele.update_ghost_values();
              }


            m_triangulation.prepare_coarsening_and_refinement();

            using SolTransBlockVector =
              typename SolutionTransferSelector<dim, VecType, is_mpi>::type;
            using SolTransVector =
              typename SolutionTransferSelector<dim, VecBType, is_mpi>::type;

            SolTransBlockVector solution_transfer(m_dof_handler);
            SolTransVector solution_transfer_history_variable(dof_handler_L2);

            if constexpr (is_mpi)
              {
#if DEAL_II_VERSION_GTE(9, 7, 0)
                solution_transfer.prepare_for_coarsening_and_refinement(
                  old_solutions_rele);

                solution_transfer_history_variable
                  .prepare_for_coarsening_and_refinement(
                    old_history_variable_field_L2_rele);
#else

                std::vector<const VecType *> old_solutions_ptrs = {
                  &old_solutions_rele[0], &old_solutions_rele[1]};

                solution_transfer.prepare_for_coarsening_and_refinement(
                  old_solutions_ptrs);
                solution_transfer_history_variable
                  .prepare_for_coarsening_and_refinement(
                    old_history_variable_field_L2_rele);
#endif
              }
            else
              {
                solution_transfer.prepare_for_coarsening_and_refinement(
                  old_solutions);

                solution_transfer_history_variable
                  .prepare_for_coarsening_and_refinement(
                    old_history_variable_field_L2);
              }

            m_triangulation.execute_coarsening_and_refinement();

            setup_system();
            solution_next_step.initialize();

            dof_handler_L2.distribute_dofs(fe_L2);
            constraints.clear();
            // Since we use discontinuous Lagrange polynomials as shape
            // functions we don't need to worry about enforcing continuity of
            // the history variable at hanging nodes.
            // DoFTools::make_hanging_node_constraints(dof_handler_L2,
            // constraints);
            constraints.close();

            std::vector<VecType> tmp_solutions(2);
            if constexpr (is_mpi)
              {
                // target vectors should have info about ghost cells
                tmp_solutions[0].reinit(*m_blocks_desc.ownedPartitionPtr(),
                                        *m_mpiInfo.mpiCommPtr());
                tmp_solutions[1].reinit(*m_blocks_desc.ownedPartitionPtr(),
                                        *m_mpiInfo.mpiCommPtr());
              }
            else
              {
                tmp_solutions[0].reinit(*m_blocks_desc.dofsPerBlockPtr());
                tmp_solutions[1].reinit(*m_blocks_desc.dofsPerBlockPtr());
              }


            if constexpr (is_mpi)
              {
                const IndexSet relevant_dofs =
                  DoFTools::extract_locally_relevant_dofs(dof_handler_L2);

                new_history_variable_field_L2.reinit(
                  dof_handler_L2.locally_owned_dofs(), *m_mpiInfo.mpiCommPtr());
                new_history_variable_field_L2_rele.reinit(
                  dof_handler_L2.locally_owned_dofs(),
                  relevant_dofs,
                  *m_mpiInfo.mpiCommPtr());
              }
            else
              {
                new_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());
              }

#if DEAL_II_VERSION_GTE(9, 7, 0)
            solution_transfer.interpolate(tmp_solutions);
#else
            // If an older version of dealII is used, for example, 9.4.0,
            // interpolate() needs to use the following interface.
            if constexpr (is_mpi)
              {
                std::vector<VecType *> tmp_solutions_ptrs = {&tmp_solutions[0],
                                                             &tmp_solutions[1]};
                solution_transfer.interpolate(tmp_solutions_ptrs);
              }
            else
              solution_transfer.interpolate(old_solutions, tmp_solutions);
#endif

#if DEAL_II_VERSION_GTE(9, 7, 0)
            solution_transfer_history_variable.interpolate(
              new_history_variable_field_L2);
#else
            // If an older version of dealII is used, for example, 9.4.0,
            // interpolate() needs to use the following interface.
            if constexpr (is_mpi)
              {
                solution_transfer_history_variable.interpolate(
                  new_history_variable_field_L2);
              }
            else
              solution_transfer_history_variable.interpolate(
                old_history_variable_field_L2, new_history_variable_field_L2);
#endif



            solution_next_step.base() = tmp_solutions[0];
            m_solution.base()         = tmp_solutions[1];



            // make sure the projected solutions still satisfy
            // hanging node constraints

            // distribute and update relevance
            solution_next_step.distributeCst(
              m_constraints);                        // ghost cells updated
            m_solution.distributeCst(m_constraints); // ghost cells updated
            // Since we use discontinuous Lagrange polynomials as shape
            // functions we don't need to worry about enforcing continuity of
            // the history variable at hanging nodes.
            // constraints.distribute(new_history_variable_field_L2);


            if constexpr (is_mpi)
              {
                new_history_variable_field_L2_rele =
                  new_history_variable_field_L2;
                new_history_variable_field_L2_rele.update_ghost_values();
              }

            // new_history_variable_field_L2 contains the history variable
            // projected onto the newly refined mesh
            FEValues<dim> fe_values(fe_L2,
                                    m_qf_cell,
                                    update_values | update_gradients |
                                      update_quadrature_points |
                                      update_JxW_values);

            for (const auto &cell : dof_handler_L2.active_cell_iterators())
              {
                if constexpr (is_mpi)
                  {
                    if (!cell->is_locally_owned())
                      continue;
                  }
                fe_values.reinit(cell);

                const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
                  m_quadrature_point_history.get_data(cell);

                std::vector<double> history_variable_values_cell(m_n_q_points);

                fe_values.get_function_values(
                  new_history_variable_field_L2_rele,
                  history_variable_values_cell);

                for (unsigned int q_point :
                     fe_values.quadrature_point_indices())
                  {
                    lqph[q_point]->assign_history_variable(
                      history_variable_values_cell[q_point]);
                  }
              }
          } // if (cell_refine_flag)
      }     // while(cell_refine_flag)

    // calculate field variables for newly refined cells
    if (!mesh_is_same)
      {
        repartition(solution_next_step,
                    new_history_variable_field_L2,
                    new_history_variable_field_L2_rele);

        auto temp_solution_delta_handle = m_vec_pool_ghosted.getHandle(false);
        BVector &temp_solution_delta    = temp_solution_delta_handle.get();


        // Since we want to map the history variable in the previous time step
        // from the coarse mesh to the refined mesh, we should not update them
        // here. update_history_field_step();

        m_logfile << "\t\tUpdate field variables" << std::endl;

        // initial guess for the resolve on the refined mesh
        LBFGS_update_refine.base() =
          solution_next_step.base() - m_solution.base();
        LBFGS_update_refine.updateRelevance();

        solution_delta.updateRelevance();
      }

    return mesh_is_same;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::print_parameter_information()
  {
    if constexpr (is_mpi)
      m_logfile << "ENABLE_CUSTOMIZED_REPARTITION_MODE = "
                << supportRepartioning << std::endl;
    m_logfile << "Scenario number = " << m_parameters.m_scenario << std::endl;
    m_logfile << "Log file = " << m_parameters.m_logfile_name << std::endl;
    m_logfile << "Write iteration history to log file? = " << std::boolalpha
              << m_parameters.m_output_iteration_history << std::endl;
    m_logfile << "Phase-field model type = " << m_parameters.m_phasefield_name
              << std::endl;


    if (m_parameters.m_phasefield_name == "AT2")
      {
        m_logfile << "\tPhase-field geometric function alpha(d) = d^2"
                  << std::endl;
        m_logfile << "\tPhase-field degradation function g(d) = (1-d)^2"
                  << std::endl;
      }
    else if (m_parameters.m_phasefield_name == "AT1")
      {
        m_logfile << "\tPhase-field geometric function alpha(d) = d"
                  << std::endl;
        m_logfile << "\tPhase-field degradation function g(d) = (1-d)^2"
                  << std::endl;
      }
    else if (m_parameters.m_phasefield_name == "AT1-Cohesive")
      {
        m_logfile << "\tPhase-field geometric function alpha(d) = d"
                  << std::endl;
        m_logfile << "\tPhase-field degradation function g(d) ="
                     " (1-d)^p / [(1-d)^p + a1*d + a1*a2*d^2 + a1*a3*d^3]"
                  << std::endl;
        m_logfile
          << "\t\tFor quasi-linear degradation function: p = 1, a2 = 0, a3 = 0;"
          << std::endl;
        m_logfile
          << "\t\tFor quasi-quadratic degradation function: p = 2, a2 >= 1, a3 = 0;"
          << std::endl;
      }
    else if (m_parameters.m_phasefield_name == "PFCZM")
      {
        m_logfile << "\tPhase-field geometric function alpha(d) = 2*d -d^2"
                  << std::endl;
        m_logfile << "\tPhase-field degradation function g(d) ="
                     " (1-d)^p / [(1-d)^p + a1*d + a1*a2*d^2 + a1*a3*d^3]"
                  << std::endl;
        m_logfile << "\t\tSuggested parameters:" << std::endl;
        m_logfile << "\t\t\tLinear softening curve: "
                  << "p = 2.0, a2 = -0.5, a3 = 0;" << std::endl;
        m_logfile << "\t\t\tExponential softening curve: "
                  << "p = 2.5, a2 = 0.1748, a3 = 0;" << std::endl;
        m_logfile << "\t\t\tCornelissen softening curve: "
                  << "p = 2.0, a2 = 1.3868, a3 = 0.9106 or 0.6566;"
                  << std::endl;
      }
    else
      {
        AssertThrow(false,
                    ExcMessage("Chosen phase-field model not implemented!"));
      }


    if (dim == 2)
      {
        if (m_parameters.m_plane_stress)
          m_logfile << "2D plane-stress case" << std::endl;
        else
          m_logfile << "2D plane-strain case" << std::endl;
      }
    m_logfile << "Nonlinear solver type = "
              << m_parameters.m_type_nonlinear_solver << std::endl;
    m_logfile << "Line search type = " << m_parameters.m_type_line_search
              << std::endl;

    m_logfile << "Linear solver type = " << m_parameters.laSolverType
              << std::endl;
    m_logfile << "Mesh refinement strategy = "
              << m_parameters.m_refinement_strategy << std::endl;

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
        m_logfile
          << "\tMaximum adaptive refinement times allowed in each step = "
          << m_parameters.m_max_adaptive_refine_times << std::endl;
        m_logfile << "\tMaximum allowed cell refinement level = "
                  << m_parameters.m_max_allowed_refinement_level << std::endl;
        m_logfile << "\tPhasefield-based refinement threshold value = "
                  << m_parameters.m_phasefield_refine_threshold << std::endl;
      }

    m_logfile << "L-BFGS_m = " << m_parameters.m_LBFGS_m << std::endl;
    m_logfile << "Global refinement times = "
              << m_parameters.m_global_refine_times << std::endl;
    m_logfile << "Local prerefinement times = "
              << m_parameters.m_local_prerefine_times << std::endl;

    m_logfile << "Allowed maximum h/l ratio = "
              << m_parameters.m_allowed_max_h_l_ratio << std::endl;
    m_logfile << "Total number of material types = "
              << m_parameters.m_total_material_regions << std::endl;
    m_logfile << "Material data file name = "
              << m_parameters.m_material_file_name << std::endl;
    if (m_parameters.m_reaction_force_face_id >= 0)
      m_logfile << "Calculate reaction forces on Face ID = "
                << m_parameters.m_reaction_force_face_id << std::endl;
    else
      m_logfile << "No need to calculate reaction forces." << std::endl;

    if (m_parameters.m_relative_residual)
      m_logfile << "Relative residual for convergence." << std::endl;
    else
      m_logfile << "Absolute residual for convergence." << std::endl;

    m_logfile << "Body force = (" << m_parameters.m_x_component << ", "
              << m_parameters.m_y_component << ", "
              << m_parameters.m_z_component << ") (N/m^3)" << std::endl;

    m_logfile << "End time = " << m_parameters.m_end_time << std::endl;
    m_logfile << "Time data file name = " << m_parameters.m_time_file_name
              << std::endl;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::run()
  {
    print_parameter_information();

    read_material_data(m_parameters.m_library_dir +
                         m_parameters.m_material_file_name,
                       m_parameters.m_total_material_regions);


    std::vector<std::array<double, 3>> time_table;
    std::vector<std::vector<double>>   factors;
    std::vector<unsigned int>          intervals;
    read_time_data(m_parameters.m_time_file_name,
                   time_table,
                   factors,
                   intervals);


    make_grid();
    setup_system();
    output_results();

    m_time.increment(time_table, factors, intervals);

    while (m_time.current() < m_time.end() + m_time.get_delta_t() * 1.0e-6)
      {
        m_logfile << std::endl
                  << "Timestep " << m_time.get_timestep() << " @ "
                  << m_time.current() << 's' << std::endl;

        bool mesh_is_same = false;

        // initial guess for the resolve on the refined mesh
        auto LBFGS_update_refine_handle = m_vec_pool_ghosted.getHandle(false);
        BVector &LBFGS_update_refine    = LBFGS_update_refine_handle.get();


        // local adaptive mesh refinement loop
        unsigned int adp_refine_iteration = 0;
        for (; adp_refine_iteration <
               m_parameters.m_max_adaptive_refine_times + 1;
             ++adp_refine_iteration)
          {
            if (m_parameters.m_refinement_strategy == "adaptive-refine")
              m_logfile << "\tAdaptive refinement-" << adp_refine_iteration
                        << ": " << std::endl;

            auto solution_delta_handle = m_vec_pool_ghosted.getHandle(false);
            BVector &solution_delta    = solution_delta_handle.get();

            if (m_parameters.m_type_nonlinear_solver == "Newton")
              {
                bool newton_success = false;
                newton_success =
                  solve_nonlinear_timestep_newton(solution_delta);
                AssertThrow(
                  newton_success,
                  ExcMessage(
                    "No convergence in Newton-Raphson nonlinear solver!"));
                /*
                 // if Newton-Raphson failed, use LBFGS solver
                 if (!newton_success)
                 {
                 solution_delta = 0.0;
                 solve_nonlinear_timestep_LBFGS(solution_delta,
                 LBFGS_update_refine);
                 }
                 */
              }
            else if (m_parameters.m_type_nonlinear_solver == "BFGS")
              solve_nonlinear_timestep_BFGS(solution_delta);
            else if (m_parameters.m_type_nonlinear_solver == "LBFGS")
              {
                solve_nonlinear_timestep_LBFGS(solution_delta,
                                               LBFGS_update_refine);
                LBFGS_update_refine.updateRelevance();
              }
            else
              AssertThrow(false,
                          ExcMessage("Nonlinear solver type not implemented"));

            solution_delta.updateRelevance();

            if (m_parameters.m_refinement_strategy == "adaptive-refine")
              {
                if (adp_refine_iteration ==
                    m_parameters.m_max_adaptive_refine_times)
                  {
                    m_solution += solution_delta;
                    m_solution.updateRelevance();
                    break;
                  }

                mesh_is_same =
                  local_refine_and_solution_transfer(solution_delta,
                                                     LBFGS_update_refine);
                solution_delta.updateRelevance();
                LBFGS_update_refine.updateRelevance();

                if (mesh_is_same)
                  {
                    m_solution += solution_delta;
                    m_solution.updateRelevance();
                    break;
                  }
              }
            else if (m_parameters.m_refinement_strategy == "pre-refine")
              {
                m_solution += solution_delta;
                m_solution.updateRelevance();
                break;
              }
            else
              {
                AssertThrow(
                  false,
                  ExcMessage(
                    "Selected mesh refinement strategy not implemented!"));
              }
          } // for (; adp_refine_iteration <
            // m_parameters.m_max_adaptive_refine_times; ++adp_refine_iteration)

        // AssertThrow(adp_refine_iteration <
        // m_parameters.m_max_adaptive_refine_times,
        //             ExcMessage("Number of local adaptive mesh refinement
        //             exceeds allowed maximum times!"));

        update_history_field_step();
        // output vtk files every 10 steps if there are too
        // many time steps
        // if (m_time.get_timestep() % 10 == 0)
        output_results();

        double energy_functional_current = calculate_energy_functional();
        m_logfile << "\t\tEnergy functional (J) = " << std::fixed
                  << std::setprecision(10) << std::scientific
                  << energy_functional_current << std::endl;

        std::pair<double, double> energy_pair =
          calculate_total_strain_energy_and_crack_energy_dissipation();
        m_logfile << "\t\tTotal strain energy (J) = " << std::fixed
                  << std::setprecision(10) << std::scientific
                  << energy_pair.first << std::endl;
        m_logfile << "\t\tCrack energy dissipation (J) = " << std::fixed
                  << std::setprecision(10) << std::scientific
                  << energy_pair.second << std::endl;

        std::pair<double, std::array<double, 3>> time_energy;
        time_energy.first     = m_time.current();
        time_energy.second[0] = energy_pair.first;
        time_energy.second[1] = energy_pair.second;
        time_energy.second[2] = energy_pair.first + energy_pair.second;
        m_history_energy.push_back(time_energy);

        int face_ID = m_parameters.m_reaction_force_face_id;
        if (face_ID >= 0)
          calculate_reaction_force(face_ID);

        write_history_data();

        m_time.increment(time_table, factors, intervals);
      } // while(m_time.current() < m_time.end() + m_time.get_delta_t()*1.0e-6)
  }
} // namespace PhaseField


int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace PhaseField;
  using namespace ::common;

  if (argc < 2)
    AssertThrow(false, ExcMessage("Usage: ./main [options] <input.prm>"));



  // read prm by input command
  Parameters::AllParameters parameters(argv[argc - 1]);

  // initialize MPI by prm settings
  MPIInfo mpiInfo(parameters.m_mpi_type == "PETSc" ||
                    parameters.m_mpi_type == "Trilinos",
                  argc - 1,
                  argv);


  /**
   *
   * Print MPI / non-MPI runtime information at rank 0.
   *
   * In serial mode, only non-MPI information is printed to the terminal.
   * In MPI mode, runtime MPI configuration information is printed.
   *
   * [ Warning ]
   * Whether the MPI functionality is initialized, the mode is only determined
   * by the settings in `.prm` via `MPIInfo`. If `Serial` is specified in the
   * `.prm` file but the executable is launched via `mpiexec` or `mpirun`, MPI
   * will NOT be initialized inside the program. In this case, the launcher will
   * start multiple independent instances of the same executable.
   *
   * As a consequence, the program is executed repeatedly for `n` times,
   * where `n` is the number of processes requested by `mpiexec` or `mpirun`.
   * The program does not automatically detect or prevent this situation.
   * If this is unintended, please terminate the job immediately.
   *
   */
  if (mpiInfo.rank() == 0)
    mpiInfo.summary(std::cout);

  // create dirctories with sub-directories in the case folder
  {
    std::vector<FileSystem::SubDir> subDirs = {
      //          name of sub-dir   output argument to store sub-dir
      FileSystem::SubDir("ori", parameters.oriDir),
      FileSystem::SubDir("hist", parameters.histDir),
      FileSystem::SubDir("results", parameters.resultsDir),
    };


    /* create subfolders in a structure under ./parameters.m_output_dir/:
     *      parameters.subDir/subDirs[0]
     *      parameters.subDir/subDirs[1]
     *          ........
     */
    FileSystem::outputDirSystem(mpiInfo,
                                parameters.m_output_dir,
                                parameters.subDir,
                                subDirs);
  }



  parameters.laSolverType = parameters.m_type_linear_solver;
  if (parameters.m_type_linear_solver != "Direct")
    {
      parameters.laSolverType += "+" + parameters.m_prec_type_linear_solver;
    }

  std::ofstream log_fstream;
  // output the directory inforamtion
  if (mpiInfo.isRankEqualsTo(0))
    {
      const std::string logFileName =
        parameters.m_logfile_name + "_" + parameters.m_mpi_type + "_" +
        std::to_string(mpiInfo.nRanks()) + "_" + parameters.laSolverType + "_" +
        parameters.m_phasefield_name + ".log";

      std::cout << "\nOutput dir: \t" << parameters.m_output_dir << std::endl;
      std::cout << "Input dir: \t" << parameters.m_library_dir << std::endl;
      std::cout << "Type: \t" << parameters.m_mpi_type << std::endl;
      std::cout << "Linear solver type: \t" << parameters.laSolverType
                << std::endl;
      std::cout << "Log: \t" << logFileName << std::endl << std::endl;


      // only rank 0 creates logfile to avoid overriding in MPI mode
      log_fstream.open(parameters.m_output_dir + logFileName);
    }
  ConditionalOStream logfile(log_fstream, mpiInfo.rank() == 0);

  // dimension by prm setting
  const unsigned int dim = parameters.m_dim;
  AssertThrow(dim == 2 || dim == 3,
              ExcMessage("Dimension has to be either 2 or 3"));



  if (dim == 2)
    {
#if ENABLE_CUSTOMIZED_REPARTITION_MODE == 1
      const auto setting = DTria<2>::no_automatic_repartitioning;
#else
      const auto setting = DTria<2>::default_setting;
#endif
      const auto smooth = RTria<2>::MeshSmoothing(
        RTria<2>::smoothing_on_refinement | RTria<2>::smoothing_on_coarsening);

      if (parameters.m_mpi_type == "PETSc")
        {
#ifdef HAVE_PETSC
          DTria<2> tria(*mpiInfo.mpiCommPtr(), smooth, setting);

          PhaseFieldMonolithicSolve<Traits<TagPETSc>, DTria<2>> Phasefield2D(
            parameters, mpiInfo, logfile, tria);
          Phasefield2D.run();
#else
#endif
        }
      else if (parameters.m_mpi_type == "Trilinos")
        {
#ifdef HAVE_TRILINOS
          DTria<2> tria(*mpiInfo.mpiCommPtr(), smooth, setting);

          PhaseFieldMonolithicSolve<Traits<TagTrilinos>, DTria<2>> Phasefield2D(
            parameters, mpiInfo, logfile, tria);
          Phasefield2D.run();
#else
          std::cout << "[ ERROR ] The selected mpi mode ("
                    << parameters.m_mpi_type << ") is not installed."
                    << std::endl;
#endif
        }
      else if (parameters.m_mpi_type == "Serial")
        {
          RTria<2> tria(Triangulation<2>::maximum_smoothing);

          PhaseFieldMonolithicSolve<Traits<TagSerial>, RTria<2>> Phasefield2D(
            parameters, mpiInfo, logfile, tria);
          Phasefield2D.run();
        }
    }
  else if (dim == 3)
    {
#if ENABLE_CUSTOMIZED_REPARTITION_MODE == 1
      const auto setting = DTria<3>::no_automatic_repartitioning;
#else
      const auto setting = DTria<3>::default_setting;
#endif
      const auto smooth = RTria<3>::MeshSmoothing(
        RTria<2>::smoothing_on_refinement | RTria<2>::smoothing_on_coarsening);
      if (parameters.m_mpi_type == "PETSc")
        {
#ifdef HAVE_PETSC
          DTria<3> tria(*mpiInfo.mpiCommPtr(), smooth, setting);

          PhaseFieldMonolithicSolve<Traits<TagPETSc>, DTria<3>> Phasefield3D(
            parameters, mpiInfo, logfile, tria);
          Phasefield3D.run();
#else
          std::cout << "[ ERROR ] The selected mpi mode ("
                    << parameters.m_mpi_type << ") is not installed."
                    << std::endl;
#endif
        }
      else if (parameters.m_mpi_type == "Trilinos")
        {
#ifdef HAVE_TRILINOS
          DTria<3> tria(*mpiInfo.mpiCommPtr(), smooth, setting);

          PhaseFieldMonolithicSolve<Traits<TagTrilinos>, DTria<3>> Phasefield3D(
            parameters, mpiInfo, logfile, tria);
          Phasefield3D.run();
#else
          std::cout << "[ ERROR ] The selected mpi mode ("
                    << parameters.m_mpi_type << ") is not installed."
                    << std::endl;
#endif
        }
      else if (parameters.m_mpi_type == "Serial")
        {
          RTria<3> tria(Triangulation<3>::maximum_smoothing);

          PhaseFieldMonolithicSolve<Traits<TagSerial>, RTria<3>> Phasefield3D(
            parameters, mpiInfo, logfile, tria);
          Phasefield3D.run();
        }
    }
  return 0;
}
