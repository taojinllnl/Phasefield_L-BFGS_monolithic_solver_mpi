//
//  MaterialConstants.h
//  main
//
//

#ifndef MaterialConstants_h
#define MaterialConstants_h


namespace PhaseField
{

  enum class PFModel
  {
    AT1,
    AT1_Cohesive,
    AT2,
    PFCZM
  };



  inline double
  phasefield_coefficient_constant(PFModel model_name)
  {
    double value = 0.0;
    if (model_name == PFModel::AT2)
      value = 2.0;
    else if (model_name == PFModel::AT1 || model_name == PFModel::AT1_Cohesive)
      value = 8.0 / 3.0;
    else if (model_name == PFModel::PFCZM)
      value = 4.0 * std::atan(1);
    else
      Assert(false,
             dealii::ExcMessage(
               "The phase-field geometric function has not been implemented!"));

    return value;
  }


  template <int dim>
  struct MaterialConstants
  {
    const PFModel model;
    const bool    plane_stress_flag;

    const double lambda;
    const double lambda_x_0_5;
    const double mu;
    const double mu_x_2;

    const double l0;
    const double gc;

    const double tensile_strength;



    const double E0;
    const double poisson_ratio;

    const double effective_lambda;
    const double effective_lambda_x_0_5;
    const double effective_E0;


    const double c_alpha; // phasefield_coefficient_constant


    const double max_strain_energy;


    const double p;

    const double a1;
    const double a2;
    const double a3;

    const double a1a2;
    const double a1a3;
    const double two_a1a2;
    const double three_a1a3;
    const double six_a1a3;


    const double residual_k;
    const double viscosity;
    const bool   has_viscosity;


    const double inv_c_alpha;
    const double inv_l0;

    const double I_over_c_alpha_l0; // 1.0 / l0 / c_alpha
    const double c_alpha_over_l0;   // c_alpha / l0
    const double l0_over_c_alpha;   // l0 / c_alpha

    const double gcxl0_over_c_alpha; // gc * l0 / c_alpha

    const double gc_over_c_alpha_l0;   // gc / (c_alpha * l0)
    const double gc_over_2_c_alpha_l0; // gc / (2.0 * c_alpha * l0)

    const double two_gc_l0_over_c_alpha; // 2.0 * gc * l0 / c_alpha


    static double
    young(const double lambda, const double mu);

    static double
    compute_a1(const PFModel pf_model,
               const double  phasefield_geo_constant,
               const double  length_scale,
               const double  gc,
               const double  E0,
               const double  tensile_strength);

    static double
    compute_max_strain_energy(const PFModel pf_model,
                              const double  phasefield_geo_constant,
                              const double  length_scale,
                              const double  gc,
                              const double  E0,
                              const double  tensile_strength);

    MaterialConstants(const PFModel model,
                      const double  lambda,
                      const double  mu,
                      const double  l0,
                      const double  gc,
                      const double  tensile_strength,
                      const double  viscosity,
                      const double  residual_k,
                      const double  p,
                      const double  a2,
                      const double  a3,
                      const bool    plane_stress);

    MaterialConstants(const MaterialConstants &other);

    template <typename StreamT>
    void
    print(StreamT &stream) const;
  };

  template <int dim>
  double
  MaterialConstants<dim>::young(const double lambda, const double mu)
  {
    return mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
  }

  template <int dim>
  double
  MaterialConstants<dim>::compute_a1(const PFModel pf_model,
                                     const double  phasefield_geo_constant,
                                     const double  length_scale,
                                     const double  gc,
                                     const double  E0,
                                     const double  tensile_strength)
  {
    double a1 = 0.0;
    if (pf_model == PFModel::PFCZM)
      a1 = 4.0 / (phasefield_geo_constant * length_scale) * gc * E0 /
           (tensile_strength * tensile_strength);
    else if (pf_model == PFModel::AT1_Cohesive)
      a1 = 2.0 / (phasefield_geo_constant * length_scale) * gc * E0 /
           (tensile_strength * tensile_strength);
    else
      a1 = 0.0;
    return a1;
  }


  template <int dim>
  double
  MaterialConstants<dim>::compute_max_strain_energy(
    const PFModel pf_model,
    const double  phasefield_geo_constant,
    const double  length_scale,
    const double  gc,
    const double  E0,
    const double  tensile_strength)
  {
    if (pf_model == PFModel::AT1)
      return gc / (2 * length_scale * phasefield_geo_constant);
    else if (pf_model == PFModel::PFCZM || pf_model == PFModel::AT1_Cohesive)
      return tensile_strength * tensile_strength / (2 * E0);
    else
      return 0.0;
  }

  template <int dim>
  MaterialConstants<dim>::MaterialConstants(const PFModel model_,
                                            const double  lambda_,
                                            const double  mu_,
                                            const double  l0_,
                                            const double  gc_,
                                            const double  tensile_strength_,
                                            const double  viscosity_,
                                            const double  residual_k_,
                                            const double  p_,
                                            const double  a2_,
                                            const double  a3_,
                                            const bool    plane_stress_)
    : model(model_)
    , plane_stress_flag(dim == 2 && plane_stress_)
    , lambda(lambda_)
    , lambda_x_0_5(lambda * 0.5)
    , mu(mu_)
    , mu_x_2(mu * 2)
    , l0(l0_)
    , gc(gc_)
    , tensile_strength(tensile_strength_)
    , E0(young(lambda, mu))
    , poisson_ratio(lambda / (2.0 * (lambda + mu)))
    , effective_lambda((dim == 2 && plane_stress_flag) ?
                         2.0 * mu * lambda / (lambda + 2.0 * mu) :
                         lambda)
    , effective_lambda_x_0_5(effective_lambda * 0.5)
    , effective_E0(young(effective_lambda, mu))
    , c_alpha(phasefield_coefficient_constant(model))
    , max_strain_energy(compute_max_strain_energy(model,
                                                  c_alpha,
                                                  l0,
                                                  gc,
                                                  effective_E0,
                                                  tensile_strength))
    , p(p_)
    , a1(compute_a1(model, c_alpha, l0, gc, effective_E0, tensile_strength))
    , a2(a2_)
    , a3(a3_)
    , a1a2(a1 * a2)
    , a1a3(a1 * a3)
    , two_a1a2(2.0 * a1 * a2)
    , three_a1a3(3.0 * a1 * a3)
    , six_a1a3(6.0 * a1 * a3)
    , residual_k(residual_k_)
    , viscosity(viscosity_)
    , has_viscosity(viscosity != 0.0)
    , inv_c_alpha(1.0 / c_alpha)
    , inv_l0(1.0 / l0)
    , I_over_c_alpha_l0(1.0 / c_alpha / l0)
    , c_alpha_over_l0(c_alpha / l0)
    , l0_over_c_alpha(l0 / c_alpha)
    , gcxl0_over_c_alpha(gc * l0 / c_alpha)
    , gc_over_c_alpha_l0(gc / (c_alpha * l0))
    , gc_over_2_c_alpha_l0(gc / (2.0 * c_alpha * l0))
    , two_gc_l0_over_c_alpha(2.0 * gc * l0 / c_alpha)
  {
    AssertThrow((poisson_ratio <= 0.5) && (poisson_ratio >= -1.0),
                dealii::ExcInternalError());
  }



  template <int dim>
  MaterialConstants<dim>::MaterialConstants(const MaterialConstants &other)
    : model(other.model)
    , plane_stress_flag(other.plane_stress_flag)
    , lambda(other.lambda)
    , lambda_x_0_5(other.lambda_x_0_5)
    , mu(other.mu)
    , mu_x_2(other.mu_x_2)
    , l0(other.l0)
    , gc(other.gc)
    , tensile_strength(other.tensile_strength)
    , E0(other.E0)
    , poisson_ratio(other.poisson_ratio)
    , effective_lambda(other.effective_lambda)
    , effective_lambda_x_0_5(other.effective_lambda_x_0_5)
    , effective_E0(other.effective_E0)
    , c_alpha(other.c_alpha)
    , max_strain_energy(other.max_strain_energy)
    , p(other.p)
    , a1(other.a1)
    , a2(other.a2)
    , a3(other.a3)
    , a1a2(other.a1a2)
    , a1a3(other.a1a3)
    , two_a1a2(other.two_a1a2)
    , three_a1a3(other.three_a1a3)
    , six_a1a3(other.six_a1a3)
    , residual_k(other.residual_k)
    , viscosity(other.viscosity)
    , has_viscosity(other.has_viscosity)
    , inv_c_alpha(other.inv_c_alpha)
    , inv_l0(other.inv_l0)
    , I_over_c_alpha_l0(other.I_over_c_alpha_l0)
    , c_alpha_over_l0(other.c_alpha_over_l0)
    , l0_over_c_alpha(other.l0_over_c_alpha)
    , gcxl0_over_c_alpha(other.gcxl0_over_c_alpha)
    , gc_over_c_alpha_l0(other.gc_over_c_alpha_l0)
    , gc_over_2_c_alpha_l0(other.gc_over_2_c_alpha_l0)
    , two_gc_l0_over_c_alpha(other.two_gc_l0_over_c_alpha)
  {}



  template <int dim>
  template <typename StreamT>
  void
  MaterialConstants<dim>::print(StreamT &stream) const
  {
    using namespace ::dealii;

    stream << "\t\tLame lambda = " << lambda << std::endl;
    stream << "\t\tLame mu = " << mu << std::endl;
    stream << "\t\tYoung's modulus (E0) = " << E0 << std::endl;
    stream << "\t\tPoisson ratio = " << poisson_ratio << std::endl;
    stream << "\t\tPhase field length scale (l) = " << l0 << std::endl;
    stream << "\t\tCritical energy release rate (gc) = " << gc << std::endl;
    stream << "\t\tViscosity for regularization (eta) = " << viscosity
           << std::endl;
    stream << "\t\tResidual_k (k) = " << residual_k << std::endl;
    stream << "\t\tTensile strength (ft) = " << tensile_strength << std::endl;


    if (model == PFModel::AT1_Cohesive || model == PFModel::PFCZM)
      {
        stream << "\t\tp (the polynomial order of the term (1-d)^p\n"
                  "\t\t\tin the degradation function) = "
               << p << std::endl;
        if (model == PFModel::PFCZM)
          stream
            << "\t\ta1 = 4.0 / (phasefield_geo_constant * length_scale) * gc * E0 / (tensile_strength * tensile_strength) = ";
        else
          stream
            << "\t\ta1 = 2.0 / (phasefield_geo_constant * length_scale) * gc * E0 / (tensile_strength * tensile_strength) = ";

        stream << a1 << std::endl;

        stream << "\t\ta2 (the coefficient of the a1*a2*d^2 term\n"
                  "\t\t\tin the denominator of the degradation function) = "
               << a2 << std::endl;
        stream << "\t\ta3 (the coefficient of the a1*a3*d^3 term\n"
                  "\t\t\tin the denominator of the degradation function) = "
               << a3 << std::endl;
      }


    if (model == PFModel::AT2)
      stream
        << "\t\tFor AT-2 model, tensile-strength (ft), p, a2, and a3 are irrelevant."
        << std::endl;

    else if (model == PFModel::AT1)
      {
        const double proper_l =
          gc * effective_E0 / (c_alpha * tensile_strength * tensile_strength);
        const double proper_ft = std::sqrt(gc * effective_E0 / (c_alpha * l0));

        stream
          << "\t\tFor AT-1 (Griffith) model, the provided tensile strength (ft) = "
          << tensile_strength << std::endl;
        stream
          << "\t\tHowever, based on the formular ft = sqrt[gc*E0/(c_alpha*l)],"
          << std::endl;
        stream << "\t\tthe actual material tensile strength should be "
               << proper_ft << std::endl;
        stream << "\t\tOr in order to use the provided strength ("
               << tensile_strength << ")," << std::endl;
        stream << "\t\tthe actual length-scale l should be " << proper_l
               << std::endl;
        stream
          << "\t\tFor AT-1 (Griffith) model, since the standard quadratic\n"
             "\t\tdegradation funciton is used, p, a2, and a3 are irrelevant."
          << std::endl;
      }
    else if (model == PFModel::AT1_Cohesive)
      {
        if (std::fabs(p - 1) < 1.0e-9)
          {
            stream
              << "\t\tFor AT-1 (cohesive) model, quasi-linear degradation is adopted:\n"
                 "\t\t\t g(d) = (1-d)/(1-d + a1*d)"
              << std::endl;
            AssertThrow((a2 == 0) && (a3 == 0),
                        ExcMessage("For AT-1 quasi-linear cohesive model, "
                                   "a2 = a3 = 0"));
            double upper_l = 3.0 * gc * effective_E0 /
                             (4.0 * tensile_strength * tensile_strength);
            stream << "\t\tThe provided length-scale l (" << l0
                   << ") should be smaller than the upper limit " << upper_l
                   << std::endl;
            AssertThrow(l0 < upper_l,
                        ExcMessage("The provided length-scale is over the "
                                   "upper limit!"));
          }
        else if (std::fabs(p - 2) < 1.0e-9)
          {
            stream
              << "\t\tFor AT-1 (cohesive) model, quasi-quadratic degradation is adopted:\n"
                 "\t\t\t g(d) = (1-d)^2/[(1-d)^2 + a1*d + a1*a2*d^2]"
              << std::endl;
            AssertThrow((a2 >= 1) && (a3 == 0),
                        ExcMessage("For AT-1 quasi-quadratic cohesive model, "
                                   "a2 >=1 and a3 = 0"));
            double upper_l =
              3.0 * gc * effective_E0 /
              (4.0 * (a2 + 2) * tensile_strength * tensile_strength);
            stream << "\t\tThe provided length-scale l (" << l0
                   << ") should be smaller than the upper limit " << upper_l
                   << std::endl;
            AssertThrow(l0 < upper_l,
                        ExcMessage("The provided length-scale is over the "
                                   "upper limit!"));
          }
        else
          {
            AssertThrow(false,
                        ExcMessage(
                          "For AT-1 cohesive model, "
                          "p = 1 (quasi-linear) or 2 (quasi-quadratic)"));
          }


        stream << "\n\t\tAT1-Cohesive degradation function type: ";
        if (std::fabs(p - 1) < 1.0e-9 && a2 == 0.0 && a3 == 0.0)
          {
            stream << "Quasi-linear degradation function." << std::endl;
          }
        else if (std::fabs(p - 2) < 1.0e-9 && a2 >= 1.0 && a3 == 0.0)
          {
            stream << "Quasi-quadratic degradation function." << std::endl;
          }
        else
          {
            stream << "Customized degradation function." << std::endl;
          }
        stream << std::endl;
      }
    else if (model == PFModel::PFCZM)
      {
        double lch = gc * effective_E0 / (tensile_strength * tensile_strength);
        double coeff   = 4.0 / (c_alpha * (a2 + p + 0.5));
        double upper_l = lch * coeff;

        stream << "\t\tThe characteristic length: lch = " << lch << "mm."
               << std::endl;

        stream << "\t\tThe provided length-scale l (" << l0
               << ") should be smaller than the upper limit " << upper_l
               << std::endl;

        stream << "\t\tIf the first step has negative total energy, "
               << "the length-scale should be reduced further" << std::endl;

        AssertThrow(l0 < upper_l,
                    ExcMessage("The provided length-scale is over the "
                               "upper limit!"));

        stream << "\n\t\tPFCZM softening curve type: ";
        if (p == 2.0 && a2 == -0.5 && a3 == 0)
          {
            stream << "Linear softening curve." << std::endl;
          }
        else if (p == 2.5 && a2 == 0.1748 && a3 == 0)
          {
            stream << "Exponential softening curve." << std::endl;
          }
        else if (p == 2.0 && a2 == 1.3868 && (a3 == 0.9106 || a3 == 0.6566))
          {
            stream << "Cornelissen softening curve." << std::endl;
          }
        else
          {
            stream << "Customized softening curve." << std::endl;
          }
        stream << std::endl;
      }
    else
      {
        AssertThrow(false,
                    ExcMessage("Chosen phase-field model not implemented!"));
      }
  }



} // namespace PhaseField

#endif /* MaterialConstants_h */
