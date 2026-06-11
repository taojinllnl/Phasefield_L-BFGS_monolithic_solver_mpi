#ifndef usrcodes_spectrum_decomposition_h
#define usrcodes_spectrum_decomposition_h
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/patterns.h>
#include <fstream>
#include <iostream>
#include <array>


namespace usr_spectrum_decomposition
{
using namespace dealii;

inline double positive_ramp_function(const double x);


inline double negative_ramp_function(const double x);

inline double heaviside_function(const double x);

// templated function has to be defined in the header file
// perform a spectrum decomposition of a symmetric tensor
// input: a symmetric tensor (SymmetricTensor<2, matrix_dimension>)
// output: eigenvalues  (Vector<double>)
//         eigenvectors (std::vector<Tensor<1, dim>>)
template <int dim>
inline void
spectrum_decomposition(SymmetricTensor<2, dim> const & symmetric_tensor,
                       std::array<double, dim> & myEigenvalues,
                       std::array<Tensor<1, dim>, dim> & myEigenvectors);


template <int dim>
inline SymmetricTensor<2, dim>
positive_tensor(std::array<double, dim> const & eigenvalues,
                std::array<Tensor<1, dim>, dim> const & eigenvectors);

template <int dim>
inline SymmetricTensor<2, dim>
negative_tensor(std::array<double, dim> const & eigenvalues,
                std::array<Tensor<1, dim>, dim> const & eigenvectors);




// -----------------------------------------------------------------------------
// optimized functions
// -----------------------------------------------------------------------------

template <int dim>
inline SymmetricTensor<2, dim>
positive_tensor(std::array<double, dim> const & eigenvalues,
                std::array<SymmetricTensor<2, dim>, dim> const & projectors_2nd);

template <int dim>
inline SymmetricTensor<2, dim>
negative_tensor(std::array<double, dim> const & eigenvalues,
                std::array<SymmetricTensor<2, dim>, dim> const & projectors_2nd);


template <int dim>
inline void
positive_negative_projectors(std::array<double, dim> const & eigenvalues,
                             std::array<Tensor<1, dim>, dim> const & eigenvectors,
                             SymmetricTensor<4, dim> & positive_projector,
                             SymmetricTensor<4, dim> & negative_projector);



template <int dim>
inline void
positive_negative_projectors(std::array<SymmetricTensor<2, dim>, dim> const & M,
                             std::array<double, dim> const & eigenvalues,
                             SymmetricTensor<4, dim> & positive_projector,
                             SymmetricTensor<4, dim> & negative_projector);






inline double positive_ramp_function(const double x)
{
    return std::fmax(x, 0.0);
}

inline double negative_ramp_function(const double x)
{
    return std::fmin(x, 0.0);
}

inline double heaviside_function(const double x)
{
    if (std::fabs(x) < 1.0e-16)
        return 0.5;
    
    if (x > 0)
        return 1.0;
    else
        return 0.0;
}


template <int dim>
inline void
spectrum_decomposition(SymmetricTensor<2, dim> const & symmetric_tensor,
                       std::array<double, dim> &         myEigenvalues,
                       std::array<Tensor<1, dim>, dim> & myEigenvectors)
{
    
    const std::array< std::pair< double, Tensor< 1, dim > >, dim >
    myEigenSystem = eigenvectors(symmetric_tensor);
    
    for (int i = 0; i < dim; i++)
    {
        myEigenvalues[i] = myEigenSystem[i].first;
        myEigenvectors[i] = myEigenSystem[i].second;
    }
}


template <int dim>
inline SymmetricTensor<2, dim>
positive_tensor(std::array<double, dim> const & eigenvalues,
                std::array<Tensor<1, dim>, dim> const & eigenvectors)
{
    SymmetricTensor<2, dim> positive_part_tensor;
    positive_part_tensor = 0;
    for (int i = 0; i < dim; i++)
        positive_part_tensor += positive_ramp_function(eigenvalues[i])
        * symmetrize(outer_product(eigenvectors[i],
                                   eigenvectors[i]));
    return positive_part_tensor;
}



template <int dim>
inline SymmetricTensor<2, dim>
negative_tensor(std::array<double, dim> const & eigenvalues,
                std::array<Tensor<1, dim>, dim> const & eigenvectors)
{
    SymmetricTensor<2, dim> negative_part_tensor;
    negative_part_tensor = 0;
    for (int i = 0; i < dim; i++)
        negative_part_tensor += negative_ramp_function(eigenvalues[i])
        * symmetrize(outer_product(eigenvectors[i],
                                   eigenvectors[i]));
    return negative_part_tensor;
}




// -----------------------------------------------------------------------------
// optimized functions
// -----------------------------------------------------------------------------

template <int dim>
inline SymmetricTensor<2, dim>
positive_tensor(std::array<double, dim> const & eigenvalues,
                std::array<SymmetricTensor<2, dim>, dim> & projectors_2nd)
{
    SymmetricTensor<2, dim> positive_part_tensor;
    positive_part_tensor = 0;
    for (int i = 0; i < dim; i++)
        positive_part_tensor += positive_ramp_function(eigenvalues[i])
        * projectors_2nd[i];
    return positive_part_tensor;
}



template <int dim>
inline SymmetricTensor<2, dim>
negative_tensor(std::array<double, dim> const & eigenvalues,
                std::array<SymmetricTensor<2, dim>, dim> & projectors_2nd)
{
    SymmetricTensor<2, dim> negative_part_tensor;
    negative_part_tensor = 0;
    for (int i = 0; i < dim; i++)
        negative_part_tensor += negative_ramp_function(eigenvalues[i])
        * projectors_2nd[i];
    return negative_part_tensor;
}


template <int dim>
inline void
positive_negative_projectors(std::array<double, dim> const & eigenvalues,
                             std::array<Tensor<1, dim>, dim> const & eigenvectors,
                             SymmetricTensor<4, dim> & positive_projector,
                             SymmetricTensor<4, dim> & negative_projector)
{
    Assert(dim <= 3,
           ExcMessage("Project tensors only work for dim <= 3."));
    
    std::array<SymmetricTensor<2, dim>, dim> M;
    for (int a = 0; a < dim; a++)
        M[a] = symmetrize(outer_product(eigenvectors[a], eigenvectors[a]));
    
    std::array<SymmetricTensor<4, dim>, dim> Q;
    for (int a = 0; a < dim; a++)
        Q[a] = outer_product(M[a], M[a]);
    
    std::array<std::array<SymmetricTensor<4, dim>, dim>, dim> G;
    for (int a = 0; a < dim; a++)
        for (int b = 0; b < dim; b++)
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    for (int k = 0; k < dim; k++)
                        for (int l = 0; l < dim; l++)
                            G[a][b][i][j][k][l] = M[a][i][k] * M[b][j][l]
                            + M[a][i][l] * M[b][j][k];
    
    positive_projector = 0;
    for (int a = 0; a < dim; a++)
    {
        double lambda_a = eigenvalues[a];
        positive_projector += heaviside_function(lambda_a)
        * Q[a];
        for (int b = 0; b < dim; b++)
        {
            if (b != a)
            {
                double lambda_b = eigenvalues[b];
                double v_ab = 0.0;
                if (std::fabs(lambda_a - lambda_b) > 1.0e-12)
                    v_ab = (positive_ramp_function(lambda_a) - positive_ramp_function(lambda_b))
                    / (lambda_a - lambda_b);
                else
                    v_ab = 0.5 * (  heaviside_function(lambda_a)
                                  + heaviside_function(lambda_b) );
                positive_projector += 0.5 * v_ab * 0.5 * (G[a][b] + G[b][a]);
            }
        }
    }
    
    negative_projector = 0;
    for (int a = 0; a < dim; a++)
    {
        double lambda_a = eigenvalues[a];
        negative_projector += heaviside_function(-lambda_a)
        * Q[a];
        for (int b = 0; b < dim; b++)
        {
            if (b != a)
            {
                double lambda_b = eigenvalues[b];
                double v_ab = 0.0;
                if (std::fabs(lambda_a - lambda_b) > 1.0e-12)
                    v_ab = (negative_ramp_function(lambda_a) - negative_ramp_function(lambda_b))
                    / (lambda_a - lambda_b);
                else
                    v_ab = 0.5 * (  heaviside_function(-lambda_a)
                                  + heaviside_function(-lambda_b) );
                negative_projector += 0.5 * v_ab * 0.5 * (G[a][b] + G[b][a]);
            }
        }
    }
    
}




template <int dim>
inline void
positive_negative_projectors(
                             std::array<SymmetricTensor<2, dim>, dim> & M,
  const std::array<double, dim>          &eigenvalues,
  SymmetricTensor<4, dim>                &positive_projector,
  SymmetricTensor<4, dim>                &negative_projector)
{
    Assert(dim <= 3,
           ExcMessage("Project tensors only work for dim <= 3."));
    

  positive_projector = 0;
  negative_projector = 0;

  const std::array<double, dim>& lambda = eigenvalues;
  std::array<double, dim> ramp_pos;
  std::array<double, dim> ramp_neg;
  std::array<double, dim> H_pos;
  std::array<double, dim> H_neg;

  for (unsigned int a = 0; a < dim; ++a)
    {
      ramp_pos[a] = positive_ramp_function(lambda[a]);
      ramp_neg[a] = negative_ramp_function(lambda[a]);

      H_pos[a] = heaviside_function(lambda[a]);
      H_neg[a] = heaviside_function(-lambda[a]);
    }

  for (unsigned int a = 0; a < dim; ++a)
    {
      const SymmetricTensor<4, dim> Q_a = outer_product(M[a], M[a]);

      positive_projector += H_pos[a] * Q_a;
      negative_projector += H_neg[a] * Q_a;
    }

  

  constexpr double eigenvalue_tol = 1.0e-12;

  for (unsigned int a = 0; a < dim; ++a)
    for (unsigned int b = a + 1; b < dim; ++b)
      {
        const double diff = lambda[a] - lambda[b];

        double v_pos = 0.0;
        double v_neg = 0.0;

        if (std::fabs(diff) > eigenvalue_tol)
          {
            const double inv_diff = 1.0 / diff;

            v_pos = (ramp_pos[a] - ramp_pos[b]) * inv_diff;
            v_neg = (ramp_neg[a] - ramp_neg[b]) * inv_diff;
          }
        else
          {
            v_pos = 0.5 * (H_pos[a] + H_pos[b]);
            v_neg = 0.5 * (H_neg[a] + H_neg[b]);
          }

        SymmetricTensor<4, dim> Gbar_ab;

        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
              for (unsigned int l = 0; l < dim; ++l)
                {
                  const double G_ab =
                    M[a][i][k] * M[b][j][l] +
                    M[a][i][l] * M[b][j][k];

                  const double G_ba =
                    M[b][i][k] * M[a][j][l] +
                    M[b][i][l] * M[a][j][k];

                  Gbar_ab[i][j][k][l] = 0.5 * (G_ab + G_ba);
                }

        positive_projector += v_pos * Gbar_ab;
        negative_projector += v_neg * Gbar_ab;
      }
}


}
#endif
