//
//  PFValues.h
//  main
//
//

#ifndef PFValues_h
#define PFValues_h

#include <type_traits>
#include <limits>
#include <cmath>

#include "MaterialConstants.h"

namespace PhaseField
{

template <unsigned int order, int dim>
class PFValues
{
public:
    
    static constexpr double INVALID_VALUE = std::numeric_limits<double>::infinity();
    
private:
    
    struct Values
    {
        double d            = INVALID_VALUE;
        double degradation  = INVALID_VALUE;
        double geometric    = INVALID_VALUE;
    };
    
public:
    Values          __values;
    
    
    
private:
    
    void
    init_quadratic_degradation(const double base)
    {
        if constexpr (order == 0)
            __values.degradation = base * base;
        
        else if constexpr (order == 1)
            __values.degradation = -2.0 * base;
        
        else if constexpr (order == 2)
            __values.degradation = 2.0;
    }
    
    
    void
    init_cohesive_degradation(const MaterialConstants<dim>& matConst,
                              const double d,
                              const double d2,
                              const double d3,
                              const double base)
    {
        const double p  = matConst.p;
        const double a1 = matConst.a1;
        
        const double b    = std::fabs(base);
        const double sign = (d > 1.0) ? -1.0 : 1.0;
        
        const double f1 = std::pow(b, p);
        const double f2 = f1 + a1 * d + matConst.a1a2 * d2 + matConst.a1a3 * d3;
        const double inv_f2  = 1.0 / f2;
        
        if constexpr (order == 0)
            __values.degradation = f1 * inv_f2;
        
        if constexpr (order == 1 || order == 2)
        {
            const double f1_1 = sign * (-p) * std::pow(b, p - 1.0);
            
            const double f2_1 = f1_1 + a1 + matConst.two_a1a2 * d + matConst.three_a1a3 * d2;
            
            const double f1_1_X_f2 = f1_1 * f2;
            const double f1_X_f2_1 = f1 * f2_1;
            
            if constexpr (order == 1)
                __values.degradation = (f1_1_X_f2 - f1_X_f2_1) * inv_f2 * inv_f2 ;
            
            if constexpr (order == 2)
            {
                const double f1_2 = p * (p - 1.0) * std::pow(b, p - 2.0);
                
                const double f2_2 = f1_2 + matConst.two_a1a2 + matConst.six_a1a3 * d;
                
                const double f3   = f1_1_X_f2- f1_X_f2_1;
                const double f4   = f2 * f2;
                const double f3_1 = f1_2 * f2 - f1 * f2_2;
                const double f4_1 = 2.0 * f2 * f2_1;
                
                __values.degradation = (f3_1 * f4 - f3 * f4_1) / (f4 * f4);
            }
        }
    }
    
    
    void
    init_geometry_AT1(const double d)
    {
        if constexpr (order == 0)
            __values.geometric = d;
        
        else if constexpr (order == 1)
            __values.geometric = 1.0;
        
        else if constexpr (order == 2)
            __values.geometric = 0.0;
    }
    
    
    void
    init_geometry_AT2(const double two_d,
                      const double d2)
    {
        if constexpr (order == 0)
            __values.geometric = d2;
        
        else if constexpr (order == 1)
            __values.geometric = two_d;
        
        else if constexpr (order == 2)
            __values.geometric = 2.0;
    }
    
    
    void
    init_geometry_PFCZM(const double two_d,
                        const double d2,
                        const double base)
    {
        if constexpr (order == 0)
            __values.geometric = two_d - d2;
        
        else if constexpr (order == 1)
            __values.geometric = 2.0 * base;
        
        else if constexpr (order == 2)
            __values.geometric = -2.0;
    }
    
    
    void
    init_AT1(const double d,
             const bool calculate_degradation,
             const bool calculate_geometric)
    {
        if(calculate_degradation)
        {
            const double base    = 1.0 - d;
            init_quadratic_degradation(base);
        }
        
        if (calculate_geometric)
            init_geometry_AT1(d);
    }
    
    
    void
    init_AT2(const double d,
             const bool calculate_degradation,
             const bool calculate_geometric)
    {
        if(calculate_degradation)
        {
            const double base    = 1.0 - d;
            init_quadratic_degradation(base);
        }
        
        if (calculate_geometric)
        {
            const double two_d = 2.0 * d;
            const double d2    = d * d;
            init_geometry_AT2(two_d, d2);
        }
    }
    
    
    void
    init_AT1_Cohesive(const MaterialConstants<dim>& matConst,
                      const double d,
                      const bool calculate_degradation,
                      const bool calculate_geometric)
    {
        if (calculate_degradation)
        {
            const double base = 1.0 - d;
            const double d2 = d * d;
            const double d3 = d2 * d;
            init_cohesive_degradation(matConst, d, d2, d3, base);
        }
        
        if (calculate_geometric)
            init_geometry_AT1(d);
    }
    
    
    void
    init_PFCZM(const MaterialConstants<dim>& matConst,
               const double d,
               const bool calculate_degradation,
               const bool calculate_geometric)
    {
        const double base  = 1.0 - d;
        const double d2    = d * d;
        
        if constexpr (order == 0 || order == 1 || order == 2)
        {
            if(calculate_degradation)
            {
                const double d3 = d2 * d;
                init_cohesive_degradation(matConst, d, d2, d3, base);
            }
        }
        
        if (calculate_geometric)
        {
            const double two_d = 2.0 * d;
            init_geometry_PFCZM(two_d, d2, base);
        }
    }
    
    
public:
    
    const Values& init(const MaterialConstants<dim>& matConst,
                       const double d,
                       const bool calculate_degradation = true,
                       const bool calculate_geometric   = true)
    {
        __values.d = d;
        __values.degradation  = INVALID_VALUE;
        __values.geometric    = INVALID_VALUE;
        
        if (matConst.model == PFModel::AT1)
            init_AT1(d, calculate_degradation, calculate_geometric);
        
        
        else if (matConst.model == PFModel::AT1_Cohesive)
            init_AT1_Cohesive(matConst, d, calculate_degradation, calculate_geometric);
        
        else if (matConst.model == PFModel::AT2)
            init_AT2(d, calculate_degradation, calculate_geometric);
        
        
        else if (matConst.model == PFModel::PFCZM)
            init_PFCZM(matConst, d, calculate_degradation, calculate_geometric);
        
        return __values;
    }
    
    double degradation() const
    {
        Assert(__values.degradation != INVALID_VALUE,
               dealii::ExcMessage("degradation was not calculated."));
        return __values.degradation;
    }
    
    
    double geometric() const
    {
        Assert(__values.geometric != INVALID_VALUE,
               dealii::ExcMessage("geometric was not calculated."));
        return __values.geometric;
    }
    
    
};

}



#endif /* PFValues_h */
