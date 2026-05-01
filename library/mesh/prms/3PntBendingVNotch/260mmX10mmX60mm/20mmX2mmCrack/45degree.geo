
// ------------------------------------------------------------
// Geometry parameters
// ------------------------------------------------------------

// ------------------------------------------------------------
// crack shape:
//                      y ^
//                       /
//    z ^               /
//      |     +---------------------+ +---------------------+ -------
//      |    /                     / /                     /|    ^
//      |   /                     / /                     / |    |
//      |  /                     / /                     /  |    |
//      | /                     / /                     /   |  height
//      |/                     / /                     /    |    |
//      +------+--------------+ +--------------+------+     |    |
//      |  ex1 |              | |              |  ex2 |     |    |
//      |<---->|              + +              |<---->|     |    v
//      |                      V                      |     + -------
//      |                                             |    /|
//      |                                             |   / |
//      |                                             |  /  |
//      |                                             | /  /|
//      |                                             |/  / |
//      +---------------------------------------------+ -/--|-------> x
//      |                                             | / thickness
//      |                                             |/
//      |                   length                    |
//      |<------------------------------------------->|
//
//
//                        notch_width_normal
//                      |<--->|
//                      +     + -------
//                      |     |    ^
//                      |     |    |
//                      |     | crack_depth
//            --------- +     +    |
//        v_height = lc  \   /     |
//                        \ /      v
//            ------------ + ----------
// The depth of the pre-existing crack is defined as crack_depth, containing a V-Shaped tip whose depth is defined as lc.
// ------------------------------------------------------------


// =====================================================
// parameters config starts.
// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

// The global geometry of the beam
length     = 260.0;
thickness  = 10.0;
height     = 60.0;

// Inward offset from both ends for supporting points as boundary conditions.
ex1 = 10.0;
ex2 = 10.0;

// The total crack depth
crack_depth = 20;
// the degree of the slanted notch
theta       = 0.7853981634; // Pi / 4.0;

x_crack_factor = 0.5;

// Notch width measured normal to the slanted notch center plane.
notch_width_normal = 2.0;

// Background characteristic length.
lc = 5.0;

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// parameters config ends.
// Do not modify the following content after this point.
// =====================================================



// if ex3 or ex4 are nagative, the three-point bending test will be launched.
ex3 = -100.0;
ex4 = -100.0;
