
// ------------------------------------------------------------
// Geometry parameters
// ------------------------------------------------------------

// ------------------------------------------------------------
// crack shape / 4-point bending layout:
//
//                      y ^
//                       /
//    z ^               /
//      |     +---------------------+ +---------------------+ -------
//      |    /                     / /                     /|    ^
//      |   /                     / /                     / |    |
//      |  /                     / /                     /  |    |
//      | /                     / /                     /   |  height
//      |/                     / /                     /    |    |
//      +------+------+-------+ +-----+--------+------+     |    |
//      |  ex1 |      |       | |     |        |  ex2 |     |    |
//      |<---->|      |       + +     |        |<---->|     |    v
//      |      ex3    |        V      |      ex4      |     + -------
//      |<----------->|               |<------------->|    /|
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
//  x_outer_left  = ex1
//  x_inner_left  = ex3
//  x_inner_right = length - ex4
//  x_outer_right = length - ex2
//
//  For a standard 4-point bending setup, the original ex1/ex2 lines
//  remain the two outer support lines, while ex3/ex4 define the two
//  inner loading lines.
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
// The depth of the pre-existing crack is defined as crack_depth, containing
// a V-shaped tip whose depth is defined as lc.
// ------------------------------------------------------------


// =====================================================
// parameters config starts.
// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

// The global geometry of the beam.
length     = 260.0;
thickness  = 60.0;
height     = 100.0;

// Outer offsets from both ends. These keep the original three-point
// bending convention and can be used as the two lower support lines.
ex1 = 10.0;
ex2 = 10.0;

// Inner offsets from both ends. These define the two upper loading lines.
// Required for the intended layout: ex1 < ex3 and ex2 < ex4.
ex3 = 40.0;
ex4 = 40.0;

// The total crack depth.
crack_depth = 20.0;

// The degree of the slanted notch.
theta       = 0.7853981634; // PI / 4.0

x_crack_factor = 0.5;

// Notch width measured normal to the slanted notch center plane.
notch_width_normal = 2.0;

// Background characteristic length.
lc = 5.0;

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// parameters config ends.
// Do not modify the following content after this point.
// =====================================================
