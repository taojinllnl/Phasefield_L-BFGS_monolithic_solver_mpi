// =====================================================
//      Geometry setting:
// =====================================================
//   y ^
//     |
//     |
//  y3 +--------+-----+--------+ --------------
//     |        |     |        |             ^
//     |    I   |  I  |    I   |             |
//     |        |     |        |             |
//     |        |     |        |             |
//     |        |     |        |             |
//  y2 >-- C1 --+-----+--------+ --------    |
//     |        |     |        |       ^     |
//     |        |     |        |       |     |
//     |    I   |  I  |    I   |       |     H
//     |        |     |        |       |     |
//     |        |     |        |       |     |
//  y1 +--------+-----+-- C2 --< ---   H1    |
//     |        |     |        |  ^    |     |
//     |        |     |        |  |    |     |
//     |        |     |        |  H2   |     |
//     |   I    |  I  |        |  |    |     |
//     |        |     |        |  v    v     v
//  y0 +--------+-----+--------+ --------------   ----> x
//     |x0      |x1   |x2      |x3
//     |<--L1-->|     |<--L2-->|
// =====================================================


// =====================================================
// parameters config starts.
// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

// global dimensions
L  = 1.0;   // total length  (x direction)
H  = 1.0;   // total height  (y direction)

// crack-plan sizes in top view
L1 = 0.4;    // x-size of C1 patch from left boundary
L2 = 0.4;    // x-size of C2 patch from right boundary

// crack heights 
// Note: the geometry code treats H1 as the upper crack height and H2 as the lower crack height.
// If H1 < H2, the two values are swapped internally to preserve this ordering.
H1 = 0.55;    // height of C1 crack
H2 = 0.45;    // height of C2 crack

// crack opening widths (vertical opening size)
C1 = 0.010;   // total opening of crack C1
C2 = 0.010;   // total opening of crack C2


lc = 0.05; // the size for elements

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// parameters config ends.
// Do not modify the following content after this point.
// =====================================================

// Unused variables retained for compatibility
W  = 0.0;
W1 = 0.0;
W2 = 0.0;



is_v_notched = 1;
