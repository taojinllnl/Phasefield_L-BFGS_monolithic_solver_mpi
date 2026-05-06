
// =====================================================
//      Geometry setting:
// =====================================================
//
//  through_crack = 1;
//     ^ y
//     |       Top View:
//     |
//  y3 +--------+-----+--------+ ---------
//     |        |     |        |       ^
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       |
//     |   C1   |  I  |   C2   |       W
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       |
//     |        |     |        |       v
//  y0 +--------+-----+--------+ --------- -----> x
//     |x0      |x1   |x2      |x3
//     |<--L1-->|     |<--L2-->|
//     |<----------L---------->|
//
//
//
//
//     ^z    Front view:
//     |
//     |
//  z3 +--------+-----+--------+ --------------
//     |        |     |        |             ^
//     |    I   |  I  |    I   |             |
//     |        |     |        |             |
//     |        |     |        |             |
//     |        |     |        |             |
//  z2 >-- C1 --+-----+--------+ --------    |
//     |        |     |        |       ^     |
//     |        |     |        |       |     |
//     |    I   |  I  |    I   |       |     H
//     |        |     |        |       |     |
//     |        |     |        |       |     |
//  z1 +--------+-----+-- C2 --< ---   H1    |
//     |        |     |        |  ^    |     |
//     |        |     |        |  |    |     |
//     |        |     |        |  H2   |     |
//     |   I    |  I  |        |  |    |     |
//     |        |     |        |  v    v     v
//  z0 +--------+-----+--------+ --------------   ----> x
//
//
// =====================================================


// =====================================================
// parameters config starts.
// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

// global dimensions
L  = 1.0;   // total length  (x direction)
W  = 1.0;   // total width   (y direction)
H  = 1.0;   // total height  (z direction)

// crack-plan sizes in top view
L1 = 0.3;    // x-size of C1 patch from left boundary
L2 = 0.3;    // x-size of C2 patch from right boundary

// crack heights in front view
// Note: the geometry code treats H1 as the upper crack height and H2 as the lower crack height.
// If H1 < H2, the two values are swapped internally to preserve this ordering.
// In order to be read by program propopally, these two variables should be explicitly prescribed.
// That means the expression, H2 = H1;, will be illegal.
H1 = 0.50;    // height of C1 crack
H2 = 0.50;    // height of C2 crack

// crack opening widths (vertical opening size)
C1 = 0.010;   // total opening of crack C1
C2 = 0.010;   // total opening of crack C2


lc = 0.05; // the size for elements

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// parameters config ends.
// Do not modify the following content after this point.
// =====================================================


// Crack type switch
// through_crack = 0: keep the original partial corner-crack geometry.
// through_crack = 1: make C1 and C2 through-thickness cracks across the full y-width W.
//
// In through-crack mode:
//   - C1 extends from x = 0 to x = L1 at height H1;
//   - C2 extends from x = L - L2 to x = L at height H2;
//   - both cracks are extruded through the full y direction, from y = 0 to y = W;
//   - W1 and W2 are kept for compatibility with the original parameter list,
//     but they are not used in the through-crack branch.
through_crack = 1;




W1 = 0;
W2 = 0; 
