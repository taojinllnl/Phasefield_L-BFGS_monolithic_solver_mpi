// ============================================================
// Two side cracks with exactly 5 regions
// Strictly following the user's reference sketch
// ============================================================
//
// Coordinate system:
//   x from left to right
//   y from bottom to top
//
// Outer domain:
//   0 <= x <= L
//   0 <= y <= H
//
// Left crack:
//   starts from left boundary x = 0
//   length = L1
//   bottom height = H2
//   crack height = Hc2
//
// Right crack:
//   starts from right boundary x = L
//   length = L2
//   bottom height = H1
//   crack height = Hc2
//
//      y ^
//        |
//        |
// ------ +-------------+------------------------------+--------------+
//    ^   |             |                              |              |
//    |   |             |                              |              |
//    |   |             |                              |              |
//    |   |             |                              |              |
//    |   |             |                              |              |
//    |   +-------------+ -----                        |              |
//    |                 |   ^                          |              |
//    |                 |   |                          |              |
//    |                 |   Hc2                        |              |
//    |                 |   |                          |              |
//    |                 |   v                          |              |
//    |   +-------------+ -----                        |              |
//    |   |             |   ^                          |              |
//    |   |      L1     |   |                          |              |
//    H   |<----------->|   |                          |              |
//    |   |             |   |                          |              |
//    |   |             |   |                          |              |
//    |   |             |   |                          +--------------+ -----
//    |   |             |   |                          |                  ^
//    |   |             |   |                          |                  |
//    |   |             |   H2                         |                  Hc2
//    |   |             |   |                          |                  |
//    |   |             |   |                          |                  v
//    |   |             |   |                          +--------------+ -----
//    |   |             |   |                          |      L2      |   ^
//    |   |             |   |                          |<------------>|   |
//    |   |             |   |                          |              |   H1
//    |   |             |   |                          |              |   |
//    v   |             |   v                          |              |   v
// -----  +-------------+------------------------------+--------------+ ----- -->
//        |                                                           |         x
//        |                            L                              |
//        |<--------------------------------------------------------->|
//        |                                                           |
//
// ============================================================


// =====================================================
// parameters config starts.
// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

L   = 10.0;
H   = 5.0;

L1  = 2.0;
L2  = 2.0;

H1  = 1.0;
H2  = 3.0;

Hc2 = 0.5;

lc  = 0.1;




// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// parameters config ends.
// Do not modify the following content after this point.
// =====================================================

// If thickness = 0, generate 2D mesh.
// If thickness > 0, generate 3D extruded mesh.
thickness = 1.0;
