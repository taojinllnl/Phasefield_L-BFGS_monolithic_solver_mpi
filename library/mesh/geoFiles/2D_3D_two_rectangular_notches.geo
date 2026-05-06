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


// ============================================================
// Two side cracks with exactly 5 regions
// Structured 2D / 3D mesh version
//
// If thickness = 0:
//   generate 2D structured quadrilateral mesh
//
// If thickness > 0:
//   extrude the 2D structured mesh along z direction
//   generate 3D structured hexahedral mesh
// ============================================================

SetFactory("Built-in");

// ----------------------
// Parameters from sketch
// ----------------------

L   = 10.0;
H   = 5.0;

L1  = 2.0;
L2  = 2.0;

H1  = 1.0;
H2  = 3.0;

Hc2 = 0.5;

lc  = 0.1;

// If thickness = 0, generate 2D mesh.
// If thickness > 0, generate 3D extruded mesh.
thickness = 1.0;

// ----------------------
// Geometry checks
// ----------------------

If (L <= 0 || H <= 0)
  Error("L and H must be positive.");
EndIf

If (L1 <= 0 || L2 <= 0)
  Error("L1 and L2 must be positive.");
EndIf

If (L1 + L2 >= L)
  Error("Need L1 + L2 < L so that the middle region has positive width.");
EndIf

If (H1 <= 0 || H2 <= 0)
  Error("H1 and H2 must be positive for the 5-region sketch topology.");
EndIf

If (Hc2 <= 0)
  Error("Hc2 must be positive.");
EndIf

If (H1 + Hc2 >= H)
  Error("Right crack exceeds the total height H.");
EndIf

If (H2 + Hc2 >= H)
  Error("Left crack exceeds the total height H.");
EndIf

If (thickness < 0)
  Error("thickness must be non-negative.");
EndIf

// ----------------------
// Derived coordinates
// ----------------------

x0 = 0.0;
x1 = L1;
x2 = L - L2;
x3 = L;

y0 = 0.0;
yH = H;

// right crack levels
yr0 = H1;
yr1 = H1 + Hc2;

// left crack levels
yl0 = H2;
yl1 = H2 + Hc2;

// ============================================================
// Structured mesh control
// ============================================================
//
// These variables are numbers of ELEMENTS.
// In Gmsh, Transfinite Curve needs number of POINTS,
// so later we use nDiv + 1.
//
// nDivZ is only used when thickness > 0.
//

nDivXLeft   = Max(1, Ceil(L1 / lc));
nDivXMiddle = Max(1, Ceil((L - L1 - L2) / lc));
nDivXRight  = Max(1, Ceil(L2 / lc));

nDivYTotal = Max(1, Ceil(H / lc));

nDivYCrack = Max(1, Floor(nDivYTotal * Hc2 / H));

// left vertical partition: 0 -> H2 -> H2+Hc2 -> H
nDivYLeftLower = Max(1, Floor(nDivYTotal * H2 / H));
nDivYLeftUpper = nDivYTotal - nDivYLeftLower - nDivYCrack;

// right vertical partition: 0 -> H1 -> H1+Hc2 -> H
nDivYRightLower = Max(1, Floor(nDivYTotal * H1 / H));
nDivYRightUpper = nDivYTotal - nDivYRightLower - nDivYCrack;

// z direction divisions
If (thickness > 0)
  nDivZ = Max(1, Ceil(thickness / lc));
EndIf

If (nDivYLeftUpper < 1)
  Error("nDivYLeftUpper < 1. Decrease lc or adjust H2/Hc2.");
EndIf

If (nDivYRightUpper < 1)
  Error("nDivYRightUpper < 1. Decrease lc or adjust H1/Hc2.");
EndIf

// ============================================================
// Points
// ============================================================

// bottom points
p00 = newp; Point(p00) = {x0, y0, 0, lc};
p10 = newp; Point(p10) = {x1, y0, 0, lc};
p20 = newp; Point(p20) = {x2, y0, 0, lc};
p30 = newp; Point(p30) = {x3, y0, 0, lc};

// top points
p0H = newp; Point(p0H) = {x0, yH, 0, lc};
p1H = newp; Point(p1H) = {x1, yH, 0, lc};
p2H = newp; Point(p2H) = {x2, yH, 0, lc};
p3H = newp; Point(p3H) = {x3, yH, 0, lc};

// left crack points
p0L0 = newp; Point(p0L0) = {x0, yl0, 0, lc};
p1L0 = newp; Point(p1L0) = {x1, yl0, 0, lc};

p0L1 = newp; Point(p0L1) = {x0, yl1, 0, lc};
p1L1 = newp; Point(p1L1) = {x1, yl1, 0, lc};

// right crack points
p2R0 = newp; Point(p2R0) = {x2, yr0, 0, lc};
p3R0 = newp; Point(p3R0) = {x3, yr0, 0, lc};

p2R1 = newp; Point(p2R1) = {x2, yr1, 0, lc};
p3R1 = newp; Point(p3R1) = {x3, yr1, 0, lc};

// ============================================================
// Lines
// ============================================================

// ----------------------
// Bottom boundary
// ----------------------

l_bottom_left   = newl; Line(l_bottom_left)   = {p00, p10};
l_bottom_middle = newl; Line(l_bottom_middle) = {p10, p20};
l_bottom_right  = newl; Line(l_bottom_right)  = {p20, p30};

// ----------------------
// Top boundary
// ----------------------

l_top_left   = newl; Line(l_top_left)   = {p0H, p1H};
l_top_middle = newl; Line(l_top_middle) = {p1H, p2H};
l_top_right  = newl; Line(l_top_right)  = {p2H, p3H};

// ----------------------
// Left outer boundary
// ----------------------

l_left_lower = newl; Line(l_left_lower) = {p00, p0L0};
l_left_upper = newl; Line(l_left_upper) = {p0L1, p0H};

// ----------------------
// Right outer boundary
// ----------------------

l_right_lower = newl; Line(l_right_lower) = {p30, p3R0};
l_right_upper = newl; Line(l_right_upper) = {p3R1, p3H};

// ----------------------
// Vertical internal partition at x = L1
// ----------------------

l_x1_lower = newl; Line(l_x1_lower) = {p10, p1L0};
l_x1_tip   = newl; Line(l_x1_tip)   = {p1L0, p1L1};
l_x1_upper = newl; Line(l_x1_upper) = {p1L1, p1H};

// ----------------------
// Vertical internal partition at x = L - L2
// ----------------------

l_x2_lower = newl; Line(l_x2_lower) = {p20, p2R0};
l_x2_tip   = newl; Line(l_x2_tip)   = {p2R0, p2R1};
l_x2_upper = newl; Line(l_x2_upper) = {p2R1, p2H};

// ----------------------
// Left crack faces
// ----------------------

l_left_crack_lower = newl; Line(l_left_crack_lower) = {p0L0, p1L0};
l_left_crack_upper = newl; Line(l_left_crack_upper) = {p1L1, p0L1};

// ----------------------
// Right crack faces
// ----------------------

l_right_crack_lower = newl; Line(l_right_crack_lower) = {p2R0, p3R0};
l_right_crack_upper = newl; Line(l_right_crack_upper) = {p3R1, p2R1};

// ============================================================
// Surfaces: exactly 5 regions
// ============================================================

// ------------------------------------------------------------
// Region 1: left lower region
// x in [0, L1], y in [0, H2]
// ------------------------------------------------------------

cl_left_lower = newll;
Curve Loop(cl_left_lower) = {
  l_bottom_left,
  l_x1_lower,
  -l_left_crack_lower,
  -l_left_lower
};

s_left_lower = news;
Plane Surface(s_left_lower) = {cl_left_lower};

// ------------------------------------------------------------
// Region 2: left upper region
// x in [0, L1], y in [H2 + Hc2, H]
// ------------------------------------------------------------

cl_left_upper = newll;
Curve Loop(cl_left_upper) = {
  -l_left_crack_upper,
  l_x1_upper,
  -l_top_left,
  -l_left_upper
};

s_left_upper = news;
Plane Surface(s_left_upper) = {cl_left_upper};

// ------------------------------------------------------------
// Region 3: middle full-height region
// x in [L1, L - L2], y in [0, H]
// ------------------------------------------------------------

cl_middle = newll;
Curve Loop(cl_middle) = {
  l_bottom_middle,
  l_x2_lower,
  l_x2_tip,
  l_x2_upper,
  -l_top_middle,
  -l_x1_upper,
  -l_x1_tip,
  -l_x1_lower
};

s_middle = news;
Plane Surface(s_middle) = {cl_middle};

// ------------------------------------------------------------
// Region 4: right lower region
// x in [L - L2, L], y in [0, H1]
// ------------------------------------------------------------

cl_right_lower = newll;
Curve Loop(cl_right_lower) = {
  l_bottom_right,
  l_right_lower,
  -l_right_crack_lower,
  -l_x2_lower
};

s_right_lower = news;
Plane Surface(s_right_lower) = {cl_right_lower};

// ------------------------------------------------------------
// Region 5: right upper region
// x in [L - L2, L], y in [H1 + Hc2, H]
// ------------------------------------------------------------

cl_right_upper = newll;
Curve Loop(cl_right_upper) = {
  -l_right_crack_upper,
  l_right_upper,
  -l_top_right,
  -l_x2_upper
};

s_right_upper = news;
Plane Surface(s_right_upper) = {cl_right_upper};

// ============================================================
// Transfinite structured mesh constraints
// ============================================================

// ----------------------
// Horizontal directions
// ----------------------

Transfinite Curve {
  l_bottom_left,
  l_top_left,
  l_left_crack_lower,
  l_left_crack_upper
} = nDivXLeft + 1;

Transfinite Curve {
  l_bottom_middle,
  l_top_middle
} = nDivXMiddle + 1;

Transfinite Curve {
  l_bottom_right,
  l_top_right,
  l_right_crack_lower,
  l_right_crack_upper
} = nDivXRight + 1;

// ----------------------
// Vertical directions: left side
// ----------------------

Transfinite Curve {
  l_left_lower,
  l_x1_lower
} = nDivYLeftLower + 1;

Transfinite Curve {
  l_x1_tip
} = nDivYCrack + 1;

Transfinite Curve {
  l_left_upper,
  l_x1_upper
} = nDivYLeftUpper + 1;

// ----------------------
// Vertical directions: right side
// ----------------------

Transfinite Curve {
  l_right_lower,
  l_x2_lower
} = nDivYRightLower + 1;

Transfinite Curve {
  l_x2_tip
} = nDivYCrack + 1;

Transfinite Curve {
  l_right_upper,
  l_x2_upper
} = nDivYRightUpper + 1;

// ----------------------
// Transfinite surfaces
// ----------------------

Transfinite Surface {s_left_lower} = {
  p00, p10, p1L0, p0L0
};

Transfinite Surface {s_left_upper} = {
  p0L1, p1L1, p1H, p0H
};

Transfinite Surface {s_middle} = {
  p10, p20, p2H, p1H
};

Transfinite Surface {s_right_lower} = {
  p20, p30, p3R0, p2R0
};

Transfinite Surface {s_right_upper} = {
  p2R1, p3R1, p3H, p2H
};

// ----------------------
// Recombine into quadrilateral elements
// ----------------------

Recombine Surface {
  s_left_lower,
  s_left_upper,
  s_middle,
  s_right_lower,
  s_right_upper
};

// ============================================================
// Mesh output branch
// ============================================================

Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder = 1;
Mesh.Smoothing = 10;

// ============================================================
// Case 1: 2D structured quadrilateral mesh
// ============================================================

If (thickness <= 0)

  // ----------------------
  // Physical surfaces
  // ----------------------

  Physical Surface("Region_Left_Lower") = {s_left_lower};
  Physical Surface("Region_Left_Upper") = {s_left_upper};
  Physical Surface("Region_Middle") = {s_middle};
  Physical Surface("Region_Right_Lower") = {s_right_lower};
  Physical Surface("Region_Right_Upper") = {s_right_upper};

  // Optional aggregate group.
  // If you use deal.II material_id from physical tags and need
  // distinct region ids, you may comment this aggregate group.
  Physical Surface("Domain") = {
    s_left_lower,
    s_left_upper,
    s_middle,
    s_right_lower,
    s_right_upper
  };

  // ----------------------
  // Physical curves
  // ----------------------

  Physical Curve("Bottom") = {
    l_bottom_left,
    l_bottom_middle,
    l_bottom_right
  };

  Physical Curve("Top") = {
    l_top_left,
    l_top_middle,
    l_top_right
  };

  Physical Curve("LeftBoundary") = {
    l_left_lower,
    l_left_upper
  };

  Physical Curve("RightBoundary") = {
    l_right_lower,
    l_right_upper
  };

  Physical Curve("LeftCrack") = {
    l_left_crack_lower,
    l_x1_tip,
    l_left_crack_upper
  };

  Physical Curve("LeftCrackLowerFace") = {l_left_crack_lower};
  Physical Curve("LeftCrackTip")       = {l_x1_tip};
  Physical Curve("LeftCrackUpperFace") = {l_left_crack_upper};

  Physical Curve("RightCrack") = {
    l_right_crack_lower,
    l_x2_tip,
    l_right_crack_upper
  };

  Physical Curve("RightCrackLowerFace") = {l_right_crack_lower};
  Physical Curve("RightCrackTip")       = {l_x2_tip};
  Physical Curve("RightCrackUpperFace") = {l_right_crack_upper};

  Physical Curve("InternalVerticalPartition_Left") = {
    l_x1_lower,
    l_x1_tip,
    l_x1_upper
  };

  Physical Curve("InternalVerticalPartition_Right") = {
    l_x2_lower,
    l_x2_tip,
    l_x2_upper
  };

  Mesh 2;

EndIf

// ============================================================
// Case 2: 3D structured hexahedral mesh
// ============================================================

If (thickness > 0)

  // ----------------------------------------------------------
  // Extrude all five structured quadrilateral surfaces together.
  //
  // Layers{nDivZ} controls the number of elements through thickness.
  // Recombine converts extruded prisms into hexahedral elements
  // because the source surfaces are recombined quadrilateral surfaces.
  // ----------------------------------------------------------

  Extrude {0, 0, thickness} {
    Surface{
      s_left_lower,
      s_left_upper,
      s_middle,
      s_right_lower,
      s_right_upper
    };
    Layers{nDivZ};
    Recombine;
  }

  // ----------------------------------------------------------
  // Select generated volumes using bounding boxes.
  // This avoids relying on the order of the Extrude output array.
  // ----------------------------------------------------------

  eps = 1.0e-8 * Max(L, Max(H, thickness));

  v_left_lower[] = Volume In BoundingBox {
    x0 - eps, y0 - eps, -eps,
    x1 + eps, yl0 + eps, thickness + eps
  };

  v_left_upper[] = Volume In BoundingBox {
    x0 - eps, yl1 - eps, -eps,
    x1 + eps, yH + eps, thickness + eps
  };

  v_middle[] = Volume In BoundingBox {
    x1 - eps, y0 - eps, -eps,
    x2 + eps, yH + eps, thickness + eps
  };

  v_right_lower[] = Volume In BoundingBox {
    x2 - eps, y0 - eps, -eps,
    x3 + eps, yr0 + eps, thickness + eps
  };

  v_right_upper[] = Volume In BoundingBox {
    x2 - eps, yr1 - eps, -eps,
    x3 + eps, yH + eps, thickness + eps
  };

  // ----------------------
  // Physical volumes
  // ----------------------

  Physical Volume("Region_Left_Lower") = {v_left_lower[]};
  Physical Volume("Region_Left_Upper") = {v_left_upper[]};
  Physical Volume("Region_Middle") = {v_middle[]};
  Physical Volume("Region_Right_Lower") = {v_right_lower[]};
  Physical Volume("Region_Right_Upper") = {v_right_upper[]};

  // Optional aggregate volume group.
  // For deal.II, if you want each volume to keep a unique material_id,
  // it is often cleaner to comment this aggregate group.
  Physical Volume("Domain") = {
    v_left_lower[],
    v_left_upper[],
    v_middle[],
    v_right_lower[],
    v_right_upper[]
  };

  // ----------------------------------------------------------
  // Boundary surfaces selected by bounding boxes
  // ----------------------------------------------------------

  s_front[] = Surface In BoundingBox {
    x0 - eps, y0 - eps, -eps,
    x3 + eps, yH + eps, eps
  };

  s_back[] = Surface In BoundingBox {
    x0 - eps, y0 - eps, thickness - eps,
    x3 + eps, yH + eps, thickness + eps
  };

  s_bottom[] = Surface In BoundingBox {
    x0 - eps, y0 - eps, -eps,
    x3 + eps, y0 + eps, thickness + eps
  };

  s_top[] = Surface In BoundingBox {
    x0 - eps, yH - eps, -eps,
    x3 + eps, yH + eps, thickness + eps
  };

  s_left_boundary[] = Surface In BoundingBox {
    x0 - eps, y0 - eps, -eps,
    x0 + eps, yH + eps, thickness + eps
  };

  s_right_boundary[] = Surface In BoundingBox {
    x3 - eps, y0 - eps, -eps,
    x3 + eps, yH + eps, thickness + eps
  };

  // Left crack surfaces:
  // lower face, crack tip face, upper face.
  s_left_crack[] = Surface In BoundingBox {
    x0 - eps, yl0 - eps, -eps,
    x1 + eps, yl1 + eps, thickness + eps
  };

  // Right crack surfaces:
  // lower face, crack tip face, upper face.
  s_right_crack[] = Surface In BoundingBox {
    x2 - eps, yr0 - eps, -eps,
    x3 + eps, yr1 + eps, thickness + eps
  };

  // Optional internal partition surfaces.
  // These are not external boundaries, but can be useful for inspection.
  s_internal_left[] = Surface In BoundingBox {
    x1 - eps, y0 - eps, -eps,
    x1 + eps, yH + eps, thickness + eps
  };

  s_internal_right[] = Surface In BoundingBox {
    x2 - eps, y0 - eps, -eps,
    x2 + eps, yH + eps, thickness + eps
  };

  // ----------------------
  // Physical boundary surfaces
  // ----------------------

  Physical Surface("Front") = {s_front[]};
  Physical Surface("Back") = {s_back[]};

  Physical Surface("Bottom") = {s_bottom[]};
  Physical Surface("Top") = {s_top[]};

  Physical Surface("LeftBoundary") = {s_left_boundary[]};
  Physical Surface("RightBoundary") = {s_right_boundary[]};

  Physical Surface("LeftCrack") = {s_left_crack[]};
  Physical Surface("RightCrack") = {s_right_crack[]};

  Physical Surface("InternalVerticalPartition_Left") = {s_internal_left[]};
  Physical Surface("InternalVerticalPartition_Right") = {s_internal_right[]};

  Mesh 3;

EndIf
