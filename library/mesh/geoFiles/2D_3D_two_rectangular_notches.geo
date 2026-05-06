Include "../prms/throughRecNotchedCube/1mmX1mmX1mm/0.1mmX0.2mmCracks/0.0mmDistance.geo";


SetFactory("Built-in");

// ============================================================
// Mesh/file options for deal.II compatibility
// ============================================================
// deal.II 9.x reads Gmsh MSH 2.2 reliably for low-order cells.
Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder   = 1;
Mesh.RecombineAll   = 1;
Mesh.RecombinationAlgorithm = 1;
Mesh.Smoothing = 10;

// ============================================================
// Thickness variable compatibility
// ============================================================
// The reference 3D file uses the parameter name "thickness".
// This file historically used "W" as the extrusion depth.
// If the included parameter file defines thickness but not W, then W is
// effectively zero/undefined and the script falls into the Mesh 2 branch.
// That produces only line/surface elements, and deal.II GridIn<3> reports
// "did not find any cells".
If (!Exists(thickness))
  If (Exists(W))
    thickness = W;
  Else
    thickness = 0.0;
  EndIf
EndIf

// From here on, W is the extrusion thickness used by this script.
W = thickness;


// ============================================================
// Geometry checks
// ============================================================

If (L <= 0 || H <= 0)
  Error("L and H must be positive.");
EndIf

If (L2 <= 0 || L1 <= 0)
  Error("L2 and L1 must be positive.");
EndIf

If (L2 + L1 >= L)
  Error("Need L2 + L1 < L so that the middle region has positive width.");
EndIf

If (H2 <= 0 || H1 <= 0)
  Error("H2 and H1 must be positive for this 5-region topology.");
EndIf

If (C2 <= 0 || C1 <= 0)
  Error("C2 and C1 must be positive.");
EndIf

If (H2 + C2 >= H)
  Error("Right crack exceeds the total height H: require H2 + C2 < H.");
EndIf

If (H1 + C1 >= H)
  Error("Left crack exceeds the total height H: require H1 + C1 < H.");
EndIf

If (W < 0)
  Error("W must be non-negative.");
EndIf

// ============================================================
// Derived coordinates
// ============================================================

x0 = 0.0;

// Left crack tip location.
// Left crack length is L1.
x1 = L1;

// Right crack tip location.
// Right crack length is L2.
x2 = L - L2;

x3 = L;

y0 = 0.0;
yH = H;

// Right crack levels
yr0 = H2;
yr1 = H2 + C2;

// Left crack levels
yl0 = H1;
yL2 = H1 + C1;

// ============================================================
// Structured mesh control
// ============================================================
//
// These variables are numbers of ELEMENTS.
// Gmsh Transfinite Curve requires number of POINTS,
// therefore each curve uses nDiv + 1 below.
//
// The middle region has two vertical sides:
//
//   left side : 0 -> H1 -> H1+C1 -> H
//   right side: 0 -> H2 -> H2+C2 -> H
//
// To keep Region_Middle structured and conforming,
// both sides must have the same total number of vertical elements.
// ============================================================

nDivXLeft   = Max(1, Ceil(L1 / lc));
nDivXMiddle = Max(1, Ceil((L - L2 - L1) / lc));
nDivXRight  = Max(1, Ceil(L2 / lc));

nDivYTotal = Max(1, Ceil(H / lc) + 1);

// Left crack vertical subdivisions
nDivYLeftCrack = Max(1, Floor(nDivYTotal * C1 / H));

// Right crack vertical subdivisions
nDivYRightCrack = Max(1, Floor(nDivYTotal * C2 / H));

// Left vertical partition: 0 -> H1 -> H1+C1 -> H
nDivYLeftUpper = Max(1, Floor(nDivYTotal * (H - H1 - C1) / H));
nDivYLeftLower = nDivYTotal - nDivYLeftUpper - nDivYLeftCrack - 1;

// Right vertical partition: 0 -> H2 -> H2+C2 -> H
nDivYRightLower = Max(1, Floor(nDivYTotal * H2 / H));
nDivYRightUpper = nDivYTotal - nDivYRightLower - nDivYRightCrack - 1;


// aligned case
If (aligned_case)
  yA0 = H2;
  yA1 = H2 + C2;

  nDivYLower = Max(1, Floor(nDivYTotal * yA0 / H));
  nDivYCrack = Max(1, Floor(nDivYTotal * (yA1 - yA0) / H));
  nDivYUpper = nDivYTotal - nDivYLower - nDivYCrack;

  If (nDivYUpper < 1)
    Error("nDivYUpper < 1.");
  EndIf
EndIf

If (nDivYLeftUpper < 1)
  Error("nDivYLeftUpper < 1. Decrease lc or adjust H1/C1.");
EndIf

If (nDivYRightUpper < 1)
  Error("nDivYRightUpper < 1. Decrease lc or adjust H2/C2.");
EndIf

If (W > 0)
  nDivZ = Max(1, Ceil(W / lc));
EndIf

// ============================================================
// Points
// ============================================================

// Bottom points
p00 = newp; Point(p00) = {x0, y0, 0, lc};
p10 = newp; Point(p10) = {x1, y0, 0, lc};
p20 = newp; Point(p20) = {x2, y0, 0, lc};
p30 = newp; Point(p30) = {x3, y0, 0, lc};

// Top points
p0H = newp; Point(p0H) = {x0, yH, 0, lc};
p1H = newp; Point(p1H) = {x1, yH, 0, lc};
p2H = newp; Point(p2H) = {x2, yH, 0, lc};
p3H = newp; Point(p3H) = {x3, yH, 0, lc};

// Left crack points
p0L0 = newp; Point(p0L0) = {x0, yl0, 0, lc};
p1L0 = newp; Point(p1L0) = {x1, yl0, 0, lc};

p0L2 = newp; Point(p0L2) = {x0, yL2, 0, lc};
p1L2 = newp; Point(p1L2) = {x1, yL2, 0, lc};

// Right crack points
p2R0 = newp; Point(p2R0) = {x2, yr0, 0, lc};
p3R0 = newp; Point(p3R0) = {x3, yr0, 0, lc};

p2R1 = newp; Point(p2R1) = {x2, yr1, 0, lc};
p3R1 = newp; Point(p3R1) = {x3, yr1, 0, lc};

// ============================================================
// Lines
// ============================================================

// Bottom boundary
l_bottom_left   = newl; Line(l_bottom_left)   = {p00, p10};
l_bottom_middle = newl; Line(l_bottom_middle) = {p10, p20};
l_bottom_right  = newl; Line(l_bottom_right)  = {p20, p30};

// Top boundary
l_top_left   = newl; Line(l_top_left)   = {p0H, p1H};
l_top_middle = newl; Line(l_top_middle) = {p1H, p2H};
l_top_right  = newl; Line(l_top_right)  = {p2H, p3H};

// Left outer boundary
// The crack mouth itself is open.
l_left_lower = newl; Line(l_left_lower) = {p00, p0L0};
l_left_upper = newl; Line(l_left_upper) = {p0L2, p0H};

// Right outer boundary
// The crack mouth itself is open.
l_right_lower = newl; Line(l_right_lower) = {p30, p3R0};
l_right_upper = newl; Line(l_right_upper) = {p3R1, p3H};

// Vertical internal partition at x = L1
l_x1_lower = newl; Line(l_x1_lower) = {p10, p1L0};
l_x1_tip   = newl; Line(l_x1_tip)   = {p1L0, p1L2};
l_x1_upper = newl; Line(l_x1_upper) = {p1L2, p1H};

// Vertical internal partition at x = L - L2
l_x2_lower = newl; Line(l_x2_lower) = {p20, p2R0};
l_x2_tip   = newl; Line(l_x2_tip)   = {p2R0, p2R1};
l_x2_upper = newl; Line(l_x2_upper) = {p2R1, p2H};

// Left crack faces
l_left_crack_lower = newl; Line(l_left_crack_lower) = {p0L0, p1L0};
l_left_crack_upper = newl; Line(l_left_crack_upper) = {p1L2, p0L2};

// Right crack faces
l_right_crack_lower = newl; Line(l_right_crack_lower) = {p2R0, p3R0};
l_right_crack_upper = newl; Line(l_right_crack_upper) = {p3R1, p2R1};

// ============================================================
// Surfaces: exactly 5 regions
// ============================================================

// ------------------------------------------------------------
// Region 1: left lower region
// x in [0, L1], y in [0, H1]
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
// x in [0, L1], y in [H1 + C1, H]
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
// x in [L - L2, L], y in [0, H2]
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
// x in [L - L2, L], y in [H2 + C2, H]
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

// Left part width = L1
Transfinite Curve {
  l_bottom_left,
  l_top_left,
  l_left_crack_lower,
  l_left_crack_upper
} = nDivXLeft + 1;

// Middle part width = L - L2 - L1
Transfinite Curve {
  l_bottom_middle,
  l_top_middle
} = nDivXMiddle + 1;

// Right part width = L2
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
} = nDivYLeftCrack + 1;

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
} = nDivYRightCrack + 1;

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
  p0L2, p1L2, p1H, p0H
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
// Case 1: 2D structured quadrilateral mesh
// ============================================================

If (W <= 0)

  Physical Surface("Region_Left_Lower") = {s_left_lower};
  Physical Surface("Region_Left_Upper") = {s_left_upper};
  Physical Surface("Region_Middle") = {s_middle};
  Physical Surface("Region_Right_Lower") = {s_right_lower};
  Physical Surface("Region_Right_Upper") = {s_right_upper};

  // If you need unique material_id in deal.II, it is usually safer
  // not to also put the same elements into an aggregate Physical Surface.
  //
  // Physical Surface("Domain") = {
  //   s_left_lower,
  //   s_left_upper,
  //   s_middle,
  //   s_right_lower,
  //   s_right_upper
  // };

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

  Physical Curve("RightCrack") = {
    l_right_crack_lower,
    l_x2_tip,
    l_right_crack_upper
  };

  // Optional separate crack-face groups.
  // Enable these only if you really need different boundary ids
  // for lower face, upper face, and tip.
  //
  // Physical Curve("LeftCrackLowerFace") = {l_left_crack_lower};
  // Physical Curve("LeftCrackTip")       = {l_x1_tip};
  // Physical Curve("LeftCrackUpperFace") = {l_left_crack_upper};
  //
  // Physical Curve("RightCrackLowerFace") = {l_right_crack_lower};
  // Physical Curve("RightCrackTip")       = {l_x2_tip};
  // Physical Curve("RightCrackUpperFace") = {l_right_crack_upper};

  Mesh 2;

EndIf

// ============================================================
// Case 2: 3D structured hexahedral mesh
// ============================================================

If (W > 0)

  // IMPORTANT:
  // Do not use "Volume In BoundingBox" to recover the volumes after extrusion.
  // It can return empty arrays depending on the geometry kernel/version/tolerance.
  // If Physical Volume groups are empty, Gmsh writes only surfaces/curves to the
  // .msh file, and deal.II GridIn<3>::read_msh() then reports
  // ExcGmshNoCellInformation.
  //
  // The robust way is to store the entities returned by Extrude directly.
  // For an extruded surface:
  //   out[0] = translated/top surface at z = W
  //   out[1] = generated volume
  //   out[2...] = lateral surfaces generated from the boundary curves

  ext_left_lower[] = Extrude {0, 0, W} {
    Surface{s_left_lower};
    Layers{nDivZ};
    Recombine;
  };

  ext_left_upper[] = Extrude {0, 0, W} {
    Surface{s_left_upper};
    Layers{nDivZ};
    Recombine;
  };

  ext_middle[] = Extrude {0, 0, W} {
    Surface{s_middle};
    Layers{nDivZ};
    Recombine;
  };

  ext_right_lower[] = Extrude {0, 0, W} {
    Surface{s_right_lower};
    Layers{nDivZ};
    Recombine;
  };

  ext_right_upper[] = Extrude {0, 0, W} {
    Surface{s_right_upper};
    Layers{nDivZ};
    Recombine;
  };

  // Volumes returned directly by Extrude.
  v_left_lower  = ext_left_lower[1];
  v_left_upper  = ext_left_upper[1];
  v_middle      = ext_middle[1];
  v_right_lower = ext_right_lower[1];
  v_right_upper = ext_right_upper[1];

  // Merge duplicate geometrical entities created by independent extrusions
  // of adjacent 2D blocks. This is not the main cause of the current deal.II
  // error, but it makes the final multi-block volume mesh cleaner.
  Coherence;

  Physical Volume("Region_Left_Lower")  = {v_left_lower};
  Physical Volume("Region_Left_Upper")  = {v_left_upper};
  Physical Volume("Region_Middle")      = {v_middle};
  Physical Volume("Region_Right_Lower") = {v_right_lower};
  Physical Volume("Region_Right_Upper") = {v_right_upper};

  // If you need unique material_id in deal.II, do not also assign the same
  // cells to an aggregate Physical Volume. Otherwise the same cell can belong
  // to more than one physical volume group.
  //
  // Physical Volume("Domain") = {
  //   v_left_lower,
  //   v_left_upper,
  //   v_middle,
  //   v_right_lower,
  //   v_right_upper
  // };

  eps = 1.0e-8 * Max(L, Max(H, W));

  // ----------------------------------------------------------
  // Front/back surfaces
  // ----------------------------------------------------------
  // The original 2D surfaces remain at z = 0.
  // The top surfaces returned by Extrude are at z = W.

  Physical Surface("Front") = {
    s_left_lower,
    s_left_upper,
    s_middle,
    s_right_lower,
    s_right_upper
  };

  Physical Surface("Back") = {
    ext_left_lower[0],
    ext_left_upper[0],
    ext_middle[0],
    ext_right_lower[0],
    ext_right_upper[0]
  };

  // ----------------------------------------------------------
  // External boundary surfaces selected by bounding boxes
  // ----------------------------------------------------------
  // These selections are less critical than the volume selection. If one of
  // these groups is empty or too broad, deal.II can still read the 3D cells,
  // but boundary ids may need further adjustment.

  s_bottom[] = Surface In BoundingBox {
    x0 - eps, y0 - eps, -eps,
    x3 + eps, y0 + eps, W + eps
  };

  s_top[] = Surface In BoundingBox {
    x0 - eps, yH - eps, -eps,
    x3 + eps, yH + eps, W + eps
  };

  s_left_boundary[] = Surface In BoundingBox {
    x0 - eps, y0 - eps, -eps,
    x0 + eps, yH + eps, W + eps
  };

  s_right_boundary[] = Surface In BoundingBox {
    x3 - eps, y0 - eps, -eps,
    x3 + eps, yH + eps, W + eps
  };

  // ----------------------------------------------------------
  // Left crack surfaces
  // ----------------------------------------------------------

  s_left_crack_lower[] = Surface In BoundingBox {
    x0 - eps, yl0 - eps, -eps,
    x1 + eps, yl0 + eps, W + eps
  };

  s_left_crack_tip[] = Surface In BoundingBox {
    x1 - eps, yl0 - eps, -eps,
    x1 + eps, yL2 + eps, W + eps
  };

  s_left_crack_upper[] = Surface In BoundingBox {
    x0 - eps, yL2 - eps, -eps,
    x1 + eps, yL2 + eps, W + eps
  };

  // ----------------------------------------------------------
  // Right crack surfaces
  // ----------------------------------------------------------

  s_right_crack_lower[] = Surface In BoundingBox {
    x2 - eps, yr0 - eps, -eps,
    x3 + eps, yr0 + eps, W + eps
  };

  s_right_crack_tip[] = Surface In BoundingBox {
    x2 - eps, yr0 - eps, -eps,
    x2 + eps, yr1 + eps, W + eps
  };

  s_right_crack_upper[] = Surface In BoundingBox {
    x2 - eps, yr1 - eps, -eps,
    x3 + eps, yr1 + eps, W + eps
  };

  Physical Surface("Bottom") = {s_bottom[]};
  Physical Surface("Top") = {s_top[]};

  Physical Surface("LeftBoundary") = {s_left_boundary[]};
  Physical Surface("RightBoundary") = {s_right_boundary[]};

  Physical Surface("LeftCrack") = {
    s_left_crack_lower[],
    s_left_crack_tip[],
    s_left_crack_upper[]
  };

  Physical Surface("RightCrack") = {
    s_right_crack_lower[],
    s_right_crack_tip[],
    s_right_crack_upper[]
  };

  // Optional separate crack-face groups.
  //
  // Physical Surface("LeftCrackLowerFace") = {s_left_crack_lower[]};
  // Physical Surface("LeftCrackTip")       = {s_left_crack_tip[]};
  // Physical Surface("LeftCrackUpperFace") = {s_left_crack_upper[]};
  //
  // Physical Surface("RightCrackLowerFace") = {s_right_crack_lower[]};
  // Physical Surface("RightCrackTip")       = {s_right_crack_tip[]};
  // Physical Surface("RightCrackUpperFace") = {s_right_crack_upper[]};

  Mesh 3;

EndIf
