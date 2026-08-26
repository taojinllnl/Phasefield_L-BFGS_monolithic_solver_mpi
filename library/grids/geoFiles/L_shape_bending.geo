Include "../prms/LShapeBending/250mmX250mmX250mmX250mm/50mmThickness30mmLoading.geo";




If (thickness > 0)

/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Geometry Generation    ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

L  = X1 + X2;   // the length of the plate
H  = Y1 + Y2;   // the distance between the crack tip and the top edge
L1 = X2 - a;    // the distance between the loading point and the inner corner

// calculate the number of the nodes along each transfinite curve
nCrackEleX1 = Floor(X1/lc)+1;          // the number of nodes along X1
nCrackEleL1 = Floor(L1/lc)+1;          // the number of nodes along L1
nCrackEleAX = Floor(a/lc)+1;           // the number of nodes along a

nCrackEleY1 = Floor(Y1/lc)+1;          // the number of nodes along Y1
nCrackEleY2 = Floor(Y2/lc)+1;          // the number of nodes along Y2
nCrackEleZ  = Floor(thickness/lc)+1;   // the number of nodes along thickness

// create the points distribution on the front face, z = 0
Point(1)  = {0,        0,    0, lc};
Point(2)  = {X1,       0,    0, lc};
Point(3)  = {X1,      Y1,    0, lc};
Point(4)  = {X1+L1,   Y1,    0, lc};
Point(5)  = {L,       Y1,    0, lc};
Point(6)  = {L,        H,    0, lc};
Point(7)  = {X1+L1,    H,    0, lc};
Point(8)  = {X1,       H,    0, lc};
Point(9)  = {0,        H,    0, lc};
Point(10) = {0,       Y1,    0, lc};

// create the points distribution on the back face, z = thickness
Point(101) = {0,        0,    thickness, lc};
Point(102) = {X1,       0,    thickness, lc};
Point(103) = {X1,      Y1,    thickness, lc};
Point(104) = {X1+L1,   Y1,    thickness, lc};
Point(105) = {L,       Y1,    thickness, lc};
Point(106) = {L,        H,    thickness, lc};
Point(107) = {X1+L1,    H,    thickness, lc};
Point(108) = {X1,       H,    thickness, lc};
Point(109) = {0,        H,    thickness, lc};
Point(110) = {0,       Y1,    thickness, lc};

// create the lines on the front face, z = 0
Line(1) =  {1,   2 };
Line(2) =  {2,   3 };
Line(3) =  {3,   4 };
Line(4) =  {4,   5 };
Line(5) =  {5,   6 };
Line(6) =  {6,   7 };
Line(7) =  {7,   8 };
Line(8) =  {8,   9 };
Line(9) =  {9,   10};
Line(10) = {10,  1 };
Line(11) = {10,  3 };
Line(12) = {3,   8 };
Line(13) = {4,   7 };

// create the corresponding lines on the back face, z = thickness
Line(101) = {101, 102};
Line(102) = {102, 103};
Line(103) = {103, 104};
Line(104) = {104, 105};
Line(105) = {105, 106};
Line(106) = {106, 107};
Line(107) = {107, 108};
Line(108) = {108, 109};
Line(109) = {109, 110};
Line(110) = {110, 101};
Line(111) = {110, 103};
Line(112) = {103, 108};
Line(113) = {104, 107};

// create the lines along the thickness direction
Line(201) = {1,   101};
Line(202) = {2,   102};
Line(203) = {3,   103};
Line(204) = {4,   104};
Line(205) = {5,   105};
Line(206) = {6,   106};
Line(207) = {7,   107};
Line(208) = {8,   108};
Line(209) = {9,   109};
Line(210) = {10,  110};

// create 4 surfaces on the front face, z = 0
Curve Loop(1) = {1, 2, -11, 10};
Plane Surface(1) = {1};
Transfinite Surface{1} = {1, 2, 3, 10};

Curve Loop(2) = {11, 12, 8, 9};
Plane Surface(2) = {2};
Transfinite Surface{2} = {10, 3, 8, 9};

Curve Loop(3) = {3, 13, 7, -12};
Plane Surface(3) = {3};
Transfinite Surface{3} = {3, 4, 7, 8};

Curve Loop(4) = {4, 5, 6, -13};
Plane Surface(4) = {4};
Transfinite Surface{4} = {4, 5, 6, 7};

// create 4 surfaces on the back face, z = thickness
Curve Loop(101) = {101, 102, -111, 110};
Plane Surface(101) = {101};
Transfinite Surface{101} = {101, 102, 103, 110};

Curve Loop(102) = {111, 112, 108, 109};
Plane Surface(102) = {102};
Transfinite Surface{102} = {110, 103, 108, 109};

Curve Loop(103) = {103, 113, 107, -112};
Plane Surface(103) = {103};
Transfinite Surface{103} = {103, 104, 107, 108};

Curve Loop(104) = {104, 105, 106, -113};
Plane Surface(104) = {104};
Transfinite Surface{104} = {104, 105, 106, 107};

// create side surfaces by connecting each front-face curve to its back-face curve
Curve Loop(201) = {1,   202, -101, -201};
Plane Surface(201) = {201};
Transfinite Surface{201} = {1, 2, 102, 101};

Curve Loop(202) = {2,   203, -102, -202};
Plane Surface(202) = {202};
Transfinite Surface{202} = {2, 3, 103, 102};

Curve Loop(203) = {3,   204, -103, -203};
Plane Surface(203) = {203};
Transfinite Surface{203} = {3, 4, 104, 103};

Curve Loop(204) = {4,   205, -104, -204};
Plane Surface(204) = {204};
Transfinite Surface{204} = {4, 5, 105, 104};

Curve Loop(205) = {5,   206, -105, -205};
Plane Surface(205) = {205};
Transfinite Surface{205} = {5, 6, 106, 105};

Curve Loop(206) = {6,   207, -106, -206};
Plane Surface(206) = {206};
Transfinite Surface{206} = {6, 7, 107, 106};

Curve Loop(207) = {7,   208, -107, -207};
Plane Surface(207) = {207};
Transfinite Surface{207} = {7, 8, 108, 107};

Curve Loop(208) = {8,   209, -108, -208};
Plane Surface(208) = {208};
Transfinite Surface{208} = {8, 9, 109, 108};

Curve Loop(209) = {9,   210, -109, -209};
Plane Surface(209) = {209};
Transfinite Surface{209} = {9, 10, 110, 109};

Curve Loop(210) = {10,  201, -110, -210};
Plane Surface(210) = {210};
Transfinite Surface{210} = {10, 1, 101, 110};

Curve Loop(211) = {11,  203, -111, -210};
Plane Surface(211) = {211};
Transfinite Surface{211} = {10, 3, 103, 110};

Curve Loop(212) = {12,  208, -112, -203};
Plane Surface(212) = {212};
Transfinite Surface{212} = {3, 8, 108, 103};

Curve Loop(213) = {13,  207, -113, -204};
Plane Surface(213) = {213};
Transfinite Surface{213} = {4, 7, 107, 104};

// create 4 volumes by the corresponding front, back, and side surfaces
Surface Loop(1) = {1, 101, 201, 202, 211, 210};
Volume(1) = {1};
Transfinite Volume{1} = {1, 2, 3, 10, 101, 102, 103, 110};

Surface Loop(2) = {2, 102, 211, 212, 208, 209};
Volume(2) = {2};
Transfinite Volume{2} = {10, 3, 8, 9, 110, 103, 108, 109};

Surface Loop(3) = {3, 103, 203, 213, 207, 212};
Volume(3) = {3};
Transfinite Volume{3} = {3, 4, 7, 8, 103, 104, 107, 108};

Surface Loop(4) = {4, 104, 204, 205, 206, 213};
Volume(4) = {4};
Transfinite Volume{4} = {4, 5, 6, 7, 104, 105, 106, 107};

// assign the specified number of nodes along the curves by the line indices
Transfinite Curve {1, 11, 8, 101, 111, 108} = nCrackEleX1 Using Progression 1;
Transfinite Curve {3, 7, 103, 107}         = nCrackEleL1 Using Progression 1;
Transfinite Curve {4, 6, 104, 106}         = nCrackEleAX Using Progression 1;
Transfinite Curve {10, 2, 110, 102}        = nCrackEleY1 Using Progression 1;
Transfinite Curve {9, 12, 13, 5, 109, 112, 113, 105} = nCrackEleY2 Using Progression 1;
Transfinite Curve {201:210}                = nCrackEleZ Using Progression 1;

// generate structured quadrilateral surfaces and hexahedral volumes
Recombine Surface{1:4, 101:104, 201:213};
Recombine Volume{1:4};


/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Boundary conditions    ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/* */
// The original 2D Physical Curves are converted to 3D Physical Surfaces.
Physical Surface("upper_edge", 14)  = {206, 207, 208};
Physical Surface("left_edge", 15)   = {209, 210};
Physical Surface("bottom_edge", 17) = {201};
/* */
/////////////////////////////////////////////////////////////////////////////////
////////////////////////       Material set-up       ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/* */
Physical Volume(19) = {1:4};
/* */

Mesh 3;



Else


/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Geometry Generation    ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

L  = X1 + X2;    // the length of the plate
H  = Y1 + Y2;    // the distance between the crack tip and the top edge
L1 = X2 - a;    // the distance between the loading point and the inner corner

// calculate the number of the elements
nCrackEleX1 = Floor(X1/lc)+1;       // the number of element along X1
nCrackEleL1 = Floor(L1/lc)+1;       // the number of element along L1
nCrackEleAX = Floor(a/lc)+1;       // the number of element along a

nCrackEleY1 = Floor(Y1/lc)+1;       // the number of element along Y1
nCrackEleY2 = Floor(Y2/lc)+1;       // the number of element along Y2

// create the points distribution
Point(1)  = {0,        0,    0, lc};
Point(2)  = {X1,       0,    0, lc};
Point(3)  = {X1,      Y1,    0, lc};
Point(4)  = {X1+L1,   Y1,    0, lc};
Point(5)  = {L,       Y1,    0, lc};
Point(6)  = {L,        H,    0, lc};
Point(7)  = {X1+L1,    H,    0, lc};
Point(8)  = {X1,       H,    0, lc};
Point(9)  = {0,        H,    0, lc};
Point(10) = {0,       Y1,    0, lc};

// create the lines for the geometry
Line(1) =  {1,   2 };
Line(2) =  {2,   3 };
Line(3) =  {3,   4 };
Line(4) =  {4,   5 };
Line(5) =  {5,   6 };
Line(6) =  {6,   7 };
Line(7) =  {7,   8 };
Line(8) =  {8,   9 };
Line(9) =  {9,   10};
Line(10) = {10,  1 };
Line(11) = {10,  3 };
Line(12) = {3,   8 };
Line(13) = {4,   7 };

// create 4 surfaces by the lines' indexes
Curve Loop(1) = {1, 2, -11, 10};          // create the curve loop by the lines' indices
Plane Surface(1) = {1};                   // create a surface by the curve loop
Transfinite Surface{1} = {1, 2, 3, 10};   // create the structured surface by points' indices

Curve Loop(2) = {11, 12, 8, 9};
Plane Surface(2) = {2};
Transfinite Surface{2} = {10, 3, 8, 9};

Curve Loop(3) = {3, 13, 7, -12};
Plane Surface(3) = {3};
Transfinite Surface{3} = {3, 4, 7, 8};

Curve Loop(4) = {4, 5, 6, -13};
Plane Surface(4) = {4};
Transfinite Surface{4} = {4, 5, 6, 7};

// assign the specified number of the elements along the curves by the lines' indices
Transfinite Curve {1, 11, 8}    = nCrackEleX1 Using Progression 1;
Transfinite Curve {3, 7}    = nCrackEleL1 Using Progression 1;
Transfinite Curve {4, 6}    = nCrackEleAX Using Progression 1;
Transfinite Curve {10, 2} = nCrackEleY1 Using Progression 1;
Transfinite Curve {9, 12, 13, 5}    = nCrackEleY2 Using Progression 1;

// generate the structured rectangulars by combining the triangles
Recombine Surface{1:4};


/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Boundary conditions    ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/* */
Physical Curve("upper_edge", 14)  = {6, 7, 8};
Physical Curve("left_edge", 15)   = {9, 10};
Physical Curve("bottom_edge", 17) = {1    };
/* */
/////////////////////////////////////////////////////////////////////////////////
////////////////////////       Material set-up       ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/* */
Physical Surface(19) = {1:4};
/* */

Mesh 2;

EndIf
