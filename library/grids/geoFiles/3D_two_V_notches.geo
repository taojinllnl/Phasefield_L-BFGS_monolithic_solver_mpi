//Include "../throughVNotchedCube/1mmX1mmX1mm/0.3X0.3mmCracks/0.0mmDistance.geo";

Include "../edgeNothcedCube/300mmX300mmX300mm/100mmX100mmCracks/30mmDistance.geo";

SetFactory("Built-in");


// =====================================================
// Through-thickness crack branch
// =====================================================
If(through_crack == 1)

// This branch builds the x-z front-view geometry first, then extrudes it along
// the full y direction. Therefore, the cracks are through-thickness cracks in y.
// The original partial corner-crack geometry is preserved in the Else branch.

If(H1 != H2)

// Preserve the height-ordering convention used by the original file:
// C1 is treated as the upper crack and C2 as the lower crack.
If(H1 < H2)
  H1Temp = H1;
  H1 = H2;
  H2 = H1Temp;
EndIf

// ---------------------
// mesh density
// ---------------------
NX1 = Ceil(L1/lc) + 1;
NXM = Ceil((L-L1-L2)/lc) + 1;
NX2 = Ceil(L2/lc) + 1;
NY  = Ceil(W/lc) + 1;

NZ1 = Ceil(H2/lc) + 1;
NZ2 = Ceil((H1-H2)/lc) + 1;
NZ3 = Ceil((H-H1)/lc) + 1;

If(NX1 < 2) NX1 = 2; EndIf
If(NXM < 2) NXM = 2; EndIf
If(NX2 < 2) NX2 = 2; EndIf
If(NY  < 2) NY  = 2; EndIf
If(NZ1 < 2) NZ1 = 2; EndIf
If(NZ2 < 2) NZ2 = 2; EndIf
If(NZ3 < 2) NZ3 = 2; EndIf

Mesh.Smoothing = 100;

// ---------------------
// coordinates
// ---------------------
x0 = 0.0;
x1 = L1;
x2 = L - L2;
x3 = L;

y0 = 0.0;

z0 = 0.0;
z1 = H2; // C2 crack height
z2 = H1; // C1 crack height
z3 = H;

d1 = C1 / 2.0;
d2 = C2 / 2.0;

// =====================================================
// 2D front-view points in the x-z plane, y = 0
// =====================================================

// z = z0
Point(100) = {x0,y0,z0,lc};
Point(101) = {x1,y0,z0,lc};
Point(102) = {x2,y0,z0,lc};
Point(103) = {x3,y0,z0,lc};

// z = z1 = H2, C2 is split at x = x3
Point(200) = {x0,y0,z1,lc};
Point(201) = {x1,y0,z1,lc};
Point(202) = {x2,y0,z1,lc};

// z = z2 = H1, C1 is split at x = x0
Point(301) = {x1,y0,z2,lc};
Point(302) = {x2,y0,z2,lc};
Point(303) = {x3,y0,z2,lc};

// z = z3
Point(400) = {x0,y0,z3,lc};
Point(401) = {x1,y0,z3,lc};
Point(402) = {x2,y0,z3,lc};
Point(403) = {x3,y0,z3,lc};

// split points for crack openings
Point(900) = {x3,y0,z1-d2,lc}; // C2 lower side
Point(901) = {x3,y0,z1+d2,lc}; // C2 upper side

Point(902) = {x0,y0,z2-d1,lc}; // C1 lower side
Point(903) = {x0,y0,z2+d1,lc}; // C1 upper side

// =====================================================
// 2D front-view lines
// =====================================================

// x-lines
Line(1000) = {100,101};
Line(1001) = {101,102};
Line(1002) = {102,103};

Line(1100) = {200,201};
Line(1101) = {201,202};
Line(1800) = {202,900}; // C2 lower crack face
Line(1801) = {202,901}; // C2 upper crack face

Line(1810) = {902,301}; // C1 lower crack face
Line(1811) = {903,301}; // C1 upper crack face
Line(1201) = {301,302};
Line(1202) = {302,303};

Line(1300) = {400,401};
Line(1301) = {401,402};
Line(1302) = {402,403};

// z-lines, layer 1: z0 -> z1
Line(1900) = {100,200};
Line(1901) = {101,201};
Line(1902) = {102,202};
Line(1903) = {103,900}; // reaches C2 lower side

// z-lines, layer 2: z1 -> z2
Line(2000) = {200,902}; // reaches C1 lower side
Line(2001) = {201,301};
Line(2002) = {202,302};
Line(2003) = {901,303}; // starts from C2 upper side

// z-lines, layer 3: z2 -> z3
Line(2100) = {903,400}; // starts from C1 upper side
Line(2101) = {301,401};
Line(2102) = {302,402};
Line(2103) = {303,403};

// =====================================================
// 2D front-view surfaces
// =====================================================

// layer 1: z0 -> z1
Curve Loop(3000) = {1000, 1901, -1100, -1900};
Plane Surface(3000) = {3000};

Curve Loop(3001) = {1001, 1902, -1101, -1901};
Plane Surface(3001) = {3001};

Curve Loop(3002) = {1002, 1903, -1800, -1902};
Plane Surface(3002) = {3002};

// layer 2: z1 -> z2
Curve Loop(3100) = {1100, 2001, -1810, -2000};
Plane Surface(3100) = {3100};

Curve Loop(3101) = {1101, 2002, -1201, -2001};
Plane Surface(3101) = {3101};

Curve Loop(3102) = {1801, 2003, -1202, -2002};
Plane Surface(3102) = {3102};

// layer 3: z2 -> z3
Curve Loop(3200) = {1811, 2101, -1300, -2100};
Plane Surface(3200) = {3200};

Curve Loop(3201) = {1201, 2102, -1301, -2101};
Plane Surface(3201) = {3201};

Curve Loop(3202) = {1202, 2103, -1302, -2102};
Plane Surface(3202) = {3202};

// =====================================================
// Transfinite curves and surfaces in the front view
// =====================================================

// x-direction
Transfinite Curve{1000,1100,1300} = NX1;
Transfinite Curve{1001,1101,1201,1301} = NXM;
Transfinite Curve{1002,1202,1302} = NX2;
Transfinite Curve{1800,1801} = NX2;
Transfinite Curve{1810,1811} = NX1;

// z-direction
Transfinite Curve{1900,1901,1902,1903} = NZ1;
Transfinite Curve{2000,2001,2002,2003} = NZ2;
Transfinite Curve{2100,2101,2102,2103} = NZ3;

For sid In {3000:3002}
  Transfinite Surface{sid};
  Recombine Surface{sid};
EndFor

For sid In {3100:3102}
  Transfinite Surface{sid};
  Recombine Surface{sid};
EndFor

For sid In {3200:3202}
  Transfinite Surface{sid};
  Recombine Surface{sid};
EndFor

// =====================================================
// Extrude the front-view surfaces through the full width W
// =====================================================

out3000[] = Extrude {0, W, 0} { Surface{3000}; Layers{NY-1}; Recombine; };
out3001[] = Extrude {0, W, 0} { Surface{3001}; Layers{NY-1}; Recombine; };
out3002[] = Extrude {0, W, 0} { Surface{3002}; Layers{NY-1}; Recombine; };

out3100[] = Extrude {0, W, 0} { Surface{3100}; Layers{NY-1}; Recombine; };
out3101[] = Extrude {0, W, 0} { Surface{3101}; Layers{NY-1}; Recombine; };
out3102[] = Extrude {0, W, 0} { Surface{3102}; Layers{NY-1}; Recombine; };

out3200[] = Extrude {0, W, 0} { Surface{3200}; Layers{NY-1}; Recombine; };
out3201[] = Extrude {0, W, 0} { Surface{3201}; Layers{NY-1}; Recombine; };
out3202[] = Extrude {0, W, 0} { Surface{3202}; Layers{NY-1}; Recombine; };

Coherence;

Physical Volume("materialID", 0) = {
  out3000[1], out3001[1], out3002[1],
  out3100[1], out3101[1], out3102[1],
  out3200[1], out3201[1], out3202[1]
};

Mesh 3;

Else // H1 == H2 in through-crack mode

Hc = 0.5 * (H1 + H2);

// ---------------------
// mesh density
// ---------------------
NX1 = Ceil(L1/lc) + 1;
NXM = Ceil((L-L1-L2)/lc) + 1;
NX2 = Ceil(L2/lc) + 1;
NY  = Ceil(W/lc) + 1;

NZ1 = Ceil(Hc/lc) + 1;
NZ2 = Ceil((H-Hc)/lc) + 1;

If(NX1 < 2) NX1 = 2; EndIf
If(NXM < 2) NXM = 2; EndIf
If(NX2 < 2) NX2 = 2; EndIf
If(NY  < 2) NY  = 2; EndIf
If(NZ1 < 2) NZ1 = 2; EndIf
If(NZ2 < 2) NZ2 = 2; EndIf

Mesh.Smoothing = 100;

// ---------------------
// coordinates
// ---------------------
x0 = 0.0;
x1 = L1;
x2 = L - L2;
x3 = L;

y0 = 0.0;

z0 = 0.0;
z1 = Hc;
z2 = H;

d1 = C1 / 2.0;
d2 = C2 / 2.0;

// =====================================================
// 2D front-view points in the x-z plane, y = 0
// =====================================================

// z = z0
Point(100) = {x0,y0,z0,lc};
Point(101) = {x1,y0,z0,lc};
Point(102) = {x2,y0,z0,lc};
Point(103) = {x3,y0,z0,lc};

// z = z1 = Hc, both C1 and C2 are split
Point(201) = {x1,y0,z1,lc};
Point(202) = {x2,y0,z1,lc};

// z = z2 = H
Point(300) = {x0,y0,z2,lc};
Point(301) = {x1,y0,z2,lc};
Point(302) = {x2,y0,z2,lc};
Point(303) = {x3,y0,z2,lc};

// split points for crack openings
Point(900) = {x3,y0,z1-d2,lc}; // C2 lower side
Point(901) = {x3,y0,z1+d2,lc}; // C2 upper side

Point(902) = {x0,y0,z1-d1,lc}; // C1 lower side
Point(903) = {x0,y0,z1+d1,lc}; // C1 upper side

// =====================================================
// 2D front-view lines
// =====================================================

// x-lines
Line(1000) = {100,101};
Line(1001) = {101,102};
Line(1002) = {102,103};

Line(1810) = {902,201}; // C1 lower crack face
Line(1811) = {903,201}; // C1 upper crack face
Line(1101) = {201,202};
Line(1800) = {202,900}; // C2 lower crack face
Line(1801) = {202,901}; // C2 upper crack face

Line(1200) = {300,301};
Line(1201) = {301,302};
Line(1202) = {302,303};

// z-lines, lower layer
Line(1900) = {100,902}; // reaches C1 lower side
Line(1901) = {101,201};
Line(1902) = {102,202};
Line(1903) = {103,900}; // reaches C2 lower side

// z-lines, upper layer
Line(2000) = {903,300}; // starts from C1 upper side
Line(2001) = {201,301};
Line(2002) = {202,302};
Line(2003) = {901,303}; // starts from C2 upper side

// =====================================================
// 2D front-view surfaces
// =====================================================

// lower layer
Curve Loop(3000) = {1000, 1901, -1810, -1900};
Plane Surface(3000) = {3000};

Curve Loop(3001) = {1001, 1902, -1101, -1901};
Plane Surface(3001) = {3001};

Curve Loop(3002) = {1002, 1903, -1800, -1902};
Plane Surface(3002) = {3002};

// upper layer
Curve Loop(3100) = {1811, 2001, -1200, -2000};
Plane Surface(3100) = {3100};

Curve Loop(3101) = {1101, 2002, -1201, -2001};
Plane Surface(3101) = {3101};

Curve Loop(3102) = {1801, 2003, -1202, -2002};
Plane Surface(3102) = {3102};

// =====================================================
// Transfinite curves and surfaces in the front view
// =====================================================

// x-direction
Transfinite Curve{1000,1200} = NX1;
Transfinite Curve{1001,1101,1201} = NXM;
Transfinite Curve{1002,1202} = NX2;
Transfinite Curve{1810,1811} = NX1;
Transfinite Curve{1800,1801} = NX2;

// z-direction
Transfinite Curve{1900,1901,1902,1903} = NZ1;
Transfinite Curve{2000,2001,2002,2003} = NZ2;

For sid In {3000:3002}
  Transfinite Surface{sid};
  Recombine Surface{sid};
EndFor

For sid In {3100:3102}
  Transfinite Surface{sid};
  Recombine Surface{sid};
EndFor

// =====================================================
// Extrude the front-view surfaces through the full width W
// =====================================================

out3000[] = Extrude {0, W, 0} { Surface{3000}; Layers{NY-1}; Recombine; };
out3001[] = Extrude {0, W, 0} { Surface{3001}; Layers{NY-1}; Recombine; };
out3002[] = Extrude {0, W, 0} { Surface{3002}; Layers{NY-1}; Recombine; };

out3100[] = Extrude {0, W, 0} { Surface{3100}; Layers{NY-1}; Recombine; };
out3101[] = Extrude {0, W, 0} { Surface{3101}; Layers{NY-1}; Recombine; };
out3102[] = Extrude {0, W, 0} { Surface{3102}; Layers{NY-1}; Recombine; };

Coherence;

Physical Volume("materialID", 0) = {
  out3000[1], out3001[1], out3002[1],
  out3100[1], out3101[1], out3102[1]
};

Mesh 3;

EndIf // H1/H2 branch in through-crack mode

Else // through_crack == 0: original partial-crack geometry

// =====================================================
// mesh density
// =====================================================
If(H1 != H2)
// adjust the H1 and H2 to guarantee the H1 is smaller than H2
If(H1 < H2)
  H1Temp = H1;
  H1 = H2;
  H2 = H1Temp;
EndIf

// the numbers of points to separate the sub-regions
NX1 = Ceil(L1/lc);            // points in x-band L1
NXM = Ceil((L-L1-L2)/lc);     // points in middle x-band
NX2 = Ceil(L2/lc);            // points in x-band L2

NY2 = Ceil(W2/lc);            // points in bottom y-band W2
NYM = Ceil((W-W1-W2)/lc);     // points in middle y-band
NY1 = Ceil(W1/lc);            // points in top y-band W1

NZ1 = Ceil(H2/lc);            // points in z-band [0, H2]
NZ2 = Ceil((H1-H2)/lc);       // points in z-band [H2, H1]
NZ3 = Ceil((H-H1)/lc);        // points in z-band [H1, H]


// at least 2 points on each sub-regions
If(NX1 < 1) NX1 = 1; EndIf
If(NXM < 1) NXM = 1; EndIf
If(NX2 < 1) NX2 = 1; EndIf

If(NY2 < 1) NY2 = 1; EndIf
If(NYM < 1) NYM = 1; EndIf
If(NY1 < 1) NY1 = 1; EndIf

If(NZ1 < 1) NZ1 = 1; EndIf
If(NZ2 < 1) NZ2 = 1; EndIf
If(NZ3 < 1) NZ3 = 1; EndIf


Mesh.Smoothing = 100;

// =====================================================
// coordinates
// =====================================================

// x partition: [0, L1] [middle] [L-L2, L]
x0 = 0.0;
x1 = L1;
x2 = L - L2;
x3 = L;

// y partition: [0, W2] [middle] [W-W1, W]
y0 = 0.0;
y1 = W2;
y2 = W - W1;
y3 = W;

// z partition: [0, H2] [H2, H1] [H1, H]
z0 = 0.0;
z1 = H2;
z2 = H1;
z3 = H;

// half openings
d1 = C1 / 2.0;
d2 = C2 / 2.0;

// =====================================================
// Point numbering convention
//
//            z      y   x
// z = z0 : 100 + 10*j + i
// z = z1 : 200 + 10*j + i
// z = z2 : 300 + 10*j + i
// z = z3 : 400 + 10*j + i
//
// i = 0..3 in x, j = 0..3 in y
//
// split points for crack surfaces:
// 900 : C2 lower-side corner  at (x3,y0,z1-d2)
// 901 : C2 upper-side corner  at (x3,y0,z1+d2)
// 902 : C1 lower-side corner  at (x0,y3,z2-d1)
// 903 : C1 upper-side corner  at (x0,y3,z2+d1)
//
//
//  The point ids on *-th plane in the z-direction:
//     ^ y
//     |
//     |(*30)   |(*31)|(*32)   |(*33)
//  y3 +--------+-----+--------+
//     |        |     |        |
//     |        |     |        |
//     |        |     |        |
//     |        |     |        |
//     |(*20)   |(*21)|(*22)   |(*23)
//  y2 +--------+-----+--------+
//     |        |     |        |
//     |        |     |        |
//     |(*10)   |(*11)|(*12)   |(*13)
//  y1 +--------+-----+--------+
//     |        |     |        |
//     |        |     |        |
//     |        |     |        |
//     |        |     |        |
//     |(*00)   |(*01)|(*02)   |(*03)
//  y0 +--------+-----+--------+  -----> x
//     |x0      |x1   |x2      |x3
// =====================================================

// ---------------------
// z = z0
// ---------------------
Point(100) = {x0,y0,z0,lc};
Point(101) = {x1,y0,z0,lc};
Point(102) = {x2,y0,z0,lc};
Point(103) = {x3,y0,z0,lc};

Point(110) = {x0,y1,z0,lc};
Point(111) = {x1,y1,z0,lc};
Point(112) = {x2,y1,z0,lc};
Point(113) = {x3,y1,z0,lc};

Point(120) = {x0,y2,z0,lc};
Point(121) = {x1,y2,z0,lc};
Point(122) = {x2,y2,z0,lc};
Point(123) = {x3,y2,z0,lc};

Point(130) = {x0,y3,z0,lc};
Point(131) = {x1,y3,z0,lc};
Point(132) = {x2,y3,z0,lc};
Point(133) = {x3,y3,z0,lc};

// ---------------------
// z = z1 = H2
// ---------------------
Point(200) = {x0,y0,z1,lc};
Point(201) = {x1,y0,z1,lc};
Point(202) = {x2,y0,z1,lc};
//Point(203) = {x3,y0,z1,lc}; // ghost point: the point at middle plane of the crack. This point can be ignored.

Point(210) = {x0,y1,z1,lc};
Point(211) = {x1,y1,z1,lc};
Point(212) = {x2,y1,z1,lc};
Point(213) = {x3,y1,z1,lc};

Point(220) = {x0,y2,z1,lc};
Point(221) = {x1,y2,z1,lc};
Point(222) = {x2,y2,z1,lc};
Point(223) = {x3,y2,z1,lc};

Point(230) = {x0,y3,z1,lc};
Point(231) = {x1,y3,z1,lc};
Point(232) = {x2,y3,z1,lc};
Point(233) = {x3,y3,z1,lc};

// ---------------------
// z = z2 = H1
// ---------------------
Point(300) = {x0,y0,z2,lc};
Point(301) = {x1,y0,z2,lc};
Point(302) = {x2,y0,z2,lc};
Point(303) = {x3,y0,z2,lc};

Point(310) = {x0,y1,z2,lc};
Point(311) = {x1,y1,z2,lc};
Point(312) = {x2,y1,z2,lc};
Point(313) = {x3,y1,z2,lc};

Point(320) = {x0,y2,z2,lc};
Point(321) = {x1,y2,z2,lc};
Point(322) = {x2,y2,z2,lc};
Point(323) = {x3,y2,z2,lc};

//Point(330) = {x0,y3,z2,lc}; // the point at middle plane of the crack
Point(331) = {x1,y3,z2,lc};
Point(332) = {x2,y3,z2,lc};
Point(333) = {x3,y3,z2,lc};

// ---------------------
// z = z3 = H
// ---------------------
Point(400) = {x0,y0,z3,lc};
Point(401) = {x1,y0,z3,lc};
Point(402) = {x2,y0,z3,lc};
Point(403) = {x3,y0,z3,lc};

Point(410) = {x0,y1,z3,lc};
Point(411) = {x1,y1,z3,lc};
Point(412) = {x2,y1,z3,lc};
Point(413) = {x3,y1,z3,lc};

Point(420) = {x0,y2,z3,lc};
Point(421) = {x1,y2,z3,lc};
Point(422) = {x2,y2,z3,lc};
Point(423) = {x3,y2,z3,lc};

Point(430) = {x0,y3,z3,lc};
Point(431) = {x1,y3,z3,lc};
Point(432) = {x2,y3,z3,lc};
Point(433) = {x3,y3,z3,lc};

// ---------------------
// points for the crack openning
// ---------------------
Point(900) = {x3,y0,z1-d2,lc}; // C2 lower side
Point(901) = {x3,y0,z1+d2,lc}; // C2 upper side

Point(902) = {x0,y3,z2-d1,lc}; // C1 lower side
Point(903) = {x0,y3,z2+d1,lc}; // C1 upper side

// =====================================================
// Lines on z = *-th planes
//
// Line numbering convension:
//
//      j: for y, i for x
//      1*ji: 0 <= * <= 3: lines parallel to x-axis
//      1*ji: 4 <= * <= 7: lines parallel to y-axis
//      18**: crack opening
// =====================================================

// x-lines on z0, z1, z2, z3
For j In {0:3}
  For i In {0:2}
    Line(1000 + 10*j + i) = {100 + 10*j + i, 100 + 10*j + i + 1};

    If(j == 0 && i == 2)
        // skip the ghost point in the middle of the crack openning
    Else
        Line(1100 + 10*j + i) = {200 + 10*j + i, 200 + 10*j + i + 1};
    EndIf
    
    If(j == 3 && i == 0)
        // skip the ghost point in the middle of the crack openning
    Else
            Line(1200 + 10*j + i) = {300 + 10*j + i, 300 + 10*j + i + 1};
    EndIf
    Line(1300 + 10*j + i) = {400 + 10*j + i, 400 + 10*j + i + 1};
  EndFor
EndFor

// y-lines on z0, z1, z2, z3
For i In {0:3}
  For j In {0:2}
    Line(1400 + 10*i + j) = {100 + 10*j + i, 100 + 10*(j+1) + i};
    If(j == 0 && i == 3)
        // skip the ghost point in the middle of the crack openning
    Else
        Line(1500 + 10*i + j) = {200 + 10*j + i, 200 + 10*(j+1) + i};
    EndIf
    
    If(j == 2 && i == 0)
        // skip the ghost point in the middle of the crack openning
    Else
        Line(1600 + 10*i + j) = {300 + 10*j + i, 300 + 10*(j+1) + i};
    EndIf
    
    Line(1700 + 10*i + j) = {400 + 10*j + i, 400 + 10*(j+1) + i};
  EndFor
EndFor

// =====================================================
// lines on crack opennings
// =====================================================

// C2 patch = bottom-right corner patch at z = H2
// patch corners: (x2,y0) (x3,y0) (x3,y1) (x2,y1)
Line(1800) = {202,900}; // south edge, lower crack face
Line(1801) = {202,901}; // south edge, upper crack face
Line(1802) = {900,213}; // east  edge, lower crack face
Line(1803) = {901,213}; // east  edge, upper crack face

// C1 patch = top-left corner patch at z = H1
// patch corners: (x0,y2) (x1,y2) (x1,y3) (x0,y3)
Line(1810) = {320,902}; // west  edge, lower crack face
Line(1811) = {320,903}; // west  edge, upper crack face
Line(1812) = {902,331}; // north edge, lower crack face
Line(1813) = {903,331}; // north edge, upper crack face

// =====================================================
// Vertical lines
// =====================================================

// layer 1 : z0 -> z1
For j In {0:3}
    For i In {0:3}
        pTop = 200 + 10*j + i;
        If(i == 3 && j == 0)
            pTop = 900; // C2 lower side
            EndIf
        Line(1900 + 10*j + i) = {100 + 10*j + i, pTop};
    EndFor
EndFor

// layer 2 : z1 -> z2
For j In {0:3}
    For i In {0:3}
        pBot = 200 + 10*j + i;
        pTop = 300 + 10*j + i;

        If(i == 3 && j == 0)
            pBot = 901; // C2 upper side
        EndIf

        If(i == 0 && j == 3)
            pTop = 902; // C1 lower side
        EndIf

        Line(2000 + 10*j + i) = {pBot, pTop};
    EndFor
EndFor

// layer 3 : z2 -> z3
For j In {0:3}
    For i In {0:3}
        pBot = 300 + 10*j + i;
        If(i == 0 && j == 3)
            pBot = 903; // C1 upper side
        EndIf
        Line(2100 + 10*j + i) = {pBot, 400 + 10*j + i};
    EndFor
EndFor

// =====================================================
// Horizontal surfaces
// =====================================================

// z = z0
For j In {0:2}
  For i In {0:2}
    sid = 3000 + 10*j + i;
    Curve Loop(sid) = {
      1000 + 10*j + i,
      1400 + 10*(i+1) + j,
     -(1000 + 10*(j+1) + i),
     -(1400 + 10*i + j)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// z = z1 intact patches except C2 patch (i=2,j=0)
For j In {0:2}
  For i In {0:2}
    If(!(i == 2 && j == 0))
      sid = 3100 + 10*j + i;
      Curve Loop(sid) = {
        1100 + 10*j + i,
        1500 + 10*(i+1) + j,
       -(1100 + 10*(j+1) + i),
       -(1500 + 10*i + j)
      };
      Plane Surface(sid) = {sid};
      Transfinite Surface{sid};
      Recombine Surface{sid};
    EndIf
  EndFor
EndFor

// z = z1, C2 lower crack face
Curve Loop(3190) = {1800, 1802, -1112, -1520};
Surface(3190) = {3190};
Transfinite Surface{3190} = {202, 900, 213, 212};
Recombine Surface{3190};

// z = z1, C2 upper crack face
Curve Loop(3191) = {1801, 1803, -1112, -1520};
Surface(3191) = {3191};
Transfinite Surface{3191} = {202, 901, 213, 212};
Recombine Surface{3191};

// z = z2 intact patches except C1 patch (i=0,j=2)
For j In {0:2}
  For i In {0:2}
    If(!(i == 0 && j == 2))
      sid = 3200 + 10*j + i;
      Curve Loop(sid) = {
        1200 + 10*j + i,
        1600 + 10*(i+1) + j,
       -(1200 + 10*(j+1) + i),
       -(1600 + 10*i + j)
      };
      Plane Surface(sid) = {sid};
      Transfinite Surface{sid};
      Recombine Surface{sid};
    EndIf
  EndFor
EndFor

// z = z2, C1 lower crack face
Curve Loop(3290) = {1220, 1612, -1812, -1810};
Surface(3290) = {3290};
Transfinite Surface{3290} = {320, 321, 331, 902};
Recombine Surface{3290};

// z = z2, C1 upper crack face
Curve Loop(3291) = {1220, 1612, -1813, -1811};
Surface(3291) = {3291};
Transfinite Surface{3291} = {320, 321, 331, 903};
Recombine Surface{3291};

// z = z3
For j In {0:2}
  For i In {0:2}
    sid = 3300 + 10*j + i;
    Curve Loop(sid) = {
      1300 + 10*j + i,
      1700 + 10*(i+1) + j,
     -(1300 + 10*(j+1) + i),
     -(1700 + 10*i + j)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// =====================================================
// Vertical x-z surfaces  (fixed y)
// =====================================================

// layer 1 : z0 -> z1
For j In {0:3}
  For i In {0:2}
    topx = 1100 + 10*j + i;
    If(j == 0 && i == 2)
      topx = 1800; // C2 lower special south edge
    EndIf

    sid = 4000 + 10*j + i;
    Curve Loop(sid) = {
      1000 + 10*j + i,
      1900 + 10*j + (i+1),
     -topx,
     -(1900 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// layer 2 : z1 -> z2
For j In {0:3}
  For i In {0:2}
    botx = 1100 + 10*j + i;
    topx = 1200 + 10*j + i;

    If(j == 0 && i == 2)
      botx = 1801; // C2 upper special south edge
    EndIf

    If(j == 3 && i == 0)
      topx = 1812; // C1 lower special north edge
    EndIf

    sid = 4100 + 10*j + i;
    Curve Loop(sid) = {
      botx,
      2000 + 10*j + (i+1),
     -topx,
     -(2000 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// layer 3 : z2 -> z3
For j In {0:3}
  For i In {0:2}
    botx = 1200 + 10*j + i;
    If(j == 3 && i == 0)
      botx = 1813; // C1 upper special north edge
    EndIf

    sid = 4200 + 10*j + i;
    Curve Loop(sid) = {
      botx,
      2100 + 10*j + (i+1),
     -(1300 + 10*j + i),
     -(2100 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// =====================================================
// Vertical y-z surfaces  (fixed x)
// =====================================================

// layer 1 : z0 -> z1
For i In {0:3}
  For j In {0:2}
    topy = 1500 + 10*i + j;
    If(i == 3 && j == 0)
      topy = 1802; // C2 lower special east edge
    EndIf

    sid = 4300 + 10*i + j;
    Curve Loop(sid) = {
      1400 + 10*i + j,
      1900 + 10*(j+1) + i,
     -topy,
     -(1900 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// layer 2 : z1 -> z2
For i In {0:3}
  For j In {0:2}
    boty = 1500 + 10*i + j;
    topy = 1600 + 10*i + j;

    If(i == 3 && j == 0)
      boty = 1803; // C2 upper special east edge
    EndIf

    If(i == 0 && j == 2)
      topy = 1810; // C1 lower special west edge
    EndIf

    sid = 4400 + 10*i + j;
    Curve Loop(sid) = {
      boty,
      2000 + 10*(j+1) + i,
     -topy,
     -(2000 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// layer 3 : z2 -> z3
For i In {0:3}
  For j In {0:2}
    boty = 1600 + 10*i + j;
    If(i == 0 && j == 2)
      boty = 1811; // C1 upper special west edge
    EndIf

    sid = 4500 + 10*i + j;
    Curve Loop(sid) = {
      boty,
      2100 + 10*(j+1) + i,
     -(1700 + 10*i + j),
     -(2100 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// =====================================================
// Transfinite curves
// =====================================================

// x-direction
For j In {0:3}
  Transfinite Curve{1000 + 10*j + 0, 1100 + 10*j + 0, 1200 + 10*j + 0, 1300 + 10*j + 0} = NX1;
  Transfinite Curve{1000 + 10*j + 1, 1100 + 10*j + 1, 1200 + 10*j + 1, 1300 + 10*j + 1} = NXM;
  Transfinite Curve{1000 + 10*j + 2, 1100 + 10*j + 2, 1200 + 10*j + 2, 1300 + 10*j + 2} = NX2;
EndFor

// y-direction
For i In {0:3}
  Transfinite Curve{1400 + 10*i + 0, 1500 + 10*i + 0, 1600 + 10*i + 0, 1700 + 10*i + 0} = NY2;
  Transfinite Curve{1400 + 10*i + 1, 1500 + 10*i + 1, 1600 + 10*i + 1, 1700 + 10*i + 1} = NYM;
  Transfinite Curve{1400 + 10*i + 2, 1500 + 10*i + 2, 1600 + 10*i + 2, 1700 + 10*i + 2} = NY1;
EndFor

// special crack-edge curves
Transfinite Curve{1800,1801} = NX2;
Transfinite Curve{1802,1803} = NY2;
Transfinite Curve{1810,1811} = NY1;
Transfinite Curve{1812,1813} = NX1;

// z-direction
For j In {0:3}
  For i In {0:3}
    Transfinite Curve{1900 + 10*j + i} = NZ1;
    Transfinite Curve{2000 + 10*j + i} = NZ2;
    Transfinite Curve{2100 + 10*j + i} = NZ3;
  EndFor
EndFor

// =====================================================
// Volumes
// =====================================================

// layer 1 : z0 -> z1
For j In {0:2}
  For i In {0:2}
    topS = 3100 + 10*j + i;
    If(i == 2 && j == 0)
      topS = 3190;
    EndIf

    sl = 7000 + 10*j + i;
    Surface Loop(sl) = {
      3000 + 10*j + i,
      topS,
      4000 + 10*j + i,
      4000 + 10*(j+1) + i,
      4300 + 10*i + j,
      4300 + 10*(i+1) + j
    };
    Volume(6000 + 10*j + i) = {sl};

    pSW = 200 + 10*j + i;
    pSE = 200 + 10*j + (i+1);
    pNE = 200 + 10*(j+1) + (i+1);
    pNW = 200 + 10*(j+1) + i;

    If(i == 2 && j == 0)
      pSE = 900;
    EndIf

    Transfinite Volume{6000 + 10*j + i} = {
      100 + 10*j + i,
      100 + 10*j + (i+1),
      100 + 10*(j+1) + (i+1),
      100 + 10*(j+1) + i,
      pSW, pSE, pNE, pNW
    };
  EndFor
EndFor

// layer 2 : z1 -> z2
For j In {0:2}
  For i In {0:2}
    botS = 3100 + 10*j + i;
    topS = 3200 + 10*j + i;

    If(i == 2 && j == 0)
      botS = 3191;
    EndIf

    If(i == 0 && j == 2)
      topS = 3290;
    EndIf

    sl = 7100 + 10*j + i;
    Surface Loop(sl) = {
      botS,
      topS,
      4100 + 10*j + i,
      4100 + 10*(j+1) + i,
      4400 + 10*i + j,
      4400 + 10*(i+1) + j
    };
    Volume(6100 + 10*j + i) = {sl};

    pSW0 = 200 + 10*j + i;
    pSE0 = 200 + 10*j + (i+1);
    pNE0 = 200 + 10*(j+1) + (i+1);
    pNW0 = 200 + 10*(j+1) + i;

    pSW1 = 300 + 10*j + i;
    pSE1 = 300 + 10*j + (i+1);
    pNE1 = 300 + 10*(j+1) + (i+1);
    pNW1 = 300 + 10*(j+1) + i;

    If(i == 2 && j == 0)
      pSE0 = 901;
    EndIf

    If(i == 0 && j == 2)
      pNW1 = 902;
    EndIf

    Transfinite Volume{6100 + 10*j + i} = {
      pSW0, pSE0, pNE0, pNW0,
      pSW1, pSE1, pNE1, pNW1
    };
  EndFor
EndFor

// layer 3 : z2 -> z3
For j In {0:2}
  For i In {0:2}
    botS = 3200 + 10*j + i;
    If(i == 0 && j == 2)
      botS = 3291;
    EndIf

    sl = 7200 + 10*j + i;
    Surface Loop(sl) = {
      botS,
      3300 + 10*j + i,
      4200 + 10*j + i,
      4200 + 10*(j+1) + i,
      4500 + 10*i + j,
      4500 + 10*(i+1) + j
    };
    Volume(6200 + 10*j + i) = {sl};

    pSW = 300 + 10*j + i;
    pSE = 300 + 10*j + (i+1);
    pNE = 300 + 10*(j+1) + (i+1);
    pNW = 300 + 10*(j+1) + i;

    If(i == 0 && j == 2)
      pNW = 903;
    EndIf

    Transfinite Volume{6200 + 10*j + i} = {
      pSW, pSE, pNE, pNW,
      400 + 10*j + i,
      400 + 10*j + (i+1),
      400 + 10*(j+1) + (i+1),
      400 + 10*(j+1) + i
    };
  EndFor
EndFor

// =====================================================
// physical groups
// =====================================================

Physical Volume("materialID", 0) = {
  6000,6001,6002,
  6010,6011,6012,
  6020,6021,6022,
  6100,6101,6102,
  6110,6111,6112,
  6120,6121,6122,
  6200,6201,6202,
  6210,6211,6212,
  6220,6221,6222
};


// =====================================================
// Mesh
// =====================================================
Mesh 3;

// =====================================================
// END
// =====================================================





Else // same H1, H2


// =====================================================
// parameters
//
//    ^ y
//    |       Top View:
//    |
//  y3+--------+-----+--------+ ---------
//    |        |     |        |  ^    ^
//    |        |     |        |  |    |
//    |   C1   |  I  |    I   |  W1   |
//    |        |     |        |  |    |
//    |        |     |        |  v    |
//  y2+--------+-----+--------+ ---   |
//    |        |     |        |       |
//    |   I    |  I  |    I   |       W
//    |        |     |        |       |
//  y1+--------+-----+--------+ ---   |
//    |        |     |        |  ^    |
//    |        |     |        |  |    |
//    |   I    |  I  |    C2  |  W2   |
//    |        |     |        |  |    |
//    |        |     |        |  v    v
//  y0+--------+-----+--------+ --------- -----> x
//    |x0      |x1   |x2      |x3
//    |<--L1-->|     |<--L2-->|
//    |<----------L---------->|
//
//
//
//
//           ^z
//           |
//           |      Front view:
// ------ z2 +--------+-----+--------+
//  ^        |        |     |        |
//  |        |        |     |        |
//  |        |    I   |  I  |    I   |
//  |        |        |     |        |
//  |        |        |     |        |
//  |     z1 >-- C1 --+-----+-- C2 --< ---
//  |        |        |     |        |  ^
//  |        |        |     |        |  |
//  |        |        |     |        |  |
//  H        |        |     |        |  |
//  |        |        |     |        |  |
//  |        |        |     |        |  Hc = (H1 + H2)/2
//  |        |   I    |  I  |     I  |  |
//  |        |        |     |        |  |
//  |        |        |     |        |  |
//  |        |        |     |        |  |
//  v        |        |     |        |  v
// ------ z0 +--------+-----+--------+ ---   ----> x
//
//
// =====================================================

Hc = 0.5 * (H1 + H2);

// =====================================================
// mesh density
// =====================================================

NX1 = Ceil(L1/lc) + 1;            // points in x-band L1
NXM = Ceil((L-L1-L2)/lc) + 1;     // points in middle x-band
NX2 = Ceil(L2/lc) + 1;            // points in x-band L2

NY2 = Ceil(W2/lc) + 1;            // points in bottom y-band W2
NYM = Ceil((W-W1-W2)/lc) + 1;     // points in middle y-band
NY1 = Ceil(W1/lc) + 1;            // points in top y-band W1

NZ1 = Ceil(Hc/lc) + 1;            // points in z-band [0, Hc]
NZ2 = Ceil((H-Hc)/lc) + 1;        // points in z-band [Hc, H]

// safety: at least 2 points on each transfinite curve
If(NX1 < 2) NX1 = 2; EndIf
If(NXM < 2) NXM = 2; EndIf
If(NX2 < 2) NX2 = 2; EndIf

If(NY2 < 2) NY2 = 2; EndIf
If(NYM < 2) NYM = 2; EndIf
If(NY1 < 2) NY1 = 2; EndIf

If(NZ1 < 2) NZ1 = 2; EndIf
If(NZ2 < 2) NZ2 = 2; EndIf

Mesh.Smoothing = 50;

// =====================================================
// coordinates
// =====================================================

// x partition: [0, L1] [middle] [L-L2, L]
x0 = 0.0;
x1 = L1;
x2 = L - L2;
x3 = L;

// y partition: [0, W2] [middle] [W-W1, W]
y0 = 0.0;
y1 = W2;
y2 = W - W1;
y3 = W;

// z partition: [0, Hc] [Hc, H]
z0 = 0.0;
z1 = Hc;
z2 = H;

// half openings
d1 = C1 / 2.0;
d2 = C2 / 2.0;

// =====================================================
// Point numbering convention
//
//            z      y   x
// z = z0 : 100 + 10*j + i
// z = z1 : 200 + 10*j + i
// z = z2 : 300 + 10*j + i
// z = z3 : 400 + 10*j + i
//
// i = 0..3 in x, j = 0..3 in y
//
// split points for crack surfaces:
// 900 : C2 lower-side corner  at (x3,y0,z1-d2)
// 901 : C2 upper-side corner  at (x3,y0,z1+d2)
// 902 : C1 lower-side corner  at (x0,y3,z2-d1)
// 903 : C1 upper-side corner  at (x0,y3,z2+d1)
//
//
//  The point ids on *-th plane in the z-direction:
//     ^ y
//     |
//     |(*30)   |(*31)|(*32)   |(*33)
//  y3 +--------+-----+--------+
//     |        |     |        |
//     |        |     |        |
//     |        |     |        |
//     |        |     |        |
//     |(*20)   |(*21)|(*22)   |(*23)
//  y2 +--------+-----+--------+
//     |        |     |        |
//     |        |     |        |
//     |(*10)   |(*11)|(*12)   |(*13)
//  y1 +--------+-----+--------+
//     |        |     |        |
//     |        |     |        |
//     |        |     |        |
//     |        |     |        |
//     |(*00)   |(*01)|(*02)   |(*03)
//  y0 +--------+-----+--------+  -----> x
//     |x0      |x1   |x2      |x3
// =====================================================

// ---------------------
// z = z0
// ---------------------
Point(100) = {x0,y0,z0,lc};
Point(101) = {x1,y0,z0,lc};
Point(102) = {x2,y0,z0,lc};
Point(103) = {x3,y0,z0,lc};

Point(110) = {x0,y1,z0,lc};
Point(111) = {x1,y1,z0,lc};
Point(112) = {x2,y1,z0,lc};
Point(113) = {x3,y1,z0,lc};

Point(120) = {x0,y2,z0,lc};
Point(121) = {x1,y2,z0,lc};
Point(122) = {x2,y2,z0,lc};
Point(123) = {x3,y2,z0,lc};

Point(130) = {x0,y3,z0,lc};
Point(131) = {x1,y3,z0,lc};
Point(132) = {x2,y3,z0,lc};
Point(133) = {x3,y3,z0,lc};

// ---------------------
// z = z1 = Hc
// ---------------------
Point(200) = {x0,y0,z1,lc};
Point(201) = {x1,y0,z1,lc};
Point(202) = {x2,y0,z1,lc};
// Point(203) omitted: split by C2

Point(210) = {x0,y1,z1,lc};
Point(211) = {x1,y1,z1,lc};
Point(212) = {x2,y1,z1,lc};
Point(213) = {x3,y1,z1,lc};

Point(220) = {x0,y2,z1,lc};
Point(221) = {x1,y2,z1,lc};
Point(222) = {x2,y2,z1,lc};
Point(223) = {x3,y2,z1,lc};

// Point(230) omitted: split by C1
Point(231) = {x1,y3,z1,lc};
Point(232) = {x2,y3,z1,lc};
Point(233) = {x3,y3,z1,lc};

// ---------------------
// z = z2 = H
// ---------------------
Point(300) = {x0,y0,z2,lc};
Point(301) = {x1,y0,z2,lc};
Point(302) = {x2,y0,z2,lc};
Point(303) = {x3,y0,z2,lc};

Point(310) = {x0,y1,z2,lc};
Point(311) = {x1,y1,z2,lc};
Point(312) = {x2,y1,z2,lc};
Point(313) = {x3,y1,z2,lc};

Point(320) = {x0,y2,z2,lc};
Point(321) = {x1,y2,z2,lc};
Point(322) = {x2,y2,z2,lc};
Point(323) = {x3,y2,z2,lc};

Point(330) = {x0,y3,z2,lc};
Point(331) = {x1,y3,z2,lc};
Point(332) = {x2,y3,z2,lc};
Point(333) = {x3,y3,z2,lc};

// ---------------------
// points for the crack openning
// ---------------------
Point(900) = {x3,y0,z1-d2,lc}; // C2 lower side
Point(901) = {x3,y0,z1+d2,lc}; // C2 upper side

Point(902) = {x0,y3,z1-d1,lc}; // C1 lower side
Point(903) = {x0,y3,z1+d1,lc}; // C1 upper side

// =====================================================
// Lines on z = *-th planes
//
// Line numbering convension:
//
//      j: for y, i for x
//      1*ji: 0 <= * <= 3: lines parallel to x-axis
//      1*ji: 4 <= * <= 7: lines parallel to y-axis
//      18**: crack opening
// =====================================================

// x-lines on z0, z1, z2
For j In {0:3}
  For i In {0:2}
    Line(1000 + 10*j + i) = {100 + 10*j + i, 100 + 10*j + i + 1};

    If(!((j == 0 && i == 2) || (j == 3 && i == 0)))
      Line(1100 + 10*j + i) = {200 + 10*j + i, 200 + 10*j + i + 1};
    EndIf

    Line(1200 + 10*j + i) = {300 + 10*j + i, 300 + 10*j + i + 1};
  EndFor
EndFor

// y-lines on z0, z1, z2
For i In {0:3}
  For j In {0:2}
    Line(1300 + 10*i + j) = {100 + 10*j + i, 100 + 10*(j+1) + i};

    If(!((i == 3 && j == 0) || (i == 0 && j == 2)))
      Line(1400 + 10*i + j) = {200 + 10*j + i, 200 + 10*(j+1) + i};
    EndIf

    Line(1500 + 10*i + j) = {300 + 10*j + i, 300 + 10*(j+1) + i};
  EndFor
EndFor

// =====================================================
// lines on crack opennings z = Hc
// =====================================================

// C2 patch = bottom-right corner patch at z = Hc
Line(1800) = {202,900}; // south edge, lower crack face
Line(1801) = {202,901}; // south edge, upper crack face
Line(1802) = {900,213}; // east  edge, lower crack face
Line(1803) = {901,213}; // east  edge, upper crack face

// C1 patch = top-left corner patch at z = Hc
Line(1810) = {220,902}; // west  edge, lower crack face
Line(1811) = {220,903}; // west  edge, upper crack face
Line(1812) = {902,231}; // north edge, lower crack face
Line(1813) = {903,231}; // north edge, upper crack face

// =====================================================
// Vertical lines
// =====================================================

// lower layer : z0 -> z1
For j In {0:3}
  For i In {0:3}
    pTop = 200 + 10*j + i;

    If(i == 3 && j == 0)
      pTop = 900; // C2 lower side
    EndIf

    If(i == 0 && j == 3)
      pTop = 902; // C1 lower side
    EndIf

    Line(1900 + 10*j + i) = {100 + 10*j + i, pTop};
  EndFor
EndFor

// upper layer : z1 -> z2
For j In {0:3}
  For i In {0:3}
    pBot = 200 + 10*j + i;

    If(i == 3 && j == 0)
      pBot = 901; // C2 upper side
    EndIf

    If(i == 0 && j == 3)
      pBot = 903; // C1 upper side
    EndIf

    Line(2000 + 10*j + i) = {pBot, 300 + 10*j + i};
  EndFor
EndFor

// =====================================================
// Horizontal surfaces
// =====================================================

// z = z0
For j In {0:2}
  For i In {0:2}
    sid = 3000 + 10*j + i;
    Curve Loop(sid) = {
      1000 + 10*j + i,
      1300 + 10*(i+1) + j,
     -(1000 + 10*(j+1) + i),
     -(1300 + 10*i + j)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// z = z1 intact patches except C2 (i=2,j=0) and C1 (i=0,j=2)
For j In {0:2}
  For i In {0:2}
    If(!((i == 2 && j == 0) || (i == 0 && j == 2)))
      sid = 3100 + 10*j + i;
      Curve Loop(sid) = {
        1100 + 10*j + i,
        1400 + 10*(i+1) + j,
       -(1100 + 10*(j+1) + i),
       -(1400 + 10*i + j)
      };
      Plane Surface(sid) = {sid};
      Transfinite Surface{sid};
      Recombine Surface{sid};
    EndIf
  EndFor
EndFor

// z = z1, C2 lower crack face
Curve Loop(3190) = {1800, 1802, -1112, -1420};
Surface(3190) = {3190};
Transfinite Surface{3190} = {202, 900, 213, 212};
Recombine Surface{3190};

// z = z1, C2 upper crack face
Curve Loop(3191) = {1801, 1803, -1112, -1420};
Surface(3191) = {3191};
Transfinite Surface{3191} = {202, 901, 213, 212};
Recombine Surface{3191};

// z = z1, C1 lower crack face
Curve Loop(3290) = {1120, 1412, -1812, -1810};
Surface(3290) = {3290};
Transfinite Surface{3290} = {220, 221, 231, 902};
Recombine Surface{3290};

// z = z1, C1 upper crack face
Curve Loop(3291) = {1120, 1412, -1813, -1811};
Surface(3291) = {3291};
Transfinite Surface{3291} = {220, 221, 231, 903};
Recombine Surface{3291};

// z = z2
For j In {0:2}
  For i In {0:2}
    sid = 3200 + 10*j + i;
    Curve Loop(sid) = {
      1200 + 10*j + i,
      1500 + 10*(i+1) + j,
     -(1200 + 10*(j+1) + i),
     -(1500 + 10*i + j)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// =====================================================
// Vertical x-z surfaces  (fixed y)
// =====================================================

// lower layer : z0 -> z1
For j In {0:3}
  For i In {0:2}
    topx = 1100 + 10*j + i;

    If(j == 0 && i == 2)
      topx = 1800; // C2 lower special south edge
    EndIf

    If(j == 3 && i == 0)
      topx = 1812; // C1 lower special north edge
    EndIf

    sid = 4000 + 10*j + i;
    Curve Loop(sid) = {
      1000 + 10*j + i,
      1900 + 10*j + (i+1),
     -topx,
     -(1900 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// upper layer : z1 -> z2
For j In {0:3}
  For i In {0:2}
    botx = 1100 + 10*j + i;

    If(j == 0 && i == 2)
      botx = 1801; // C2 upper special south edge
    EndIf

    If(j == 3 && i == 0)
      botx = 1813; // C1 upper special north edge
    EndIf

    sid = 4100 + 10*j + i;
    Curve Loop(sid) = {
      botx,
      2000 + 10*j + (i+1),
     -(1200 + 10*j + i),
     -(2000 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// =====================================================
// Vertical y-z surfaces  (fixed x)
// =====================================================

// lower layer : z0 -> z1
For i In {0:3}
  For j In {0:2}
    topy = 1400 + 10*i + j;

    If(i == 3 && j == 0)
      topy = 1802; // C2 lower special east edge
    EndIf

    If(i == 0 && j == 2)
      topy = 1810; // C1 lower special west edge
    EndIf

    sid = 4200 + 10*i + j;
    Curve Loop(sid) = {
      1300 + 10*i + j,
      1900 + 10*(j+1) + i,
     -topy,
     -(1900 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// upper layer : z1 -> z2
For i In {0:3}
  For j In {0:2}
    boty = 1400 + 10*i + j;

    If(i == 3 && j == 0)
      boty = 1803; // C2 upper special east edge
    EndIf

    If(i == 0 && j == 2)
      boty = 1811; // C1 upper special west edge
    EndIf

    sid = 4300 + 10*i + j;
    Curve Loop(sid) = {
      boty,
      2000 + 10*(j+1) + i,
     -(1500 + 10*i + j),
     -(2000 + 10*j + i)
    };
    Plane Surface(sid) = {sid};
    Transfinite Surface{sid};
    Recombine Surface{sid};
  EndFor
EndFor

// =====================================================
// Transfinite curves
// =====================================================

// x-direction on z0 and z2
For j In {0:3}
  Transfinite Curve{1000 + 10*j + 0, 1200 + 10*j + 0} = NX1;
  Transfinite Curve{1000 + 10*j + 1, 1200 + 10*j + 1} = NXM;
  Transfinite Curve{1000 + 10*j + 2, 1200 + 10*j + 2} = NX2;
EndFor

// x-direction on z1 (existing curves only)
For j In {0:3}
  If(j != 3)
    Transfinite Curve{1100 + 10*j + 0} = NX1;
  EndIf
  Transfinite Curve{1100 + 10*j + 1} = NXM;
  If(j != 0)
    Transfinite Curve{1100 + 10*j + 2} = NX2;
  EndIf
EndFor

// y-direction on z0 and z2
For i In {0:3}
  Transfinite Curve{1300 + 10*i + 0, 1500 + 10*i + 0} = NY2;
  Transfinite Curve{1300 + 10*i + 1, 1500 + 10*i + 1} = NYM;
  Transfinite Curve{1300 + 10*i + 2, 1500 + 10*i + 2} = NY1;
EndFor

// y-direction on z1 (existing curves only)
For i In {0:3}
  If(i != 3)
    Transfinite Curve{1400 + 10*i + 0} = NY2;
  EndIf
  Transfinite Curve{1400 + 10*i + 1} = NYM;
  If(i != 0)
    Transfinite Curve{1400 + 10*i + 2} = NY1;
  EndIf
EndFor

// special crack-edge curves
Transfinite Curve{1800,1801} = NX2;
Transfinite Curve{1802,1803} = NY2;
Transfinite Curve{1810,1811} = NY1;
Transfinite Curve{1812,1813} = NX1;

// z-direction
For j In {0:3}
  For i In {0:3}
    Transfinite Curve{1900 + 10*j + i} = NZ1;
    Transfinite Curve{2000 + 10*j + i} = NZ2;
  EndFor
EndFor

// =====================================================
// Volumes
// =====================================================

// lower layer : z0 -> z1
For j In {0:2}
  For i In {0:2}
    topS = 3100 + 10*j + i;

    If(i == 2 && j == 0)
      topS = 3190;
    EndIf

    If(i == 0 && j == 2)
      topS = 3290;
    EndIf

    sl = 7000 + 10*j + i;
    Surface Loop(sl) = {
      3000 + 10*j + i,
      topS,
      4000 + 10*j + i,
      4000 + 10*(j+1) + i,
      4200 + 10*i + j,
      4200 + 10*(i+1) + j
    };
    Volume(6000 + 10*j + i) = {sl};

    pSW = 200 + 10*j + i;
    pSE = 200 + 10*j + (i+1);
    pNE = 200 + 10*(j+1) + (i+1);
    pNW = 200 + 10*(j+1) + i;

    If(i == 2 && j == 0)
      pSE = 900;
    EndIf

    If(i == 0 && j == 2)
      pNW = 902;
    EndIf

    Transfinite Volume{6000 + 10*j + i} = {
      100 + 10*j + i,
      100 + 10*j + (i+1),
      100 + 10*(j+1) + (i+1),
      100 + 10*(j+1) + i,
      pSW, pSE, pNE, pNW
    };
  EndFor
EndFor

// upper layer : z1 -> z2
For j In {0:2}
  For i In {0:2}
    botS = 3100 + 10*j + i;

    If(i == 2 && j == 0)
      botS = 3191;
    EndIf

    If(i == 0 && j == 2)
      botS = 3291;
    EndIf

    sl = 7100 + 10*j + i;
    Surface Loop(sl) = {
      botS,
      3200 + 10*j + i,
      4100 + 10*j + i,
      4100 + 10*(j+1) + i,
      4300 + 10*i + j,
      4300 + 10*(i+1) + j
    };
    Volume(6100 + 10*j + i) = {sl};

    pSW = 200 + 10*j + i;
    pSE = 200 + 10*j + (i+1);
    pNE = 200 + 10*(j+1) + (i+1);
    pNW = 200 + 10*(j+1) + i;

    If(i == 2 && j == 0)
      pSE = 901;
    EndIf

    If(i == 0 && j == 2)
      pNW = 903;
    EndIf

    Transfinite Volume{6100 + 10*j + i} = {
      pSW, pSE, pNE, pNW,
      300 + 10*j + i,
      300 + 10*j + (i+1),
      300 + 10*(j+1) + (i+1),
      300 + 10*(j+1) + i
    };
  EndFor
EndFor

// =====================================================
// Optional physical groups
// =====================================================

Physical Volume("materialID", 0) = {
  6000,6001,6002,
  6010,6011,6012,
  6020,6021,6022,
  6100,6101,6102,
  6110,6111,6112,
  6120,6121,6122
};



// =====================================================
// Mesh
// =====================================================
Mesh 3;


EndIf

EndIf // through_crack

