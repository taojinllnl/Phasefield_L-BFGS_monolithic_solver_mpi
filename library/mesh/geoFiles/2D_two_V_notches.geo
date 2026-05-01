Include "../prms/throughNotchedCube/1mmX1mm/0.4mmCracks/0.0mmDistance.geo";

SetFactory("Built-in");

// =====================================================
// H1 != H2 branch
// =====================================================
If(H1 != H2)

// Preserve the original convention:
// C1 is the upper crack and C2 is the lower crack.
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

NY1 = Ceil(H2/lc) + 1;
NY2 = Ceil((H1-H2)/lc) + 1;
NY3 = Ceil((H-H1)/lc) + 1;

If(NX1 < 2) NX1 = 2; EndIf
If(NXM < 2) NXM = 2; EndIf
If(NX2 < 2) NX2 = 2; EndIf
If(NY1 < 2) NY1 = 2; EndIf
If(NY2 < 2) NY2 = 2; EndIf
If(NY3 < 2) NY3 = 2; EndIf

Mesh.Smoothing = 100;

// ---------------------
// coordinates
// ---------------------
x0 = 0.0;
x1 = L1;
x2 = L - L2;
x3 = L;

y0 = 0.0;
y1 = H2; // C2 crack height

y2 = H1; // C1 crack height

y3 = H;

d1 = C1 / 2.0;
d2 = C2 / 2.0;

// =====================================================
// 2D points in the reduced x-y plane
// (where y corresponds to the original z direction)
// =====================================================

// y = y0
Point(100) = {x0,y0,0,lc};
Point(101) = {x1,y0,0,lc};
Point(102) = {x2,y0,0,lc};
Point(103) = {x3,y0,0,lc};

// y = y1 = H2, C2 is split at x = x3
Point(200) = {x0,y1,0,lc};
Point(201) = {x1,y1,0,lc};
Point(202) = {x2,y1,0,lc};

// y = y2 = H1, C1 is split at x = x0
Point(301) = {x1,y2,0,lc};
Point(302) = {x2,y2,0,lc};
Point(303) = {x3,y2,0,lc};

// y = y3
Point(400) = {x0,y3,0,lc};
Point(401) = {x1,y3,0,lc};
Point(402) = {x2,y3,0,lc};
Point(403) = {x3,y3,0,lc};

// split points for crack openings
Point(900) = {x3,y1-d2,0,lc}; // C2 lower side
Point(901) = {x3,y1+d2,0,lc}; // C2 upper side

Point(902) = {x0,y2-d1,0,lc}; // C1 lower side
Point(903) = {x0,y2+d1,0,lc}; // C1 upper side

// =====================================================
// 2D lines
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

// y-lines, layer 1: y0 -> y1
Line(1900) = {100,200};
Line(1901) = {101,201};
Line(1902) = {102,202};
Line(1903) = {103,900}; // reaches C2 lower side

// y-lines, layer 2: y1 -> y2
Line(2000) = {200,902}; // reaches C1 lower side
Line(2001) = {201,301};
Line(2002) = {202,302};
Line(2003) = {901,303}; // starts from C2 upper side

// y-lines, layer 3: y2 -> y3
Line(2100) = {903,400}; // starts from C1 upper side
Line(2101) = {301,401};
Line(2102) = {302,402};
Line(2103) = {303,403};

// =====================================================
// 2D surfaces
// =====================================================

// layer 1: y0 -> y1
Curve Loop(3000) = {1000, 1901, -1100, -1900};
Plane Surface(3000) = {3000};

Curve Loop(3001) = {1001, 1902, -1101, -1901};
Plane Surface(3001) = {3001};

Curve Loop(3002) = {1002, 1903, -1800, -1902};
Plane Surface(3002) = {3002};

// layer 2: y1 -> y2
Curve Loop(3100) = {1100, 2001, -1810, -2000};
Plane Surface(3100) = {3100};

Curve Loop(3101) = {1101, 2002, -1201, -2001};
Plane Surface(3101) = {3101};

Curve Loop(3102) = {1801, 2003, -1202, -2002};
Plane Surface(3102) = {3102};

// layer 3: y2 -> y3
Curve Loop(3200) = {1811, 2101, -1300, -2100};
Plane Surface(3200) = {3200};

Curve Loop(3201) = {1201, 2102, -1301, -2101};
Plane Surface(3201) = {3201};

Curve Loop(3202) = {1202, 2103, -1302, -2102};
Plane Surface(3202) = {3202};

// =====================================================
// Transfinite curves and surfaces
// =====================================================

// x-direction
Transfinite Curve{1000,1100,1300} = NX1;
Transfinite Curve{1001,1101,1201,1301} = NXM;
Transfinite Curve{1002,1202,1302} = NX2;
Transfinite Curve{1800,1801} = NX2;
Transfinite Curve{1810,1811} = NX1;

// y-direction
Transfinite Curve{1900,1901,1902,1903} = NY1;
Transfinite Curve{2000,2001,2002,2003} = NY2;
Transfinite Curve{2100,2101,2102,2103} = NY3;

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

Physical Surface("materialID", 0) = {
  3000, 3001, 3002,
  3100, 3101, 3102,
  3200, 3201, 3202
};

Mesh 2;

Else // H1 == H2

Hc = 0.5 * (H1 + H2);

// ---------------------
// mesh density
// ---------------------
NX1 = Ceil(L1/lc) + 1;
NXM = Ceil((L-L1-L2)/lc) + 1;
NX2 = Ceil(L2/lc) + 1;

NY1 = Ceil(Hc/lc) + 1;
NY2 = Ceil((H-Hc)/lc) + 1;

If(NX1 < 2) NX1 = 2; EndIf
If(NXM < 2) NXM = 2; EndIf
If(NX2 < 2) NX2 = 2; EndIf
If(NY1 < 2) NY1 = 2; EndIf
If(NY2 < 2) NY2 = 2; EndIf

Mesh.Smoothing = 100;

// ---------------------
// coordinates
// ---------------------
x0 = 0.0;
x1 = L1;
x2 = L - L2;
x3 = L;

y0 = 0.0;
y1 = Hc;
y2 = H;

d1 = C1 / 2.0;
d2 = C2 / 2.0;

// =====================================================
// 2D points in the reduced x-y plane
// =====================================================

// y = y0
Point(100) = {x0,y0,0,lc};
Point(101) = {x1,y0,0,lc};
Point(102) = {x2,y0,0,lc};
Point(103) = {x3,y0,0,lc};

// y = y1 = Hc, both C1 and C2 are split
Point(201) = {x1,y1,0,lc};
Point(202) = {x2,y1,0,lc};

// y = y2 = H
Point(300) = {x0,y2,0,lc};
Point(301) = {x1,y2,0,lc};
Point(302) = {x2,y2,0,lc};
Point(303) = {x3,y2,0,lc};

// split points for crack openings
Point(900) = {x3,y1-d2,0,lc}; // C2 lower side
Point(901) = {x3,y1+d2,0,lc}; // C2 upper side

Point(902) = {x0,y1-d1,0,lc}; // C1 lower side
Point(903) = {x0,y1+d1,0,lc}; // C1 upper side

// =====================================================
// 2D lines
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

// y-lines, lower layer
Line(1900) = {100,902}; // reaches C1 lower side
Line(1901) = {101,201};
Line(1902) = {102,202};
Line(1903) = {103,900}; // reaches C2 lower side

// y-lines, upper layer
Line(2000) = {903,300}; // starts from C1 upper side
Line(2001) = {201,301};
Line(2002) = {202,302};
Line(2003) = {901,303}; // starts from C2 upper side

// =====================================================
// 2D surfaces
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
// Transfinite curves and surfaces
// =====================================================

// x-direction
Transfinite Curve{1000,1200} = NX1;
Transfinite Curve{1001,1101,1201} = NXM;
Transfinite Curve{1002,1202} = NX2;
Transfinite Curve{1810,1811} = NX1;
Transfinite Curve{1800,1801} = NX2;

// y-direction
Transfinite Curve{1900,1901,1902,1903} = NY1;
Transfinite Curve{2000,2001,2002,2003} = NY2;

For sid In {3000:3002}
  Transfinite Surface{sid};
  Recombine Surface{sid};
EndFor

For sid In {3100:3102}
  Transfinite Surface{sid};
  Recombine Surface{sid};
EndFor

Physical Surface("materialID", 0) = {
  3000, 3001, 3002,
  3100, 3101, 3102
};

Mesh 2;

EndIf
