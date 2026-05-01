Include "../prms/throughNotchedCube/1mmX1mm/0.4mmCracks/0.0mmDistance.geo";

SetFactory("Built-in");

// =====================================================
// 2D double rectangular notches, cleaned block layout
// =====================================================
// Parameter convention:
//   L, H  : specimen length and height in the 2D x-y plane
//   L1    : depth of the left rectangular notch, measured from x = 0
//   L2    : depth of the right rectangular notch, measured from x = L
//   H1    : center height of the left notch
//   H2    : center height of the right notch
//   lc    : mesh size and rectangular notch width
//
// Rectangular notches:
//   left notch  = [0, L1]       x [H1 - lc/2, H1 + lc/2]
//   right notch = [L - L2, L]   x [H2 - lc/2, H2 + lc/2]
//
// Important mesh-layout change:
//   - The vertical partition lines at x=L1 and x=L-L2 are only used
//     where they are required by the notch geometry.
//   - The top and bottom regions are generated as single full-width
//     transfinite blocks instead of being split into left/middle/right
//     blocks.
//   - The ligament between two notches has exactly one element through
//     the notch width direction because the notch width is lc and NYN=2.
// =====================================================

Mesh.Smoothing = 100;
Mesh.SaveAll = 0;

// ---------------------
// coordinates
// ---------------------
x0 = 0.0;
x1 = L1;
x2 = L - L2;
x3 = L;

y0 = 0.0;
yH = H;

notch_width = lc;
half_width  = 0.5 * notch_width;

// ---------------------
// mesh density in x-direction
// ---------------------
NX1 = Ceil((x1 - x0)/lc) + 1;
NXM = Ceil((x2 - x1)/lc) + 1;
NX2 = Ceil((x3 - x2)/lc) + 1;

If(NX1 < 2) NX1 = 2; EndIf
If(NXM < 2) NXM = 2; EndIf
If(NX2 < 2) NX2 = 2; EndIf

// Combined point counts for compound sides.
NXL  = NX1 + NXM - 1;       // x0 -> x2
NXR  = NXM + NX2 - 1;       // x1 -> x3
NXA  = NX1 + NXM + NX2 - 2; // x0 -> x3

// Exactly one element through the rectangular notch width.
NYN = 2;

// =====================================================
// Branch 1: left notch is above right notch, H1 > H2
// =====================================================
If(H1 > H2)

// y1/y2: right notch lower/upper surfaces
// y3/y4: left notch lower/upper surfaces
y1 = H2 - half_width;
y2 = H2 + half_width;
y3 = H1 - half_width;
y4 = H1 + half_width;
y5 = yH;

NY0 = Ceil((y1 - y0)/lc) + 1;
NYM = Ceil((y3 - y2)/lc) + 1;
NY4 = Ceil((y5 - y4)/lc) + 1;

If(NY0 < 2) NY0 = 2; EndIf
If(NYM < 2) NYM = 2; EndIf
If(NY4 < 2) NY4 = 2; EndIf

// ---------------------
// points
// ---------------------
Point(100) = {x0,y0,0,lc};
Point(103) = {x3,y0,0,lc};

Point(200) = {x0,y1,0,lc};
Point(202) = {x2,y1,0,lc};
Point(203) = {x3,y1,0,lc};

Point(300) = {x0,y2,0,lc};
Point(302) = {x2,y2,0,lc};
Point(303) = {x3,y2,0,lc};

Point(400) = {x0,y3,0,lc};
Point(401) = {x1,y3,0,lc};
Point(403) = {x3,y3,0,lc};

Point(500) = {x0,y4,0,lc};
Point(501) = {x1,y4,0,lc};
Point(503) = {x3,y4,0,lc};

Point(600) = {x0,y5,0,lc};
Point(603) = {x3,y5,0,lc};

// ---------------------
// horizontal lines
// ---------------------
Line(1000) = {100,103};

Line(1100) = {200,202};
Line(1101) = {202,203};

Line(1200) = {300,302};
Line(1201) = {302,303};

Line(1300) = {400,401};
Line(1301) = {401,403};

Line(1400) = {500,501};
Line(1401) = {501,503};

Line(1500) = {600,603};

// ---------------------
// vertical lines
// ---------------------
Line(2000) = {100,200};
Line(2003) = {103,203};

Line(2100) = {200,300};
Line(2102) = {202,302}; // right notch inner vertical face
Line(2103) = {203,303};

Line(2200) = {300,400};
Line(2203) = {303,403};

Line(2300) = {400,500};
Line(2301) = {401,501}; // left notch inner vertical face
Line(2303) = {403,503};

Line(2400) = {500,600};
Line(2403) = {503,603};

// ---------------------
// surfaces
// ---------------------
// lower full-width block
Curve Loop(3000) = {1000, 2003, -1101, -1100, -2000};
Plane Surface(3000) = {3000};

// right-notch band: material remains from x0 to x2
Curve Loop(3010) = {1100, 2102, -1200, -2100};
Plane Surface(3010) = {3010};

// middle full-width block between the two notch levels
Curve Loop(3020) = {1200, 1201, 2203, -1301, -1300, -2200};
Plane Surface(3020) = {3020};

// left-notch band: material remains from x1 to x3
Curve Loop(3030) = {1301, 2303, -1401, -2301};
Plane Surface(3030) = {3030};

// upper full-width block
Curve Loop(3040) = {1400, 1401, 2403, -1500, -2400};
Plane Surface(3040) = {3040};

// ---------------------
// transfinite curves and surfaces
// ---------------------
Transfinite Curve{1000,1500} = NXA;
Transfinite Curve{1100,1200} = NXL;
Transfinite Curve{1101,1201} = NX2;
Transfinite Curve{1300,1400} = NX1;
Transfinite Curve{1301,1401} = NXR;

Transfinite Curve{2000,2003} = NY0;
Transfinite Curve{2100,2102,2103} = NYN;
Transfinite Curve{2200,2203} = NYM;
Transfinite Curve{2300,2301,2303} = NYN;
Transfinite Curve{2400,2403} = NY4;

Transfinite Surface{3000} = {100,103,203,200};
Transfinite Surface{3010} = {200,202,302,300};
Transfinite Surface{3020} = {300,303,403,400};
Transfinite Surface{3030} = {401,403,503,501};
Transfinite Surface{3040} = {500,503,603,600};

Recombine Surface{3000,3010,3020,3030,3040};

Physical Surface("materialID", 0) = {3000,3010,3020,3030,3040};

// =====================================================
// Branch 2: right notch is above left notch, H2 > H1
// =====================================================
ElseIf(H2 > H1)

// y1/y2: left notch lower/upper surfaces
// y3/y4: right notch lower/upper surfaces
y1 = H1 - half_width;
y2 = H1 + half_width;
y3 = H2 - half_width;
y4 = H2 + half_width;
y5 = yH;

NY0 = Ceil((y1 - y0)/lc) + 1;
NYM = Ceil((y3 - y2)/lc) + 1;
NY4 = Ceil((y5 - y4)/lc) + 1;

If(NY0 < 2) NY0 = 2; EndIf
If(NYM < 2) NYM = 2; EndIf
If(NY4 < 2) NY4 = 2; EndIf

// ---------------------
// points
// ---------------------
Point(100) = {x0,y0,0,lc};
Point(103) = {x3,y0,0,lc};

Point(200) = {x0,y1,0,lc};
Point(201) = {x1,y1,0,lc};
Point(203) = {x3,y1,0,lc};

Point(300) = {x0,y2,0,lc};
Point(301) = {x1,y2,0,lc};
Point(303) = {x3,y2,0,lc};

Point(400) = {x0,y3,0,lc};
Point(402) = {x2,y3,0,lc};
Point(403) = {x3,y3,0,lc};

Point(500) = {x0,y4,0,lc};
Point(502) = {x2,y4,0,lc};
Point(503) = {x3,y4,0,lc};

Point(600) = {x0,y5,0,lc};
Point(603) = {x3,y5,0,lc};

// ---------------------
// horizontal lines
// ---------------------
Line(1000) = {100,103};

Line(1100) = {200,201};
Line(1101) = {201,203};

Line(1200) = {300,301};
Line(1201) = {301,303};

Line(1300) = {400,402};
Line(1301) = {402,403};

Line(1400) = {500,502};
Line(1401) = {502,503};

Line(1500) = {600,603};

// ---------------------
// vertical lines
// ---------------------
Line(2000) = {100,200};
Line(2003) = {103,203};

Line(2100) = {200,300};
Line(2101) = {201,301}; // left notch inner vertical face
Line(2103) = {203,303};

Line(2200) = {300,400};
Line(2203) = {303,403};

Line(2300) = {400,500};
Line(2302) = {402,502}; // right notch inner vertical face
Line(2303) = {403,503};

Line(2400) = {500,600};
Line(2403) = {503,603};

// ---------------------
// surfaces
// ---------------------
// lower full-width block
Curve Loop(3000) = {1000, 2003, -1101, -1100, -2000};
Plane Surface(3000) = {3000};

// left-notch band: material remains from x1 to x3
Curve Loop(3010) = {1101, 2103, -1201, -2101};
Plane Surface(3010) = {3010};

// middle full-width block between the two notch levels
Curve Loop(3020) = {1200, 1201, 2203, -1301, -1300, -2200};
Plane Surface(3020) = {3020};

// right-notch band: material remains from x0 to x2
Curve Loop(3030) = {1300, 2302, -1400, -2300};
Plane Surface(3030) = {3030};

// upper full-width block
Curve Loop(3040) = {1400, 1401, 2403, -1500, -2400};
Plane Surface(3040) = {3040};

// ---------------------
// transfinite curves and surfaces
// ---------------------
Transfinite Curve{1000,1500} = NXA;
Transfinite Curve{1100,1200} = NX1;
Transfinite Curve{1101,1201} = NXR;
Transfinite Curve{1300,1400} = NXL;
Transfinite Curve{1301,1401} = NX2;

Transfinite Curve{2000,2003} = NY0;
Transfinite Curve{2100,2101,2103} = NYN;
Transfinite Curve{2200,2203} = NYM;
Transfinite Curve{2300,2302,2303} = NYN;
Transfinite Curve{2400,2403} = NY4;

Transfinite Surface{3000} = {100,103,203,200};
Transfinite Surface{3010} = {201,203,303,301};
Transfinite Surface{3020} = {300,303,403,400};
Transfinite Surface{3030} = {400,402,502,500};
Transfinite Surface{3040} = {500,503,603,600};

Recombine Surface{3000,3010,3020,3030,3040};

Physical Surface("materialID", 0) = {3000,3010,3020,3030,3040};

// =====================================================
// Branch 3: H1 == H2, two notches are at the same height
// =====================================================
Else

Hc = 0.5 * (H1 + H2);

y1 = Hc - half_width;
y2 = Hc + half_width;
y3 = yH;

NY0 = Ceil((y1 - y0)/lc) + 1;
NY2 = Ceil((y3 - y2)/lc) + 1;

If(NY0 < 2) NY0 = 2; EndIf
If(NY2 < 2) NY2 = 2; EndIf

// ---------------------
// points
// ---------------------
Point(100) = {x0,y0,0,lc};
Point(103) = {x3,y0,0,lc};

Point(200) = {x0,y1,0,lc};
Point(201) = {x1,y1,0,lc};
Point(202) = {x2,y1,0,lc};
Point(203) = {x3,y1,0,lc};

Point(300) = {x0,y2,0,lc};
Point(301) = {x1,y2,0,lc};
Point(302) = {x2,y2,0,lc};
Point(303) = {x3,y2,0,lc};

Point(400) = {x0,y3,0,lc};
Point(403) = {x3,y3,0,lc};

// ---------------------
// horizontal lines
// ---------------------
Line(1000) = {100,103};

Line(1100) = {200,201};
Line(1101) = {201,202};
Line(1102) = {202,203};

Line(1200) = {300,301};
Line(1201) = {301,302};
Line(1202) = {302,303};

Line(1300) = {400,403};

// ---------------------
// vertical lines
// ---------------------
Line(2000) = {100,200};
Line(2003) = {103,203};

Line(2100) = {200,300};
Line(2101) = {201,301}; // left notch inner vertical face
Line(2102) = {202,302}; // right notch inner vertical face
Line(2103) = {203,303};

Line(2200) = {300,400};
Line(2203) = {303,403};

// ---------------------
// surfaces
// ---------------------
// lower full-width block. The top side is split only to represent the two notch bottoms.
Curve Loop(3000) = {1000, 2003, -1102, -1101, -1100, -2000};
Plane Surface(3000) = {3000};

// middle ligament between the two rectangular notches.
// Because NYN = 2 and y2-y1 = lc, this is exactly one element through the notch width.
Curve Loop(3010) = {1101, 2102, -1201, -2101};
Plane Surface(3010) = {3010};

// upper full-width block. The lower side is split only to represent the two notch tops.
Curve Loop(3020) = {1200, 1201, 1202, 2203, -1300, -2200};
Plane Surface(3020) = {3020};

// ---------------------
// transfinite curves and surfaces
// ---------------------
Transfinite Curve{1000,1300} = NXA;
Transfinite Curve{1100,1200} = NX1;
Transfinite Curve{1101,1201} = NXM;
Transfinite Curve{1102,1202} = NX2;

Transfinite Curve{2000,2003} = NY0;
Transfinite Curve{2100,2101,2102,2103} = NYN;
Transfinite Curve{2200,2203} = NY2;

Transfinite Surface{3000} = {100,103,203,200};
Transfinite Surface{3010} = {201,202,302,301};
Transfinite Surface{3020} = {300,303,403,400};

Recombine Surface{3000,3010,3020};

Physical Surface("materialID", 0) = {3000,3010,3020};

EndIf

Mesh 2;
