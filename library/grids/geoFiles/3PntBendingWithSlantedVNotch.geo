Include "../prms/3PntBendingVNotch/260mmX10mmX60mm/20mmX2mmCrack/45degree.geo";

SetFactory("Built-in");

Mesh.MshFileVersion = 2.2;
Mesh.ElementOrder   = 1;
Mesh.RecombineAll   = 1;
Mesh.RecombinationAlgorithm = 1;



x_min = 0.0;
x_support_left  = ex1;
x_support_right = length - ex2;
x_max = length;

half_length     = x_crack_factor * length;
half_thickness  = 0.5 * thickness;
half_height     = 0.5 * height;

tan_theta = Tan(theta);





// If the real notch width along y is measured normal to the slanted plane
slot_gap_offset = notch_width_normal / Cos(theta);
slot_half       = 0.5 * slot_gap_offset;

// Local crack characteristic length for points close to notch faces.
lc_crack = lc;
If (lc > notch_width_normal)
    lc_crack = notch_width_normal;
EndIf

// ------------------------------------------------------------
// the loading point is at the center of the intact surface
// the grid under the tip of the slanted crack will be twisted to center of the intact surface.
// In order to guarantee the smoothness, three extra layers are add for the transiency.
//      ^
//      |
//      |
//      |
//      +---------------------+ +---------------------+ -- ztop
//      |                     | |                     |
//      |                     + +---------------------+ -- zcut
//      |                      V                      |
//      |                      + ---------------------+ -- ztip
//      |                                             |
//      |                                             + -- zmid
//      |                                             |
//      |              transient region               |
//      |                      |                      + -- zflat
//      |                      |                      |
//      +----------------------*----------------------+ -- z0       -----> x
//                        loading point
// ------------------------------------------------------------

v_height = lc;

ztop  = height;
ztip  = ztop  - crack_depth;
zcut  = ztip + v_height;
zmid  = ztip / 3.0 * 2.0;
zflat = ztip / 3.0;
z0    = 0.0;



// Important indices:
//   iz = 0: z0
//   iz = 1: zflat
//   iz = 2: zmid
//   iz = 3: ztip
//   iz = 4: zcut
//   iz = 5: ztop
zLevel[]    = {z0,  zflat, zmid, ztip,  zcut, ztop};
// The factors to correct the angles
// 0: the points will be on the middle plane
// 1: the points will not be corrected
betaLevel[] = {0.0, 0.33,   0.66,  1.0,   1.0,  1.0};

iz_tip = 3;
iz_cut = 4;
iz_top = 5;

// Number of transfinite points in each z-band.
nZ[] = {
    Floor((zflat - z0)    / lc) + 1,
    Floor((zmid  - zflat) / lc) + 1,
    Floor((ztip  - zmid)  / lc) + 1,
    Floor((zcut  - ztip)  / lc) + 1,
    Floor((ztop  - zcut)  / lc) + 1
};

// ------------------------------------------------------------
// Transfinite counts in x and y directions
// ------------------------------------------------------------
// x-bands:
//   ib = 0: [0, ex]
//   ib = 1: [ex, left notch/symmetry line]
//   ib = 2: central notch opening / collapsed gap, always skipped
//   ib = 3: [right notch/symmetry line, length-ex]
//   ib = 4: [length-ex, length]
// ------------------------------------------------------------

nX[] = {
    Floor((x_support_left - x_min) / lc) + 1,
    Floor((half_length - slot_half - x_support_left) / lc) + 1,
    2, // two elements for the crack tip
    Floor((x_support_right - (half_length + slot_half)) / lc) + 1,
    Floor((x_max - x_support_right) / lc) + 1
};

nY = Floor(thickness / lc) + 1;

// ============================================================
// Point generation
// ============================================================
// For each z-level and y={0,thickness}:
//
//   x_center(y,z) = half_length
//                 + beta(z) * (y - half_thickness) * tan_theta
//
// For iz < iz_cut:
//   ix = 2 and ix = 3 are both placed at x_center.
//
// For iz >= iz_cut:
//   ix = 2 is the left wall of the opened slot.
//   ix = 3 is the right wall of the opened slot.
// ============================================================

p[] = {};

For iz In {0:5}

    z    = zLevel[iz];
    beta = betaLevel[iz];

    For iy In {0:1}

        y = iy * thickness;
        x_center = half_length + beta * (y - half_thickness) * tan_theta;

        For ix In {0:5}

            If (ix == 0)
                x = x_min;
            EndIf

            If (ix == 1)
                x = x_support_left;
            EndIf

            If (ix == 2)
                If (iz >= iz_cut)
                    x = x_center - slot_half;
                Else
                    x = x_center;
                EndIf
            EndIf

            If (ix == 3)
                If (iz >= iz_cut)
                    x = x_center + slot_half;
                Else
                    x = x_center;
                EndIf
            EndIf

            If (ix == 4)
                x = x_support_right;
            EndIf

            If (ix == 5)
                x = x_max;
            EndIf

            id = ix + 6*iy + 12*iz;

            // Below the opened slot, ix=2 and ix=3 must be the same
            // geometrical point. Do not create two coincident points;
            // reuse the ix=2 point tag for ix=3. This keeps the lower
            // center partition connected as one symmetry line.
            If (ix == 3 && iz < iz_cut)

                p[id] = p[2 + 6*iy + 12*iz];

            Else

                p[id] = newp;

                If (ix == 2 || ix == 3)
                    Point(p[id]) = {x, y, z, lc_crack};
                Else
                    Point(p[id]) = {x, y, z, lc};
                EndIf

            EndIf

        EndFor
    EndFor
EndFor

// ============================================================
// Curves
// ============================================================

lx[] = {};
ly[] = {};
lz[] = {};

// ------------------------------------------------------------
// x-direction curves
// lx[ib + 5*iy + 10*iz]
// ib = 0..4, iy = 0..1, iz = 0..5
//
// The central curve ib = 2 is always skipped because it is either
// a collapsed zero-thickness gap or an actual notch opening.
// ------------------------------------------------------------
For iz In {0:5}
    For iy In {0:1}
        For ib In {0:4}

            If (ib != 2)

                A = p[ib     + 6*iy + 12*iz];
                B = p[ib + 1 + 6*iy + 12*iz];

                id = ib + 5*iy + 10*iz;

                lx[id] = newl;
                Line(lx[id]) = {A, B};

                Transfinite Curve{lx[id]} = nX[ib];

            EndIf

        EndFor
    EndFor
EndFor

// ------------------------------------------------------------
// y-direction curves
// ly[ix + 6*iz]
// ix = 0..5, iz = 0..5
// ------------------------------------------------------------
For iz In {0:5}
    For ix In {0:5}

        A = p[ix + 6*0 + 12*iz];
        B = p[ix + 6*1 + 12*iz];

        id = ix + 6*iz;

        // At collapsed z-levels, ix=3 is the same center line as ix=2.
        If (ix == 3 && iz < iz_cut)

            ly[id] = ly[2 + 6*iz];

        Else

            ly[id] = newl;
            Line(ly[id]) = {A, B};

            Transfinite Curve{ly[id]} = nY;

        EndIf

    EndFor
EndFor

// ------------------------------------------------------------
// z-direction curves
// lz[ix + 6*iy + 12*iz]
// ix = 0..5, iy = 0..1, iz = 0..4
// ------------------------------------------------------------
For iz In {0:4}
    For iy In {0:1}
        For ix In {0:5}

            A = p[ix + 6*iy + 12*iz];
            B = p[ix + 6*iy + 12*(iz+1)];

            id = ix + 6*iy + 12*iz;

            // Below the crack tip, ix=3 is identical to ix=2 through the
            // whole z-band. Reuse the same z-curve instead of creating a
            // duplicate coincident curve.
            If (ix == 3 && iz < iz_tip)

                lz[id] = lz[2 + 6*iy + 12*iz];

            Else

                lz[id] = newl;
                Line(lz[id]) = {A, B};

                Transfinite Curve{lz[id]} = nZ[iz];

            EndIf

        EndFor
    EndFor
EndFor

// ============================================================
// Surfaces
// ============================================================

sxy[] = {};
sxz[] = {};
syz[] = {};

surf_z0[]     = {};
surf_ztop[]   = {};
surf_y0[]     = {};
surf_yW[]     = {};
surf_xmin[]   = {};
surf_xmax[]   = {};
surf_notch[]  = {};

// ------------------------------------------------------------
// xy surfaces at each z-level
// sxy[ib + 5*iz]
// ib = 0..4, iz = 0..5
//
// The central xy surface ib = 2 is skipped everywhere.
// There is no artificial bottom cap at zcut because the V-tip layer
// connects the opened slot to the sharp tip.
// ------------------------------------------------------------
For iz In {0:5}
    For ib In {0:4}

        If (ib != 2)

            A = p[ib     + 6*0 + 12*iz];
            B = p[ib + 1 + 6*0 + 12*iz];
            C = p[ib + 1 + 6*1 + 12*iz];
            D = p[ib     + 6*1 + 12*iz];

            l1 = lx[ib + 5*0 + 10*iz];
            l2 = ly[(ib+1) + 6*iz];
            l3 = lx[ib + 5*1 + 10*iz];
            l4 = ly[ib + 6*iz];

            cl = newll;
            Curve Loop(cl) = {l1, l2, -l3, -l4};

            id = ib + 5*iz;

            sxy[id] = news;
            Ruled Surface(sxy[id]) = {cl};

            Transfinite Surface{sxy[id]} = {A, B, C, D};
            Recombine Surface{sxy[id]};

            If (iz == 0)
                surf_z0[] += {sxy[id]};
            EndIf

            If (iz == iz_top)
                surf_ztop[] += {sxy[id]};
            EndIf

        EndIf

    EndFor
EndFor

// ------------------------------------------------------------
// xz surfaces at y = 0 and y = thickness
// sxz[ib + 5*iy + 10*iz]
// ib = 0..4, iy = 0..1, iz = 0..4
//
// The central xz surface ib = 2 is skipped everywhere. On the side
// faces y=0 and y=thickness this leaves the visible V-shaped notch opening.
// ------------------------------------------------------------
For iz In {0:4}
    For iy In {0:1}
        For ib In {0:4}

            If (ib != 2)

                A = p[ib     + 6*iy + 12*iz];
                B = p[ib + 1 + 6*iy + 12*iz];
                C = p[ib + 1 + 6*iy + 12*(iz+1)];
                D = p[ib     + 6*iy + 12*(iz+1)];

                l1 = lx[ib + 5*iy + 10*iz];
                l2 = lz[(ib+1) + 6*iy + 12*iz];
                l3 = lx[ib + 5*iy + 10*(iz+1)];
                l4 = lz[ib + 6*iy + 12*iz];

                cl = newll;
                Curve Loop(cl) = {l1, l2, -l3, -l4};

                id = ib + 5*iy + 10*iz;

                sxz[id] = news;
                Ruled Surface(sxz[id]) = {cl};

                Transfinite Surface{sxz[id]} = {A, B, C, D};
                Recombine Surface{sxz[id]};

                If (iy == 0)
                    surf_y0[] += {sxz[id]};
                EndIf

                If (iy == 1)
                    surf_yW[] += {sxz[id]};
                EndIf

            EndIf

        EndFor
    EndFor
EndFor

// ------------------------------------------------------------
// yz surfaces at each x-line
// syz[ix + 6*iz]
// ix = 0..5, iz = 0..4
//
// ix = 2 and ix = 3 form the two notch faces only for iz >= iz_tip.
// For iz < iz_tip they are coincident internal partition surfaces on
// the symmetry line and are not placed into the physical notch group.
// ------------------------------------------------------------
For iz In {0:4}
    For ix In {0:5}

        A = p[ix + 6*0 + 12*iz];
        B = p[ix + 6*1 + 12*iz];
        C = p[ix + 6*1 + 12*(iz+1)];
        D = p[ix + 6*0 + 12*(iz+1)];

        l1 = ly[ix + 6*iz];
        l2 = lz[ix + 6*1 + 12*iz];
        l3 = ly[ix + 6*(iz+1)];
        l4 = lz[ix + 6*0 + 12*iz];

        id = ix + 6*iz;

        // Below the crack tip, ix=3 and ix=2 are the same internal symmetry
        // surface. Reuse the ix=2 surface tag so the two neighboring
        // volumes share the same surface instead of two coincident surfaces.
        If (ix == 3 && iz < iz_tip)

            syz[id] = syz[2 + 6*iz];

        Else

            cl = newll;
            Curve Loop(cl) = {l1, l2, -l3, -l4};

            syz[id] = news;
            Ruled Surface(syz[id]) = {cl};

            Transfinite Surface{syz[id]} = {A, B, C, D};
            Recombine Surface{syz[id]};

            If (ix == 0)
                surf_xmin[] += {syz[id]};
            EndIf

            If (ix == 5)
                surf_xmax[] += {syz[id]};
            EndIf

            // Slanted V faces in the taper layer and vertical slot faces above.
            If (iz >= iz_tip && (ix == 2 || ix == 3))
                surf_notch[] += {syz[id]};
            EndIf

        EndIf

    EndFor
EndFor

// ============================================================
// Volumes
// ============================================================

volumes[] = {};

For iz In {0:4}
    For ib In {0:4}

        // The central block is removed everywhere:
        //   below ztip: zero-thickness collapsed central strip is not meshed;
        //   ztip->zcut: V-shaped notch void;
        //   zcut->ztop: straight opened notch void.
        If (ib != 2)

            A = p[ib     + 6*0 + 12*iz];
            B = p[ib + 1 + 6*0 + 12*iz];
            C = p[ib + 1 + 6*1 + 12*iz];
            D = p[ib     + 6*1 + 12*iz];

            E = p[ib     + 6*0 + 12*(iz+1)];
            F = p[ib + 1 + 6*0 + 12*(iz+1)];
            G = p[ib + 1 + 6*1 + 12*(iz+1)];
            H = p[ib     + 6*1 + 12*(iz+1)];

            s_bottom = sxy[ib + 5*iz];
            s_top    = sxy[ib + 5*(iz+1)];

            s_y0     = sxz[ib + 5*0 + 10*iz];
            s_yW     = sxz[ib + 5*1 + 10*iz];

            s_left   = syz[ib     + 6*iz];
            s_right  = syz[ib + 1 + 6*iz];

            sl = newsl;
            Surface Loop(sl) = {
                -s_bottom,
                 s_top,
                 s_y0,
                -s_yW,
                -s_left,
                 s_right
            };

            v = newv;
            Volume(v) = {sl};

            Transfinite Volume{v} = {A, B, C, D, E, F, G, H};

            volumes[] += {v};

        EndIf

    EndFor
EndFor

Coherence;

// ============================================================
// Physical groups
// ============================================================

Physical Volume("beam") = {volumes[]};

Physical Surface("bottom_z0") = {surf_z0[]};
Physical Surface("top_zH")    = {surf_ztop[]};

Physical Surface("front_y0")  = {surf_y0[]};
Physical Surface("back_yW")   = {surf_yW[]};

Physical Surface("left_x0")       = {surf_xmin[]};
Physical Surface("right_xLength") = {surf_xmax[]};

// Contains the two V faces in the taper layer and the two vertical
// faces of the upper straight slot.
Physical Surface("notch_faces") = {surf_notch[]};

// Reserved loading/support curves on the bottom surface z = 0.
Physical Curve("reserved_line_xEx_z0")            = {ly[1]};
Physical Curve("reserved_line_xLengthMinusEx_z0") = {ly[4]};

// The sharp crack-tip line at z = ztip.
Physical Curve("sharp_crack_tip_line") = {ly[2 + 6*iz_tip]};

// ============================================================
// Mesh
// ============================================================
Mesh 3;
