import cadquery as cq
import cadquery.occ_impl.shape_protocols as cq_sp
import numpy as np
from scipy.special import erf
from typing import Sequence, List, cast

width = 26
depth = 8 
diffuser_thickness=1.6
lip_thickness = 1
lip_depth = 1
min_radius = 30
radius_scale = 75
z_scale = 75
hole_dia=6

tolerance = 0.01
diffuser_tol = 0.1
diffuser_gap = 0.6
diffuser_depth = 3
diffuser_curve=3

# profile_thickness: float, profile_width: float, vscale: float=1.5,
loops: float = 3


def gen_spiral(
    erf_stddev: float = 1.4,
    min_rad: float = 75,
    zscale: float = 100,
    rscale: float = 100,
    loops=3.5,
):
    def eqn(t: float) -> tuple[float, float, float]:

        m = min_rad
        # linear version
        # z = -(t/np.pi/loops)

        # nonlinear version
        l = 2 * loops * np.pi
        # find integral of zd
        erf_out = -erf(t / (l / (erf_stddev * 2)) - erf_stddev)
        z = erf_out * zscale
        y = (-erf_out * rscale + m + rscale) * np.sin(t)
        x = (-erf_out * rscale + m + rscale) * np.cos(t)
        return x, y, z

    return eqn


spiral = gen_spiral(
    erf_stddev=1.2, min_rad=min_radius, zscale=z_scale, rscale=radius_scale, loops=loops
)
xs, ys, zs = spiral(0)
xe, ye, ze = spiral(loops * np.pi * 2)
spiral_path = (
    cq.Workplane("XY").parametricCurve(spiral, start=0, stop=loops * np.pi * 2).val()
)

cs_width = lip_thickness * 2 + diffuser_thickness*2 + width
cs_height = depth + lip_depth +tolerance
edge_height = cs_height/2

main_cross_section = (
    cq.Workplane("XZ", origin=(xs, ys, zs))
    .moveTo(-cs_width/2, cs_height/ 2)
    .line(cs_width, 0)
    .line(0, -edge_height)
    .line(-diffuser_thickness, 0)
    .line(0, -(cs_height-edge_height))
    .line(-lip_thickness, 0)
    .line(0, lip_depth)
    .line(-width, 0)
    .line(0, -lip_depth)
    .line(-lip_thickness, 0)
    .line(0, (cs_height-edge_height))
    .line(-diffuser_thickness, 0)
    .line(0, edge_height)
    .close()
)


neg_cross_section_short_len = xs + lip_thickness + width / 2
neg_cross_section_long_len = neg_cross_section_short_len + width

neg_cross_section = (
    cq.Workplane("XZ", origin=(xs, ys, zs))
    .moveTo(-cs_width/2, cs_height/ 2)
    .line(cs_width, 0)
    .line(0, -edge_height)
    .line(-diffuser_thickness, 0)
    .line(0, -(cs_height-edge_height))
    .line(-lip_thickness, 0)
    .line(0, lip_depth)
    .line(-width, 0)
    .line(0, -lip_depth)
    .line(-lip_thickness, 0)
    .line(0, lip_depth)
    .line(-xs,0)
    .line(0, -depth)
    .line(neg_cross_section_long_len+width,0)
    .line(0, depth*3)
    .line(-neg_cross_section_long_len-width, 0)
    .line(0, -depth)360mm in feet
    .close()
)


spiral_solid = main_cross_section.sweep(spiral_path)
neg_spiral_solid = neg_cross_section.sweep(
    cq.Workplane("XY").parametricCurve(spiral, start=-np.pi / 2, stop=np.pi / 2)
)

spiral_support = (
    cq.Workplane("YZ", origin=(width/2, 0, 0))
    .center(0, zs)
    .rect(width + lip_thickness * 2, depth * 1.8)
    .extrude(xs+width*.5)
    .val()
)

lamp = spiral_solid.union(spiral_support.cut(neg_spiral_solid.val()))

#display(lamp)


lamp = lamp.union(
    cq.Solid.extrudeLinear(
        lamp.faces(cq.NearestToPointSelector((width/2, 0, zs))).val(),
        cq.Vector(-width, 0, 0),
    )
)


class ApproxParallelDirSelector(cq.ParallelDirSelector):
    def filter(
        self, objectList: Sequence[cq.selectors.Shape]
    ) -> List[cq.selectors.Shape]:
        """
        There are lots of kinds of filters, but for planes they are always
        based on the normal of the plane, and for edges on the tangent vector
        along the edge
        """
        r = []
        for o in objectList:
            # no really good way to avoid a switch here, edges and faces are simply different!
            if o.ShapeType() == "Face" and o.geomType() == "PLANE":
                # a face is only parallel to a direction if it is a plane, and
                # its normal is parallel to the dir
                test_vector = cast(cq_sp.FaceProtocol, o).normalAt(None)
            elif o.ShapeType() == "Edge" and (
                o.geomType() == "LINE" or o.geomType() == "BSPLINE"
            ):
                # an edge is parallel to a direction if its underlying geometry is plane or line
                test_vector = cast(cq_sp.Shape1DProtocol, o).tangentAt()
            else:
                continue

            if self.test(test_vector):
                r.append(o)

        return r


class MinLengthSelector(cq.ParallelDirSelector):
    def __init__(self, min_length: float):
        self.min_length = min_length

    def filter(
        self, objectList: Sequence[cq.selectors.Shape]
    ) -> List[cq.selectors.Shape]:
        """
        There are lots of kinds of filters, but for planes they are always
        based on the normal of the plane, and for edges on the tangent vector
        along the edge
        """
        r = []
        for o in objectList:
            # no really good way to avoid a switch here, edges and faces are simply different!
            if o.ShapeType() == "Face" and o.geomType() == "PLANE":
                raise ValueError(
                    f"MinLengthSelector supports only Edges and Wires, not {type(obj).__name__}"
                )
            elif o.ShapeType() == "Edge" and (
                o.geomType() == "LINE" or o.geomType() == "BSPLINE"
            ):
                # an edge is parallel to a direction if its underlying geometry is plane or line
                if cast(cq_sp.Shape1DProtocol, o).Length() > self.min_length:
                    r.append(o)
            else:
                continue
        return r
    
hole = cq.Workplane("XY",origin=(0,0,zs)).sketch().circle(hole_dia/2).finalize().extrude(depth, both=True)


lamp = lamp.edges("|Z").fillet(4).cut(hole)



def gen_diffuser(diffuser_thickness, offset_inner=0)->cq.Workplane:
    import copy

    w1 = (cs_width-diffuser_thickness*2+diffuser_tol)
    h1 = (depth+lip_depth)
    ph1 = h1/2-edge_height -diffuser_gap
    h2 = cs_height - edge_height -diffuser_gap + diffuser_tol
    ph2 = ph1-h2

    diffuser_core = (
        cq.Workplane("XZ", origin=(xs, ys, zs))
        .moveTo(-w1/2, ph1)
        .line(0, -h2)
        .threePointArc((0, -h2-diffuser_curve), (w1/2, ph2  ))
        .line(0,h2).close()
    )
    
    diffuser_core.rotate((0,0,0),(1,0,0), -90)
    
    diffuser_shell = (
        copy.copy(diffuser_core)
        ).offset2D(diffuser_thickness)
    
      
    cutout_cap = (
        cq.Workplane("XZ", origin=(xs, ys, zs))
        .moveTo(-w1/2-diffuser_thickness*1.1, ph1-tolerance)
        .rect(w1+2*diffuser_thickness*1.1, diffuser_thickness*1.1, centered=False)
    )

    d_spiral_path = cq.Workplane("XY").parametricCurve(spiral, start=-np.pi / 2, stop=loops * np.pi * 2+np.pi/16)
    longer_spiral_path = cq.Workplane("XY").parametricCurve(spiral, start=-np.pi / 4, stop=loops * np.pi * 2)
    diffuser_shell = diffuser_shell.sweep(longer_spiral_path)
    cutout = diffuser_core.wires().toPending().sweep(d_spiral_path).union(cutout_cap.sweep(d_spiral_path))
    
    return diffuser_shell.cut(cutout)

#
diffuser = gen_diffuser(diffuser_thickness)
trim = (
    cq.Workplane("XZ", origin=(xs, ys, zs))
    .moveTo(-cs_width/2, cs_height/ 2)
    .line(cs_width-diffuser_thickness-diffuser_tol, 0)
    .line(0, -depth-diffuser_tol)
    .line(-width*2, 0)
    .close().sweep(
    cq.Workplane("XY").parametricCurve(spiral, start=-np.pi / 2, stop=np.pi *.25)
))

diffuser = diffuser.cut(trim)

diffuser.val().exportStep("lamp_diffuser.step")

lamp.val().exportStep("lamp.step")

assembly = cq.Assembly().add(lamp, color = cq.Color("blue")).add(diffuser, color = cq.Color("red"))
#display(assembly)
