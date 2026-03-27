"""
Mesh generation for the Schäfer-Turek DFG 2D-1/2D-2 benchmark.

Geometry (all dimensions in metres):
    Channel  :  [0, 2.2] x [0, 0.41]
    Cylinder :  centre (0.2, 0.2),  radius 0.05
    Note     :  cylinder is off-centre (midline = 0.205), breaking symmetry.

Boundary markers:
    1  - Inflow   (x = 0)
    2  - Outflow  (x = 2.2)
    3  - Walls    (y = 0 and y = 0.41)
    4  - Cylinder

References:
    Schäfer, M. & Turek, S. (1996). "Benchmark computations of laminar flow
    around a cylinder." Notes on Numerical Fluid Mechanics, 52, 547-566.

Usage:
    python generate_dfg_mesh.py              # default refinement
    python generate_dfg_mesh.py --lc 0.02    # custom element size
    python generate_dfg_mesh.py --order 2    # Q2 elements
    python generate_dfg_mesh.py --gui        # open GMSH GUI after meshing
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import gmsh
except ImportError:
    raise ImportError("gmsh python API required: pip install gmsh")


def generate_dfg_mesh(
    lc_global: float = 0.04,
    lc_cylinder: float = 0.008,
    lc_wake: float = 0.015,
    order: int = 2,
    output: str = "dfg_benchmark.msh",
    gui: bool = False,
    recombine: bool = True,
):
    """
    Generate the DFG flow-past-cylinder benchmark mesh.

    Parameters
    ----------
    lc_global : float
        Characteristic element size far from the cylinder.
    lc_cylinder : float
        Element size on the cylinder surface.
    lc_wake : float
        Element size in the wake refinement region.
    order : int
        Element order (1 or 2).
    output : str
        Output mesh filename.
    gui : bool
        If True, open the GMSH GUI before finalising.
    recombine : bool
        If True, recombine triangles into quads.
    """

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("dfg_benchmark")

    geo = gmsh.model.geo

    # ── Geometry parameters ───────────────────────────────────────────
    L  = 2.2       # channel length
    L  = 1.0       # channel length
    H  = 0.5      # channel height 0.41
    xc = 0.2       # cylinder centre x
    yc = 0.2       # cylinder centre y
    r  = 0.05      # cylinder radius

    # ── Channel corners ───────────────────────────────────────────────
    p1 = geo.addPoint(0, 0, 0, lc_global)      # bottom-left
    p2 = geo.addPoint(L, 0, 0, lc_global)      # bottom-right
    p3 = geo.addPoint(L, H, 0, lc_global)      # top-right
    p4 = geo.addPoint(0, H, 0, lc_global)      # top-left

    # ── Cylinder (4 quarter-arcs) ─────────────────────────────────────
    pc = geo.addPoint(xc, yc, 0, lc_cylinder)  # centre

    p5 = geo.addPoint(xc + r, yc,     0, lc_cylinder)   # east
    p6 = geo.addPoint(xc,     yc + r, 0, lc_cylinder)   # north
    p7 = geo.addPoint(xc - r, yc,     0, lc_cylinder)   # west
    p8 = geo.addPoint(xc,     yc - r, 0, lc_cylinder)   # south

    # ── Channel edges ─────────────────────────────────────────────────
    l_bottom = geo.addLine(p1, p2)    # bottom wall
    l_right  = geo.addLine(p2, p3)    # outflow
    l_top    = geo.addLine(p3, p4)    # top wall
    l_left   = geo.addLine(p4, p1)    # inflow

    # ── Cylinder arcs (oriented CCW viewed from +z) ───────────────────
    c1 = geo.addCircleArc(p5, pc, p6)   # E -> N
    c2 = geo.addCircleArc(p6, pc, p7)   # N -> W
    c3 = geo.addCircleArc(p7, pc, p8)   # W -> S
    c4 = geo.addCircleArc(p8, pc, p5)   # S -> E

    # ── Surface (channel minus cylinder hole) ─────────────────────────
    outer_loop    = geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
    cylinder_loop = geo.addCurveLoop([c1, c2, c3, c4])
    surface = geo.addPlaneSurface([outer_loop, cylinder_loop])

    geo.synchronize()

    # ── Physical groups (boundary markers) ────────────────────────────
    gmsh.model.addPhysicalGroup(1, [l_left],                     tag=1, name="left")
    gmsh.model.addPhysicalGroup(1, [l_right],                    tag=2, name="right")
    gmsh.model.addPhysicalGroup(1, [l_bottom],                   tag=3, name="bottom")
    gmsh.model.addPhysicalGroup(1, [l_top],                      tag=4, name="top")
    gmsh.model.addPhysicalGroup(1, [c1, c2, c3, c4],            tag=5, name="cylinder")
    gmsh.model.addPhysicalGroup(2, [surface],                    tag=10, name="Fluid")

    # ── Mesh size field: refine near cylinder and in the wake ─────────
    # Distance from cylinder
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [c1, c2, c3, c4])
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

    # Threshold: ramp from lc_cylinder near surface to lc_global far away
    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", lc_cylinder)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", lc_global)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 0.5)

    # Wake refinement box: downstream of the cylinder
    f_box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_box, "VIn", lc_wake)
    gmsh.model.mesh.field.setNumber(f_box, "VOut", lc_global)
    gmsh.model.mesh.field.setNumber(f_box, "XMin", xc)
    gmsh.model.mesh.field.setNumber(f_box, "XMax", xc + 10 * r)    # 10 radii downstream
    gmsh.model.mesh.field.setNumber(f_box, "YMin", yc - 3 * r)
    gmsh.model.mesh.field.setNumber(f_box, "YMax", yc + 3 * r)

    # Take the minimum of all fields
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thresh, f_box])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # Disable default size computation from points
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # ── Meshing ───────────────────────────────────────────────────────
    if recombine:
        gmsh.option.setNumber("Mesh.Algorithm", 8)               # Frontal-Delaunay for quads
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # simple full-quad
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 6)               # Frontal-Delaunay

    gmsh.model.mesh.generate(2)

    if order > 1:
        gmsh.model.mesh.setOrder(order)

    # ── Extract mesh data ─────────────────────────────────────────────
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)[:, :2]

    # Build tag -> index map
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    # Extract 2D elements from the physical surface
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    connectivity = []
    for et, etags, entags in zip(elem_types, elem_tags, elem_node_tags):
        name, dim, eorder, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(et)
        raw = np.array(entags, dtype=int).reshape(-1, n_nodes)
        for row in raw:
            connectivity.append([tag_to_idx[t] for t in row])
    connectivity = np.array(connectivity, dtype=int)

    # ── Remove orphan nodes (e.g. circle centre) ─────────────────────
    referenced = set(connectivity.ravel())
    keep = np.array(sorted(referenced), dtype=int)
    old_to_new = np.full(nodes.shape[0], -1, dtype=int)
    old_to_new[keep] = np.arange(len(keep))

    n_removed = nodes.shape[0] - len(keep)
    if n_removed > 0:
        print(f"  Removed {n_removed} orphan node(s)")
        nodes = nodes[keep]
        connectivity = old_to_new[connectivity]

    # ── Build node-to-element map for parent lookup ─────────────────
    node_to_elems = {}
    for eidx, elem in enumerate(connectivity):
        for n in elem:
            node_to_elems.setdefault(n, set()).add(eidx)

    # ── Extract boundary edges as ((n1, n2, ...), parent_elem_id) ──
    #    For each 1D element on a physical boundary curve, retrieve its
    #    node indices, then find the unique 2D parent element that
    #    contains all of them.
    boundary_edges = {}
    for phys_tag, phys_name in [(1, "left"),   (2, "right"),
                                (3, "bottom"), (4, "top"),
                                (5, "cylinder")]:
        edges = []
        entities = gmsh.model.getEntitiesForPhysicalGroup(1, phys_tag)
        for ent in entities:
            et_list, _, en_list = gmsh.model.mesh.getElements(dim=1, tag=ent)
            for et, entags in zip(et_list, en_list):
                _, _, _, n_nodes_per_edge, _, _ = gmsh.model.mesh.getElementProperties(et)
                raw = np.array(entags, dtype=int).reshape(-1, n_nodes_per_edge)
                for row in raw:
                    edge_nodes = tuple(
                        old_to_new[tag_to_idx[int(t)]]
                        for t in row
                        if int(t) in tag_to_idx and old_to_new[tag_to_idx[int(t)]] >= 0
                    )
                    if len(edge_nodes) < 2:
                        continue
                    # Parent = unique 2D element containing all edge nodes
                    candidates = set.intersection(
                        *(node_to_elems.get(n, set()) for n in edge_nodes)
                    )
                    if candidates:
                        parent = candidates.pop()
                        edges.append((edge_nodes, parent))
        boundary_edges[phys_name] = edges


    
    new = Path(output)
    
    info = {
        "n_nodes": nodes.shape[0],
        "n_elements": connectivity.shape[0],
        "nodes_per_element": connectivity.shape[1],
        "order": order,
        "boundary_edges": boundary_edges,
        "filepath": new.with_stem(new.stem + "_Ne{}".format(connectivity.shape[0])),
    }
    # ── Save and finalise ────────────────────────────────────────────
    gmsh.write(str(info['filepath']))
    

    print(f"Mesh written to: {info['filepath']}")
    print(f"  Nodes:             {info['n_nodes']}")
    print(f"  Elements:          {info['n_elements']}")
    print(f"  Nodes/element:     {info['nodes_per_element']}")
    print(f"  Order:             {info['order']}")
    for name, edge_list in boundary_edges.items():
        print(f"  {name:16s}:  {len(edge_list)} edges")

    if gui:
        gmsh.fltk.run()

    gmsh.finalize()

    return nodes, connectivity, boundary_edges, info


def save_mesh(filepath, nodes, connectivity, boundary_edges, info):
    """
    Save mesh to a single .npz file.

    Parameters
    ----------
    filepath : str
        Output path (e.g. 'dfg_mesh.npz').
    nodes : np.ndarray
    connectivity : np.ndarray
    boundary_edges : dict of str -> list of ((n1, n2, ...), elem_id)
    info : dict
    """
    data = {
        "nodes": nodes,
        "connectivity": connectivity,
        "order": np.array(info["order"]),
    }

    # Store each boundary as two arrays:
    #   be_<name>_nodes  : (n_edges, nodes_per_edge)  — edge node indices
    #   be_<name>_elems  : (n_edges,)                 — parent element ids
    for name, edge_list in boundary_edges.items():
        if edge_list:
            edge_nodes = np.array([e[0] for e in edge_list], dtype=int)
            edge_elems = np.array([e[1] for e in edge_list], dtype=int)
        else:
            edge_nodes = np.empty((0, 0), dtype=int)
            edge_elems = np.empty((0,), dtype=int)
        data[f"be_{name}_nodes"] = edge_nodes
        data[f"be_{name}_elems"] = edge_elems

    np.savez(filepath, **data)
    print(f"Mesh saved to: {filepath}")


def load_mesh(filepath):
    """
    Load mesh from .npz file.

    Returns
    -------
    nodes : np.ndarray, shape (n_nodes, 2)
    connectivity : np.ndarray, shape (n_elements, nodes_per_elem)
    boundary_edges : dict of str -> list of ((n1, n2, ...), elem_id)
    order : int
    """
    f = np.load(filepath)

    nodes = f["nodes"]
    connectivity = f["connectivity"]
    order = int(f["order"])

    # Reconstruct boundary_edges dict
    boundary_edges = {}
    names = set()
    for key in f.files:
        if key.startswith("be_") and key.endswith("_nodes"):
            names.add(key[3:-6])

    for name in sorted(names):
        edge_nodes = f[f"be_{name}_nodes"]
        edge_elems = f[f"be_{name}_elems"]
        edge_list = []
        for i in range(len(edge_elems)):
            edge_list.append((tuple(int(_) for _ in edge_nodes[i]), int(edge_elems[i])))
        boundary_edges[name] = edge_list

    print(f"Mesh loaded from: {filepath}")

    return nodes, connectivity, boundary_edges, order


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFG benchmark mesh generator")
    
    parser.add_argument("--lc",          type=float, default=0.04,   help="Global element size")
    parser.add_argument("--lc-cylinder", type=float, default=0.008,  help="Cylinder element size")
    parser.add_argument("--lc-wake",     type=float, default=0.015,  help="Wake region element size")
    
    # parser = argparse.ArgumentParser(description="DFG benchmark mesh generator")
    # parser.add_argument("--lc",          type=float, default=0.05,   help="Global element size")
    # parser.add_argument("--lc-cylinder", type=float, default=0.005,  help="Cylinder element size")
    # parser.add_argument("--lc-wake",     type=float, default=0.0075,  help="Wake region element size")
    # parser.add_argument("--order",       type=int,   default=2,      help="Element order")
    
    
    parser.add_argument("--order",       type=int,   default=2,      help="Element order")
    parser.add_argument("--output",      type=str,   default="dfg_benchmark.msh")
    parser.add_argument("--gui",         action="store_true")
    parser.add_argument("--no-quad",     action="store_true",        help="Keep triangles")

    
    args = parser.parse_args()
    
    path = Path(__file__).parents[0] / 'meshes' / args.output
    nodes, conn, bn, info = generate_dfg_mesh(
        lc_global=args.lc,
        lc_cylinder=args.lc_cylinder,
        lc_wake=args.lc_wake,
        order=args.order,
        output= path,
        gui=args.gui,
        recombine=not args.no_quad,
    )
    save_mesh(str(info['filepath']).replace(".msh",".npz"), nodes, conn, bn, info)

    from fem.incompressibleNS.incNS_solver import _plot_mesh
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 3))
    _plot_mesh(nodes, conn, bn, ax =ax)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()