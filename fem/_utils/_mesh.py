import pygmsh
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterable

#####################################################
# AUXILIARY MESH GENERATION AND FUNCTIONS


def generate_rectangular_domain(height, width, mesh_size = 0.08):
    poly_pts = [
            [0.0,   0.0,    0.0],
            [width, 0.0,    0.0],
            [width, height, 0.0],
            [0.0,   height, 0.0]]

    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(poly_pts, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
    
    nodes = mesh.points 
    connectivity = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            connectivity = cell_block.data
            break
    return nodes, connectivity
        
def generate_circular_domain(radius, mesh_size = 0.08):
    with pygmsh.geo.Geometry() as geom:
            geom.add_circle([0.0, 0.0, 0.0], radius, mesh_size=mesh_size)
            mesh = geom.generate_mesh()

    nodes = mesh.points[1:, :2]

    # Extract triangle cells
    connectivity = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            connectivity = cell_block.data
            break
    connectivity -= 1
    return nodes, connectivity


def generate_rect_mesh(nx, ny, width, h1, h2=None, order=1, element='complete'):
    """
    Generate a rectangular or trapezium quad mesh.

    Behavior:
      - If `h2` is None: behaves like original rectangle generator using `h1` as uniform height.
      - If `h2` is provided: generates a trapezium whose bottom is y=0 and whose top edge
        has height `h_left` at x=0 and `h_right` at x=width (linearly varying top).

    Parameters
    ----------
    nx, ny : int
        Number of elements in x and y directions (elements, not nodes).
    width : float
        Size of the domain in x.
    h1 : float
        If `h2` is None: uniform height of rectangle.
        If `h2` provided: this is the left-top height (h_left).
    h2 : float or None
        If provided, the right-top height (h_right). If None, a rectangle is produced.
    order : int
        Polynomial order: 1 or 2 supported.
    element : str
        For order==2: 'serendipity' (8-node) or 'complete' (9-node). Ignored for order==1.

    Returns
    -------
    nodes : np.ndarray shape (n_nodes, 2)
        Node coordinates (x,y) as floats.
    conn : np.ndarray shape (n_elems, nodes_per_elem)
        Element connectivity lists (node indices).
    """
    if order not in (1, 2):
        raise ValueError("Only order 1 and 2 are supported.")
    if order == 2 and element not in ('serendipity', 'complete'):
        raise ValueError("element must be 'serendipity' or 'complete' for order==2")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers.")
    if width <= 0:
        raise ValueError("width must be positive.")

    # Determine if trapezium or rectangle
    if h2 is None:
        # rectangle mode: single uniform height
        h_left = h_right = float(h1)
    else:
        h_left = float(h1)
        h_right = float(h2)

    # spacing in x and (reference) element height fractions
    dx = width / nx

    # function to compute top height at a given x (linear interpolation)
    def top_height_at_x(x):
        return h_left + (h_right - h_left) * (x / width)

    if order == 1:
        # nodes: (nx+1) by (ny+1)
        nxn = nx + 1
        nyn = ny + 1

        nodes = []
        for j in range(nyn):                       # j = 0..ny
            s = j / ny                             # fraction in vertical direction (0..1)
            for i in range(nxn):                   # i = 0..nx
                x = i * dx
                h_at_x = top_height_at_x(x)
                y = s * h_at_x                    # bottom is y=0, top is y = h_at_x
                nodes.append([x, y])

        nodes = np.array(nodes, dtype=float)

        def idx(i, j):
            return j * nxn + i

        conn = []
        for ey in range(ny):
            by = ey
            for ex in range(nx):
                bl = idx(ex, by)
                br = idx(ex + 1, by)
                tr = idx(ex + 1, by + 1)
                tl = idx(ex, by + 1)
                # order: [bottom-left, bottom-right, top-right, top-left]
                conn.append([bl, br, tr, tl])

        return nodes, np.array(conn, dtype=int)

    else:  # order == 2
        # refined grid: 2*nx + 1 by 2*ny + 1
        nxn = 2 * nx + 1
        nyn = 2 * ny + 1

        nodes = []
        # note: refined steps in x are (0.5*dx) increments; in y fractional step is (j/ (2*ny))
        for j in range(nyn):
            s = j / (2 * ny)   # fraction between bottom (0) and top (1)
            for i in range(nxn):
                x = (i * 0.5) * dx
                h_at_x = top_height_at_x(x)
                y = s * h_at_x
                nodes.append([x, y])

        nodes = np.array(nodes, dtype=float)

        def idx(i, j):
            return j * nxn + i

        conn = []
        for ey in range(ny):
            by = 2 * ey
            for ex in range(nx):
                bx = 2 * ex
                # corners
                n00 = idx(bx + 0, by + 0)  # bottom-left
                n20 = idx(bx + 2, by + 0)  # bottom-right
                n22 = idx(bx + 2, by + 2)  # top-right
                n02 = idx(bx + 0, by + 2)  # top-left

                # midsides
                n10 = idx(bx + 1, by + 0)  # mid-bottom
                n21 = idx(bx + 2, by + 1)  # mid-right
                n12 = idx(bx + 1, by + 2)  # mid-top
                n01 = idx(bx + 0, by + 1)  # mid-left

                # center
                n11 = idx(bx + 1, by + 1)

                if element == 'serendipity':
                    # 8-node: [bl, br, tr, tl, mid-bottom, mid-right, mid-top, mid-left]
                    conn.append([n00, n20, n22, n02, n10, n21, n12, n01])
                else:
                    # 9-node complete: add center last
                    conn.append([n00, n20, n22, n02, n10, n21, n12, n01, n11])

        return nodes, np.array(conn, dtype=int)


Segment = Tuple[int, int]
SegmentWithElem = Tuple[Segment, int]
EdgesDict = Dict[str, List[SegmentWithElem]]

def boundary_edges_connectivity(conn: np.ndarray, nx: int, ny: int,
                                 order: int = 1, element: str = 'complete') -> EdgesDict:
    """
    Return linear boundary edge segments with the element id each segment belongs to.

    Parameters
    ----------
    conn : array-like, shape (n_elems, nodes_per_elem)
        Element connectivity. Assumes first 4 entries are corners [bl, br, tr, tl].
    nx, ny : int
        Number of elements in x and y directions.
    order : int
        1 or 2.
    element : str
        For order==2: 'serendipity' or 'complete' (affects nodal layout).

    Returns
    -------
    edges : dict
        keys: 'bottom','right','top','left'
        values: lists of (segment, elem_id) where segment is (n_start, n_end)
                 and elem_id is the index of the element in conn that owns that edge.
    """
    conn = np.asarray(conn, dtype=int)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if order == 2 and element not in ('serendipity', 'complete'):
        raise ValueError("element must be 'serendipity' or 'complete' for order==2")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")

    def elem_index(ex: int, ey: int) -> int:
        return ey * nx + ex

    edges: EdgesDict = {'bottom': [], 'right': [], 'top': [], 'left': []}

    # bottom: ey=0, ex=0..nx-1, direction left->right
    ey = 0
    for ex in range(0, nx):
        e = conn[elem_index(ex, ey)]
        elem_id = elem_index(ex, ey)
        n_bl, n_br = int(e[0]), int(e[1])
        if order == 1:
            edges['bottom'].append(((n_bl, n_br), elem_id))
        else:
            nmid = int(e[4])   # mid-bottom at index 4 in our ordering
            edges['bottom'].append(((n_bl, nmid), elem_id))
            edges['bottom'].append(((nmid, n_br), elem_id))

    # right: ex=nx-1, ey=0..ny-1, bottom->top
    ex = nx - 1
    for ey in range(0, ny):
        e = conn[elem_index(ex, ey)]
        elem_id = elem_index(ex, ey)
        n_br, n_tr = int(e[1]), int(e[2])
        if order == 1:
            edges['right'].append(((n_br, n_tr), elem_id))
        else:
            nmid = int(e[5])   # mid-right at index 5
            edges['right'].append(((n_br, nmid), elem_id))
            edges['right'].append(((nmid, n_tr), elem_id))

    # top: ey=ny-1, ex=nx-1..0, direction right->left (keep consistent orientation)
    ey = ny - 1
    for ex in range(nx - 1, -1, -1):
        e = conn[elem_index(ex, ey)]
        elem_id = elem_index(ex, ey)
        n_tr, n_tl = int(e[2]), int(e[3])
        if order == 1:
            edges['top'].append(((n_tr, n_tl), elem_id))
        else:
            nmid = int(e[6])   # mid-top at index 6
            edges['top'].append(((n_tr, nmid), elem_id))
            edges['top'].append(((nmid, n_tl), elem_id))

    # left: ex=0, ey=ny-1..0, direction top->bottom
    ex = 0
    for ey in range(ny - 1, -1, -1):
        e = conn[elem_index(ex, ey)]
        elem_id = elem_index(ex, ey)
        n_tl, n_bl = int(e[3]), int(e[0])
        if order == 1:
            edges['left'].append(((n_tl, n_bl), elem_id))
        else:
            nmid = int(e[7])   # mid-left at index 7
            edges['left'].append(((n_tl, nmid), elem_id))
            edges['left'].append(((nmid, n_bl), elem_id))

    return edges

def find_corners_fromSegmentsWithElem(edges_list:List[List[SegmentWithElem]]):


    seen_index = set()
    vertex_index = set()
    for edges in edges_list:
        for segWele in edges:
            print(segWele)
            index = segWele[0][1]
            print('\t', index)
            if index in seen_index:
                if index in vertex_index:
                    raise RuntimeError("WTF is this edge connectivity")
                vertex_index.add(index)
                seen_index.remove(index)
            else:
                seen_index.add(index)

    return vertex_index



def _cluster_x_stations(nodes: np.ndarray, indices: np.ndarray, tol: float):
    """
    Helper: cluster the x-values of `nodes[indices]` into stations using tol.
    Returns list of representative x (mean) and list of lists of node indices (sorted by y).
    """
    xs = nodes[indices, 0]
    ys = nodes[indices, 1]
    # sort by x then y
    order = np.lexsort((ys, xs))
    xs_s = xs[order]
    idx_s = indices[order]

    stations = []
    station_lists = []
    for xval, nidx in zip(xs_s, idx_s):
        if not stations:
            stations.append(xval)
            station_lists.append([int(nidx)])
            continue
        if abs(xval - stations[-1]) <= tol:
            station_lists[-1].append(int(nidx))
        else:
            stations.append(xval)
            station_lists.append([int(nidx)])

    reps = []
    final_groups = []
    for idx_list in station_lists:
        idx_arr = np.array(idx_list, dtype=int)
        # sort by y ascending
        order_y = np.argsort(nodes[idx_arr, 1], kind='stable')
        sorted_idx = idx_arr[order_y].tolist()
        final_groups.append(sorted_idx)
        reps.append(float(np.mean(nodes[idx_arr, 0])))

    return np.array(reps, dtype=float), final_groups

def group_nodes_by_x(
    nodes: np.ndarray,
    conn: Optional[np.ndarray] = None,
    order: int = 1,
    tol: float = 1e-12) -> Tuple[Dict[float, List[int]], Dict[float, List[int]]
]:
    """
    Group nodes by x-station (all nodes) and also return the subset of stations
    that correspond to element vertical edges (i.e. x-values of corner nodes).

    Parameters
    ----------
    nodes : (N,2) array-like
        Node coordinates [[x0,y0], [x1,y1], ...].
    conn : optional (n_elems, nodes_per_elem) connectivity array
        If provided, corner node indices are taken from conn[:, :4] to compute edge stations.
        If not provided and order==2, the function will attempt to infer edge x-stations by
        selecting unique x-values that align with a coarse spacing (less robust).
    order : int
        Element order: 1 or 2. For order==1 edge stations == all stations.
    tol : float
        Tolerance for grouping x-values.

    Returns
    -------
    all_x_vals, all_groups, all_map, edge_x_vals, edge_groups, edge_map
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must be shape (N,2)")

    N = nodes.shape[0]

    # --- all stations (using all nodes) ---
    all_indices = np.arange(N, dtype=int)
    all_x_vals, all_groups = _cluster_x_stations(nodes, all_indices, tol)
    # sort ascending (already should be sorted)
    sort_idx = np.argsort(all_x_vals)
    all_x_vals = all_x_vals[sort_idx]
    all_groups = [all_groups[i] for i in sort_idx]
    all_map = {float(all_x_vals[i]): np.array(all_groups[i]) for i in range(len(all_x_vals))}

    # --- edge/corner stations ---
    if order == 1:
        # identical
        edge_x_vals = all_x_vals.copy()
        edge_groups = list(all_groups)
        edge_map = dict(all_map)
        return all_x_vals, all_groups, all_map, edge_x_vals, edge_groups, edge_map

    # order == 2: prefer to compute edge stations from corner nodes in conn[:,:4]
    edge_x_vals = None
    edge_groups = None

    if conn is not None:
        conn = np.asarray(conn, dtype=int)
        if conn.ndim != 2 or conn.shape[1] < 4:
            raise ValueError("conn must be shape (n_elems, >=4) with corners in first 4 entries")

        corner_idx = np.unique(conn[:, :4].ravel())
        # cluster only the corner nodes
        reps, groups = _cluster_x_stations(nodes, corner_idx, tol)
        # sort
        sort_idx = np.argsort(reps)
        edge_x_vals = reps[sort_idx]
        edge_groups = [groups[i] for i in sort_idx]
    else:
        # fallback: attempt to infer edge stations from all_x_vals by selecting a subset
        # (less robust). We choose every other station if pattern fits.
        nstations = len(all_x_vals)
        if (nstations - 1) % 2 == 0:
            expected = (nstations - 1) // 2
            # pick even-indexed stations
            indices = np.arange(0, nstations, 2, dtype=int)
            edge_x_vals = all_x_vals[indices].copy()
            edge_groups = [all_groups[i] for i in indices.tolist()]
        else:
            # last fallback: pick unique x among nodes with minimal count equal to len(all)/2 approx
            # use unique x among nodes that have maximum y-span (likely corners)
            # heuristic: pick nodes with min y or max y (bottom/top) as corners, collect their x
            bottom_nodes = np.where(nodes[:,1] == nodes[:,1].min())[0]
            top_nodes = np.where(nodes[:,1] == nodes[:,1].max())[0]
            candidate = np.unique(np.concatenate([bottom_nodes, top_nodes]))
            if candidate.size > 0:
                reps, groups = _cluster_x_stations(nodes, candidate, tol)
                sort_idx = np.argsort(reps)
                edge_x_vals = reps[sort_idx]
                edge_groups = [groups[i] for i in sort_idx]
            else:
                # fallback to all
                edge_x_vals = all_x_vals.copy()
                edge_groups = list(all_groups)

    edge_map = {float(edge_x_vals[i]): np.array(edge_groups[i]) for i in range(len(edge_groups))}
    return all_map, edge_map


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # linear mesh example
    nodes, conn = generate_rect_mesh(nx=3, ny=2, width=3.0, h1=2.0, order=1)
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    
    print("\nRectangular Domain")
    print("Order 1: nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (4 nodes):", conn[0])

    # quadratic serendipity example
    nodes, conn = generate_rect_mesh(nx=3, ny=2, width=3.0, h1=2.0, order=2, element='serendipity')
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    print("\nRectangular Domain")
    print("\nOrder 2 (serendipity): nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (8 nodes):", conn[0])

    # quadratic complete example
    nodes, conn = generate_rect_mesh(nx=3, ny=2, width=3.0, h1=2.0, order=2, element='complete')
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    print("\nRectangular Domain")
    print("Order 2 (complete): nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (9 nodes):", conn[0])

    # trapezium: left top = 1.0, right top = 0.5
    nodes, conn = generate_rect_mesh(nx=4, ny=2, width=2.0, h1=1.0, h2=0.5, order=1)
    print("\nTrapezoidal Domain")
    print("Order 1 (complete): nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (9 nodes):", conn[0])
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    
    # second-order serendipity trapezium
    nodes, conn = generate_rect_mesh(nx=3, ny=2, width=3.0, h1=1.2, h2=0.8, order=2, element='serendipity')
    print("\nTrapezoidal Domain")
    print("Order 2 (complete): nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (9 nodes):", conn[0])
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')

    

    # generate a 2nd-order rectangular mesh (for example)
    nodes, conn = generate_rect_mesh(nx=4, ny=2, width=2.0, h1=1.0, h2=0.5, order=2)
    edges = boundary_edges_connectivity(conn, nx=4, ny=2, order=2, element='complete')
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '--')
    for k,v in edges.items():
        for edge in v:
            temp = np.vstack(nodes[edge[0],:]).T
            plt.plot(*temp, '-')
    print("Bottom edges (count):", len(edges['bottom']))
    print("First bottom edge nodes:", edges['bottom'][0])
    print("Right edges (count):", len(edges['right']))
    print("Top edges (count):", len(edges['top']))
    print("Left edges (count):", len(edges['left']))




    all_map, edge_map = \
        group_nodes_by_x(nodes, conn, order=2, tol=1e-9)
    for k,v in all_map.items():
        plt.plot(nodes[v,0], nodes[v,1], 'o', markerfacecolor = 'none', ms = 10, markeredgewidth = 2)
    for k,v in edge_map.items():
        plt.plot(nodes[v,0], nodes[v,1], 's', markerfacecolor = 'none', ms = 15, markeredgewidth = 2)

    plt.show()