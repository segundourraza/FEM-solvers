import pygmsh
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterable

#####################################################
# AUXILIARY MESH GENERATION AND FUNCTIONS

def generate_uniform_rect_mesh(nx, ny, width, h1, h2=None, order=1, element='complete'):
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


def generate_nonuniform_rect_mesh(nx: int,
                                  ny: int,
                                  width: float,
                                  h1: float,
                                  h2: float = None,
                                  order: int = 1,
                                  element: str = 'complete',
                                  dx: list = None,
                                  dy: list = None,
                                  origin: Tuple[float,float] = (0.0, 0.0)
                                 ):
    """
    Generate a non-uniform rectangular / trapezium quadrilateral mesh while matching
    the calling convention of generate_uniform_rect_mesh.

    Parameters
    ----------
    nx, ny : int
        Number of elements in x and y directions.
    width : float
        Full domain width in x (sum(dx) must equal this or dx will be scaled).
    h1 : float
        If h2 is None -> uniform rectangle height.
        If h2 provided -> left-top height (h_left).
    h2 : float or None
        If provided -> right-top height (h_right). If None, rectangle mode.
    order : int
        1 or 2
    element : str
        For order==2: 'serendipity' (8-node) or 'complete' (9-node).
    dx : list or None
        Optional list of `nx` element widths. If None, uniform widths used (width/nx).
        If provided and sums != width, dx will be scaled to sum to `width`.
    dy : list or None
        Optional list of `ny` vertical weights (relative heights). Values must be positive.
        They are normalized so only relative sizes matter. If None, uniform vertical spacing used.
    origin : (ox, oy)
        Lower-left corner coordinates (default (0,0))

    Returns
    -------
    nodes : np.ndarray (n_nodes, 2)
    conn  : np.ndarray (n_elems, nodes_per_elem)
    """
    element = element.lower()
    if order not in (1,2):
        raise ValueError("Only order 1 and 2 are supported.")
    if order == 2 and element not in ('serendipity','complete'):
        raise ValueError("element must be 'serendipity' or 'complete' for order==2")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers.")
    if width <= 0:
        raise ValueError("width must be positive.")

    ox, oy = origin

    # Determine trapezium / rectangle heights
    if h2 is None:
        h_left = h_right = float(h1)
    else:
        h_left = float(h1)
        h_right = float(h2)

    # --- handle dx (x element widths) ---
    if dx is None:
        dx_arr = np.full(nx, width / nx, dtype=float)
    else:
        if len(dx) != nx:
            raise ValueError("dx must have length nx")
        dx_arr = np.asarray(dx, dtype=float)
        if np.any(dx_arr <= 0):
            raise ValueError("dx entries must be positive.")
        s = dx_arr.sum()
        if s <= 0:
            raise ValueError("Sum of dx must be positive.")
        if not np.isclose(s, width):
            # scale to match width so user doesn't accidentally provide inconsistent sum
            dx_arr = dx_arr * (width / s)

    # corner x coordinates
    x_corners = ox + np.concatenate(([0.0], np.cumsum(dx_arr)))

    # --- handle dy (vertical weights -> fractions) ---
    if dy is None:
        dy_weights = np.ones(ny, dtype=float)
    else:
        if len(dy) != ny:
            raise ValueError("dy must have length ny")
        dy_weights = np.asarray(dy, dtype=float)
        if np.any(dy_weights <= 0):
            raise ValueError("dy entries must be positive.")

    # normalized cumulative fractions at element corners in y: length ny+1, from 0..1
    dy_cum = np.concatenate(([0.0], np.cumsum(dy_weights)))
    total_dy = dy_cum[-1]
    if total_dy <= 0:
        raise ValueError("Sum of dy must be positive.")
    y_fracs = dy_cum / total_dy   # fractions [0, ..., 1]

    # helper to compute top height at given absolute x (linear interpolation)
    def top_height_at_x(x_abs: float) -> float:
        # if width == 0 (shouldn't happen), avoid division by zero
        if width == 0:
            return h_left
        t = (x_abs - ox) / width
        return h_left + (h_right - h_left) * t

    if order == 1:
        # node grid is (nx+1) by (ny+1) but y depends on x via top_height_at_x
        nxn = nx + 1
        nyn = ny + 1
        nodes = []
        for j in range(nyn):           # j = 0..ny
            s = y_fracs[j]             # fraction (0..1) for this horizontal line
            for i in range(nxn):       # i = 0..nx
                x = x_corners[i]
                h_at_x = top_height_at_x(x)
                y = oy + s * h_at_x
                nodes.append([x, y])
        nodes = np.asarray(nodes, dtype=float)

        def idx(i,j):
            return j * nxn + i

        conn = []
        for ey in range(ny):
            by = ey
            for ex in range(nx):
                bl = idx(ex, by)
                br = idx(ex + 1, by)
                tr = idx(ex + 1, by + 1)
                tl = idx(ex, by + 1)
                conn.append([bl, br, tr, tl])

        return nodes, np.asarray(conn, dtype=int)

    else:
        # order == 2
        # Build refined x positions: length 2*nx + 1 (even indices = corners, odd = midpoints)
        nxn = 2*nx + 1
        x_ref = np.empty(nxn, dtype=float)
        for k in range(nxn):
            if k % 2 == 0:  # corner
                x_ref[k] = x_corners[k//2]
            else:            # midpoint between corner k//2 and k//2 + 1
                x_ref[k] = 0.5*(x_corners[(k-1)//2] + x_corners[(k+1)//2])

        # Build refined y fractions (length 2*ny + 1). Even: corner fractions; Odd: midpoint fractions
        nyn = 2*ny + 1
        y_ref_frac = np.empty(nyn, dtype=float)
        for k in range(nyn):
            if k % 2 == 0:
                y_ref_frac[k] = y_fracs[k//2]
            else:
                low = y_fracs[(k-1)//2]
                high = y_fracs[(k+1)//2]
                y_ref_frac[k] = 0.5*(low + high)

        # Build node grid: x_ref (0..2*nx) and y_ref_frac (0..2*ny)
        nodes = []
        for j in range(nyn):
            for i in range(nxn):
                x = x_ref[i]
                h_at_x = top_height_at_x(x)
                y = oy + y_ref_frac[j] * h_at_x
                nodes.append([x, y])
        nodes = np.asarray(nodes, dtype=float)

        def idx(i,j):
            return j * nxn + i

        conn = []
        for ey in range(ny):
            by = 2*ey
            for ex in range(nx):
                bx = 2*ex
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
                    conn.append([n00, n20, n22, n02, n10, n21, n12, n01])
                else:
                    conn.append([n00, n20, n22, n02, n10, n21, n12, n01, n11])

        return nodes, np.asarray(conn, dtype=int)



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
        for id in [0,-1]:
            index = edges[id][0][id]
            if index in seen_index:
                if index in vertex_index:
                    raise RuntimeError("WTF is this edge connectivity")
                vertex_index.add(index)
                seen_index.remove(index)
            else:
                seen_index.add(index)

    return list(sorted(vertex_index))


def group_array(arr:np.ndarray, tol = 1e-9):
    arr = np.asarray(arr)
    sort_idx = np.argsort(arr)
    sorted_arr = arr[sort_idx]

    breaks = np.where(np.diff(sorted_arr) > tol)[0] + 1
    groups = np.split(sort_idx, breaks)

    reps = [arr[g[0]] for g in groups]  # representative value

    return {float(rep): g.tolist() for rep, g in zip(reps, groups)}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # linear mesh example
    nodes, conn = generate_uniform_rect_mesh(nx=3, ny=2, width=3.0, h1=2.0, order=1)
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    
    print("\nRectangular Domain")
    print("Order 1: nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (4 nodes):", conn[0])

    # quadratic serendipity example
    nodes, conn = generate_uniform_rect_mesh(nx=3, ny=2, width=3.0, h1=2.0, order=2, element='serendipity')
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    print("\nRectangular Domain")
    print("\nOrder 2 (serendipity): nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (8 nodes):", conn[0])

    # quadratic complete example
    nodes, conn = generate_uniform_rect_mesh(nx=3, ny=2, width=3.0, h1=2.0, order=2, element='complete')
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    print("\nRectangular Domain")
    print("Order 2 (complete): nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (9 nodes):", conn[0])

    # trapezium: left top = 1.0, right top = 0.5
    nodes, conn = generate_uniform_rect_mesh(nx=4, ny=2, width=2.0, h1=1.0, h2=0.5, order=1)
    print("\nTrapezoidal Domain")
    print("Order 1 (complete): nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (9 nodes):", conn[0])
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    
    # second-order serendipity trapezium
    nodes, conn = generate_uniform_rect_mesh(nx=3, ny=2, width=3.0, h1=1.2, h2=0.8, order=2, element='serendipity')
    print("\nTrapezoidal Domain")
    print("Order 2 (complete): nodes:", len(nodes), "elements:", len(conn))
    print("First element connectivity (9 nodes):", conn[0])
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '-')

    

    # generate a 2nd-order rectangular mesh (for example)
    nodes, conn = generate_uniform_rect_mesh(nx=4, ny=2, width=2.0, h1=1.0, h2=0.5, order=2)
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

    # Example 1: nonuniform in x, uniform in y, rectangle
    dx = [0.5,0.5,1.0,1.0,1.0, 0.5, 0.5, 0.25, 0.25]   # sum = 3.0
    nodes, conn = generate_nonuniform_rect_mesh(nx=9, ny=2, width=3.0, h1=1.0, h2=None,
                                                order=2, element='complete', dx=dx, dy=None)
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '--')
    
    
    # Example 2: uniform x, nonuniform y, trapezium (top left 1.0 top right 2.0)
    dy = [1.0, 2.0]  # relative heights -> middle row twice as tall as bottom
    nodes, conn = generate_nonuniform_rect_mesh(nx=4, ny=2, width=2.0, h1=1.0, h2=2.0,
                                                  order=1, element='complete', dx=None, dy=dy)
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(conn):
        plt.plot(nodes[con,0], nodes[con,1], '--')
    
    
    plt.show()