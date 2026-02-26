import pygmsh
import numpy as np

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

    plt.show()