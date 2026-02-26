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


def generate_rect_mesh(nx, ny, width, height, order=1, element='complete'):
    """
    Generate a rectangular quad mesh.

    Parameters
    ----------
    nx, ny : int
        Number of elements in x and y directions (elements, not nodes).
    width, height : float
        Size of the rectangle in x and y.
    order : int
        Polynomial order: 1 or 2 supported.
    element : str
        For order==2: 'serendipity' (8-node) or 'complete' (9-node). Ignored for order==1.

    Returns
    -------
    nodes : list of (x, y)
        Node coordinates (floats), zero-based index into connectivity.
    conn : list of list[int]
        Element connectivity lists (node indices).
    """
    if order not in (1, 2):
        raise ValueError("Only order 1 and 2 are supported.")
    if order == 2 and element not in ('serendipity', 'complete'):
        raise ValueError("element must be 'serendipity' or 'complete' for order==2")

    # spacing
    dx = width / nx
    dy = height / ny

    if order == 1:
        # regular grid (nx+1) x (ny+1)
        nxn = nx + 1
        nyn = ny + 1

        nodes = []
        for j in range(nyn):                       # y index 0..ny
            y = j * dy
            for i in range(nxn):                   # x index 0..nx
                x = i * dx
                nodes.append([x, y])

        # helper to convert grid coords -> node idx
        def idx(i, j):
            return j * nxn + i

        conn = []
        for ey in range(ny):
            base_y = ey
            for ex in range(nx):
                base_x = ex
                # element nodes in CCW order starting bottom-left:
                # [bottom-left, bottom-right, top-right, top-left]
                bl = idx(base_x, base_y)
                br = idx(base_x + 1, base_y)
                tr = idx(base_x + 1, base_y + 1)
                tl = idx(base_x, base_y + 1)
                conn.append([bl, br, tr, tl])

        return np.array(nodes), np.array(conn)

    else:  # order == 2
        # refined grid has 2*nx + 1 nodes in x, similarly in y
        # grid indices 0..2*nx (integers). Element lower-left corners at (2*ex, 2*ey).
        nxn = 2 * nx + 1
        nyn = 2 * ny + 1

        nodes = []
        for j in range(nyn):
            y = (j * 0.5) * dy    # because j steps are half-element increments
            for i in range(nxn):
                x = (i * 0.5) * dx
                nodes.append([x, y])

        def idx(i, j):
            return j * nxn + i

        conn = []
        for ey in range(ny):
            by = 2 * ey
            for ex in range(nx):
                bx = 2 * ex
                # nodes positions inside the refined 3x3 block:
                # (i,j) with i=0..2, j=0..2 where (0,0) is bottom-left of the element
                # mapping to global index: idx(bx + i, by + j)

                # corners:
                n00 = idx(bx + 0, by + 0)  # bottom-left
                n20 = idx(bx + 2, by + 0)  # bottom-right
                n22 = idx(bx + 2, by + 2)  # top-right
                n02 = idx(bx + 0, by + 2)  # top-left

                # midsides:
                n10 = idx(bx + 1, by + 0)  # mid-bottom
                n21 = idx(bx + 2, by + 1)  # mid-right
                n12 = idx(bx + 1, by + 2)  # mid-top
                n01 = idx(bx + 0, by + 1)  # mid-left

                # center:
                n11 = idx(bx + 1, by + 1)  # center

                if element == 'serendipity':
                    # 8-node serendipity ordering (commonly used):
                    # [bl, br, tr, tl, mid-bottom, mid-right, mid-top, mid-left]
                    conn.append([n00, n20, n22, n02, n10, n21, n12, n01])
                else:
                    # 'complete' 9-node ordering:
                    # [bl, br, tr, tl, mid-bottom, mid-right, mid-top, mid-left, center]
                    conn.append([n00, n20, n22, n02, n10, n21, n12, n01, n11])

        return np.array(nodes), np.array(conn)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # linear mesh example
    nodes, connectivity = generate_rect_mesh(nx=3, ny=2, width=3.0, height=2.0, order=1)
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(connectivity):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    
    print("Order 1: nodes:", len(nodes), "elements:", len(connectivity))
    print("First 5 nodes:", nodes[:5])
    print("First element connectivity (4 nodes):", connectivity[0])

    # quadratic serendipity example
    nodes, connectivity = generate_rect_mesh(nx=3, ny=2, width=3.0, height=2.0, order=2, element='serendipity')
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(connectivity):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    
    print("\nOrder 2 (serendipity): nodes:", len(nodes), "elements:", len(connectivity))
    print("First 9 nodes:", nodes[:9])
    print("First element connectivity (8 nodes):", connectivity[0])

    # quadratic complete example
    nodes, connectivity = generate_rect_mesh(nx=3, ny=2, width=3.0, height=2.0, order=2, element='complete')
    plt.figure()
    plt.plot(nodes[:,0], nodes[:,1], '.')
    for e, con in enumerate(connectivity):
        plt.plot(nodes[con,0], nodes[con,1], '-')
    
    print("\nOrder 2 (complete): nodes:", len(nodes), "elements:", len(connectivity))
    print("First element connectivity (9 nodes):", connectivity[0])


    plt.show()