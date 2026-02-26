from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterable, List, Optional, Tuple, Union, Any
import numpy as np

# typing helpers
Point = Tuple[float, float] 
NodeIndex = int
Segment = Tuple[int, int]             # linear segment (n0, n1)
SegmentWithElem = Tuple[Segment, int] # (segment, elem_id)
SegmentsList = List[Union[Segment, SegmentWithElem]]

# A function that returns a value at (x,y,t). Returns either scalar or 2-tuple for velocity.
BCFunc = Callable[[float, float, float], Union[float, Tuple[float, float]]]


class BCType(Enum):
    """Type of boundary condition"""
    DIRICHLET = auto()    # prescribe values (velocity or pressure)
    NEUMANN = auto()      # prescribe flux/traction (stress)
    ROBIN = auto()        # mixed (alpha*u + beta*du/dn = g)

class Var(Enum):
    """Which variable(s) the BC applies to"""
    VELOCITY = auto()     # (u, v)
    PRESSURE = auto()     # p
    BOTH = auto()         # both

@dataclass
class BoundaryCondition:
    """
    Describe boundary conditions applied on a geometric boundary (edge).
    Store either constant values or spatial/time-dependent functions.

    Attributes
    ----------
    name : str
        Logical name of the boundary (e.g., "inlet", "wall-left", "outlet", "top").
    boundary_key : str
        One of 'bottom','right','top','left' or any user-defined label.
    segments : Optional[SegmentsList]
        List of segments that make up this boundary. Each segment is either:
           - (n0, n1) tuple (node indices) or
           - ((n0,n1), elem_id) tuple if you use element ownership info.
        This is optional: you can instead use 'tag' or 'boundary_key' to attach BCs.
    bc_type : BCType
        Type of BC (DIRICHLET, NEUMANN, ...).
    variable : Var
        Which variable(s) the BC acts on (VELOCITY, PRESSURE, or BOTH).
    value : Optional[Union[float, Tuple[float, float], BCFunc]]
        The prescribed value. For VELOCITY: provide (u,v) or function(x,y,t)->(u,v).
        For PRESSURE: provide scalar or function(x,y,t)->p.
    traction : Optional[BCFunc]
        If bc_type==NEUMANN and variable==VELOCITY, this function should return the traction
        (t_x, t_y) at (x,y,t). If None and NEUMANN specified, solver may compute natural BC.
    apply_strong : bool
        If True apply BC strongly (dirichlet enforced directly); otherwise planned for weak/penalty.
    active : bool
        If False this boundary is ignored (useful for toggling BCs in tests).
    metadata : dict
        Arbitrary user metadata (units, notes, reference ids).
    """
    name: str
    boundary_key: str
    segments: Optional[SegmentsList] = None
    bc_type: BCType = BCType.DIRICHLET
    variable: Var = Var.VELOCITY
    value: Optional[Union[float, Tuple[float, float], BCFunc]] = None
    traction: Optional[BCFunc] = None
    apply_strong: bool = True
    active: bool = True
    metadata: dict = field(default_factory=dict)

    def is_dirichlet_velocity(self) -> bool:
        return self.bc_type == BCType.DIRICHLET and self.variable in (Var.VELOCITY, Var.BOTH)

    def is_dirichlet_pressure(self) -> bool:
        return self.bc_type == BCType.DIRICHLET and self.variable in (Var.PRESSURE, Var.BOTH)

    def is_neumann(self) -> bool:
        return self.bc_type == BCType.NEUMANN

def evaluate_bc_at_point(bc: BoundaryCondition, x: float, y: float, t: float = 0.0) -> Any:
    """
    Evaluate the BC value at (x,y,t).

    Returns:
      - For velocity Dirichlet: (u, v)
      - For pressure Dirichlet: scalar p
      - For Neumann traction: (tx, ty)
      - None if bc.value/traction is None
    """
    if not bc.active:
        return None

    if bc.bc_type == BCType.DIRICHLET:
        if callable(bc.value):
            return bc.value(x, y, t)
        else:
            return bc.value
    elif bc.bc_type == BCType.NEUMANN:
        if bc.traction is not None:
            return bc.traction(x, y, t)
        else:
            return None
    else:
        return None

# 1) No-slip wall on left boundary (velocity = (0,0))
bc_wall_left = BoundaryCondition(
    name="no-slip-left",
    boundary_key="left",
    bc_type=BCType.DIRICHLET,
    variable=Var.VELOCITY,
    value=(0.0, 0.0),
    apply_strong=True,
    metadata={"description": "No-slip wall"}
)

# 2) Parabolic inlet on bottom boundary (u = Umax * 4*y*(H - y)/H^2, v = 0)
def parabolic_inlet(x, y, t, Umax=1.0, H=1.0):
    # here y is the coordinate normal to the inflow direction only if your bottom is horizontal.
    u = Umax * 4.0 * y * (H - y) / (H * H)
    return (u, 0.0)

bc_inlet = BoundaryCondition(
    name="parabolic-inlet",
    boundary_key="bottom",
    bc_type=BCType.DIRICHLET,
    variable=Var.VELOCITY,
    value=lambda x, y, t: parabolic_inlet(x, y, t, Umax=1.0, H=1.0),
    metadata={"Umax": 1.0, "profile": "parabolic"}
)

# 3) Stress-free (traction-free) outlet on right boundary (Neumann: traction = (0,0))
bc_outlet = BoundaryCondition(
    name="outlet-stressfree",
    boundary_key="right",
    bc_type=BCType.NEUMANN,
    variable=Var.VELOCITY,
    traction=lambda x, y, t: (0.0, 0.0),
    apply_strong=False,
    metadata={"description": "do-nothing / traction-free outlet"}
)

# 4) Fixed pressure (Dirichlet) at a single boundary if required:
bc_pressure_ref = BoundaryCondition(
    name="pressure-ref",
    boundary_key="right",
    bc_type=BCType.DIRICHLET,
    variable=Var.PRESSURE,
    value=0.0,
    apply_strong=True,
    metadata={"note": "Reference pressure to fix nullspace"}
)

# Pack into a list for solver
boundary_conditions = [bc_wall_left, bc_inlet, bc_outlet, bc_pressure_ref]

# --- Example usage with mesh nodes & edges you already compute ---
def attach_segments_from_edges(bc: BoundaryCondition, edges_dict: dict):
    """
    If you computed `edges = boundary_edges_with_elements(...)`,
    attach the segments for the bc automatically by matching boundary_key.
    """
    if bc.boundary_key in edges_dict:
        bc.segments = edges_dict[bc.boundary_key]
    else:
        bc.segments = None

# Example snippet:
# edges = boundary_edges_with_elements(conn, nx, ny, order=2, element='serendipity')
# attach_segments_from_edges(bc_inlet, edges)
# Now bc_inlet.segments contains the (segment, elem_id) list for 'bottom'

# helper to evaluate bc at midpoint of a segment
def evaluate_bc_on_segment_midpoint(bc: BoundaryCondition, nodes: np.ndarray, seg: Segment, t: float = 0.0):
    n0, n1 = seg
    x0, y0 = nodes[n0]
    x1, y1 = nodes[n1]
    xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return evaluate_bc_at_point(bc, float(xm), float(ym), t)