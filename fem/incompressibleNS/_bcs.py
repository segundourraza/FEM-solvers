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

class BCVar(Enum):
    """Which variable(s) the BC applies to"""
    VELOCITY = auto()     # (u, v)
    PRESSURE = auto()     # p
    TRACTION = auto()

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
    type: BCType = BCType.DIRICHLET
    variable: BCVar = BCVar.VELOCITY
    value: Optional[Union[float, Tuple[Optional[float], Optional[float]], BCFunc]] = None
    apply_strong: bool = False
    active: bool = True
    metadata: dict = field(default_factory=dict)
        
    def attach_segments_from_edges(self, edges_dict: dict):
        """
        If you computed `edges = boundary_edges_with_elements(...)`,
        attach the segments for the bc automatically by matching boundary_key.
        """
        if self.boundary_key in edges_dict:
            self.segments = edges_dict[self.boundary_key]
        else:
            self.segments = None

    
    def __call__(self, x:float, y:float, t:float = 0.0):
        """
        Evaluate the BC value at (x,y,t).

        Returns:
        - For velocity Dirichlet: (u, v)
        - For pressure Dirichlet: scalar p
        - For Neumann traction: (tx, ty)
        - None if bc.value/traction is None
        """
        if not self.active:
            return None

        if callable(self.value):
            return self.value(x, y, t)
        else:
            return self.value

@dataclass
class PressureReferenceNode:

    value: float
    index: Optional[int] = None


if __name__ == '__main__':
    ###############################################
    # EXAMPLES

    # 1) No-slip wall on left boundary (velocity = (0,0))
    bc_wall_bottom = BoundaryCondition(
        name="no-slip-left",
        boundary_key="bottom",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value=(0.0, 0.0),
        apply_strong=True,
        metadata={"description": "No-slip wall"}
    )

    # 2) Parabolic inlet on bottom boundary (u = Umax * 4*y*(H - y)/H^2, v = 0)
    def parabolic_inlet(x:float, y:float, t:float, Umax=1.0, H=1.0):
        """Parabolic inlet profile with flow parallel to x-direction

        Parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinate
        t : float
            time elapsed
        Umax : float, optional
            Maximum velocity of parabolic profile, by default 1.0
        H : float, optional
            height of parabolic profile, by default 1.0

        Returns
        -------
        Tuple
            Velocity vector
        """
        u = Umax * 4.0 * y * (H - y) / (H * H)
        return np.select([x<0, x<=H, x>H], [(0.0,0.0), (u,0.0), (0.0,0.0)])

    bc_inlet = BoundaryCondition(
        name="inlet-parabolic-velocity",
        boundary_key="left",
        type=BCType.DIRICHLET,
        variable=BCVar.VELOCITY,
        value=lambda x, y, t: parabolic_inlet(x, y, t, Umax=1.0, H=1.0),
        metadata={"Umax": 1.0, "profile": "parabolic"}
    )

    # 3) Stress-free (traction-free) outlet on right boundary (Neumann: traction = (0,0))
    bc_stressfree_outlet = BoundaryCondition(
        name="outlet-stressfree",
        boundary_key="right",
        type=BCType.NEUMANN,
        variable=BCVar.VELOCITY,
        traction=lambda x, y, t: (0.0, 0.0),
        apply_strong=False,
        metadata={"description": "do-nothing / traction-free outlet"}
    )

    # 4) Fixed pressure (Dirichlet) 
    bc_pressure_outlet_right = BoundaryCondition(
        name="outlet-pressure",
        boundary_key="right",
        type=BCType.DIRICHLET,
        variable=BCVar.PRESSURE,
        value=0.0,
        apply_strong=True,
        metadata={"note": "Reference pressure to fix nullspace"}
    )
