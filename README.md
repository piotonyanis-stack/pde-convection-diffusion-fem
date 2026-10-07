# P1 Finite Elements for Convection-Diffusion-Reaction Problems

Python implementation (NumPy / SciPy / Matplotlib) of a conforming **P1 Lagrange finite element** solver for the convection-diffusion-reaction equation on triangular meshes, written from scratch (no FEM library). Final project of the course *MU4MA329 — Approximation of elliptic PDEs and numerical simulation* (Sorbonne Université, 2026).

## Problem

Find $u \in H^1(\Omega)$ such that

$$
-\Delta u + \mathbf{b}\cdot\nabla u + c\,u = f \quad \text{in } \Omega, \qquad u = g \quad \text{on } \partial\Omega,
$$

with $c > 0$ and $\mathbf{b} \in \mathbb{R}^2$.

**Variational formulation** (homogeneous case, $V = H^1_0(\Omega)$): find $u \in V$ such that for all $v \in V$

$$
\int_\Omega \nabla u\cdot\nabla v + (\mathbf{b}\cdot\nabla u)\,v + c\,uv \,dx = \int_\Omega f v \,dx.
$$

Discretising on $V_h$ (P1 elements) gives the linear system $(K + C(\mathbf{b}) + cM)\,U = M F$, where $K$ is the stiffness matrix, $C(\mathbf{b})$ the convection matrix and $M$ the mass matrix. All matrices are assembled in sparse format from elementary matrices.

## Contents

| Part | What is done |
|------|--------------|
| **1. Validation on a manufactured solution** | Square domain $(0,1)^2$, homogeneous Dirichlet conditions, $f(x,y)=e^{(b_x x+b_y y)/2}\sin(p\pi x)\sin(q\pi y)$. The exact solution is $u_{ex}=\alpha f$ with $\alpha = \big(\|\mathbf{b}\|^2/4 + \pi^2(p^2+q^2) + c\big)^{-1}$. Convergence study of the relative $L^2$ and $H^1$ errors with respect to $h$. |
| **2. Domain with holes** | Mesh `barwith4holes.msh` (rectangle with 4 circular holes). Detection of the connected components of the boundary (union-find), colored plot of each component, and resolution of $-\Delta u + \mathbf{b}\cdot\nabla u + cu = 0$ with $u = j$ on $\Gamma_j$ (Dirichlet data handled by lifting), for increasing $\|\mathbf{b}\|$. |

### Main routines (`projet.py`)

- `ElementaryConvectionMatrix`, `ConvectionMatrix`: assembly of the convection term (gradients obtained from the Jacobian of the affine map to the reference triangle).
- `Matrice_EDP`: assembles $A = K + C + cM$ (shared by both parts).
- `Sol_EDP1`: solves part 1 on a uniform mesh of $2N_xN_y$ triangles (reduced system on interior nodes, `spsolve`).
- `erreurL2`, `erreurH1`, `trace_erreurs`: relative errors and log-log convergence plots, with least-squares estimate of the order.
- `ConnectedComponents`: labels the boundary edges by connected component using a union-find structure with path compression.
- `PlotMeshComponents`: mesh plot with one color per boundary component.
- `Sol_EDP2`: solves the problem with piecewise-constant Dirichlet data on the holes and the outer boundary.
- Mesh utilities: `LoadMesh`, `GenerateMeshRectangle`, `ExtractBoundary`, `RefineMesh`.

## Results

### Convergence (part 1, $\mathbf{b}=(1,1)$, $c=1$)

Errors are measured relative to the P1 interpolant $\Pi_h u_{ex}$ (as asked in the project statement), on uniform meshes with $N = 10, 50, 100, 150$ subdivisions per side.

| $(p,q)$ | Fitted order, $L^2$ | Fitted order, $H^1$ |
|---------|---------------------|---------------------|
| (2, 3)  | 1.946               | 1.952               |
| (1, 6)  | 1.895               | 1.899               |

Both errors converge at a rate close to **2**. For the $H^1$ norm this is notable: the error with respect to $u_{ex}$ itself is only $O(h)$ for P1 elements, but the distance to the *interpolant* is of higher order on these structured meshes (**superconvergence**, or "superclose" behavior). This behavior was observed here, and the results were checked by the course instructor.

<p align="center">
  <img src="figures/convergence_p1_q6.png" width="48%" alt="Convergence for (p,q)=(1,6)">
  <img src="figures/convergence_p2_q3.png" width="48%" alt="Convergence for (p,q)=(2,3)">
</p>

*(Figure labels are in French; dashed line = reference slope 2.)*

### Numerical solution (part 1, $(p,q)=(2,3)$, 5000 triangles)

<p align="center">
  <img src="figures/solution_p2_q3_N50.png" width="55%" alt="Numerical solution">
</p>

### Domain with holes (part 2)

The solution follows the direction of the advection field $\mathbf{b}$. As $\|\mathbf{b}\|$ grows, the problem becomes advection-dominated and the solution decays to 0 very quickly away from the holes. The standard Galerkin method used here is **not stabilized**; for large mesh Péclet numbers, a stabilized scheme (e.g. SUPG) would be the natural next step.

## Usage

```bash
pip install numpy scipy matplotlib meshio
python projet.py
```

By default, `projet.py` runs the convergence study of part 1 (question g). The other tests are commented out at the bottom of the file: uncomment the ones you want (solution vector, solution plot, boundary components, problem with holes). Part 2 requires the mesh file `barwith4holes.msh` in the same folder.

## Repository structure

```
.
├── projet.py        # full implementation and tests
├── project.pdf      # problem statement
├── barwith4holes.msh  # mesh for part 2
├── figures/         # convergence plots and solution plot
└── README.md
```

## Known limitations

- `erreurL2` and `erreurH1` use the exact solution for $\mathbf{b}=(1,1)$, $c=1$ (factor $3/2$ hard-coded); they would need $\alpha$ to be passed as a parameter for other coefficients.
- Matrix assembly uses Python loops over elements, which is fine for the mesh sizes used here (up to 45,000 triangles) but would benefit from vectorization on finer meshes.
- Reading `.msh` files requires `meshio`.
