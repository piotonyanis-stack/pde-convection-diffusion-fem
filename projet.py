# Prénom     : Yanis
# Nom        : Pioton
# N°étudiant : 21217092

import meshio
import numpy as np
from scipy import sparse 
import matplotlib.pyplot as plt

########################################## Codes des TP utiles pour le projet #######################################################

# TP 1 (les codes ont été adaptés)

# Les codes de :
# erreurL2(Sol,vertex,id_free,Mass,p,q) 
# erreurH1(Sol,vertex,id_free,Mass,Stiffness,p,q) 
# viennent du TP1 mais ont été adapté pour le projet.


# TP 2

def LoadMesh(fichier) :
    mesh = meshio.read(fichier)
    vtx = mesh.points[:,:-1]
    elt = mesh.cells_dict['triangle']
    return vtx, elt

def GenerateMeshRectangle(sizex, sizey, subx, suby) :
    nx = subx+1
    ny = suby+1
    vtx = np.zeros((nx*ny,2))
    elt = np.zeros((2*subx*suby,3), dtype = int)
    for i in range(0,ny) :
        for j in range(0,nx) :
            vtx[ i*nx+j, 0] = j*sizex/subx
            vtx[ i*nx+j,1] = i*sizey/suby
    for i in range(0,suby) :
        for j in range(0,subx) :
            elt[ 2*i*subx + 2*j, 0] = i*nx + j
            elt[ 2*i*subx + 2*j, 1] = i*nx + j+1
            elt[ 2*i*subx + 2*j, 2] = (i+1)*nx + j+1
            elt[ 2*i*subx + 2*j+1, 0] = i*nx + j
            elt[ 2*i*subx + 2*j+1, 1] = (i+1)*nx + j+1
            elt[ 2*i*subx + 2*j+1, 2] = (i+1)*nx + j
    return vtx, elt

def ExtractBoundary(connect) :
    arretes_tot = {}

    for triangle in connect :
        arretes = [tuple(sorted((triangle[0],triangle[1]))),
                       tuple(sorted((triangle[1],triangle[2]))),
                       tuple(sorted((triangle[2],triangle[0])))]
        for arrete in arretes :
            arretes_tot[arrete] = arretes_tot.get(arrete,0)+1
    
    frontiere = [elem for elem, num in arretes_tot.items() if num == 1]
    return np.array(frontiere)

def PlotMesh_func(coord, connect, f, titre='z=f(x,y)') :
    x = coord[:,0]
    y = coord[:,1]
    im = plt.tripcolor(x,y,connect,f, shading = 'gouraud', cmap = 'jet')
    plt.colorbar(im)
    plt.triplot(x,y,connect,color ='black', lw = 0.8)
    plt.title(titre)

def RefineMesh(vtx, elt):
    new_vtx = list(vtx)
    new_elt = []
    # Dictionnaire pour stocker les milieux : {(idx1, idx2): nouvel_indice_vtx}
    midpoints = {}

    def get_midpoint_idx(i, j):
        # On trie les indices pour que l'arête (i,j) et (j,i) soit la même clé
        edge = tuple(sorted((i, j)))
        if edge not in midpoints:
            # Calcul du point milieu
            m = (vtx[i] + vtx[j]) / 2.0
            # Ajout au tableau des sommets
            midpoints[edge] = len(new_vtx)
            new_vtx.append(m)
        return midpoints[edge]

    for tri in elt:
        # Indices des sommets originaux
        p0, p1, p2 = tri
        
        # Création (ou récupération) des indices des milieux des 3 arêtes
        m01 = get_midpoint_idx(int(p0), int(p1))
        m12 = get_midpoint_idx(int(p1), int(p2))
        m20 = get_midpoint_idx(int(p2), int(p0))
        
        # Division du triangle original en 4 nouveaux triangles
        new_elt.append([p0, m01, m20])  # Triangle coin p0
        new_elt.append([p1, m12, m01])  # Triangle coin p1
        new_elt.append([p2, m20, m12])  # Triangle coin p2
        new_elt.append([m01, m12, m20]) # Triangle central

    return np.array(new_vtx), np.array(new_elt)


# TP 3

def ElementaryMassMatrix(vertex) :
    #vertex est la matrice des coordonnées du triangle
    x, y = vertex[:,0], vertex[:,1]
    Aire_T = abs((x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))/2
    return (Aire_T/12)*np.array([[2,1,1],
                                 [1,2,1],
                                 [1,1,2]])

def MassMatrix(vtx,elt) :
    n_vtx = len(vtx)
    n_elt = len(elt)
    total = 9*n_elt
    data, rows, colls = np.zeros(total), np.zeros(total, dtype = int), np.zeros(total, dtype = int)

    for k in range(0,n_elt) :
        tri = elt[k]
        temp_mat = ElementaryMassMatrix(vtx[tri])
        debut = k*9
        fin = (k+1)*9

        rows[debut:fin] = np.repeat(tri,3)
        colls[debut:fin] = np.tile(tri,3)
        data[debut:fin] = temp_mat.flatten()
    return sparse.csr_matrix((data,(rows,colls)), shape=(n_vtx,n_vtx))

def ElementaryStiffnessMatrix(vtx) :
    x, y = vtx[:,0], vtx[:,1]
    aire = abs((x[1]-x[0])*(y[2]-y[0]) - (y[1]-y[0])*(x[2]-x[0]))/2
    A = np.ones((3,3))
    A[:,1:] = vtx 

    try :
        grad = np.linalg.inv(A) [1:,:]
    except np.linalg.LinAlgError :
        return np.zeros((3,3))
    
    return aire*(grad.T @ grad)

def StiffnessMatrix(vtx, elt) :
    n_vtx = len(vtx)
    n_elt = len(elt)
    total = 9*n_elt

    rows, colls, data = np.zeros(total, dtype=int), np.zeros(total, dtype=int), np.zeros(total)
    for i in range(0,n_elt) :
        tri = elt[i]
        temp_K = ElementaryStiffnessMatrix(vtx[tri])

        debut = i*9
        fin = (i+1)*9
        rows[debut:fin] = np.repeat(tri,3)
        colls[debut:fin] = np.tile(tri,3)

        data[debut:fin] = temp_K.flatten()
    
    return sparse.csr_matrix((data,(rows,colls)), shape=(n_vtx,n_vtx))



##################################################### PARTIE 1 ######################################################################



# Question (a)


# Sur V := H^1_{0}(Ω), on a la formulation variationnelle suivante :
# Trouver u dans V tel que pour tout v dans V on ait :
#              a(u,v) = l(v) , où :
# a(u,v) = ∫(du.Tdv + (b.Tdu)v + cuv)dx
# l(v) = ∫(fv)dx
# La matrice du problème sera A = K + C(b) + c*M , et le membre de droite :
# F = M @ f , où :
# K = (∫(dƒ_{i}.Tdƒ_{j})dx)_(i,j) où les ƒ_{i} forment la base de l'espace P_{1}(Ω) est la matrice de rigidité
# C(b) = (∫((b.Tdƒ_{j})ƒ_{i})dx)_(i,j) dépend de b est la matrice de convection
# M = (∫(ƒ_{i}ƒ_{j})dx)_(i,j) est la matrice de masse
# f est une interpolation dans P_{1}(Ω) de la fonction dans le membre de droite de l'équation.



# Question (b)


def ElementaryConvectionMatrix(vtx,b) :
    # Matrice Jacobienne du changement de variable
    A = np.array([vtx[1]-vtx[0],vtx[2]-vtx[0]]).T
    aire = (0.5)*np.abs(np.linalg.det(A))
    invA = np.linalg.inv(A) 

    # Calcul des gradients
    gradsref = np.array([[-1,-1],[1,0],[0,1]])
    grads = gradsref @ invA  #grads[i] est le gradient de ph_i

    # Calcul de la matrice élémentaire
    bgrads = grads @ b
    result = (aire/3.)*np.ones((3,3))

    return result * bgrads 



# Question (c)


def ConvectionMatrix(vtx, elt, b) : #b est un vecteur de R^2
    nvtx = len(vtx)
    N = len(elt)
    total = 9*N
    data, rows, colls = np.zeros(total), np.zeros(total, dtype= int), np.zeros(total, dtype= int)

    for k in range(0,N) :
        tri = elt[k]
        debut, fin = 9*k, 9*(k+1)
        rows[debut:fin] = np.repeat(tri,3)
        colls[debut:fin] = np.tile(tri,3)
        data[debut:fin] = ElementaryConvectionMatrix(vtx[tri],b).flatten()

    return sparse.csr_matrix((data,(rows,colls)), shape=(nvtx,nvtx))



# Question(d) 


# Un calcul explicite des dérivées partielles de f nous donne que (on note (.,.)
# le produit scalaire et ||.|| la norme euclidienne de R^2):

# d^{2}f(x,y) = (df(x,y),b) - (||b||^2/4 + pi^2*(p^2 + q^2))*f(x,y)

# Et donc que :

# alpha = 1/(||b||^2/4 + pi^2*(p^2+q^2) + c)



# Question (e)


# Constitution générique de la matrice du problème, utile pour les 2 problèmes du projet
def Matrice_EDP(vertex, element, b=np.array([1.,1.]), c=1.) : # b est toujours dans R^2 et c > 0
    # Domaine de l'équation et sa frontière

    # Constitution des matrices du problème
    M = MassMatrix(vertex, element) #matrice de masse
    K = StiffnessMatrix(vertex, element) #matrice de rigidité
    C = ConvectionMatrix(vertex, element, b) # Ajout de la matrice de convection 

    A = K + C + c*M

    return A, M, K, C

# terme de droite
def S(x,y,p,q,b=np.array([1.,1.])) : # b est un vecteur de R^2
    return np.exp((b[0]*x + b[1]*y)/2)*np.sin(p*np.pi*x)*np.sin(q*np.pi*y)

def Sol_EDP1(p, q, Nx, Ny, b=np.array([1.,1.]), c=1.) : #b vecteur de R^2, c > 0 et (p,q) couple d'entiers de N^2, et
    # Nx resp. Ny sont les nombres de subdivisions du domaines sur l'axe (Ox) resp.(Oy) et on obtient 
    # le nombre de triangles dans le maillage : 2*Nx*Ny
    vertex, element = GenerateMeshRectangle(1,1,Nx,Ny)

    #Matrices du problème
    A, M, K, C = Matrice_EDP(vertex, element, b, c)

    # Géométrie du problème : la solution est nulle sur le bords
    boundary_edges = ExtractBoundary(element)
    indices_int = np.setdiff1d(np.arange(0,len(vertex)), boundary_edges) # vecteur des indices intérieurs au domaine
    Ar = A[indices_int,:][:,indices_int] # matrice réduite

    F = S(vertex[:,0], vertex[:,1], p, q, b) # interpolation de la fonction du membre de droite
    Fr = ( M @ F )[indices_int] # membre de droite réduit

    # Solution du problème
    res = sparse.linalg.spsolve(Ar,Fr)
    return res, vertex, element, boundary_edges, indices_int, M, K, C



# Question (f)


def Plot_approximation(p,q,Nx,Ny) :
    Us, vertex, element, _, id_int, _, _, _ = Sol_EDP1(p,q,Nx,Ny)
    Sol = np.zeros(len(vertex))
    Sol[id_int] = Us
    PlotMesh_func(vertex, element, Sol, f'Solution approchée du problème 1 pour (p,q) = {p,q} et '
                   f'{2*Nx*Ny} mailles dans le domaine')
    plt.show() 



# Question (g)


def erreurL2(Sol,vertex,id_free,Mass,p,q) : 
    #Solution exacte
    f = S(vertex[:,0], vertex[:,1], p, q)/(3/2 + (np.pi**2)*(p**2 + q**2))
    U = np.zeros(len(vertex))
    U[id_free] = Sol

    error = U - f
    return np.sqrt(error @ Mass @ error)/np.sqrt(f @ Mass @ f)

def erreurH1(Sol,vertex,id_free,Mass,Stiffness,p,q) :
    #Solution exacte
    f = S(vertex[:,0], vertex[:,1], p, q)/(3/2 + np.pi**2*(p**2 + q**2))
    U = np.zeros(len(vertex))
    U[id_free] = Sol

    error = U - f
    val_L2 , div_L2 = error @ Mass @ error , f @ Mass @ f
    val_grad , div_grad = error @ Stiffness @ error , f @ Stiffness @ f
    return np.sqrt(val_L2 + val_grad)/np.sqrt(div_L2 + div_grad)


def trace_erreurs(p,q) :
    A = [10,50,100,150]

    Z = []
    G = []
    for N in A :
        u, vtx, _, _, indices_libres, M, K, _ = Sol_EDP1(p,q,N,N)
        Z.append(erreurL2(u, vtx, indices_libres, M, p, q)) # Tracé de l'erreur L^2
        G.append(erreurH1(u, vtx, indices_libres, M, K, p, q)) # Tracé de l'erreur H^1


    fig, axes = plt.subplots(2, 1, figsize=(8,10), constrained_layout=True)

    axes[0].loglog(A,Z, label='erreur relative L^2', color='blue')
    axes[0].set_title("Évolution de l'erreur dans $L^2$")
    axes[0].set_xlabel('N (Nombre de subdivisions)')
    axes[0].set_ylabel(' Erreur ')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].loglog(A,G, label='erreur relative H^1', color='blue')
    axes[1].set_title("Évolution de l'erreur dans $H^1$")
    axes[1].set_xlabel('N (Nombre de subdivisions)')
    axes[1].set_ylabel(' Erreur ')
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle( "Analyse de la convergence relative avec l'interpolant de Lagrange,\n pour N = 10, 50, 100 et 150 "
                 f"et (p,q) = {p,q} , en échelle logarithmique" )

    plt.show()



##################################################### PARTIE 2 ######################################################################
 
# La formulation variationnelle change mais de peux : le second membre est nul, l'espace considéré pour la discrétisation
# ne change pas (c'est H^{1}_{0}(Ω)), mais on applique la formulation variationnelle à u' + v où u' appartient à H^{1}_{0}(Ω) 
# et v est un relèvement de la fonction aux bords du domaine. La solution est alors dans l'espace affine g + H^{1}_{0}(Ω).


# Question (a)


def ConnectedComponents(bnd):
    # 1. Initialisation : chaque sommet est son propre parent
    n_vtx = np.max(bnd) + 1
    parent = np.arange(n_vtx)

    # Fonction auxiliaire 
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i


    for k in range(len(bnd)):
        u, v = bnd[k ,:]
        parent_u, parent_v = find(u), find(v)
        if parent_u != parent_v:
            parent[parent_u] = parent_v # On fusionne les deux groupes (même composante connexe)


    vfind = np.vectorize(find)
    edge_roots = vfind(bnd[:, 0])
    
    _, labels = np.unique(edge_roots, return_inverse=True)
    
    return labels



# Question (b)


def PlotMeshComponents(vtx, elt, bnd, labels): # Représente le maillage et colorie chaque composante connexe du bord

    # Affichage du maillage global 
    plt.triplot(vtx[:, 0], vtx[:, 1], elt, lw=0.5)
    
    # Identification des labels uniques (combien de trous + bord extérieur)
    unique_labels = np.unique(labels)
    
    cmap = plt.get_cmap('tab10') 
    
    for i, label in enumerate(unique_labels):
        # On extrait les indices des arêtes appartenant à la composante 'label'
        idx = np.where(labels == label)[0]
        
        # Pour chaque segment de cette composante, on trace la ligne
        for k in idx:
            node1, node2 = bnd[k, :]
            plt.plot([vtx[node1, 0], vtx[node2, 0]], 
                     [vtx[node1, 1], vtx[node2, 1]], 
                     color=cmap(i % 10), lw=2, 
                     label=f"$\Gamma_{{{label}}}$" if k == idx[0] else "")

    plt.legend(bbox_to_anchor=(1.01,1),loc='upper left')
    plt.gca().set_aspect('equal')
    plt.title(f"Maillage et composantes connexes du bord pour {len(elt)} mailles dans le domaine")
    plt.show()



# Question (c)


def Sol_EDP2(vertex, element, b=np.array([1.,1.]), c=1.) : # b est un vecteur de R^2 et c un réel > 0
    # Assemblage de la matrice globale et domaine
    A, M, K, C = Matrice_EDP(vertex, element, b, c)
    n_vtx = len(vertex)
    
    # Topologie et identification des bords 
    bnd = ExtractBoundary(element)
    labels = ConnectedComponents(bnd) 
    
    # Préparation des conditions de Dirichlet 
    u_frontiere = np.zeros(n_vtx)
    u_frontiere[bnd[:, 0]] = labels  # Assigne le label j aux sommets de l'arête
    u_frontiere[bnd[:, 1]] = labels  # Idem pour l'autre sommet
    
    # Séparation noeuds Dirichlet et intérieurs
    ext_nodes = np.unique(bnd)
    nodes = np.unique(element) # on regarde les noeuds utilisés dans le maillage (et on ignore donc ceux dans les trous)
                               # si on utilise la même méthode que pour la 1ère EDP i.e. en définissant int_nodes avec
                               # setdiff1d avec np.arange(O,n_vtx) on aura les coeff/lignes/colonnes correpondant aux
                               # trou nulles
    int_nodes = np.setdiff1d(nodes, ext_nodes)
    
    # Application des conditions de Dirichlet et Résolution 
    # On résout le système réduit : Ar * u_i = f_i - A_id * u_d
    # Ici f = 0, donc on gère juste le relevé de Dirichlet
    
    Ar = A[int_nodes,:][:,int_nodes]
    A_id = A[int_nodes,:][:,ext_nodes]
    
    # Calcul du second membre 
    rhs = - A_id @ u_frontiere[ext_nodes]
    
    # Résolution et reconstruction du vecteur solution
    u_h = np.zeros(n_vtx)
    u_h[ext_nodes] = u_frontiere[ext_nodes]
    u_h[int_nodes] = sparse.linalg.spsolve(Ar, rhs)

    return u_h, bnd, labels



##################################################### Tests #########################################################################

# Vous pouvez dé-commenter les test des questions au fur et à mesure ou tous directement
# Les tests sont réalisés pour (p,q) dans {(2,3),(1,6)} et Nx = Ny dans {10,50,100,150}
# On pose donc :
E = [(2,3),(1,6)] # et :
F = [10,50,100,150]



# PARTIE 1


# Question (e) - Vecteur solution du problème

#for (a,b) in E :
    #for N in F :
        #u_h,_,_,_,_,_,_,_ = Sol_EDP1(a,b,N,N)
        #print("Approximation P_1 de la solution à l'EDP de la partie 1 pour "
        #f"{2*N**2} triangles dans le maillage et pour (p,q) = {a,b} : ",u_h)



# Question (f) - Tracé de l'approximation numérique u_h

#for (a,b) in E :
    #for N in F :
        #Plot_approximation(a,b,N,N)



# Question (g) - Tracé de la convergence des erreurs L^{2}(Ω) et H^{1}(Ω)

for (a,b) in E :
    trace_erreurs(a,b)




# PARTIE 2 


# Tests effectués avec barwith4holes.msh

# Questions (a) & (b)

#vertex2, element2 = LoadMesh('barwith4holes.msh')
#boundary = ExtractBoundary(element2)
#lab = ConnectedComponents(boundary)
#PlotMeshComponents(vertex2, element2, boundary, lab)



# Question (c)


B = [np.array([1.,1.]),np.array([0.,5.]),np.array([50.,0.]),np.array([70.,70.])] # différentes valeurs pour le vecteur b
#for b in B :
    #vertex3, element3 = LoadMesh('barwith4holes.msh')
    #v_h, bound2, _ = Sol_EDP2(vertex3, element3, b)
    #PlotMesh_func(vertex3, element3, v_h, f'Solution approchée du problème pour ||b|| = {np.linalg.norm(b):.2f} et '
                  #f'pour {len(element3)} mailles dans le domaine')
    #plt.show()
    
# On remarque que la solution a tendance à "suivre" en un sens la direction du vecteur b, et que plus b est grand, plus la solution est instables
# pour de grandes valeurs de b au sens où elle s'applatit très vite (vers 0) lorsqu'elle s'éloigne des trous du domaine
