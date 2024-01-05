import numpy as np
import itertools

# #%% Matrices
# def V(n):
#   """
#   Input: dimension of the n-simplex
#   Output: matrix whose rows are the coordinates of a n-simplex centered
#   at the origin in the unit n-sphere.
#   """
#   M= np.zeros((n+1,n))
#   for i in range(n+1):
#     for j in range(n):
#       M[i,j]=entry_matrix(i,j,n)
#   return M# n*M

# def A(j,n):
#   A=1
#   while j<=n-1:
#     A=(np.sqrt((j+1)^2-1)/(j+1))*A
#     j+=1
#   return A

# def entry_matrix(i,j,n):
#   val=A(j+1,n)
#   if j>=i:
#     entry=-val/(j+1)
#   if j==i-1:
#     entry = val
#   if j<i-1:
#     entry = 0
#   if j==n-1 & i<n:
#     val=-1/n
#   if j==n-1 & i==n:
#     val = 1
#   return entry
# #############################################################################


# # Matriz para obtener las coordenadas baricéntricas
# def M(n):#,Vm):
#   """
#   Input:
#     dimension n
#     Matrix of the maximal simplex.
#   Output:
#     Matrix to compute the barycentric coordinates.
#   """
#   m=np.zeros((n+1,n+1))
#   for j in range(n+1):
#     m[0,j]=1/(n+1)
#   for i in range(1,n):
#     m[i,i]=i/((i+1)*A(i,n))
#     for j in range(i):
#       m[i,j]=-1/((i+1)*A(i,n))
#   for j in range(n):
#     m[n,j]= -1/(n+1)
#   m[n,n]=n/(n+1)
#   return m
#   #return np.linalg.inv(np.c_[np.ones(n+1), Vm] )

def V(n):
    M= np.vstack((np.zeros(n),n*np.diag(np.ones(n))))
    return M

def M(n):
    M= np.hstack(((-1/n)*np.ones((n+1,1)),np.vstack((np.zeros(n),np.diag(np.ones(n))/n))))
    M[0,0]=1
    return M
    

# Matriz para reordenar

def Q(n):
  q=np.zeros((n+1,n+1))
  l = [1/j for j in range(1,n+2)]
  for j in range(n+1):
    for i in range(j,n+1):
      q[i,j]=l[i]
  return q


def P(n):
  p = np.zeros((n+1,n+1))
  p[0,0]=1
  for i in range(1,n+1):
    p[i,i]=i+1
    p[i,i-1]=-i
  return p
  #return np.linalg.inv(Q(n))

#%% Barycentrics and indices managing

def sorting_with_indices(l):
  n = len(l)
  ordered_indices =[y for (x,y) in sorted(zip(l,range(n)),key=lambda x: x[0],reverse=True)]
  return ordered_indices

def abstract_bar_simplex_indices(indices):
  simplices = [indices[0:i] for i in range(1,len(indices)+1)]
  return simplices

def list2string(ls):
  return '.'.join([str(x) for x in ls])
def list2string_2(ls):
  return '|'.join([str(x) for x in ls])

def bar(x,bar_ite, Vm, Pm, Mm):
  d = dict()
  d2=dict()
  n=len(x)
  dim = n+1
  x=np.concatenate(([1],x))
  barycentric_cords = np.matmul(x,Mm)
  indices = list(range(0,dim))
  indices_str = list2string(indices)
  d[indices_str]=barycentric_cords
  d2[indices_str]=indices
  list_indices = indices
  list_indices_str=indices_str
  for j in range(1,bar_ite+1):
    indices = sorting_with_indices(barycentric_cords) 
    # sort the barycentric coordinates providing the indices for the simplex where
    # data belongs
    list_indices = list_indices+indices
    list_indices_str = list_indices_str+"|"+list2string(indices)
    barycentric_cords = barycentric_cords[indices]
    barycentric_cords = np.matmul(barycentric_cords,Pm)
    d[list_indices_str]=barycentric_cords
    d2[list_indices_str]=composition(list_indices_str[len(indices_str+"|")::],dim-1)
  return d, d2

def itek_barycentrics(data,k):
  l = len(data)
  n = len(data[0])
  Vm=V(n)
  Pm=P(n)
  Mm=M(n)
  return {i: bar(data[i],k,Vm, Pm, Mm) for i in range(l)}

def vertices(st):
  l = len(st)+1
  return [sorted(st[:n]) for n in range(1,l)]

def undo_permutation(subd_simp,vertices):
  subd_simp_list = list(subd_simp)
  out = [vertices[int(i)] for i in subd_simp_list]
  return out

def composition(perm,dim):
  seq = perm.split("|")
  vertices_seq = list(range(0,dim+1))
  for s in seq:
      vertices_seq = vertices(undo_permutation(s.split("."),vertices_seq))
  return vertices_seq

#Example:
#composition("0.1.2|0.2.1.|1.2.0",2)
#[[[0, 2]], [[0, 2], [0, 2, 1]], [[0, 2], [0, 2, 1], [0]]]

# pairs barycentrics and vertices
# def auxiliary(d1,d2,dim,iter):
#   # iter: number of times barycentric applied
#   # dim: dimension
#   dim+=1
#   iter+=1 # the reason is that we start with 0.1.2 for 0 barycentric subdivisions
#   d1keys=list(d1.keys())
#   d1_key = [v for v in d1keys if len(v.split("|"))==iter][0]
#   list_bars = d1[d1_key]
#   list_vertices = list(d2[d1_key])
#   z=list(zip(list_vertices,list_bars))
  # return z

# Example:
#auxiliary(d1,d2,2,0)
#[(0, 0.10239322565748302), (1, 0.5642734410091838), (2, 0.3333333333333333)]
#auxiliary(d1,d2,2,1)
#[([0], 0.23094010767585044),
# ([0, 1], 0.4618802153517006),
# ([0, 1, 2], 0.3071796769724491)]




from itertools import combinations, permutations
def generate_vertices_level_0(dim):
  indices = range(0,dim+1)
  simplices = permutations(indices)
  vertices = []
  for s in simplices:
    for i in range(1,dim+2):
      cs = [sorted(list(c)) for c in combinations(s,i)]
      for c in cs:
        if c not in vertices:
          vertices.append(c)
  return vertices






def cartesian_coordinates(vertex,dim,Vm):
  vs = []
  if type(vertex)!=list:
      vs.append(Vm[vertex,:])
  elif any(isinstance(i, list) for i in vertex):
    for v in vertex:
      if any(isinstance(i, list) for i in v):
        vs.append(cartesian_coordinates(v,dim,Vm))
      else:
        ls = [Vm[vi,:] for vi in v]
        vs.append(np.sum(ls,axis=0)/len(ls))
  else:
      ls = [Vm[vi,:] for vi in vertex]
      vs.append(np.sum(ls,axis=0)/len(ls))
  return list(np.sum(vs,axis=0)/len(vs))




#%% 


def generate_vertices(n_iter,dim):
  #str0 = ''.join([str(i) for i in range(dim+1)])#"012"
  init_vs = list(range(dim+1))
  vertices=[]
  if n_iter == 0:
    vertices = list(range(dim+1)) 
  else:
    ps = [[".".join([str(y) for y in x]) for x in permutations(init_vs)]]*n_iter#[['.'.join(p) for p in permutations(str0)] for i in range(1,n_iter+1)]
    # Tener cuidado con la permutación de str0
    possible_perms=['|'.join(p) for p in itertools.product(*ps)]
    for p in possible_perms:
      comp=p
      c=composition(comp,dim)
      for vertex in c:
        if vertex not in vertices:
          vertices.append(vertex)
  return vertices

def supports(ite,dim):
  sups = []
  Vm=V(dim)
  for i in range(1,ite+1):
    supporti = [cartesian_coordinates(v,dim,Vm) for v in generate_vertices(i,dim)]
    sups.append(supporti)
  return sups

def dic_supports(sups):
  ds = []
  for (i,sup) in enumerate(sups):
    ds.append(itek_barycentrics(sup,i))
  return ds

# def vert_ordering(d,dim,i):
#   vertices_ordering=[]
#   for j in d.keys():
#     xy=auxiliary(d[j][0],d[j][1],dim,i)
#     for x,y in xy:
#       if x not in vertices_ordering:
#         vertices_ordering.append(x)
#   return vertices_ordering

    
def bis_cons(d,bar_iterations,dim):
  support_vertices = generate_vertices(bar_iterations,dim)
  l = len(d.keys())
  m = len(support_vertices)
  bis = np.zeros((l,m))
  for i in range(l):
      vs_bs = list(zip(d[i][1].values(),d[i][0].values()))[bar_iterations]
      vs = vs_bs[0]
      bs = vs_bs[1]
      if bar_iterations==0:
          bis[i]=bs
      else:
          for k in range(len(vs)):
              j=support_vertices.index(vs[k])
              bis[i,j]=bs[k]
  return bis

        
def matching_vertices(ite,dim):
    v_ord_ite = generate_vertices(ite-1,dim)
    v_T_ite = generate_vertices(ite,dim)
    indices = []
    for j in range(len(v_ord_ite)):
      v=v_ord_ite[j]
      for i in range(len(v_T_ite)):
        if v_T_ite[i]==[v]:
          indices.append((j,i))
    return indices


# def general_matching(d,dim,ite):
#   v_ords = []
#   bis = []
#   matchings = []
#   for i in range(ite+1):
#     bisi, v_ordi = bis_cons(d,dim,i)
#     matching_i = matching_vertices(v_ordi,generate_vertices(i,dim))
#     bis.append(bisi)
#     v_ords.append(v_ordi)
#     matchings.append(matching_i)
#   return bis, v_ords, matchings

# def reorder_matchings(matchings, bis,ite,dim):
#   bis_ordered = []
#   n_samples = len(bis[0])
#   for index in range(ite+1):
#     bs = bis[index]
#     matching = matchings[index]
#     bs_ordered=np.zeros((n_samples,len(generate_vertices(index,dim))))
#     for (i,j) in matching:
#       for k in range(n_samples):
#         bs_ordered[k,j] = bs[k,i]
#     bis_ordered.append(bs_ordered)
#   return bis_ordered



#######################


import tensorflow as tf

def SMNN(bis,y,epochs,init_weights=[],verbose=False):
    #n_features = np.shape(bis)[1]

    #%% labels update
    n_classes = len(set(y))
    y_hot=tf.one_hot(y,depth=n_classes)
    y_hot=np.array(y_hot)

    input_dim = np.shape(bis)[1]

    #%%
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax",use_bias=False, input_shape=(input_dim,)))
    #loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
    if init_weights!=[]:
      model.set_weights(np.reshape(init_weights,(1,input_dim,n_classes)))
    print("Training neural network...")
    history = model.fit(bis,y_hot,epochs=epochs, verbose = verbose,batch_size=15) 

    return (model, history)
