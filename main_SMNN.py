import numpy as np
import itertools


def V(n):
    M= np.vstack((np.zeros(n),n*np.diag(np.ones(n))))
    return M

def M(n):
    M= np.hstack(((-1/n)*np.ones((n+1,1)),np.vstack((np.zeros(n),np.diag(np.ones(n))/n))))
    M[0,0]=1
    return M
    

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
  init_vs = list(range(dim+1))
  vertices=[]
  if n_iter == 0:
    vertices = list(range(dim+1)) 
  else:
    ps = [[".".join([str(y) for y in x]) for x in permutations(init_vs)]]*n_iter
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



import tensorflow as tf

def SMNN(bis,y,epochs,init_weights=[],verbose=False):

    #%% 
    n_classes = len(set(y))
    y_hot=tf.one_hot(y,depth=n_classes)
    y_hot=np.array(y_hot)

    input_dim = np.shape(bis)[1]

    #%%
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax",use_bias=False, input_shape=(input_dim,)))

    model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
    if init_weights!=[]:
      model.set_weights(np.reshape(init_weights,(1,input_dim,n_classes)))
    print("Training neural network...")
    history = model.fit(bis,y_hot,epochs=epochs, verbose = verbose,batch_size=15) 

    return (model, history)
