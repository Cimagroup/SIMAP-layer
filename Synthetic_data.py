from main_SMNN import *
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X,y = datasets.make_classification(n_samples=100,
                                   n_features=2,
                                   n_informative=1,
                                   n_classes = 2
                                   ,n_redundant =0,
                                   n_clusters_per_class=1,
                                   class_sep=0.4)

# X_centered = X-X.mean()
# X_scaled = X_centered/(np.abs(X_centered).max())/2+1/2

X_centered = X-X.mean()
X_scaled = (X_centered-X_centered.min())/(X_centered.max()-X_centered.min())

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2)



plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

plt.plot(V(2)[:,0],V(2)[:,1],'.')
t1 = plt.Polygon(V(2),edgecolor="blue",facecolor="none")
plt.gca().add_patch(t1)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.show()

plt.plot(V(2)[:,0],V(2)[:,1],'.')
t1 = plt.Polygon(V(2),edgecolor="blue",facecolor="none")
plt.gca().add_patch(t1)
plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
plt.show()
bar_iterations=3
dim=2

dic_sups = dic_supports(supports(bar_iterations,dim))

sups = supports(bar_iterations,dim)

data = X_train

d=itek_barycentrics(data,bar_iterations)

bis = [bis_cons(d,ite,dim) for ite in range(bar_iterations+1)]

n_samples = len(X_train)

n_classes = len(set(y_train))
y_hot=tf.one_hot(y_train,depth=n_classes)
y_hot=np.array(y_hot)


verbose = False
epochs = 1000
model0,history0=SMNN(bis[0],y_train,epochs,verbose =verbose)
print(model0.evaluate(bis[0],y_hot))
plt.plot(history0.history['loss'])
#plt.show()
vsi = [bis_cons(itek_barycentrics(sups[i],i),i,dim) for i in range(bar_iterations)]
vs0 = np.matmul(vsi[0],model0.get_weights())
weights0=vs0 #np.matmul(P(2),model0.get_weights())#vs0


model1,history1=SMNN(bis[1],y_train,epochs,weights0,verbose =verbose)
print(model1.evaluate(bis[1],y_hot))

#plt.show()
vs1=np.matmul(vsi[1],model1.get_weights())
weights1=vs1

model2,history2=SMNN(bis[2],y_train,epochs,weights1,verbose =verbose)
print(model2.evaluate(bis[2],y_hot))

vs2=np.matmul(vsi[2],model2.get_weights())
weights2=vs2

model3,history3=SMNN(bis[3],y_train,epochs,weights2,verbose =verbose)
print(model3.evaluate(bis[3],y_hot))

plt.plot(history0.history['loss'])
plt.plot(range(epochs,2*epochs),history1.history['loss'])
plt.plot(range(2*epochs,3*epochs),history2.history['loss'])
plt.plot(range(3*epochs,4*epochs),history3.history['loss'])
plt.legend(["model0","model1","model2","model3"])
plt.title("Loss")
plt.show()


plt.plot(history0.history['accuracy'])
plt.plot(range(epochs,2*epochs),history1.history['accuracy'])
plt.plot(range(2*epochs,3*epochs),history2.history['accuracy'])
plt.plot(range(3*epochs,4*epochs),history3.history['accuracy'])
plt.legend(["model0","model1","model2","model3"])
plt.title("Accuracy")
plt.show()
n_classes = len(set(y_train))
y_hot=tf.one_hot(y_train,depth=n_classes)
y_hot=np.array(y_hot)


# def plot_model_out(model):
#   """
#   x,y: 2D MeshGrid input
#   model: Keras Model API Object
#   """
#   a = np.linspace(-1, 1, 100)
#   xx, yy = np.meshgrid(a,a)
#   z = 
#   plt.contourf(xx, yy, z,)
#   plt.show()
  
#   plt.show()

data = X_test

d_test=itek_barycentrics(data,bar_iterations)

bis_test = [bis_cons(d_test,ite,dim) for ite in range(bar_iterations+1)]



# data_test = X_test
# bar_iterations=2
# dim=2
# d_test=itek_barycentrics(data,bar_iterations)
# bis_test, v_ords_test, matchings_test = general_matching(d_test,bar_iterations)
# n_samples_test = len(X_test)
# bis_ordered_test = reorder_matchings(n_samples_test, matchings_test, bis_test, bar_iterations)
# n_classes = len(set(y_test))
yt_hot=tf.one_hot(y_test,depth=n_classes)
yt_hot=np.array(yt_hot)
print("Evaluation on test")
print("Model 0")
model0.evaluate(bis_test[0],yt_hot)
print("Model 1")
model1.evaluate(bis_test[1],yt_hot)
print("Model 2")
model2.evaluate(bis_test[2],yt_hot)

