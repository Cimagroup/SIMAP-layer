import tensorflow as tf
from main_SMNN import *
import matplotlib.pyplot as plt

(train_images, y_train), (test_images, y_test) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


num_classes = 10
input_shape = (28, 28, 1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(4, (5, 5), activation='relu'))
model.add(tf.keras.layers.Flatten(name="flatten"))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, y_train, epochs=1,
                    validation_data=(test_images, y_test))


from keras.models import Model

layer_name = "flatten"
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(train_images)

X=intermediate_output
dim = np.shape(X[0])[0]
print(dim)

X_centered = X-X.mean()
X_scaled = (dim/2)*(X_centered-X_centered.min())/(X_centered.max()-X_centered.min())

X_train=X_scaled
data = X_train
bar_iterations=2

d=itek_barycentrics(data,bar_iterations)


bis = [bis_cons(d,ite,dim) for ite in range(bar_iterations+1)]
dic_sups = dic_supports(supports(bar_iterations,dim))
sups = supports(bar_iterations,dim)
n_samples = len(X_train)

n_classes = len(set(y_train))
y_hot=tf.one_hot(y_train,depth=n_classes)
y_hot=np.array(y_hot)


verbose = True
epochs = 10
model0,history0=SMNN(bis[0],y_train,epochs,verbose =verbose)
print(model0.evaluate(bis[0],y_hot))
vsi = [bis_cons(itek_barycentrics(sups[i],i),i,dim) for i in range(bar_iterations)]
vs0 = np.matmul(vsi[0],model0.get_weights())
weights0=vs0 

model1,history1=SMNN(bis[1],y_train,epochs,weights0,verbose =verbose)
print(model1.evaluate(bis[1],y_hot))
vs1=np.matmul(vsi[1],model1.get_weights())
weights1=vs1

model2,history2=SMNN(bis[2],y_train,epochs,weights1,verbose =verbose)
print(model2.evaluate(bis[2],y_hot))

#
plt.plot(history0.history['loss'])
plt.plot(range(epochs,2*epochs),history1.history['loss'])
plt.plot(range(2*epochs,3*epochs),history2.history['loss'])
#plt.plot(range(3*epochs,4*epochs),history3.history['loss'])
plt.legend(["model0","model1","model2","model3"])
plt.title("Loss")
plt.show()


plt.plot(history0.history['accuracy'])
plt.plot(range(epochs,2*epochs),history1.history['accuracy'])
plt.plot(range(2*epochs,3*epochs),history2.history['accuracy'])
#plt.plot(range(3*epochs,4*epochs),history3.history['accuracy'])
plt.legend(["model0","model1","model2","model3"])
plt.title("Accuracy")
plt.show()
#

#%% Test evaluation

intermediate_output_test = intermediate_layer_model.predict(test_images)

X_test=intermediate_output_test
X_centered_test = X_test-X_test.mean()
X_scaled_test = (dim/2)*(X_centered_test-X_centered.min())/(X_centered_test.max()-X_centered_test.min())
X_test=X_scaled_test

data_test = X_test

d_test=itek_barycentrics(data_test,bar_iterations)

bis_test = [bis_cons(d_test,ite,dim) for ite in range(bar_iterations+1)]



yt_hot=tf.one_hot(y_test,depth=n_classes)
yt_hot=np.array(yt_hot)
print("Evaluation on test")
print("Model 0")
model0.evaluate(bis_test[0],yt_hot)
print("Model 1")
model1.evaluate(bis_test[1],yt_hot)
print("Model 2")
model2.evaluate(bis_test[2],yt_hot)


