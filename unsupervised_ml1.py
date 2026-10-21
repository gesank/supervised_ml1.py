from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# x=spending,shopping frequency
X=np.array([
    [70,5],
    [80,7],
    [20,2],
    [40,5],
    [16,2],
    [60,6],
    [10,1]    
])
print(X.shape) #must be (7,2) - not(1,7,2)
#scale feature
scaler=StandardScaler()
X_scaler=scaler.fit_transform(X)

#training model
model=KMeans(n_clusters=2,random_state=42)
labels=model.fit_predict(X_scaler)
print(labels)

inertia=[]
k_range= range(1,6)
for k in k_range:
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X_scaler)
    inertia.append(model.inertia_)  #<--KMeans dot inertia
    
plt.figure()
plt.plot(k_range,inertia,marker='o')
plt.title('estalish best k')
plt.xlabel('k= number of cluster')
plt.ylabel('inertia')
plt.savefig("elbow.png")
plt.show()


final_model=KMeans(n_clusters=2,random_state=42,n_init=10)
final_labels=final_model.fit_predict(X_scaler)

real_center=scaler.inverse_transform(final_model.cluster_centers_)

plt.figure(figsize=(7,5))
plt.scatter(X[:,0],X[:,1],c=final_labels,cmap="viridis",label="customers")
plt.scatter(real_center[:,0],real_center[:,1],color="red",marker="X",s=200,label="centroids")
#label each point
for i ,(amount,freq) in enumerate(X):
    plt.text(amount+1,freq+0.1,f"c{final_labels[i]}")


plt.xlabel('spending')
plt.ylabel('shopping frequency')
plt.title('customer segmentation')
plt.legend()
plt.grid()
plt.savefig("cluster.png")
plt.show()



print(f"centroids 0:spends ${real_center[0,0]:.1f},frequency{real_center[0,1]:.1f}")

print(f"centroids 1: spends ${real_center[1,0]:.1f},frequency{real_center[1,1]}:1f" )