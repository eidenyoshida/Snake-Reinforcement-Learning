#%%
hasSomething = set()
maxVals = []
hasSomethingLen = len(hasSomething)
hasSomethingList = []
            
for i in range(256):
    for j in range(4):
        for k in range(len(Qs)):
            if Qs[k][i,j] != 0:
                hasSomething.add((i,j))
                if len(hasSomething) > hasSomethingLen:
                    hasSomethingLen += 1
                    hasSomethingList.append((i,j))
                #if abs(Qs[k][i,j]) > maxVal:
                 #   maxVal = abs(Qs[k][i,j])

#%%
for i in range(len(Qs)):
    Qs[i] = np.abs(Qs[i])
    
              #%%   
for i in range(len(hasSomethingList)): # for every i,j that has data
    maxVals.append(0)
    for j in range(len(Qs)): # for every epoch
        if abs(Qs[j][hasSomethingList[i][0],hasSomethingList[i][1]]) > abs(maxVals[i]):
            maxVals[i] = Qs[j][hasSomethingList[i][0],hasSomethingList[i][1]]
                
#hasSomething = list(hasSomething)

#%%
import matplotlib.pyplot as plt
plt.close("all")

x = [x for x in range(5000)]
for i in range(len(hasSomethingList)):
    y = []
    for j in range(len(Qs)):
        #y.append((Qs[j][hasSomethingList[i][0],hasSomethingList[i][1]] - maxVals[i])/maxVals[i])
        y.append((Qs[j][hasSomethingList[i][0],hasSomethingList[i][1]])/maxVals[i])
    plt.plot(y,color="black",alpha=0.01)
"""x = [1,2,3,4]
y = [5,6,7,8]"""
plt.title("Convergence of Q Table")
plt.ylabel("Normalized Q Value")
plt.xlabel("Epoch Number")
plt.show()