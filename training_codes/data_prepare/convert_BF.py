import numpy as np
type_="val"
bf_data=np.load("../data_trainlist/2Dlist_"+type_+"_HSCMuSC.npy",allow_pickle=True)

data_allnew=[]
for aa in bf_data:
    print (aa)

    iti=[]
    for i in range(len(aa)):
        itd = aa[i]

        if i<(len(aa)-1):
            #print (itd)
            itnew=itd.split('train')[1]
        else:
            itnew=itd
        #print (itnew)
        iti.append(itnew)

    data_allnew.append(iti)
print (len(data_allnew))
np.save("2Dlist_"+type_+"_HSCMuSC.npy",data_allnew)


