__authors__ = ['1491578','1496496','1495282']
__group__ = 'DJ.15'

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval,Plot3DCloud
import matplotlib.pyplot as plt
import cv2
import time


def retrieval_by_color(eti,answ):
    result=[]
    for c in answ:
        correct=np.array([len(np.argwhere(x==c)) for x in eti])
        result.append(np.argwhere(correct!=0))

    return result
def retrieval_by_shape(eti,answ):
    result=[]
    for c in answ:
         correct=np.array([len(np.argwhere(x==c)) for x in eti])
         result.append(np.argwhere(correct!=0))
    return result

def retrieval_combined(indexColor,indexShape):
    result=np.intersect1d(indexColor,indexShape)
    return result

def isCorrectAnsw(correct,answ):
    result=[]
    for col in answ:
        lngC=np.array([len(np.argwhere( np.isin(x,str(col)))) for x in correct])
        result.append(np.argwhere(lngC!=0))
    return result
def isCorrectAnswShape(correct,answ):
    result=[]
    for col in answ:        
        result.append(np.argwhere( np.isin(correct,str(col))))
    return result
def isCorrectAnswConbined(color,label):
    result=np.intersect1d(color,label)
    return result

def get_shape_accuracy(knnShape,correctShape):    
    result=np.array([(np.asarray(knnShape)[i]==np.asarray(correctShape)[i]) for i in range(len(knnShape))])
    return result
def get_color_accuracy(KmeColor,CorrectColor):
    line=np.array([np.count_nonzero(np.in1d(np.asarray(KmeColor[i]),np.asarray(CorrectColor)[i]))/len(KmeColor[i]) for i in range(len(KmeColor))])
    result=sum(line)/len(line)
    return result*100
def kmean_statistics(Kmeans,Kmax):
    
    result=[]
    resultTimer=[]
    for i in Kmeans:
       wcd=[]
       timer=[]
       for x in range(2,Kmax):
           i.K=x
           start_time=time.time()
           i.fit()
           end_time=time.time()
           wcd.append(i.whitinClassDistance()) 
           timer.append(end_time-start_time)
       result.append(wcd)
       resultTimer.append(timer)
    return result,resultTimer

def featuresKnn(imgs):
    res= np.array([cv2.resize(i, dsize=(30, 40), interpolation=cv2.INTER_CUBIC) for i in imgs])
    return res
if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes   
    
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    
    """Test Parameters"""
    optimze=False #Optimize Knn
    nTestDo=2  #Number of test image   
    doWCDPlots=True #Kmeans Plots WCD 
    doTimePlots=True #Kmeans Plots Time 
    doColorPlot=True #Plot of centroids 
    doAccuaricyAnswColor=True #Pie Plot Accuaricy for answer
    doAccuaricyColorTotal=True #Pie Plot Total Accuaricy
    
    #Knn Pie Chart plot
    doAccuracyAnsw=True #Pie Plot Accuaricy for answer
    doAccuracyShapeTotal=True #Pie Plot Total Accuaricy
    #Combined Pie Char Plo
    doAccuracyCombinedAnswTotal=True
    
    """Answers Color and Shape"""
    
    answC=["Blue","Black","Grey"]   # Is necesary that Colors and shape have the same lenght
    answS=["Jeans","Dresses","Handbags"]
    
    """Parametros Knn"""
    
    startTrain=0  #Lenght of trian_data that use Knn
    endTrain=50
    nPuntosKnn=3
    
    """Parametros Kmeans"""
    
    k=5 #Numero de Clusters      
    options = {
            "km_init": "first" , #Como podemos inicializar Centorides first,random,custom
            "verbose": False ,
            "tolerance": 20,   #Tolerancia que podemos modificar
            "max_iter": np.inf, #Iteraciones maximas
            "fitting":'WCD' #WCD o WOCD
            }
    
    
    
    
    """Inicializamos Kmeans"""
    Kmeanslist=[]
    imgs=[]
    for i in test_imgs:
        kme=Kmeans.KMeans(i,k,options)
        kme.find_bestK(k)
        Kmeanslist.append(kme)
        imgs.append(Kmeans.get_colors(kme.centroids))         
    
    if optimze:
        train_imgs=featuresKnn(train_imgs)
        test_imgs=featuresKnn(test_imgs)   
        
    labels = 'Correctos', 'Incorrectos'  #Labels Pie Plots 
    """Analisis Kmeans"""
    print("----------------Kmeans-------------------")
    resultRetri=retrieval_by_color(imgs,answC)
    indexCorrect=isCorrectAnsw(test_color_labels,answC)
    
    c=0
    for i in resultRetri:
        isCorrect=np.isin(resultRetri[c],indexCorrect[c])
        i.resize((i.shape[1],i.shape[0]))
        visualize_retrieval(test_imgs[i[0]],nTestDo,test_color_labels[i[0]],isCorrect,answC[c])
        if doColorPlot:
            for f in range(nTestDo):
                Plot3DCloud(Kmeanslist[i[0][f]])
        c=c+1
        if doTimePlots or doWCDPlots:
            WCDs,fitTimes=kmean_statistics(np.asarray(Kmeanslist)[i[0]],k)
        if doWCDPlots:
            for p in range(nTestDo):                
                plt.plot(list(range(2,k)),WCDs[p],marker='o', linestyle='--',label='WCD/k')
                plt.xticks(list(range(2,k)), range(2,k))
                plt.xlabel("K")
                plt.ylabel("WCD")
                plt.title("WCD/k Plot Imagen "+str(p))
                plt.show()
        if doTimePlots:
            for p in range(nTestDo):                
                plt.plot(list(range(2,k)),fitTimes[p],marker='o', linestyle='--',label='TimeD/k')
                plt.xticks(list(range(2,k)), range(2,k))
                plt.xlabel("K")
                plt.ylabel("Time")
                plt.title("Time/k Plot Imagen "+str(p))
                plt.show()
        if doAccuaricyAnswColor:
            CountTrueColor=np.count_nonzero(isCorrect)
            plt.pie([(CountTrueColor/len(isCorrect))*100,100-((CountTrueColor/len(isCorrect))*100)],None,labels=labels,autopct='%1.1f%%')
            plt.title("Ratio Color Answare Correctos/Incorrectos")
            plt.show()
    if doAccuaricyColorTotal:
        TrueColorPercent=get_color_accuracy(imgs,test_color_labels)
        plt.pie([TrueColorPercent,100-TrueColorPercent],None,labels=labels,autopct='%1.1f%%')
        plt.title("Total Ratio Color Correctos/Incorrectos")
        plt.show()
    """Inicializamos Knn y Parmetros"""
    knn= KNN.KNN(train_imgs.mean(3)[startTrain:endTrain],train_class_labels[startTrain:endTrain])
    knnresult=knn.predict(test_imgs.mean(3),nPuntosKnn)
    
    time.sleep(2)
    """Analisis KNN"""
    print("--------------Knn---------------------")
    resultShape=retrieval_by_shape(knnresult,answS)
    indexCorrectShape=isCorrectAnswShape(test_class_labels,answS)
    c=0
    for i in resultShape:
        isCorrectShape=np.isin(resultShape[c],indexCorrectShape[c])       
        i.resize((i.shape[1],i.shape[0]))
        visualize_retrieval(test_imgs[i[0]],nTestDo,test_class_labels[i[0]],isCorrectShape,answS[c])
        
        
        if doAccuracyAnsw:            
            CountTrue=np.count_nonzero(isCorrectShape)
            plt.pie([(CountTrue/len(indexCorrectShape[c]))*100,100-((CountTrue/len(indexCorrectShape[c]))*100)],None,labels=labels,autopct='%1.1f%%')
            plt.title("Ratio Shape Answare Correctos/Incorrectos")
            plt.show()
        c=c+1
    if doAccuracyShapeTotal:    
        acuaricyShape=get_shape_accuracy(knnresult,test_class_labels)
        acuricyShapeTrue=np.count_nonzero(acuaricyShape)
        plt.pie([(acuricyShapeTrue/len(acuaricyShape))*100,100-((acuricyShapeTrue/len(acuaricyShape))*100)],None,labels=labels,autopct='%1.1f%%')
        plt.title("Total Ratio Shape Correctos/Incorrectos")
        plt.show()
    time.sleep(2)    
    """Convined Knn Kmeans"""
    print("-----------Knn Kmeans Convinados--------------")
    colorLabels=[" , ".join(item) for item in test_color_labels]   
    for i in range(0,len(resultRetri)):
        indexCorrectConvined=isCorrectAnswConbined(indexCorrect[i],indexCorrectShape[i])     
        resultCombined=retrieval_combined(resultRetri[i],resultShape[i])
        isCorrectConbined=np.isin(resultCombined,indexCorrectConvined)
        etiSha=np.core.defchararray.add(test_class_labels[:]," ")
        eti=np.core.defchararray.add(etiSha[:],colorLabels[:])
        visualize_retrieval(test_imgs[resultCombined],nTestDo,eti[resultCombined],isCorrectConbined,str(answS[i])+" "+str(answC[i]))
        if doAccuracyCombinedAnswTotal:
            CountTrueCombined=np.count_nonzero(isCorrectConbined)
            plt.pie([(CountTrueCombined/len(indexCorrectConvined))*100,100-((CountTrueCombined/len(indexCorrectConvined))*100)],None,labels=labels,autopct='%1.1f%%')
            plt.title("Ratio Combined Answare Correctos/Incorrectos")
            plt.show()
    