__authors__ = ['1491578','1496496','1495282']
__group__ = 'DJ.15'

import numpy as np
import utils
from scipy.special import comb

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K     
        self.WCD=0
        self._init_X(X)
        """self.labels=np.empty((0,),int)"""
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################






    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
               
        X=X.astype(float)
        if len(X.shape)==3:
          self.X=np.resize(X,(X.shape[0]*X.shape[1],3))
        else:
            self.X=X
                  
        

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 20
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################




    def _init_centroids(self):
        """
        Initialization of centroids
        """
        
        """self.old_centroids=np.empty((0,self.K),float) np.arange(1/self.K,1+1/self.K,1/self.K)"""        
        if self.options['km_init'].lower() == 'first':
            X=np.unique(self.X,axis=0,return_index=True)[1]
            B=np.array([self.X[index] for index in sorted(X)])
            self.centroids=B[0:self.K]
        elif self.options['km_init'].lower()== 'random':
            X=np.unique(self.X,axis=0,return_index=True)
            B=np.array([X[0][index] for index in np.random.randint(len(X[1]), size=self.K)])
            while len(np.unique(B,axis=0))!=self.K:
                B=np.array([X[0][index] for index in np.random.randint(len(X[1]), size=self.K)])
            self.centroids=B
            
        else:           
            quant=np.arange(100/self.K, 101, 100/self.K)
            X=np.unique(self.X,axis=0,return_index=True)                       
            result=[]
            for i in range(self.K):  
                if i==0:
                    result.append(X[0][np.random.randint(0,len(X[1])*(quant[i]/100), size=1)])
                else:
                    result.append(X[0][np.random.randint(len(X[1])*(quant[i-1]/100),len(X[1])*(quant[i]/100), size=1)])            
               
            self.centroids=np.array(result).reshape(self.K,3)
            
            
        


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        self.labels=np.empty((0,),int)
        for point in distance(self.X,self.centroids):            
            self.labels=np.argmin(point)
        """
        
                   
        self.labels=np.argmin(distance(self.X,self.centroids),axis=1)
       
        

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        self.centroids=np.empty((0,3),float)
        self.centroids[x]=np.append(self.centroids,[np.array([sx/leng,sy/leng,sz/leng],float)],axis=0) 
        """
        self.old_centroids=np.array(self.centroids)
        sx=0
        sy=0
        sz=0
        for x in range(self.K):
            indices = np.array(np.where(self.labels == x))            
            values=np.take(self.X, indices,axis=0)
            leng=indices.shape[1]
            sx=np.sum(values[0,:][:,0])
            sy=np.sum(values[0,:][:,1])
            sz=np.sum(values[0,:][:,2])
            self.centroids[x]=np.array([sx/leng,sy/leng,sz/leng],float) 
        pass

    def converges(self):
        """
        Checks if there is a difference between current and old centroids len(self.X)
        """
        if np.array_equal(self.centroids,self.old_centroids):
           return True
        elif self.num_iter==self.options['max_iter']:
            return True
        else:
            return False
        

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """  
        self._init_centroids()
        while True:
            self.get_labels()
            self.get_centroids()
            self.num_iter+=1
            isOK=self.converges()
            if isOK:
                break
        pass

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
         aux+=sum(np.power(np.linalg.norm(np.take(self.X,np.where(self.labels == x),axis=0)[0]-self.centroids[x],axis=1),2))
        """
        aux=0
        for x in range(self.K):
          aux+=sum(np.power(np.linalg.norm(np.take(self.X,np.where(self.labels == x),axis=0)[0]-self.centroids[x],axis=1),2))
        aux=aux/len(self.X)
        
        return aux
    
    def whitOutClassDistance(self):
        """
         returns the whithin class distance of the current clustering
         aux+=sum(np.power(np.linalg.norm(np.take(self.X,np.where(self.labels == x),axis=0)[0]-self.centroids[x],axis=1),2))
        """
        aux=0
        for x in range(self.K):
          aux+=sum(np.power(np.linalg.norm(self.centroids[x]-self.centroids[x:],axis=1),2))
        aux=aux/comb(self.K,2)
        
        return aux
    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        if self.options['fitting']=='WCD':
            self.K=2
            self.fit()
            wcd=self.whitinClassDistance()
            for x in range(3,max_K):
                self.K=x
                self.fit()
                self.WCD = self.whitinClassDistance()
                if(100-((self.WCD/wcd)*100))<self.options['tolerance']:
                    self.K=self.K-1
                    break
                else:
                    wcd=self.WCD
        if self.options['fitting']=='WOCD':
            self.K=2
            self.fit()
            wcd=self.whitOutClassDistance()
            for x in range(3,max_K):
                self.K=x
                self.fit()
                self.WCD = self.whitOutClassDistance()
                if(100-((self.WCD/wcd)*100))<self.options['tolerance']:
                    self.K=self.K-1
                    break
                else:
                    wcd=self.WCD      
            
        pass


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
        
    result=np.empty((0,C.shape[0]), float)
    for i in X:
        a=np.empty((0,C.shape[0]), float)
        for j in C:
            
            a=np.append(a,np.linalg.norm(i-j))
        result=np.append(result,[a],axis=0)
    return result 
    """
        
    
    return np.linalg.norm(C-X[:,None],axis=2)        
     


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    colors=np.array(['Red','Orange','Brown','Yellow','Green','Blue','Purple','Pink','Black','Grey','White'])
    
    return np.take(colors, np.where(np.amax(utils.get_color_prob(centroids),axis=1)[:,None] == utils.get_color_prob(centroids))[1])

