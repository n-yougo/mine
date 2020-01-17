class nyc:
    # updata:200109
    import numpy as np
    import numpy.linalg as la
    from scipy.stats import norm
################################
    def eig_sort(self, S):
        value, vec = la.eig(S)
        idx = value.argsort()[::-1]
        P = value[idx]
        Q = vec.T[idx]

        return (P, Q.T)
################################
    def  svd_sort(self, S):
        vec_r, value, vec_l = la.svd(S)
        idx = value.argsort()[::-1]
        P = value[idx]
        Q_r = vec_r.T[idx]
        Q_l = vec_l.T[idx]
        
        return (Q_r.T, P, Q_l.T)

    def data(self, X, bias=False):#False is not normalized yet
        self.bias = bias
        self.N, self.D = X.shape
        self.X = X
        self.mean = np.zeros(self.N)
        if bias == False:
            self.mean = np.array([np.mean(self.X, axis=0)]*self.N)
            self.S = np.dot(self.X - self.mean, (self.X - self.mean).T) / (self.N - 1)
            self.value, self.vec = self.eig_sort(self.S)
        elif bias == True:
            self.S = np.dot(self.X, self.X.T) / self.N
            self.value, self.vec = self.eig_sort(self.S)
    
    def Cov(self):
        return self.S
    
    def Mean(self):
        return self.mean
    
    def sign_ad(self, Z): 
        X = Z.T[:self.N+1]
        Y = self.vec.T[:self.N+1]
        for i in range(self.N):
            sign = np.dot(X[i], Y[i])
            if sign < 0:
                Y[i] = - Y[i]

        return Y.T
    
    
    
    def P(self):
        one = np.array([1 for i in range(self.N)])
        mat = np.eye(self.N) - np.einsum('i, j -> ij', one, one) / self.N
        return mat
    
    
    
###################### check SSE Model ###############################
   
    def check_sse(self):
        value = self.CDM_value()
        SDcross = self.SDcross
        phi, tau = [0]*self.n2, [0]*self.n2
        mhat = self.n2 - 2
        
        def kappa(n):
            return np.sqrt(1 / n * np.log(n))
    
        for j in range(self.n2):
            if j == 0:
                phi[j] = np.trace(np.dot(SDcross, SDcross.T))
            else:
                phi[j] = np.trace(np.dot(SDcross, SDcross.T)) - np.sum(value[i]**2 for i in range(j))
        
        for j in range(self.n2 - 1):
            tau[j] = phi[j + 1] / phi[j]
        
        for j in range(self.n2 - 1):
            crit = tau[j] * (1 + (j + 1) * kappa(self.N))
            if crit > 1:
                mhat = j 
                break
        mhat = np.min([mhat, self.n2 - 2])
        
        return mhat

######################## estimation of eigen vector ########################    
    def et_vec(self, value, M):
        vec = np.zeros((self.D, M))    
        
        if self.bias == False:
            for m in range(M):
                vec.T[m] = np.dot(self.vec.T[m], self.X - self.mean) / np.sqrt((self.N - 1) * value[m])
                
        if self.bias == True:
            for m in range(M):
                vec.T[m] = np.dot(self.vec.T[m], self.X) / np.sqrt(self.N * value[m])
                
        return vec  
    
    
    
    def et_vec_self(self, value, M):
        vec = np.zeros((M, self.N, self.D))  
        
        if self.bias == False:
            c = np.sqrt(self.N - 1) / (self.N - 2)
            for m in range(M):
                u_hat = self.vec.T[m]
                for n in range(self.N):
                    u_hat[n] = -u_hat[n] / (self.N - 1)
                    vec[m, n] = c / np.sqrt(value[m]) * np.dot(u_hat, self.X - self.mean,)

        if self.bias == True:
            c = np.sqrt(self.N) / (self.N - 1)
            for m in range(M):
                u_hat = self.vaec.T[m]
                for n in range(self.N):
                    u_hat[n] = -u_hat[n] / self.N
                    vec[m, n] = c / np.sqrt(value[m]) * np.dot(u_hat, self.X)
                    
        return vec
    
    
################################ projection matrix #########################################

    def prj(self, vec, M):
        prj = np.eye(self.D) - np.sum(np.einsum('i, j -> ij', vec.T[i], vec.T[i]) for i in range(M))
        return prj

    
######################################### dual covariance matrix #######################################
    def dual(self):
        return (self.value, self.vec)
    
    def dual_value(self):
        return self.value

    
    
########################################### noise-reduction ##############################################
    
    def NRM_value(self): 
        value = np.zeros(self.N)
        
        if self.bias == False:
            for j in range(self.N - 2):
                value[j] = self.value[j] - (np.trace(self.S) - np.sum(self.value[0 :(j+1)])) / (self.N - j - 2)
        
        elif self.bias == True:
            for j in range(self.N - 1):
                value[j] = self.value[j] - (np.trace(self.S) - np.sum(self.value[0 :(j+1)])) / (self.N - j - 1)
            
        return value
    
    
    
    def NRM(self): 
        
        M = self.check_sse()
        value = self.NRM_value()
        
        spiked_vec = self.et_vec(value, M)
        spiked_vec_self = self.et_vec_self(value, M)
        
        return (value, spiked_vec, spiked_vec_self, M)
    
    
   
               
        
####################################### cross-data-methodology #############################################
        
    
    def CDM_value(self):
        n1 = int(np.ceil(self.N / 2))
        n2 = self.N - n1
        self.n2 = n2
        X1, X2 = self.X[:n1], self.X[-n2:]
        
        if self.bias == False:    
            mean1 = np.array([np.mean(X1, axis=0)] * n1)
            mean2 = np.array([np.mean(X2, axis=0)] * n2)
            
            self.SDcross = np.dot(X1 - mean1, (X2 - mean2).T) / np.sqrt((n1 - 1) * (n2 - 1))
            value = self.svd_sort(self.SDcross)[1]
            
        if self.bias == True:
            self.SDcross = np.dot(X1, X2.T) / np.sqrt(n1 * n2)
            value = self.svd_sort(self.SDcross)[1]
            
        return value
    
    
    
    
    def CDM(self):
        M = self.check_sse()
        value = self.CDM_value()
        
        spiked_vec = self.et_vec(value, M)
        spiked_vec_self = self.et_vec_self(value, M)
        
        return (value, spiked_vec, spiked_vec_self, M)
    
    
    

############################### estiomation of tr(Sigma^2) #################################
    
        
    def ECDM_data(self, X):
        N, D = X.shape
        n = []
        n.append(int(np.ceil(N / 2)))
        n.append(N - n[0])

        K = [i for i in range(3, 2*N)]
        index =  [i for i in range(len(K))]


        V = [[], []]
        Y = np.zeros((2, len(K), D))
        for k, pc in zip(K, index):
            dv = int(np.floor(k / 2))

            if dv < n[0]:
                V[0].append([i for i in range(dv)] + [i for i in range(dv + n[1], N)])  
            else:
                V[0].append([i for i in range(dv - n[0], dv)])

            if dv <= n[0]:
                V[1].append([i for i in range(dv, dv + n[1])])
            else:
                V[1].append([i for i in range(dv - n[0])] + [i for i in range(dv, N)])

            for i in range(2):
                Y[i, pc] = np.sum(X[V[i][pc]], axis=0) / n[i]

        w = 0
        for j in range(N):
            for i in range(j):
                w += np.dot(X[i] - Y[0][i + j - 1], X[j] - Y[1][i + j - 1]) ** 2

        u =  n[0] * n[1] / ((n[0] - 1) * (n[1] - 1))
        W = 2 * u / (N * (N - 1)) * w

        return W
    
    
    
    def ECDM(self):
        n = []
        n.append(int(np.ceil(self.N / 2)))
        n.append(self.N - n[0])

        K = [i for i in range(3, 2*self.N)]
        index =  [i for i in range(len(K))]


        V = [[], []]
        Y = np.zeros((2, len(K), self.D))
        for k, pc in zip(K, index):
            dv = int(np.floor(k / 2))

            if dv < n[0]:
                V[0].append([i for i in range(dv)] + [i for i in range(dv + n[1], self.N)])  
            else:
                V[0].append([i for i in range(dv - n[0], dv)])

            if dv <= n[0]:
                V[1].append([i for i in range(dv, dv + n[1])])
            else:
                V[1].append([i for i in range(dv - n[0])] + [i for i in range(dv, self.N)])

            for i in range(2):
                Y[i, pc] = np.sum(self.X[V[i][pc]], axis=0) / n[i]

        w = 0
        for j in range(self.N):
            for i in range(j):
                w += np.dot(self.X[i] - Y[0][i + j - 1], self.X[j] - Y[1][i + j - 1]) ** 2

        u =  n[0] * n[1] / ((n[0] - 1) * (n[1] - 1))
        W = 2 * u / (self.N * (self.N - 1)) * w

        return W
    
    
    
############### T_hat #####################################
    def T_hat(self, X, x):
        N, p = X.shape

        n = []
        n.append(int(np.ceil(N / 2)))
        n.append(N - n[0])

        K = [i for i in range(3, 2*N)]
        index =  [i for i in range(len(K))]


        V = [[], []]
        Y = np.zeros((2, len(K), p))
        y = np.zeros((2, len(K)))
        for k, pc in zip(K, index):
            dv = int(np.floor(k / 2))

            if dv < n[0]:
                V[0].append([i for i in range(dv)] + [i for i in range(dv + n[1], N)])   
            else:
                V[0].append([i for i in range(dv - n[0], dv)])

            if dv <= n[0]:
                V[1].append([i for i in range(dv, dv + n[1])])
            else:
                V[1].append([i for i in range(dv - n[0])] + [i for i in range(dv, N)])

            for i in range(2):
                Y[i, pc] = np.sum(X[V[i][pc]], axis=0) / n[i]
                y[i, pc] = np.sum(x[V[i][pc]]) / n[i]

        w = 0
        for j in range(N):
            for i in range(j):
                w += np.dot(X[i] - Y[0][i + j - 1], X[j] - Y[1][i + j - 1]) * (x[i] - y[0][i + j - 1]) * (x[j] - y[1][i + j - 1])

        u =  n[0] * n[1] / ((n[0] - 1) * (n[1] - 1))
        T = 2 * u / (N * (N - 1)) * w

        return T
###################################################################
###################### Classication ###############################
###################################################################
    def linear(self, X,Y):
        return(np.dot(X, Y))

    def rbf(self, gm):
        def in_rbf(X, Y):
            #n, p = X.shape
            #p = len(X)
            #X, Y = X.reshape(-1, p), Y.reshape(-1, p)
            #return np.array([[la.norm(X[i] - Y[j]) for i in range(p)] for j in range(p)])
            return np.e**(-gm * la.norm(X-Y)**2)
        return in_rbf

    # polynomial kernel
    def poly(self, degree, gamma, coef0):
        def in_poly(X, Y):
            return (coef0*np.dot(X, Y.T) + gamma)**degree
        return in_poly

    # laplace kernel
    def lap(self, gm):
        def in_lap(X, Y):
            return np.e**(-gm * la.norm(X-Y, ord = 1))
        return in_lap

###################################################################
###################### Classication ###############################
###################################################################
    def clf(self, X, y):
        label = np.unique(y) #Eliminate duplicates
        nclass = len(label)
        label_sub = [0]*nclass
        for i in range(nclass):
            for j in range(len(y)):
                if y[j]== label[i]:
                   label_sub[i].append(j)

        train = [X[label_sub[i], :] for i in range(nclass)]
        m = [ len(train[i]) for i in range(nclass)]
        mean_clf = [np.mean(train[i], axis = 0) for i in range(nclass)]
        cov_clf = [np.cov(train[i], rowvar=False) for i in range(nclass)]
        trS_clf = [np.trace(cov[i]) for i in range(nclass)]
        mean_clf = [np.diag(cov[i]) for i in range(nclass)]
        mean_clf = [np.diag(diag[i]**(-1)) for i in range(nclass)]
        diag_0 = np.diag((((n[0]-1) * diag[0] + (n[1]-1) * diag[1]) / (n[0]+n[1]-2))**(-1))
        
        def DBDA(x0):#X = [n, d, label]
            Y = []
            for i in range(nclass):
                term = la.norm(x0 - mean_clf[i])**2 - trS_clf[i]/m[i]
                Y.append(term)
            val = np.array(Y)
            cls = max(np.where(Y==min(Y)))
            return (cls, val)

