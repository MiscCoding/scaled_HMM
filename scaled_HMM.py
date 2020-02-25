from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


#It crates valid random markov matrix
def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x/x.sum(axis=1, keepdims=True)


class HMM:
    def __init__(self, M):
        #this class gets number of hidden states in constructor
        self.M = M

        #iteration limit is set by max_iter variable.
    def fit(self, X, max_iter=30):

        t0 = datetime.now()
        np.random.seed(123)#test
        #
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        # determine V, the vocabulary size
        # assume observables are already integers from 0 to 1
        # X is a jagged array of observed sequences
        V = max(max(x) for x in X) + 1
        N = len(X)

        self.pi = np.ones(self.M) / self.M # initial state which is uniform distribution
        self.A = random_normalized(self.M, self.M) # state transition matrix
        self.B = random_normalized(self.M, V) # output distribution

        print("initial A:", self.A)
        print("initial B:", self.B)

        costs = []
        for it in range(max_iter):
            # prints every 10 cycle
            if it % 10 == 0:
                print("it:", it)

            #
            alphas = []
            betas = []
            P = np.zeros(N)
            #loop through each observation
            for n in range(N):
                # x is nth observation
                x = X[n]
                # T length of each little x
                T = len(x)

                alpha = np.zeros((T, self.M)) #fill T by M matrix with zeros
                # 1st value of alpha is pi * 1st observation
                alpha[0] = self.pi*self.B[:,x[0]]
                # loop through each after 1st observation
                for t in range(1, T):
                    #
                    tmp1 = alpha[t-1].dot(self.A) * self.B[:, x[t]]

                    alpha[t] = tmp1
                P[n] = alpha[-1].sum()
                alphas.append(alpha)

                #beta is also T by M matrix
                #beta counts backwards
                beta = np.zeros((T, self.M))
                beta[-1] = 1#initial value of beta is 1
                for t in range(T - 2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1]) #calculate backwards
                betas.append(beta)


            assert(np.all(P > 0))
            # calculate the total log likelihood
            cost = np.sum(np.log(P))
            # appended to list of cost
            costs.append(cost)

            # now re-estimate pi, A, B
            self.pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in range(N)) / N
            # print "self.pi:", self.pi
            # break

            # keep track of all denominators and numerators for A and B updates
            den1 = np.zeros((self.M, 1))
            den2 = np.zeros((self.M, 1))
            a_num = 0
            b_num = 0
            #
            for n in range(N):
                x = X[n]
                T = len(x)
                # accumulate everything except the last one
                den1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
                #sum of all things
                den2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / P[n]



                # updating numerator for A
                a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M): #looping through all states twice
                    for j in range(self.M):
                        for t in range(T-1): # and all times except the last one.
                            a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * self.B[j, x[t+1]] * betas[n][t+1,j]
                a_num += a_num_n / P[n]  # all it to the numerator

                #updating numerator for B
                b_num_n2 = np.zeros((self.M, V))
                for i in range(self.M):# loop through every state
                    for t in range(T): # loop through possible observation
                        b_num_n2[i,x[t]] += alphas[n][t,i] * betas[n][t,i]#
                b_num += b_num_n2 / P[n]


            self.A = a_num / den1
            self.B = b_num / den2
            # print "P:", P
            # break
        print("A:", self.A)
        print("B:", self.B)
        print("pi:", self.pi)

        print("Fit duration:", (datetime.now() - t0))

        plt.plot(costs)
        plt.show()

    #
    def likelihood(self, x):
            T = len(x) # get length of x
            alpha = np.zeros((T, self.M))
            alpha[0] = self.pi * self.B[:, x[0]]

            for t in range(1, T):
                alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]] #update alpha

            return alpha[-1].sum()


    def likelihood_multi(self, X): # calculate all the likelihoods of every observation
            return np.array([self.likelihood(x) for x in X])

    #log print
    def log_likelihood_multi(self, X):
            return np.log(self.likelihood_multi(X)) #return logs above.

    #viterbi algorithms
    def get_state_sequence(self, x): # takes in one observable sequence
            T = len(x)
            delta = np.zeros((T, self.M)) # size setup
            psi = np.zeros((T, self.M))# size setup
            delta[0] = self.pi * self.B[:, x[0]] #set an initial data
            for t in range(1, T): # loop through every other time and all the states.
                for j in range(self.M):
                    delta[t, j] = np.max(delta[t-1]*self.A[:,j] * self.B[j, x[t]])
                    psi[t, j] = np.argmax(delta[t-1]*self.A[:,j])


            #backtrack
            states = np.zeros(T, dtype= np.int32)
            states[T-1] = np.argmax(delta[T-1]) # argmax delta for the last one
            for t in range(T-2, -1, -1): #loop through the rest of time in descending order
                states[t] = psi[t+1, states[t+1]] # use psi which stores the states
            return states


## load coin data and train Hidden Markov model
def fit_coin():
            X = []
            for line in open("coin_data.txt"): # loop through coin data.txt
                x = [1 if e == "M" else 0 for e in line.rstrip()]
                X.append(x)

            ##creates an object and its number of hidden states to 2 (head, or tail)
            hmm = HMM(2)
            hmm.fit(X)
            L= hmm.log_likelihood_multi(X).sum() # return likelihood and sum it cause they are separate.

            print("LL with fitted params: ", L)

            hmm.pi = np.array([0.5, 0.5])
            hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
            hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])

            L = hmm.log_likelihood_multi(X).sum()
            print ("LL with true params:", L)

            # try viterbi algorithms to figure out the best state sequence
            print ("Best state sequence for: ", X[0])
            print (hmm.get_state_sequence(X[0]))


if __name__ == "__main__":
    fit_coin()
