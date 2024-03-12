import numpy as np

def bootstrapping(x,y,M,f,weight=None):
    if len(x) != len(y):
        print('Error: The sizes of x and y are not the same.')
        quit()

    N = len(x)
    if N == 1:
        muB = f(x[0],y[0])
        sigmaB = None
    else:
        vals = np.zeros(M)
        fails = []
        for k in range(M):
            if weight is None:
                R = np.random.randint(0,N,N)
            else:
                events = range(N)
                R = np.random.choice(events, size=N, p=weight)

            X = x[R]
            Y = y[R]        
            muX = np.average(X)
            muY = np.average(Y)
            val = f(muX, muY)
            if val is not None:
                vals[k] = val
            else:
                fails.append(k)

        # Deleting the failed resamples
        if len(fails) > 0:
            print('Number of failures =', len(fails))
        vals = np.delete(vals, fails, 0)    
        muB = np.average(vals)
        sigmaB = np.std(vals)
        sigmaB *= np.sqrt(N/(N-1))  # An unbiased estimator

    return muB, sigmaB
