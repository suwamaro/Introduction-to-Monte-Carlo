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
            muX = np.mean(X)
            muY = np.mean(Y)
            try:
                val = f(muX, muY)
            except Exception as e:
                print(f'An error occurred in resampling: {e}')                
                fails.append(k)
            else:
                vals[k] = val

        # Deleting the failed resamples
        if len(fails) > 0:
            print('Number of failures =', len(fails))
        vals = np.delete(vals, fails, 0)    
        muB = np.mean(vals)
        sigmaB = np.std(vals)
        sigmaB *= np.sqrt(N/(N-1))  # An unbiased estimator

    return muB, sigmaB
