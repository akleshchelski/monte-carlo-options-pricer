import numpy as np
st = 100 #original stock price
K = 100 #strike price
r = .05 #risk-free interest rate
CallOrPut = 'put'
sigma = 0.2 #volatility
T = 1 #time (let's call this "one year")
number_of_steps = 252 #time steps per path (252 trading days per year)
number_of_paths = 1000 #total number of monte carlo paths we simulate
dt = T / number_of_steps #takes T and splits into number of steps per path
Z = np.random.randn(number_of_paths,number_of_steps) #makes empty matrix that is number_of_steps wide and number_of_paths long for computing each step value

#GBM formula
ManiupulatingEachZ = (r - (0.5)*(sigma**2))*dt + sigma*(np.sqrt(dt)*Z) #broadcasts each operation to every Z in the matrix
SumOfEachRow = np.cumsum(ManiupulatingEachZ, axis = 1) #sums each row in the matrix
ActualReturn = st * np.exp(SumOfEachRow) #multiplies stock price for actual return of each path (can be found in the last column of each row)

listOfAllActualReturns = ActualReturn[:,-1] #goes down the last column of the matrix and stores in array
if CallOrPut == 'put':
    payoff = np.maximum(K-listOfAllActualReturns,0) #calculates payoff if put option
else:
    payoff = np.maximum(listOfAllActualReturns-K,0) #calculates payoff if call option

price = np.exp(-r*T) * np.mean(payoff)
print(f"European {CallOrPut} option price: {price:.4f}") 