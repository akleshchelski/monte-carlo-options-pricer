import numpy as np
def monte_carlo_options_pricer(st,K,r,T,CallOrPut,sigma,number_of_steps,number_of_paths):
    dt = T / number_of_steps #takes T and splits into number of steps per path
    Z = np.random.randn(number_of_paths,number_of_steps) #makes empty matrix that is number_of_steps wide and number_of_paths long for computing each step value

    #GBM formula
    ManiupulatingEachZ = (r - (0.5)*(sigma**2))*dt + sigma*(np.sqrt(dt)*Z) #broadcasts each operation to every Z in the matrix
    SumOfEachRow = np.cumsum(ManiupulatingEachZ, axis = 1) #sums each row in the matrix
    ActualReturn = st * np.exp(SumOfEachRow) #multiplies stock price for actual return of each path (can be found in the last column of each row)

    listOfAllActualReturns = ActualReturn[:,-1] #goes down the last column of the matrix and stores in array
    if CallOrPut == 'put':
        payoff = np.maximum(K-listOfAllActualReturns,0) #calculates payoff if put option
    elif CallOrPut == 'call':
        payoff = np.maximum(listOfAllActualReturns-K,0) #calculates payoff if call option
    else:
        raise ValueError("type of option needs to be 'call' or 'put'")
    price = np.exp(-r*T) * np.mean(payoff)
    print(f"European {CallOrPut} option price: {price:.4f}") 
    return price

if __name__ == "__main__":
    monte_carlo_options_pricer(
        st=100,
        K=100,
        r=0.05,
        T=1,
        CallOrPut='put',
        sigma=0.2,
        number_of_steps=252,
        number_of_paths=100000
    )