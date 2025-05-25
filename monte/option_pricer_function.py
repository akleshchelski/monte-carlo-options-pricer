import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def path_simulate(st,r,T,sigma,number_of_steps,number_of_paths):
    dt = T / number_of_steps #takes T and splits into number of steps per path
    Z = np.random.randn(number_of_paths,number_of_steps) #makes empty matrix that is number_of_steps wide and number_of_paths long for computing each step value

    #GBM formula
    manipulating_each_Z = (r - (0.5)*(sigma**2))*dt + sigma*(np.sqrt(dt)*Z) #broadcasts each operation to every Z in the matrix
    sum_of_each_row = np.cumsum(manipulating_each_Z, axis = 1) #sums each row in the matrix
    sum_of_paths = st * np.exp(sum_of_each_row) #multiplies stock price for actual return of each path (can be found in the last column of each row)
    return sum_of_paths

def price_from_paths(sumPaths,K,T,r,call_or_put):
    listOfAllActualReturns = sumPaths[:,-1] #goes down the last column of the matrix and stores each value in array
    if call_or_put == 'put':
        payoff = np.maximum(K-listOfAllActualReturns,0) #calculates payoff if put option
    elif call_or_put == 'call':
        payoff = np.maximum(listOfAllActualReturns-K,0) #calculates payoff if call option
    else:
        raise ValueError("type of option needs to be 'call' or 'put'")
    price = np.exp(-r*T) * np.mean(payoff)
    # print(f"European {CallOrPut} option price: {price:.4f}") 
    return price


def plot_paths(sum_paths, T):
    number_of_steps = sum_paths.shape[1]
    time_split = np.linspace(0,T,number_of_steps)
    plt.figure(figsize=(6.4,4.8))
    for i in range((len(sum_paths))):
        plt.plot(time_split, sum_paths[i], lw=1)

    plt.title("Simulated Stock Price Paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("monte_carlo_paths.png")
    print("plot saved as monte_carlo_paths.png")


if __name__ == "__main__":
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y", interval="1d")
    st=data["Close"][-1]
    log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    daily_vol = np.std(log_returns)
    sigma = daily_vol * np.sqrt(252)
    K=st
    r=0.05
    T=1
    call_or_put='put'
    number_of_steps=252
    number_of_paths=100000
    sum_paths = path_simulate(st,r,T,sigma,number_of_steps,number_of_paths)
    plot_paths(sum_paths, T)