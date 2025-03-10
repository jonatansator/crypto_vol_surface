import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ccxt
from datetime import datetime
from scipy.optimize import minimize

# Constants
COLOR_BTC = '#FF9900'  # Brighter Bitcoin orange
COLOR_ETH = '#00B7EB'  # Brighter Ethereum cyan
START_DATE = "2024-09-01"  # Recent data for options-like simulation
END_DATE = "2024-10-12"

def fetch_spot_data(ticker):
    """Fetch historical spot data from Binance using ccxt."""
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(f"{START_DATE}T00:00:00Z")
    end_ts = exchange.parse8601(f"{END_DATE}T00:00:00Z")
    ohlcv = exchange.fetch_ohlcv(ticker, timeframe='1h', since=start_ts, limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[df['timestamp'] <= pd.to_datetime(END_DATE)]
    return df['close'].values

def simulate_option_chain(spot_price, expirations, strikes):
    """Simulate implied volatility for an option chain (since ccxt options data is limited)."""
    np.random.seed(42)  # For reproducibility
    vol_surface = np.zeros((len(expirations), len(strikes)))
    
    for i, expiry in enumerate(expirations):
        tau = expiry / 365  # Time to expiration in years
        for j, strike in enumerate(strikes):
            # Simulate a basic volatility smile (to be refined with SABR)
            atm_vol = 0.5  # Base at-the-money volatility (50%)
            skew = 0.1 * (strike / spot_price - 1)  # Linear skew
            time_effect = np.sqrt(tau)  # Volatility increases with time
            vol_surface[i, j] = atm_vol + skew + 0.05 * time_effect + np.random.normal(0, 0.02)
            vol_surface[i, j] = max(0.1, vol_surface[i, j])  # Minimum vol of 10%
    
    return vol_surface

def sabr_volatility(strike, forward, tau, alpha, beta, rho, nu):
    """Calculate implied volatility using the SABR model."""
    if strike <= 0 or forward <= 0:
        return 0.0
    X = strike / forward
    if abs(X - 1) < 1e-6:  # At-the-money case
        return alpha / (forward ** (1 - beta))
    
    z = (nu / alpha) * (forward * strike) ** ((1 - beta) / 2) * np.log(forward / strike)
    chi = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    if chi == 0:
        return 0.0
    
    term1 = alpha / ((forward * strike) ** ((1 - beta) / 2) * (1 + (1 - beta)**2 / 24 * np.log(forward / strike)**2 + (1 - beta)**4 / 1920 * np.log(forward / strike)**4))
    term2 = 1 + (((1 - beta)**2 / 24 * alpha**2 / (forward * strike)**(1 - beta) + rho * beta * nu * alpha / (4 * (forward * strike)**((1 - beta) / 2)) + nu**2 * (2 - 3 * rho**2) / 24) * tau)
    return term1 * (z / chi) * term2

def fit_sabr(vol_surface, strikes, forward, tau):
    """Fit SABR model parameters to the volatility surface."""
    def objective(params):
        alpha, rho, nu = params  # beta is fixed at 0.7 for crypto
        beta = 0.7
        fitted_vols = [sabr_volatility(k, forward, tau[i], alpha, beta, rho, nu) for i, expiry in enumerate(tau) for k in strikes]
        target_vols = vol_surface.flatten()
        return np.sum((np.array(fitted_vols) - target_vols) ** 2)
    
    # Initial guess: alpha=0.2, rho=0, nu=0.3
    result = minimize(objective, [0.2, 0, 0.3], bounds=[(0.01, 1), (-0.99, 0.99), (0.01, 1)], method='L-BFGS-B')
    alpha, rho, nu = result.x
    return alpha, 0.7, rho, nu

def plot_volatility_surface(vol_surface, strikes, expirations, crypto_name, color):
    """Create a quant-style 3D volatility surface plot."""
    X, Y = np.meshgrid(strikes, expirations)
    fig = go.Figure(data=[go.Surface(
        z=vol_surface * 100,  # Convert to percentage
        x=X,
        y=Y,
        colorscale=[[0, 'rgb(40,40,40)'], [0.5, color], [1, 'white']],
        opacity=0.9,
        contours={
            'z': {'show': True, 'start': 10, 'end': 100, 'size': 5, 'color': 'white'}
        }
    )])
    
    fig.update_layout(
        title=dict(text=f'{crypto_name} Implied Volatility Surface', font_color='white', x=0.5),
        scene=dict(
            xaxis_title='Strike Price (USD)',
            yaxis_title='Days to Expiration',
            zaxis_title='Implied Volatility (%)',
            xaxis=dict(backgroundcolor='rgb(40,40,40)', gridcolor='rgba(255,255,255,0.2)', color='white'),
            yaxis=dict(backgroundcolor='rgb(40,40,40)', gridcolor='rgba(255,255,255,0.2)', color='white'),
            zaxis=dict(backgroundcolor='rgb(40,40,40)', gridcolor='rgba(255,255,255,0.2)', color='white'),
        ),
        plot_bgcolor='rgb(40,40,40)',
        paper_bgcolor='rgb(40,40,40)',
        font_color='white',
        margin=dict(l=50, r=50, t=50, b=50),
        annotations=[dict(
            text=f"Latest Spot: ${spot_price:,.2f}",
            x=0.05, y=0.95, xref="paper", yref="paper",
            showarrow=False, font=dict(color='white', size=14)
        )]
    )
    
    fig.show()

if __name__ == "__main__":
    crypto_data = {
        'BTC': {'ticker': 'BTC/USDT', 'color': COLOR_BTC},
        'ETH': {'ticker': 'ETH/USDT', 'color': COLOR_ETH}
    }
    
    expirations = np.array([7, 14, 30, 60, 90])  # Days to expiration
    
    for crypto, params in crypto_data.items():
        try:
            # Fetch spot data
            spot_prices = fetch_spot_data(params['ticker'])
            spot_price = spot_prices[-1]
            
            # Simulate strikes around the spot price
            strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, 10)
            
            # Simulate initial volatility surface
            vol_surface = simulate_option_chain(spot_price, expirations, strikes)
            
            # Fit SABR model
            alpha, beta, rho, nu = fit_sabr(vol_surface, strikes, spot_price, expirations / 365)
            print(f"{crypto} SABR Parameters: alpha={alpha:.3f}, beta={beta:.3f}, rho={rho:.3f}, nu={nu:.3f}")
            
            # Generate fitted volatility surface
            fitted_vol_surface = np.array([[sabr_volatility(k, spot_price, t / 365, alpha, beta, rho, nu) 
                                            for k in strikes] for t in expirations])
            
            # Plot
            plot_volatility_surface(fitted_vol_surface, strikes, expirations, crypto, params['color'])
        except Exception as e:
            print(f"Error processing {crypto}: {str(e)}")