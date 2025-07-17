### Black Scholes Option Pricing Model

import warnings
warnings.filterwarnings('ignore')

# Base libraries
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta

# Option Strategy plotting
import opstrat as op

# Set max row to 300
pd.set_option('display.max_rows', 300)

class BS:
    """
    This is a class for Options contract for pricing European options on stocks without dividends.
    Attributes: 
        spot (spot price)                   : int or float
        strike (Exercise price)             : int or float 
        rate (risk-free rate)               : float
        dte (time to expiration in years)   : int or float [days to expiration in number of years]
        volatility                          : float
    """    
    
    def __init__(self, spot: float, strike: float, rate: float, dte: float, volatility: float):
        assert strike > 0 or strike < 0, f"Strike {strike} cannot be zero 0"
        
        self.spot = spot # Spot Price
        self.strike = strike # Option Strike
        self.rate = rate # Interest Rate    
        self.dte = dte # Days To Expiration
        self.volatility = volatility # Volatility
        
        # Utility 
        self._a_ = self.volatility * np.sqrt(self.dte)
        
        # if self.strike == 0:
        #     raise ZeroDivisionError('The strike price cannot be zero')
        # else:
        self._d1_ = (np.log(self.spot / self.strike) + (self.rate + (self.volatility**2) / 2) * self.dte) / self._a_
        self._d2_ = self._d1_ - self._a_
        self._b_ = np.e**-(self.rate * self.dte)
        
        
        # The __dict__ attribute Contains all the attributes defined for the object itself. It maps the attribute name to its value.
        for i in ['callPrice', 'putPrice', 'callDelta', 'putDelta', 'callTheta', 'putTheta', 'callRho', 'putRho', 'vega', 'gamma']:
            self.__dict__[i] = None
        
        [self.callPrice, self.putPrice] = self._price
        [self.callDelta, self.putDelta] = self._delta
        [self.callTheta, self.putTheta] = self._theta
        [self.callRho, self.putRho] = self._rho
        self.vega = self._vega
        self.gamma = self._gamma
        
    def __repr__(self):
        return (f"Black_Scholes_Object(spot:{self.spot}, strike:{self.strike}, rate:{self.rate}, "
                f"days_to_expiration(in_years):{self.dte}, volatility:{self.volatility} )")
    
        
    # Option Price
    @property
    def _price(self):
        """Returns the option price: [Call price, Put price]"""
        if self.volatility == 0 or self.dte == 0:
            call = np.maximum(0.0, self.spot - self.strike)
            put = np.maximum(0.0, self.strike - self.spot)
        else:
            call = self.spot * norm.cdf(self._d1_) - self.strike * np.e**(-self.rate * self.dte) * norm.cdf(self._d2_)
            put = self.strike * np.e**(-self.rate * self.dte) * norm.cdf(-self._d2_) - self.spot * norm.cdf(-self._d1_)
        return [call, put]


    # Option Delta
    @property
    def _delta(self):
        '''Returns the option delta: [Call delta, Put delta]'''
        if self.volatility == 0 or self.dte == 0:
            call = 1.0 if self.spot > self.strike else 0.0
            put = -1.0 if self.spot < self.strike else 0.0
        else:
            call = norm.cdf(self._d1_)
            put = -norm.cdf(-self._d1_)
        return [call, put]


    # Option Gamma
    @property
    def _gamma(self):
        '''Returns the option gamma'''
        return norm.pdf(self._d1_) / (self.spot * self._a_)


    # Option Vega
    @property
    def _vega(self):
        '''Returns the option vega'''
        if self.volatility == 0 or self.dte == 0:
            return 0.0
        else:
            return self.spot * norm.pdf(self._d1_) * np.sqrt(self.dte) / 100


    # Option Theta
    @property
    def _theta(self):
        '''Returns the option theta: [Call theta, Put theta]'''
        call = -self.spot * norm.pdf(self._d1_) * self.volatility / (2 * np.sqrt(self.dte)) - self.rate * self.strike * self._b_ * norm.cdf(self._d2_)

        put = -self.spot * norm.pdf(self._d1_) * self.volatility / (2 * self.dte**0.5) + self.rate * self.strike * self._b_ * norm.cdf(-self._d2_)
        return [call / 365, put / 365]


    # Option Rho
    @property
    def _rho(self):
        '''Returns the option rho: [Call rho, Put rho]'''
        call = self.strike * self.dte * self._b_ * norm.cdf(self._d2_) / 100
        put = -self.strike * self.dte * self._b_ * norm.cdf(-self._d2_) / 100

        return [call, put]

    # Implied Vol
    @property
    def _IV(self):
        pass
