"""
Arbitrage strategies for the Goalkick Ltd Trading Bot.
"""

import pandas as pd
import numpy as np
import time
import threading

from config.logging_config import get_logger
from config.strategy_params import ARBITRAGE_PARAMS
from src.strategies.base_strategy import Strategy
from src.utils.error_handling import handle_error

logger = get_logger("strategies.arbitrage")

class TriangularArbitrageStrategy(Strategy):
    """Triangular arbitrage strategy."""
    
    def __init__(self, timeframes=None, symbols=None, params=None):
        """
        Initialize the strategy.
        
        Args:
            timeframes (list): List of timeframes to use (not used for arbitrage)
            symbols (list): List of symbols to trade
            params (dict): Strategy parameters
        """
        default_params = ARBITRAGE_PARAMS.get('triangular', {})
        params = {**default_params, **(params or {})}
        
        # Arbitrage doesn't use timeframes in the traditional sense
        super().__init__('TriangularArbitrage', ['1m'], symbols, params)
        
        self.pairs_info = {}  # Symbol info cache
        self.exchange_api = None  # Will be set later
        self.running = False
        self.thread = None
    
    def set_exchange_api(self, exchange_api):
        """
        Set the exchange API.
        
        Args:
            exchange_api: Exchange API instance
        """
        self.exchange_api = exchange_api
    
    def generate_signals(self, data, symbol, timeframe):
        """
        Generate trading signals based on triangular arbitrage opportunities.
        Arbitrage doesn't use historical data; instead, it uses current ticker prices.
        
        Args:
            data (pd.DataFrame): Not used for arbitrage
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe (not used for arbitrage)
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            if not self.exchange_api:
                logger.error("Exchange API not set")
                return []
            
            # Get current tickers for all pairs
            tickers = self.exchange_api.get_tickers()
            
            if not tickers:
                logger.warning("No ticker data available")
                return []
            
            # Find triangular arbitrage opportunities
            opportunities = self._find_arbitrage_opportunities(tickers)
            
            # Generate signals for opportunities
            signals = []
            
            for opp in opportunities:
                if opp['profit_pct'] > self.params.get('min_profit_pct', 0.5) / 100:
                    # Create a signal with the arbitrage details
                    # Create a signal with the arbitrage details
                    signal = self.create_signal(
                        symbol=opp['pair1'],  # Use the first pair as the primary symbol
                        timeframe='1m',
                        direction="ARBITRAGE",
                        strength=min(1.0, opp['profit_pct'] * 10),
                        price=None,  # Not applicable for arbitrage
                        metadata={
                            'type': 'triangular',
                            'pair1': opp['pair1'],
                            'pair2': opp['pair2'],
                            'pair3': opp['pair3'],
                            'profit_pct': opp['profit_pct'],
                            'path': opp['path'],
                            'prices': opp['prices']
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"Triangular arbitrage opportunity: {opp['path']} with {opp['profit_pct']:.2%} profit")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating arbitrage signals: {e}")
            handle_error(e, "Failed to generate arbitrage signals")
            return []
    
    def _find_arbitrage_opportunities(self, tickers):
        """
        Find triangular arbitrage opportunities in the market.
        
        Args:
            tickers (dict): Current ticker data
            
        Returns:
            list: List of arbitrage opportunities
        """
        try:
            opportunities = []
            
            # Get all trading pairs
            if not self.pairs_info:
                self._update_pairs_info()
            
            # Find trading paths
            trading_paths = self._find_trading_paths()
            
            # Check each path for arbitrage opportunities
            for path in trading_paths:
                # Get ticker prices for the path
                prices = {}
                valid_path = True
                
                for pair in path['pairs']:
                    if pair not in tickers:
                        valid_path = False
                        break
                    
                    ticker = tickers[pair]
                    # Use mid price for calculations
                    mid_price = (float(ticker.get('bid', 0)) + float(ticker.get('ask', 0))) / 2
                    
                    if mid_price <= 0:
                        valid_path = False
                        break
                    
                    prices[pair] = mid_price
                
                if not valid_path:
                    continue
                
                # Calculate potential profit
                profit_pct = self._calculate_arbitrage_profit(path, prices)
                
                # Consider fees and slippage
                profit_pct -= self.params.get('max_slippage_pct', 0.1) / 100
                
                # Add to opportunities if profitable
                if profit_pct > 0:
                    opportunities.append({
                        'pair1': path['pairs'][0],
                        'pair2': path['pairs'][1],
                        'pair3': path['pairs'][2],
                        'path': path['path'],
                        'profit_pct': profit_pct,
                        'prices': prices
                    })
            
            # Sort by profit percentage
            opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
            
            return opportunities
        except Exception as e:
            logger.error(f"Error finding arbitrage opportunities: {e}")
            handle_error(e, "Failed to find arbitrage opportunities")
            return []
    
    def _update_pairs_info(self):
        """Update the trading pairs information."""
        try:
            if not self.exchange_api:
                logger.error("Exchange API not set")
                return
            
            # Get exchange info
            exchange_info = self.exchange_api.get_exchange_info()
            
            if not exchange_info or 'symbols' not in exchange_info:
                logger.error("Failed to get exchange info")
                return
            
            # Update pairs info
            self.pairs_info = {}
            
            for symbol, info in exchange_info['symbols'].items():
                base_asset = info.get('baseAsset', '').upper()
                quote_asset = info.get('quoteAsset', '').upper()
                
                if base_asset and quote_asset:
                    self.pairs_info[symbol] = {
                        'base': base_asset,
                        'quote': quote_asset
                    }
            
            logger.debug(f"Updated pairs info with {len(self.pairs_info)} pairs")
        except Exception as e:
            logger.error(f"Error updating pairs info: {e}")
            handle_error(e, "Failed to update pairs info")
    
    def _find_trading_paths(self):
        """
        Find possible triangular arbitrage paths.
        
        Returns:
            list: List of trading paths
        """
        try:
            paths = []
            
            # Get unique assets
            assets = set()
            for symbol, info in self.pairs_info.items():
                assets.add(info['base'])
                assets.add(info['quote'])
            
            # Find triangular paths
            for asset1 in assets:
                for asset2 in assets:
                    if asset1 == asset2:
                        continue
                    
                    for asset3 in assets:
                        if asset1 == asset3 or asset2 == asset3:
                            continue
                        
                        # Find pairs for each step
                        pair1 = self._find_pair(asset1, asset2)
                        pair2 = self._find_pair(asset2, asset3)
                        pair3 = self._find_pair(asset3, asset1)
                        
                        if pair1 and pair2 and pair3:
                            # Calculate direction for each step
                            dir1 = 'buy' if self.pairs_info[pair1]['base'] == asset2 else 'sell'
                            dir2 = 'buy' if self.pairs_info[pair2]['base'] == asset3 else 'sell'
                            dir3 = 'buy' if self.pairs_info[pair3]['base'] == asset1 else 'sell'
                            
                            paths.append({
                                'pairs': [pair1, pair2, pair3],
                                'assets': [asset1, asset2, asset3],
                                'directions': [dir1, dir2, dir3],
                                'path': f"{asset1} → {asset2} → {asset3} → {asset1}"
                            })
            
            logger.debug(f"Found {len(paths)} potential triangular arbitrage paths")
            return paths
        except Exception as e:
            logger.error(f"Error finding trading paths: {e}")
            handle_error(e, "Failed to find trading paths")
            return []
    
    def _find_pair(self, asset1, asset2):
        """
        Find a trading pair for two assets.
        
        Args:
            asset1 (str): First asset
            asset2 (str): Second asset
            
        Returns:
            str: Trading pair symbol or None if not found
        """
        for symbol, info in self.pairs_info.items():
            if (info['base'] == asset1 and info['quote'] == asset2) or \
               (info['base'] == asset2 and info['quote'] == asset1):
                return symbol
        
        return None
    
    def _calculate_arbitrage_profit(self, path, prices):
        """
        Calculate the potential profit for a triangular arbitrage path.
        
        Args:
            path (dict): Trading path
            prices (dict): Current prices
            
        Returns:
            float: Profit percentage
        """
        try:
            # Start with 1 unit of the first asset
            quantity = 1.0
            
            # Simulate trading along the path
            for i, pair in enumerate(path['pairs']):
                price = prices[pair]
                direction = path['directions'][i]
                
                if direction == 'buy':
                    # Buy: spend quote to get base
                    quantity = quantity / price
                else:
                    # Sell: spend base to get quote
                    quantity = quantity * price
            
            # Calculate profit percentage
            profit_pct = quantity - 1.0
            
            return profit_pct
        except Exception as e:
            logger.error(f"Error calculating arbitrage profit: {e}")
            handle_error(e, "Failed to calculate arbitrage profit")
            return 0
    
    def start_monitoring(self, check_interval=None):
        """
        Start monitoring for arbitrage opportunities in a separate thread.
        
        Args:
            check_interval (int): Check interval in seconds
        """
        if self.running:
            logger.warning("Arbitrage monitoring already running")
            return
        
        check_interval = check_interval or self.params.get('check_interval_seconds', 5)
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, args=(check_interval,), daemon=True)
        self.thread.start()
        
        logger.info(f"Started arbitrage monitoring with {check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop monitoring for arbitrage opportunities."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Stopped arbitrage monitoring")
    
    def _monitoring_loop(self, check_interval):
        """
        Monitoring loop for arbitrage opportunities.
        
        Args:
            check_interval (int): Check interval in seconds
        """
        while self.running:
            try:
                # Generate signals (which will find arbitrage opportunities)
                signals = self.generate_signals(None, None, None)
                
                # Process signals
                if signals:
                    # In a real implementation, these signals would be sent to the signal processor
                    pass
            except Exception as e:
                logger.error(f"Error in arbitrage monitoring loop: {e}")
                handle_error(e, "Error in arbitrage monitoring loop")
            
            # Sleep for the check interval
            time.sleep(check_interval)


class StatisticalArbitrageStrategy(Strategy):
    """Statistical arbitrage (pairs trading) strategy."""
    
    def __init__(self, timeframes=None, symbols=None, params=None):
        """
        Initialize the strategy.
        
        Args:
            timeframes (list): List of timeframes to use
            symbols (list): List of symbols to trade
            params (dict): Strategy parameters
        """
        default_params = ARBITRAGE_PARAMS.get('statistical', {})
        params = {**default_params, **(params or {})}
        
        super().__init__('StatisticalArbitrage', timeframes, symbols, params)
        
        self.pairs = {}  # Dictionary of correlated pairs
    
    def generate_signals(self, data, symbol, timeframe):
        """
        Generate trading signals for a symbol and timeframe.
        
        Args:
            data (pd.DataFrame): Historical price data
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            if data.empty:
                logger.warning(f"Empty data for {symbol} {timeframe}")
                return []
            
            # Preprocess data
            df = self.preprocess_data(data)
            
            if df.empty or len(df) < 100:
                return []
            
            # If we don't have pairs defined yet, find them
            if not self.pairs:
                # In a real implementation, the pairs would be found by analyzing
                # correlations between multiple symbols
                logger.warning("No pairs defined for statistical arbitrage")
                return []
            
            # Check if this symbol is part of a defined pair
            pair_info = None
            pair_symbol = None
            
            for pair_id, info in self.pairs.items():
                if info['symbol1'] == symbol:
                    pair_info = info
                    pair_symbol = info['symbol2']
                    break
                elif info['symbol2'] == symbol:
                    pair_info = info
                    pair_symbol = info['symbol1']
                    break
            
            if not pair_info or not pair_symbol:
                # This symbol is not part of a defined pair
                return []
            
            # In a real implementation, we would need the data for the pair symbol as well
            # For now, we'll simulate it
            
            # Get parameters
            z_score_threshold = self.params.get('z_score_threshold', 2.0)
            
            # Generate signals
            signals = []
            
            # In a real implementation, we would calculate the spread between the two symbols,
            # then calculate the z-score of the spread, and generate signals when the z-score
            # crosses the threshold
            
            # For now, just simulate a signal based on random conditions
            if np.random.random() < 0.05:  # 5% chance of generating a signal
                # Decide signal direction based on simulated z-score
                direction = "BUY" if np.random.random() < 0.5 else "SELL"
                
                # Create signal
                signal = self.create_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    strength=0.6,
                    price=df['close'].iloc[-1],
                    metadata={
                        'type': 'statistical_arbitrage',
                        'pair_symbol': pair_symbol,
                        'z_score': np.random.normal() * z_score_threshold,
                        'correlation': pair_info['correlation']
                    }
                )
                
                signals.append(signal)
                logger.info(f"{direction} signal for {symbol} (pair with {pair_symbol})")
            
            return signals
        except Exception as e:
            logger.error(f"Error generating statistical arbitrage signals for {symbol} {timeframe}: {e}")
            handle_error(e, f"Failed to generate statistical arbitrage signals for {symbol} {timeframe}")
            return []
    
    def find_pairs(self, price_data, min_correlation=None):
        """
        Find correlated pairs for statistical arbitrage.
        
        Args:
            price_data (dict): Price data for multiple symbols
            min_correlation (float): Minimum correlation threshold
            
        Returns:
            dict: Dictionary of correlated pairs
        """
        try:
            min_correlation = min_correlation or self.params.get('correlation_threshold', 0.8)
            
            # Create DataFrame with closing prices for all symbols
            symbols = list(price_data.keys())
            
            if len(symbols) < 2:
                logger.warning("Not enough symbols to find pairs")
                return {}
            
            # Extract closing prices
            closes = {}
            for symbol, data in price_data.items():
                closes[symbol] = data['close']
            
            # Create a DataFrame
            df = pd.DataFrame(closes)
            
            # Calculate correlations
            corr_matrix = df.corr()
            
            # Find pairs with high correlation
            pairs = {}
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i >= j:
                        continue
                    
                    correlation = corr_matrix.loc[symbol1, symbol2]
                    
                    if abs(correlation) >= min_correlation:
                        pair_id = f"{symbol1}_{symbol2}"
                        pairs[pair_id] = {
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation
                        }
            
            # Update pairs
            self.pairs = pairs
            
            logger.info(f"Found {len(pairs)} correlated pairs for statistical arbitrage")
            return pairs
        except Exception as e:
            logger.error(f"Error finding pairs: {e}")
            handle_error(e, "Failed to find pairs")
            return {}