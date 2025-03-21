"""
Bayesian optimization module for the Goalkick Ltd Trading Bot.
Implements efficient parameter tuning using Bayesian optimization with Gaussian Processes.
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Optional imports for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG
from analysis.backtest.backtest_engine import BacktestEngine
from analysis.backtest.performance import calculate_performance_metrics
from src.utils.error_handling import handle_error

logger = get_logger("optimization.bayesian_optimization")

class BayesianOptimizer:
    """Class for optimizing trading strategies using Bayesian optimization."""
    
    def __init__(self, datastore=None, exchange_api=None):
        """
        Initialize the BayesianOptimizer.
        
        Args:
            datastore: DataStore instance
            exchange_api: Exchange API instance
        """
        self.datastore = datastore
        self.exchange_api = exchange_api
        self.backtest_engine = BacktestEngine(datastore=datastore, exchange_api=exchange_api)
        
        if not SKOPT_AVAILABLE:
            logger.warning("scikit-optimize not available. Install with 'pip install scikit-optimize'")
    
    def optimize(self, strategy_class, param_space, symbols, timeframes, 
                n_calls=50, n_initial_points=10, noise=0.1, 
                fitness_metric='sharpe_ratio', start_date=None, end_date=None, 
                verbose=True, save_results=True):
        """
        Run Bayesian optimization.
        
        Args:
            strategy_class (class): Strategy class to optimize
            param_space (dict): Parameter search space definition
            symbols (list): Symbols to use for backtesting
            timeframes (list): Timeframes to use for backtesting
            n_calls (int): Number of optimization iterations
            n_initial_points (int): Number of initial random explorations
            noise (float): Expected noise in the objective function
            fitness_metric (str): Metric to optimize
            start_date (str): Start date for backtesting (YYYY-MM-DD)
            end_date (str): End date for backtesting (YYYY-MM-DD)
            verbose (bool): Whether to print progress information
            save_results (bool): Whether to save optimization results
            
        Returns:
            dict: Optimization results
        """
        if not SKOPT_AVAILABLE:
            logger.error("Cannot perform Bayesian optimization: scikit-optimize not available")
            return None
        
        try:
            logger.info(f"Starting Bayesian optimization for {strategy_class.__name__}")
            
            # Convert dates to timestamps if provided
            start_timestamp = None
            end_timestamp = None
            
            if start_date:
                start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            
            if end_date:
                end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            # Fetch historical data for backtesting
            data = {}
            for symbol in symbols:
                data[symbol] = {}
                for timeframe in timeframes:
                    # Fetch data from datastore or exchange
                    if self.datastore:
                        historical_data = self.datastore.get_market_data(
                            symbol, timeframe, start_time=start_timestamp, end_time=end_timestamp
                        )
                    else:
                        # Fetch from exchange directly
                        candles = self.exchange_api.get_candles(
                            symbol, timeframe, start_time=start_timestamp, end_time=end_timestamp
                        )
                        
                        # Convert to DataFrame
                        historical_data = pd.DataFrame(
                            candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        
                        # Set timestamp as index
                        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms')
                        historical_data.set_index('timestamp', inplace=True)
                    
                    if not historical_data.empty:
                        data[symbol][timeframe] = historical_data
                    else:
                        logger.warning(f"No historical data for {symbol} {timeframe}, skipping")
            
            # Convert parameter space to skopt format
            dimensions = []
            dimension_names = []
            
            for param_name, param_def in param_space.items():
                if isinstance(param_def, list):
                    # Categorical parameter
                    dimensions.append(Categorical(param_def, name=param_name))
                    dimension_names.append(param_name)
                elif isinstance(param_def, tuple) and len(param_def) == 2:
                    # Numerical parameter
                    low, high = param_def
                    if isinstance(low, int) and isinstance(high, int):
                        dimensions.append(Integer(low, high, name=param_name))
                    else:
                        dimensions.append(Real(low, high, name=param_name))
                    dimension_names.append(param_name)
                else:
                    logger.warning(f"Invalid parameter definition for {param_name}: {param_def}")
            
            # Define objective function for optimization
            @use_named_args(dimensions=dimensions)
            def objective(**params):
                try:
                    # Create strategy with current parameters
                    current_params = {name: params[name] for name in dimension_names}
                    strategy = strategy_class(timeframes=timeframes, symbols=symbols, params=current_params)
                    
                    # Run backtest
                    backtest_results = self.backtest_engine.run_backtest(
                        strategies=[strategy],
                        data=data,
                        symbols=symbols,
                        timeframes=timeframes,
                        initial_capital=TRADING_CONFIG.get('start_capital', 10000)
                    )
                    
                    # Calculate performance metrics
                    performance = calculate_performance_metrics(backtest_results)
                    
                    # Get fitness metric
                    fitness = performance.get(fitness_metric, 0)
                    
                    # For minimization, negate metrics where higher is better
                    if self._is_higher_better(fitness_metric):
                        fitness = -fitness
                    
                    if verbose:
                        logger.debug(f"Params: {current_params}, Fitness: {-fitness if self._is_higher_better(fitness_metric) else fitness}")
                    
                    return fitness
                    
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
                    handle_error(e, f"Failed to evaluate parameters: {params}")
                    
                    # Return a very poor fitness for failed evaluations
                    return 1000 if self._is_higher_better(fitness_metric) else -1000
            
            # Run Bayesian optimization
            start_time = time.time()
            
            result = gp_minimize(
                objective,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                noise=noise,
                verbose=verbose,
                random_state=42
            )
            
            elapsed_time = time.time() - start_time
            
            # Convert result to dictionary for easy access
            best_params = {dim.name: result.x[i] for i, dim in enumerate(dimensions)}
            
            # Fix the sign if necessary
            best_fitness = -result.fun if self._is_higher_better(fitness_metric) else result.fun
            
            logger.info(f"Bayesian optimization complete - Best {fitness_metric}: {best_fitness:.4f}")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
            
            # Save results if requested
            if save_results:
                self._save_optimization_results(
                    strategy_class.__name__, param_space, result, best_params, best_fitness,
                    symbols, timeframes, fitness_metric, n_calls, n_initial_points
                )
            
            # Collect and return the results
            optimization_result = {
                'strategy': strategy_class.__name__,
                'best_params': best_params,
                'best_fitness': best_fitness,
                'fitness_metric': fitness_metric,
                'n_calls': n_calls,
                'n_initial_points': n_initial_points,
                'all_evaluations': [
                    {'params': {dim.name: x[i] for i, dim in enumerate(dimensions)}, 
                     'fitness': -y if self._is_higher_better(fitness_metric) else y}
                    for x, y in zip(result.x_iters, result.func_vals)
                ],
                'convergence': result.func_vals,
                'elapsed_time': elapsed_time
            }
            
            return optimization_result
        
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            handle_error(e, "Bayesian optimization failed")
            return None
    
    def _is_higher_better(self, metric):
        """
        Determine if higher values of a metric are better.
        
        Args:
            metric (str): Performance metric
            
        Returns:
            bool: True if higher values are better, False otherwise
        """
        # Metrics where higher values are better
        higher_better = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio',
            'win_rate', 'profit_factor', 'expectancy', 'annual_return',
            'total_return', 'profit', 'total_profit'
        ]
        
        # Metrics where lower values are better
        lower_better = [
            'max_drawdown', 'max_drawdown_pct', 'drawdown', 'volatility',
            'downside_risk', 'var', 'cvar', 'turnover', 'loss', 'total_loss'
        ]
        
        return metric in higher_better
    
    def _save_optimization_results(self, strategy_name, param_space, result, best_params, best_fitness,
                                symbols, timeframes, fitness_metric, n_calls, n_initial_points):
        """
        Save optimization results to disk.
        
        Args:
            strategy_name (str): Name of the strategy
            param_space (dict): Parameter search space definition
            result: Optimization result object
            best_params (dict): Best parameters found
            best_fitness (float): Best fitness score
            symbols (list): Symbols used for backtesting
            timeframes (list): Timeframes used for backtesting
            fitness_metric (str): Fitness metric used
            n_calls (int): Number of optimization iterations
            n_initial_points (int): Number of initial random points
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = Path("data/optimization_results")
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_name}_bayesian_{fitness_metric}_{timestamp}.json"
            filepath = results_dir / filename
            
            # Prepare the results in a serializable format
            evaluations = []
            for x, y in zip(result.x_iters, result.func_vals):
                params = {dim.name: val for dim, val in zip(result.space.dimensions, x)}
                evaluations.append({
                    'params': params,
                    'fitness': float(-y if self._is_higher_better(fitness_metric) else y)
                })
            
            results_data = {
                'strategy': strategy_name,
                'timestamp': timestamp,
                'metric': fitness_metric,
                'param_space': self._serialize_param_space(param_space),
                'symbols': symbols,
                'timeframes': timeframes,
                'n_calls': n_calls,
                'n_initial_points': n_initial_points,
                'best_params': best_params,
                'best_fitness': float(best_fitness),
                'evaluations': evaluations,
                'convergence': [float(x) for x in result.func_vals]
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Saved Bayesian optimization results to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving Bayesian optimization results: {e}")
            handle_error(e, "Failed to save Bayesian optimization results")
    
    def _serialize_param_space(self, param_space):
        """
        Convert parameter space to a JSON-serializable format.
        
        Args:
            param_space (dict): Parameter search space
            
        Returns:
            dict: Serializable parameter space
        """
        serialized = {}
        
        for param, space in param_space.items():
            if isinstance(space, list):
                # Already serializable
                serialized[param] = space
            elif isinstance(space, tuple) and len(space) == 2:
                # Convert tuple to list
                serialized[param] = list(space)
            else:
                # Try to convert to string
                serialized[param] = str(space)
        
        return serialized
    
    def load_optimization_results(self, filepath):
        """
        Load previously saved optimization results.
        
        Args:
            filepath (str): Path to the results file
            
        Returns:
            dict: Optimization results
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded Bayesian optimization results from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading Bayesian optimization results: {e}")
            handle_error(e, "Failed to load Bayesian optimization results")
            return None
    
    def visualize_convergence(self, result, title=None):
        """
        Visualize optimization convergence.
        
        Args:
            result: Optimization result object or loaded results
            title (str): Plot title
            
        Returns:
            object: Plot object
        """
        if not SKOPT_AVAILABLE:
            logger.error("Cannot visualize results: scikit-optimize not available")
            return None
        
        try:
            # This is a placeholder for implementing visualization
            # In a real implementation, this would use skopt's plotting functions
            logger.warning("Visualization not fully implemented - requires matplotlib")
            
            # Check if result is a dict (loaded from file) or skopt result object
            if isinstance(result, dict):
                # Loaded from file
                convergence = result.get('convergence', [])
                
                # Fix sign for metrics where higher is better
                if self._is_higher_better(result.get('metric', 'sharpe_ratio')):
                    convergence = [-x for x in convergence]
                
                # Here we would create a plot
                print(f"Convergence values: {convergence}")
                
                # Return None as placeholder
                return None
            else:
                # skopt result object
                plot_title = title or "Convergence plot"
                
                # Here we would use plot_convergence from skopt
                print("Would plot convergence from skopt result object")
                
                # Return None as placeholder
                return None
            
        except Exception as e:
            logger.error(f"Error visualizing convergence: {e}")
            handle_error(e, "Failed to visualize convergence")
            return None
    
    def visualize_importance(self, result, title=None):
        """
        Visualize parameter importance.
        
        Args:
            result: Optimization result object or loaded results
            title (str): Plot title
            
        Returns:
            object: Plot object
        """
        if not SKOPT_AVAILABLE:
            logger.error("Cannot visualize results: scikit-optimize not available")
            return None
        
        try:
            # This is a placeholder for implementing visualization
            # In a real implementation, this would analyze parameter importance
            logger.warning("Parameter importance visualization not fully implemented")
            
            # Check if result is a dict (loaded from file) or skopt result object
            if isinstance(result, dict):
                # Loaded from file
                print("Would analyze parameter importance from loaded results")
                
                # Return None as placeholder
                return None
            else:
                # skopt result object - could use partial dependence plots
                print("Would analyze parameter importance from skopt result object")
                
                # Return None as placeholder
                return None
            
        except Exception as e:
            logger.error(f"Error visualizing parameter importance: {e}")
            handle_error(e, "Failed to visualize parameter importance")
            return None