"""
Parameter tuning module for the Goalkick Ltd Trading Bot.
Implements optimization techniques for strategy parameters.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from copy import deepcopy
from pathlib import Path

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG, BotMode
from config.strategy_params import TREND_FOLLOWING_PARAMS, MEAN_REVERSION_PARAMS, ARBITRAGE_PARAMS, STATISTICAL_PARAMS
from src.strategies.base_strategy import Strategy, StrategyManager
from analysis.backtest.backtest_engine import BacktestEngine
from analysis.backtest.performance import calculate_performance_metrics
from src.utils.error_handling import handle_error

logger = get_logger("optimization.parameter_tuning")

class ParameterOptimizer:
    """Class for optimizing strategy parameters."""
    
    def __init__(self, datastore=None, exchange_api=None):
        """
        Initialize the ParameterOptimizer.
        
        Args:
            datastore: DataStore instance
            exchange_api: Exchange API instance
        """
        self.datastore = datastore
        self.exchange_api = exchange_api
        self.backtest_engine = BacktestEngine(datastore=datastore, exchange_api=exchange_api)
        self.strategy_manager = StrategyManager()
        self.optimization_results = {}
        
    def optimize_strategy(self, strategy_name, strategy_class, symbols, timeframes, param_ranges, 
                          start_date=None, end_date=None, metric='sharpe_ratio', method='grid', 
                          max_workers=None, save_results=True):
        """
        Optimize parameters for a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_class (class): Strategy class
            symbols (list): List of symbols to optimize for
            timeframes (list): List of timeframes to optimize for
            param_ranges (dict): Dictionary of parameter ranges to optimize
            start_date (str): Start date for optimization (YYYY-MM-DD)
            end_date (str): End date for optimization (YYYY-MM-DD)
            metric (str): Performance metric to optimize for
            method (str): Optimization method (grid, random, bayesian)
            max_workers (int): Maximum number of worker processes
            save_results (bool): Whether to save optimization results
            
        Returns:
            tuple: (best_params, best_score, all_results)
        """
        try:
            logger.info(f"Starting parameter optimization for {strategy_name}")
            
            # Prepare parameter combinations
            parameter_combinations = self._generate_parameter_combinations(param_ranges, method)
            
            logger.info(f"Testing {len(parameter_combinations)} parameter combinations for {strategy_name}")
            
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
            
            # Determine the number of worker processes
            if max_workers is None:
                max_workers = max(1, multiprocessing.cpu_count() - 1)
            
            # Use Process Pool for parallel optimization
            results = []
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit jobs
                futures = []
                for params in parameter_combinations:
                    futures.append(
                        executor.submit(
                            self._evaluate_parameters,
                            strategy_name,
                            strategy_class,
                            params,
                            symbols,
                            timeframes,
                            data,
                            metric
                        )
                    )
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        params, score = future.result()
                        results.append((params, score))
                        
                        # Log progress periodically
                        if len(results) % 10 == 0 or len(results) == len(parameter_combinations):
                            logger.info(f"Completed {len(results)}/{len(parameter_combinations)} parameter combinations")
                    except Exception as e:
                        logger.error(f"Error in parameter evaluation: {e}")
                        handle_error(e, "Parameter evaluation failed")
            
            # Sort results based on optimization metric
            results.sort(key=lambda x: x[1], reverse=self._is_higher_better(metric))
            
            # Get best parameters
            best_params, best_score = results[0] if results else (None, None)
            
            # Log best parameters
            logger.info(f"Best parameters for {strategy_name}: {best_params} with {metric}={best_score:.4f}")
            
            # Save results if requested
            if save_results:
                self._save_optimization_results(
                    strategy_name, param_ranges, results, metric, symbols, timeframes
                )
            
            # Store results
            self.optimization_results[strategy_name] = {
                'best_params': best_params,
                'best_score': best_score,
                'all_results': results,
                'optimization_timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'timeframes': timeframes,
                'metric': metric,
                'param_ranges': param_ranges
            }
            
            return best_params, best_score, results
        
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            handle_error(e, "Strategy parameter optimization failed")
            return None, None, []
    
    def _evaluate_parameters(self, strategy_name, strategy_class, params, symbols, timeframes, data, metric):
        """
        Evaluate a set of parameters using backtesting.
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_class (class): Strategy class
            params (dict): Parameters to evaluate
            symbols (list): List of symbols
            timeframes (list): List of timeframes
            data (dict): Historical data (symbol -> timeframe -> DataFrame)
            metric (str): Performance metric
            
        Returns:
            tuple: (params, score)
        """
        try:
            # Create strategy instance with parameters
            strategy = strategy_class(timeframes=timeframes, symbols=symbols, params=params)
            
            # Add strategy to manager
            strategy_manager = StrategyManager()
            strategy_id = strategy_manager.add_strategy(strategy)
            
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
            
            # Get optimization metric
            score = performance.get(metric, 0)
            
            return params, score
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {e}")
            handle_error(e, f"Parameter evaluation failed for {params}")
            
            # Return very poor score to avoid these parameters
            if self._is_higher_better(metric):
                return params, float('-inf')
            else:
                return params, float('inf')
    
    def _generate_parameter_combinations(self, param_ranges, method='grid', num_samples=100):
        """
        Generate parameter combinations for optimization.
        
        Args:
            param_ranges (dict): Dictionary of parameter ranges
            method (str): Method for generating combinations (grid, random)
            num_samples (int): Number of random samples (if method='random')
            
        Returns:
            list: List of parameter dictionaries
        """
        if method == 'grid':
            # Generate grid of all combinations
            keys = list(param_ranges.keys())
            values = list(param_ranges.values())
            
            combinations = []
            for combo in itertools.product(*values):
                combinations.append(dict(zip(keys, combo)))
            
            return combinations
        
        elif method == 'random':
            # Generate random samples from parameter ranges
            combinations = []
            
            for _ in range(num_samples):
                params = {}
                for key, values in param_ranges.items():
                    if isinstance(values, list):
                        # Discrete values
                        params[key] = np.random.choice(values)
                    elif isinstance(values, tuple) and len(values) == 2:
                        # Continuous range
                        min_val, max_val = values
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            # Integer range
                            params[key] = np.random.randint(min_val, max_val + 1)
                        else:
                            # Float range
                            params[key] = np.random.uniform(min_val, max_val)
                    else:
                        # Default to first value
                        params[key] = values[0] if isinstance(values, list) and values else None
                
                combinations.append(params)
            
            return combinations
        
        elif method == 'bayesian':
            # This would use Bayesian optimization (requires skopt)
            # For now, default to grid search
            logger.warning("Bayesian optimization not implemented, using grid search instead")
            return self._generate_parameter_combinations(param_ranges, 'grid')
        
        else:
            logger.warning(f"Unknown optimization method: {method}, using grid search")
            return self._generate_parameter_combinations(param_ranges, 'grid')
    
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
    
    def _save_optimization_results(self, strategy_name, param_ranges, results, metric, symbols, timeframes):
        """
        Save optimization results to disk.
        
        Args:
            strategy_name (str): Name of the strategy
            param_ranges (dict): Parameter ranges that were optimized
            results (list): List of (params, score) tuples
            metric (str): Performance metric used
            symbols (list): List of symbols optimized for
            timeframes (list): List of timeframes optimized for
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = Path("data/optimization_results")
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_name}_{metric}_{timestamp}.json"
            filepath = results_dir / filename
            
            # Prepare results in a serializable format
            serializable_results = []
            for params, score in results:
                serializable_results.append({
                    'parameters': params,
                    'score': float(score) if not pd.isna(score) else None
                })
            
            # Create results dictionary
            results_data = {
                'strategy': strategy_name,
                'timestamp': timestamp,
                'metric': metric,
                'param_ranges': param_ranges,
                'symbols': symbols,
                'timeframes': timeframes,
                'results': serializable_results
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Saved optimization results to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
            handle_error(e, "Failed to save optimization results")
    
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
            
            logger.info(f"Loaded optimization results from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading optimization results: {e}")
            handle_error(e, "Failed to load optimization results")
            return None
    
    def get_default_param_ranges(self, strategy_type):
        """
        Get default parameter ranges for a strategy type.
        
        Args:
            strategy_type (str): Type of strategy
            
        Returns:
            dict: Default parameter ranges
        """
        if strategy_type == 'trend_following':
            return {
                'moving_average': {
                    'short_window': [5, 7, 9, 11, 13],
                    'long_window': [21, 25, 30, 35, 40],
                    'signal_window': [5, 7, 9, 11]
                },
                'macd': {
                    'fast_length': [8, 10, 12, 14, 16],
                    'slow_length': [21, 24, 26, 28, 30],
                    'signal_length': [7, 9, 11, 13]
                },
                'parabolic_sar': {
                    'step': [0.01, 0.015, 0.02, 0.025, 0.03],
                    'max_step': [0.1, 0.15, 0.2, 0.25, 0.3]
                },
                'adx': {
                    'length': [10, 12, 14, 16, 18],
                    'threshold': [20, 22, 25, 28, 30]
                },
                'ichimoku': {
                    'tenkan_period': [7, 9, 11, 13],
                    'kijun_period': [22, 24, 26, 28, 30],
                    'senkou_span_b_period': [44, 48, 52, 56, 60]
                }
            }
        elif strategy_type == 'mean_reversion':
            return {
                'rsi': {
                    'length': [10, 12, 14, 16, 18],
                    'overbought': [65, 70, 75, 80],
                    'oversold': [20, 25, 30, 35]
                },
                'bollinger_bands': {
                    'length': [15, 18, 20, 22, 25],
                    'std_dev': [1.8, 2.0, 2.2, 2.4, 2.6]
                },
                'stochastic': {
                    'k_period': [10, 12, 14, 16, 18],
                    'd_period': [3, 4, 5, 6],
                    'overbought': [75, 80, 85],
                    'oversold': [15, 20, 25]
                }
            }
        elif strategy_type == 'arbitrage':
            return {
                'triangular': {
                    'min_profit_pct': [0.2, 0.3, 0.4, 0.5, 0.6],
                    'max_slippage_pct': [0.05, 0.1, 0.15, 0.2]
                },
                'statistical': {
                    'z_score_threshold': [1.5, 1.8, 2.0, 2.2, 2.5],
                    'correlation_threshold': [0.7, 0.75, 0.8, 0.85, 0.9]
                }
            }
        elif strategy_type == 'statistical':
            return {
                'linear_regression': {
                    'lookback_period': [80, 90, 100, 110, 120],
                    'prediction_periods': [15, 20, 25, 30]
                },
                'machine_learning': {
                    'training_window': [800, 900, 1000, 1100, 1200]
                },
                'kalman_filter': {
                    'process_variance': [1e-6, 1e-5, 1e-4, 1e-3],
                    'measurement_variance': [0.05, 0.1, 0.15, 0.2]
                }
            }
        else:
            return {}

    def optimize_strategy_hyperparameters(self, strategy_name, model_type, symbols, timeframes, 
                                          start_date=None, end_date=None, n_trials=50):
        """
        Optimize hyperparameters for machine learning models in strategies.
        
        Args:
            strategy_name (str): Name of the strategy
            model_type (str): Type of machine learning model
            symbols (list): List of symbols to optimize for
            timeframes (list): List of timeframes to optimize for
            start_date (str): Start date for optimization (YYYY-MM-DD)
            end_date (str): End date for optimization (YYYY-MM-DD)
            n_trials (int): Number of trials for optimization
            
        Returns:
            dict: Best hyperparameters
        """
        try:
            # This is a placeholder for implementing Bayesian hyperparameter optimization
            # with Optuna or a similar framework for ML models
            logger.warning("ML hyperparameter optimization is not fully implemented")
            
            # For now, return a default set of hyperparameters
            if model_type == 'xgboost':
                return {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'min_child_weight': 1
                }
            elif model_type == 'random_forest':
                return {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt'
                }
            elif model_type == 'lightgbm':
                return {
                    'n_estimators': 100,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'min_data_in_leaf': 20
                }
            else:
                return {}
        
        except Exception as e:
            logger.error(f"Error optimizing ML hyperparameters: {e}")
            handle_error(e, "ML hyperparameter optimization failed")
            return {}
    
    def walk_forward_optimization(self, strategy_class, symbols, timeframes, param_ranges,
                                 start_date, end_date, window_size=90, step_size=30, metric='sharpe_ratio'):
        """
        Perform walk-forward optimization to prevent overfitting.
        
        Args:
            strategy_class (class): Strategy class
            symbols (list): List of symbols to optimize for
            timeframes (list): List of timeframes to optimize for
            param_ranges (dict): Dictionary of parameter ranges to optimize
            start_date (str): Start date for optimization (YYYY-MM-DD)
            end_date (str): End date for optimization (YYYY-MM-DD)
            window_size (int): Size of the optimization window in days
            step_size (int): Step size in days
            metric (str): Performance metric to optimize for
            
        Returns:
            dict: Dictionary of window -> best parameters
        """
        try:
            logger.info(f"Starting walk-forward optimization for {strategy_class.__name__}")
            
            # Parse dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Generate optimization windows
            windows = []
            current_start = start_dt
            
            while current_start < end_dt:
                current_end = min(current_start + pd.Timedelta(days=window_size), end_dt)
                out_of_sample_end = min(current_end + pd.Timedelta(days=step_size), end_dt)
                
                # Add window
                windows.append({
                    'in_sample_start': current_start.strftime("%Y-%m-%d"),
                    'in_sample_end': current_end.strftime("%Y-%m-%d"),
                    'out_of_sample_start': current_end.strftime("%Y-%m-%d"),
                    'out_of_sample_end': out_of_sample_end.strftime("%Y-%m-%d")
                })
                
                # Move to next window
                current_start = current_start + pd.Timedelta(days=step_size)
            
            logger.info(f"Created {len(windows)} windows for walk-forward optimization")
            
            # Perform optimization for each window
            results = {}
            
            for i, window in enumerate(windows):
                logger.info(f"Optimizing window {i+1}/{len(windows)}: {window['in_sample_start']} to {window['in_sample_end']}")
                
                # Run optimization on in-sample data
                best_params, best_score, _ = self.optimize_strategy(
                    f"{strategy_class.__name__}_window_{i+1}",
                    strategy_class,
                    symbols,
                    timeframes,
                    param_ranges,
                    start_date=window['in_sample_start'],
                    end_date=window['in_sample_end'],
                    metric=metric,
                    save_results=False
                )
                
                # Create strategy with best parameters
                strategy = strategy_class(timeframes=timeframes, symbols=symbols, params=best_params)
                
                # Run backtest on out-of-sample data
                backtest_results = self.backtest_engine.run_backtest(
                    strategies=[strategy],
                    symbols=symbols,
                    timeframes=timeframes,
                    start_date=window['out_of_sample_start'],
                    end_date=window['out_of_sample_end'],
                    initial_capital=TRADING_CONFIG.get('start_capital', 10000)
                )
                
                # Calculate performance metrics
                performance = calculate_performance_metrics(backtest_results)
                out_of_sample_score = performance.get(metric, 0)
                
                # Store results
                results[i] = {
                    'window': window,
                    'best_params': best_params,
                    'in_sample_score': best_score,
                    'out_of_sample_score': out_of_sample_score
                }
                
                logger.info(f"Window {i+1} results - In-sample {metric}: {best_score:.4f}, Out-of-sample {metric}: {out_of_sample_score:.4f}")
            
            # Analyze robustness across windows
            in_sample_scores = [window['in_sample_score'] for window in results.values()]
            out_of_sample_scores = [window['out_of_sample_score'] for window in results.values()]
            
            in_sample_mean = np.mean(in_sample_scores)
            out_of_sample_mean = np.mean(out_of_sample_scores)
            
            logger.info(f"Walk-forward optimization complete - Average {metric}: In-sample {in_sample_mean:.4f}, Out-of-sample {out_of_sample_mean:.4f}")
            
            # Save results
            self._save_walk_forward_results(
                strategy_class.__name__, param_ranges, results, metric, symbols, timeframes
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            handle_error(e, "Walk-forward optimization failed")
            return {}
    
    def _save_walk_forward_results(self, strategy_name, param_ranges, results, metric, symbols, timeframes):
        """
        Save walk-forward optimization results to disk.
        
        Args:
            strategy_name (str): Name of the strategy
            param_ranges (dict): Parameter ranges that were optimized
            results (dict): Dictionary of window -> results
            metric (str): Performance metric used
            symbols (list): List of symbols optimized for
            timeframes (list): List of timeframes optimized for
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = Path("data/optimization_results")
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_name}_walkforward_{metric}_{timestamp}.json"
            filepath = results_dir / filename
            
            # Create results dictionary
            results_data = {
                'strategy': strategy_name,
                'timestamp': timestamp,
                'metric': metric,
                'param_ranges': param_ranges,
                'symbols': symbols,
                'timeframes': timeframes,
                'results': results
            }
            
            # Ensure results are JSON serializable
            for window_id, window_results in results.items():
                if isinstance(window_results['best_params'], np.ndarray):
                    results_data['results'][window_id]['best_params'] = window_results['best_params'].tolist()
                if isinstance(window_results['in_sample_score'], np.ndarray):
                    results_data['results'][window_id]['in_sample_score'] = float(window_results['in_sample_score'])
                if isinstance(window_results['out_of_sample_score'], np.ndarray):
                    results_data['results'][window_id]['out_of_sample_score'] = float(window_results['out_of_sample_score'])
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Saved walk-forward optimization results to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving walk-forward results: {e}")
            handle_error(e, "Failed to save walk-forward results")
    
    def analyze_optimization_results(self, results):
        """
        Analyze optimization results to find patterns and insights.
        
        Args:
            results (list): List of (params, score) tuples
            
        Returns:
            dict: Analysis results
        """
        try:
            if not results:
                return {}
            
            # Extract parameters and scores
            params_list = [r[0] for r in results]
            scores = [r[1] for r in results]
            
            # Create DataFrame for analysis
            df = pd.DataFrame(params_list)
            df['score'] = scores
            
            # Calculate parameter importance
            importance = {}
            for param in df.columns:
                if param == 'score':
                    continue
                
                # Calculate correlation with score
                correlation = df[param].corr(df['score'])
                importance[param] = abs(correlation)
            
            # Sort by importance
            importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
            
            # Calculate optimal ranges for each parameter
            top_n = max(int(len(df) * 0.1), 10)  # Top 10% or at least 10
            top_performers = df.nlargest(top_n, 'score')
            
            optimal_ranges = {}
            for param in df.columns:
                if param == 'score':
                    continue
                
                values = top_performers[param]
                optimal_ranges[param] = {
                    'min': values.min(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'median': values.median()
                }
            
            # Overall statistics
            stats = {
                'best_score': max(scores),
                'worst_score': min(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'num_combinations': len(results)
            }
            
            return {
                'parameter_importance': importance,
                'optimal_ranges': optimal_ranges,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing optimization results: {e}")
            handle_error(e, "Failed to analyze optimization results")
            return {}