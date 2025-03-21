"""
Genetic algorithm optimization module for the Goalkick Ltd Trading Bot.
Implements evolutionary optimization techniques for strategy parameters.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime
import json
import multiprocessing
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

from config.logging_config import get_logger
from config.bot_config import TRADING_CONFIG
from analysis.backtest.backtest_engine import BacktestEngine
from analysis.backtest.performance import calculate_performance_metrics
from src.utils.error_handling import handle_error

logger = get_logger("optimization.genetic_algorithm")

class GeneticOptimizer:
    """Class for optimizing trading strategies using genetic algorithms."""
    
    def __init__(self, datastore=None, exchange_api=None):
        """
        Initialize the GeneticOptimizer.
        
        Args:
            datastore: DataStore instance
            exchange_api: Exchange API instance
        """
        self.datastore = datastore
        self.exchange_api = exchange_api
        self.backtest_engine = BacktestEngine(datastore=datastore, exchange_api=exchange_api)
        self.optimization_history = []
    
    def optimize(self, strategy_class, param_ranges, symbols, timeframes, 
                 population_size=50, generations=10, tournament_size=3, 
                 mutation_rate=0.2, crossover_rate=0.8, elitism=0.1, 
                 fitness_metric='sharpe_ratio', start_date=None, end_date=None, 
                 max_workers=None, save_history=True):
        """
        Run genetic algorithm optimization.
        
        Args:
            strategy_class (class): Strategy class to optimize
            param_ranges (dict): Parameter ranges for optimization
            symbols (list): Symbols to use for backtesting
            timeframes (list): Timeframes to use for backtesting
            population_size (int): Size of the population
            generations (int): Number of generations to run
            tournament_size (int): Size of tournament selection
            mutation_rate (float): Probability of mutation
            crossover_rate (float): Probability of crossover
            elitism (float): Percentage of top performers to keep unchanged
            fitness_metric (str): Metric to optimize
            start_date (str): Start date for backtesting (YYYY-MM-DD)
            end_date (str): End date for backtesting (YYYY-MM-DD)
            max_workers (int): Maximum number of worker processes
            save_history (bool): Whether to save optimization history
            
        Returns:
            tuple: (best_individual, fitness_history, best_fitness_history)
        """
        try:
            logger.info(f"Starting genetic optimization for {strategy_class.__name__}")
            
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
            
            # Initialize population
            population = self._initialize_population(param_ranges, population_size)
            
            # Initialize history tracking
            fitness_history = []
            best_fitness_history = []
            best_individual = None
            best_fitness = float('-inf') if self._is_higher_better(fitness_metric) else float('inf')
            
            # Run genetic algorithm for specified number of generations
            for generation in range(generations):
                logger.info(f"Generation {generation+1}/{generations}")
                
                # Evaluate fitness of each individual in the population
                fitness_scores = self._evaluate_population(
                    population, strategy_class, symbols, timeframes, data, 
                    fitness_metric, max_workers
                )
                
                # Update history
                avg_fitness = np.mean(fitness_scores)
                fitness_history.append(avg_fitness)
                
                # Find best individual in current generation
                best_idx = np.argmax(fitness_scores) if self._is_higher_better(fitness_metric) else np.argmin(fitness_scores)
                generation_best = population[best_idx]
                generation_best_fitness = fitness_scores[best_idx]
                
                # Update best individual overall if better
                if self._is_higher_better(fitness_metric):
                    if generation_best_fitness > best_fitness:
                        best_individual = deepcopy(generation_best)
                        best_fitness = generation_best_fitness
                else:
                    if generation_best_fitness < best_fitness:
                        best_individual = deepcopy(generation_best)
                        best_fitness = generation_best_fitness
                
                best_fitness_history.append(best_fitness)
                
                logger.info(f"Generation {generation+1} - Avg Fitness: {avg_fitness:.4f}, Best Fitness: {best_fitness:.4f}")
                
                # If this is the last generation, skip evolution
                if generation == generations - 1:
                    break
                
                # Create new population through selection, crossover, and mutation
                population = self._evolve_population(
                    population, fitness_scores, param_ranges,
                    tournament_size, crossover_rate, mutation_rate, elitism
                )
            
            # Log final results
            logger.info(f"Genetic optimization complete - Best {fitness_metric}: {best_fitness:.4f}")
            logger.info(f"Best parameters: {best_individual}")
            
            # Save history if requested
            if save_history:
                self._save_optimization_history(
                    strategy_class.__name__, param_ranges, population, fitness_scores,
                    fitness_history, best_fitness_history, best_individual, best_fitness,
                    symbols, timeframes, fitness_metric
                )
            
            # Store history
            self.optimization_history.append({
                'strategy': strategy_class.__name__,
                'timestamp': datetime.now().isoformat(),
                'param_ranges': param_ranges,
                'symbols': symbols,
                'timeframes': timeframes,
                'metric': fitness_metric,
                'population_size': population_size,
                'generations': generations,
                'best_individual': best_individual,
                'best_fitness': best_fitness,
                'fitness_history': fitness_history,
                'best_fitness_history': best_fitness_history
            })
            
            return best_individual, fitness_history, best_fitness_history
        
        except Exception as e:
            logger.error(f"Error in genetic optimization: {e}")
            handle_error(e, "Genetic optimization failed")
            return None, [], []
    
    def _initialize_population(self, param_ranges, population_size):
        """
        Initialize a random population.
        
        Args:
            param_ranges (dict): Parameter ranges
            population_size (int): Size of the population
            
        Returns:
            list: List of individuals (parameter dictionaries)
        """
        population = []
        
        for _ in range(population_size):
            # Create individual with random parameters within ranges
            individual = {}
            
            for param, value_range in param_ranges.items():
                if isinstance(value_range, list):
                    # Discrete values
                    individual[param] = random.choice(value_range)
                elif isinstance(value_range, tuple) and len(value_range) == 2:
                    # Continuous range
                    min_val, max_val = value_range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer range
                        individual[param] = random.randint(min_val, max_val)
                    else:
                        # Float range
                        individual[param] = random.uniform(min_val, max_val)
                else:
                    # Default to first value if available
                    individual[param] = value_range[0] if isinstance(value_range, list) and value_range else None
            
            population.append(individual)
        
        return population
    
    def _evaluate_population(self, population, strategy_class, symbols, timeframes, data, fitness_metric, max_workers):
        """
        Evaluate fitness of all individuals in the population using parallel processing.
        
        Args:
            population (list): List of individuals
            strategy_class (class): Strategy class
            symbols (list): Symbols for backtesting
            timeframes (list): Timeframes for backtesting
            data (dict): Historical data
            fitness_metric (str): Metric to optimize
            max_workers (int): Maximum number of worker processes
            
        Returns:
            list: Fitness scores for each individual
        """
        fitness_scores = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            futures = []
            for individual in population:
                futures.append(
                    executor.submit(
                        self._evaluate_individual,
                        individual,
                        strategy_class,
                        symbols,
                        timeframes,
                        data,
                        fitness_metric
                    )
                )
            
            # Collect results
            for future in as_completed(futures):
                try:
                    fitness = future.result()
                    fitness_scores.append(fitness)
                except Exception as e:
                    logger.error(f"Error evaluating individual: {e}")
                    handle_error(e, "Individual evaluation failed")
                    
                    # Assign very poor fitness
                    if self._is_higher_better(fitness_metric):
                        fitness_scores.append(float('-inf'))
                    else:
                        fitness_scores.append(float('inf'))
        
        return fitness_scores
    
    def _evaluate_individual(self, individual, strategy_class, symbols, timeframes, data, fitness_metric):
        """
        Evaluate fitness of an individual.
        
        Args:
            individual (dict): Individual parameters
            strategy_class (class): Strategy class
            symbols (list): Symbols for backtesting
            timeframes (list): Timeframes for backtesting
            data (dict): Historical data
            fitness_metric (str): Metric to optimize
            
        Returns:
            float: Fitness score
        """
        try:
            # Create strategy instance with parameters
            strategy = strategy_class(timeframes=timeframes, symbols=symbols, params=individual)
            
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
            
            return fitness
        
        except Exception as e:
            logger.error(f"Error evaluating individual {individual}: {e}")
            handle_error(e, f"Individual evaluation failed for {individual}")
            
            # Return very poor fitness
            if self._is_higher_better(fitness_metric):
                return float('-inf')
            else:
                return float('inf')
    
    def _evolve_population(self, population, fitness_scores, param_ranges, 
                          tournament_size, crossover_rate, mutation_rate, elitism):
        """
        Create a new population through selection, crossover, and mutation.
        
        Args:
            population (list): Current population
            fitness_scores (list): Fitness scores for current population
            param_ranges (dict): Parameter ranges
            tournament_size (int): Size of tournament selection
            crossover_rate (float): Probability of crossover
            mutation_rate (float): Probability of mutation
            elitism (float): Percentage of top performers to keep unchanged
            
        Returns:
            list: New population
        """
        population_size = len(population)
        new_population = []
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)
        if self._is_higher_better(fitness_metric := 'sharpe_ratio'):  # Default to sharpe ratio
            sorted_indices = sorted_indices[::-1]  # Reverse for higher-is-better
        
        # Apply elitism - keep top performers
        elites_count = max(1, int(population_size * elitism))
        for i in range(elites_count):
            elite_idx = sorted_indices[i]
            new_population.append(deepcopy(population[elite_idx]))
        
        # Fill the rest of the population with crossover and mutation
        while len(new_population) < population_size:
            # Select parents through tournament selection
            parent1 = self._tournament_selection(population, fitness_scores, tournament_size)
            parent2 = self._tournament_selection(population, fitness_scores, tournament_size)
            
            # Perform crossover with probability crossover_rate
            if random.random() < crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            # Perform mutation with probability mutation_rate
            if random.random() < mutation_rate:
                child1 = self._mutate(child1, param_ranges)
            
            if random.random() < mutation_rate:
                child2 = self._mutate(child2, param_ranges)
            
            # Add children to new population
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        return new_population
    
    def _tournament_selection(self, population, fitness_scores, tournament_size):
        """
        Select individual through tournament selection.
        
        Args:
            population (list): Current population
            fitness_scores (list): Fitness scores
            tournament_size (int): Size of tournament
            
        Returns:
            dict: Selected individual
        """
        # Select random individuals for tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Find best individual in tournament
        if self._is_higher_better(fitness_metric := 'sharpe_ratio'):  # Default to sharpe ratio
            best_idx = tournament_indices[np.argmax(tournament_fitness)]
        else:
            best_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return population[best_idx]
    
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Args:
            parent1 (dict): First parent
            parent2 (dict): Second parent
            
        Returns:
            tuple: (child1, child2)
        """
        child1 = {}
        child2 = {}
        
        # For each parameter, randomly choose from which parent to inherit
        for param in parent1.keys():
            if random.random() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def _mutate(self, individual, param_ranges):
        """
        Mutate an individual.
        
        Args:
            individual (dict): Individual to mutate
            param_ranges (dict): Parameter ranges
            
        Returns:
            dict: Mutated individual
        """
        # Clone the individual
        mutated = deepcopy(individual)
        
        # Select a random parameter to mutate
        param = random.choice(list(param_ranges.keys()))
        value_range = param_ranges[param]
        
        # Determine new value based on parameter type
        if isinstance(value_range, list):
            # Discrete values
            mutated[param] = random.choice(value_range)
        elif isinstance(value_range, tuple) and len(value_range) == 2:
            # Continuous range
            min_val, max_val = value_range
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer range
                mutated[param] = random.randint(min_val, max_val)
            else:
                # Float range
                mutated[param] = random.uniform(min_val, max_val)
        
        return mutated
    
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
    
    def _save_optimization_history(self, strategy_name, param_ranges, population, fitness_scores,
                                 fitness_history, best_fitness_history, best_individual, best_fitness,
                                 symbols, timeframes, fitness_metric):
        """
        Save optimization history to disk.
        
        Args:
            strategy_name (str): Name of the strategy
            param_ranges (dict): Parameter ranges
            population (list): Final population
            fitness_scores (list): Fitness scores of final population
            fitness_history (list): Average fitness by generation
            best_fitness_history (list): Best fitness by generation
            best_individual (dict): Best individual found
            best_fitness (float): Best fitness score
            symbols (list): Symbols used
            timeframes (list): Timeframes used
            fitness_metric (str): Fitness metric used
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = Path("data/optimization_results")
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_name}_genetic_{fitness_metric}_{timestamp}.json"
            filepath = results_dir / filename
            
            # Prepare population in a serializable format
            serializable_population = []
            for individual, fitness in zip(population, fitness_scores):
                serializable_population.append({
                    'parameters': individual,
                    'fitness': float(fitness) if not pd.isna(fitness) else None
                })
            
            # Create results dictionary
            results_data = {
                'strategy': strategy_name,
                'timestamp': timestamp,
                'metric': fitness_metric,
                'param_ranges': param_ranges,
                'symbols': symbols,
                'timeframes': timeframes,
                'population': serializable_population,
                'fitness_history': [float(x) for x in fitness_history],
                'best_fitness_history': [float(x) for x in best_fitness_history],
                'best_individual': best_individual,
                'best_fitness': float(best_fitness)
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Saved genetic optimization results to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving genetic optimization results: {e}")
            handle_error(e, "Failed to save genetic optimization results")
    
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
            
            logger.info(f"Loaded genetic optimization results from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading genetic optimization results: {e}")
            handle_error(e, "Failed to load genetic optimization results")
            return None
    
    def visualize_optimization_progress(self, fitness_history, best_fitness_history):
        """
        Visualize optimization progress.
        
        Args:
            fitness_history (list): Average fitness by generation
            best_fitness_history (list): Best fitness by generation
            
        Returns:
            bool: True if visualized successfully, False otherwise
        """
        try:
            # This is a placeholder for implementing visualization
            # In a real implementation, this would create a plot of fitness over generations
            logger.warning("Visualization of optimization progress not implemented")
            
            # For now, just print metrics
            for i, (avg_fitness, best_fitness) in enumerate(zip(fitness_history, best_fitness_history)):
                print(f"Generation {i+1}: Avg Fitness = {avg_fitness:.4f}, Best Fitness = {best_fitness:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing optimization progress: {e}")
            handle_error(e, "Failed to visualize optimization progress")
            return False