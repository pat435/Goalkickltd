"""
Error handling utilities for the Goalkick Ltd Trading Bot.
"""

import sys
import traceback
import threading
import time
from datetime import datetime
import logging
import functools
import inspect
import json
import requests

from config.logging_config import get_logger
from config.bot_config import NOTIFICATION_CONFIG

logger = get_logger("utils.error_handling")


# Custom exception classes
class BotError(Exception):
    """Base exception for all trading bot errors."""
    pass
# Add this with the other custom exception classes (around line 20-30)
class DataStoreError(BotError):
    """Exception for database/storage-related errors."""
    pass
class ExchangeError(BotError):
    """Exception for exchange-related errors."""
    pass

class DataError(BotError):
    """Exception for data-related errors."""
    pass

class ConfigError(BotError):
    """Exception for configuration-related errors."""
    pass

class NetworkError(BotError):
    """Exception for network-related errors."""
    pass

class StrategyError(BotError):
    """Exception for strategy-related errors."""
    pass

class ExecutionError(BotError):
    """Exception for execution-related errors."""
    pass

# Error registry to track unique errors and their frequency
error_registry = {}
error_lock = threading.RLock()

# Error severity levels
class ErrorSeverity:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

def handle_error(error, message=None, severity=ErrorSeverity.MEDIUM, notify=True, retry_func=None, retry_args=None, max_retries=3, retry_delay=5):
    """
    Handle an error with appropriate logging, notification, and retry logic.
    
    Args:
        error (Exception): The exception that occurred
        message (str): Additional context about the error
        severity (str): Error severity level
        notify (bool): Whether to send a notification
        retry_func (callable): Function to retry (optional)
        retry_args (dict): Arguments for retry function (optional)
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        bool: True if error was handled or retry succeeded, False otherwise
    """
    # Get error details
    error_type = type(error).__name__
    error_message = str(error)
    
    if message:
        error_msg = f"{message}: {error_type} - {error_message}"
    else:
        error_msg = f"{error_type} - {error_message}"
    
    # Get traceback
    tb = traceback.format_exc()
    
    # Get caller information
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    function = frame.f_code.co_name
    line_number = frame.f_lineno
    
    # Create error record
    timestamp = datetime.now().isoformat()
    error_record = {
        'timestamp': timestamp,
        'type': error_type,
        'message': error_message,
        'context': message,
        'severity': severity,
        'file': filename,
        'function': function,
        'line': line_number,
        'traceback': tb
    }
    
    # Log the error
    if severity == ErrorSeverity.CRITICAL:
        logger.critical(error_msg)
        logger.critical(f"Traceback: {tb}")
    elif severity == ErrorSeverity.HIGH:
        logger.error(error_msg)
        logger.error(f"Traceback: {tb}")
    elif severity == ErrorSeverity.MEDIUM:
        logger.error(error_msg)
    else:
        logger.warning(error_msg)
    
    # Register the error
    _register_error(error_record)
    
    # Send notification if enabled
    if notify and _should_notify(error_record):
        _send_error_notification(error_record)
    
    # Retry logic if provided
    if retry_func and callable(retry_func):
        return _retry_operation(retry_func, retry_args or {}, max_retries, retry_delay, error_record)
    
    return True

def send_error_notification(error_message):
    """
    Send error notification via Telegram.
    
    Args:
        error_message (str): Error message to send
    """
    try:
        if not NOTIFICATION_CONFIG.get("TELEGRAM_TOKEN") or not NOTIFICATION_CONFIG.get("TELEGRAM_CHAT_ID"):
            return
            
        url = f"https://api.telegram.org/bot{NOTIFICATION_CONFIG['TELEGRAM_TOKEN']}/sendMessage"
        data = {
            "chat_id": NOTIFICATION_CONFIG["TELEGRAM_CHAT_ID"],
            "text": f"Trading Bot Error:\n{error_message}",
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        
    except Exception as e:
        # Log locally since notification failed
        logger.error(f"Failed to send notification: {str(e)}")

def handle_error(error, context=""):
    """
    Handle and log errors, optifying if configured.
    
    Args:
        error (Exception): The error that occurred
        context (str): Context where the error occurred
    """
    error_message = f"{context}: {error.__class__.__name__} - {str(error)}"
    logger.error(error_message)
    
    if NOTIFICATION_CONFIG.get("NOTIFY_ON_ERROR", False):
        send_error_notification(error_message)
    
    # Log full traceback for debugging
    logger.debug(f"Full traceback:\n{''.join(traceback.format_tb(error.__traceback__))}")

def _register_error(error_record):
    """
    Register an error in the error registry.
    
    Args:
        error_record (dict): Error record to register
    """
    global error_registry
    
    with error_lock:
        # Create a key from error type, message, and context
        error_key = f"{error_record['type']}:{error_record['message']}:{error_record['context']}"
        
        if error_key in error_registry:
            # Update existing error record
            error_registry[error_key]['count'] += 1
            error_registry[error_key]['last_occurred'] = error_record['timestamp']
            
            # Only keep the most recent occurrences
            occurrences = error_registry[error_key]['occurrences']
            occurrences.append(error_record)
            if len(occurrences) > 10:  # Keep last 10 occurrences
                error_registry[error_key]['occurrences'] = occurrences[-10:]
        else:
            # Create new error record
            error_registry[error_key] = {
                'type': error_record['type'],
                'message': error_record['message'],
                'context': error_record['context'],
                'severity': error_record['severity'],
                'first_occurred': error_record['timestamp'],
                'last_occurred': error_record['timestamp'],
                'count': 1,
                'occurrences': [error_record]
            }

def _should_notify(error_record):
    """
    Determine if a notification should be sent for an error.
    
    Args:
        error_record (dict): Error record
        
    Returns:
        bool: True if notification should be sent, False otherwise
    """
    # Check if notifications are enabled
    if not NOTIFICATION_CONFIG.get('enable_telegram', False) or not NOTIFICATION_CONFIG.get('notify_on_error', True):
        return False
    
    # Check error severity
    if error_record['severity'] == ErrorSeverity.LOW:
        return False
    
    # Check quiet hours
    quiet_hours_start = NOTIFICATION_CONFIG.get('quiet_hours_start')
    quiet_hours_end = NOTIFICATION_CONFIG.get('quiet_hours_end')
    
    if quiet_hours_start and quiet_hours_end:
        now = datetime.now().time()
        if quiet_hours_start < quiet_hours_end:
            if quiet_hours_start <= now <= quiet_hours_end:
                # Only send notifications for critical errors during quiet hours
                return error_record['severity'] == ErrorSeverity.CRITICAL
        else:
            # Quiet hours span midnight
            if now >= quiet_hours_start or now <= quiet_hours_end:
                # Only send notifications for critical errors during quiet hours
                return error_record['severity'] == ErrorSeverity.CRITICAL
    
    # Check error frequency to avoid notification spam
    with error_lock:
        error_key = f"{error_record['type']}:{error_record['message']}:{error_record['context']}"
        if error_key in error_registry:
            error_info = error_registry[error_key]
            # If this error has occurred more than 5 times in the last hour, limit notifications
            if error_info['count'] > 5:
                first_time = datetime.fromisoformat(error_info['first_occurred'])
                last_time = datetime.fromisoformat(error_info['last_occurred'])
                time_diff = (last_time - first_time).total_seconds()
                
                # If errors are coming in faster than one per minute, limit notifications
                if time_diff < 60 * error_info['count']:
                    # Only send notification every 10 occurrences for medium severity
                    if error_record['severity'] == ErrorSeverity.MEDIUM and error_info['count'] % 10 != 0:
                        return False
    
    return True

def _send_error_notification(error_record):
    """
    Send a notification about an error.
    
    Args:
        error_record (dict): Error record
    """
    try:
        token = NOTIFICATION_CONFIG.get('telegram_token')
        chat_id = NOTIFICATION_CONFIG.get('telegram_chat_id')
        
        if not token or not chat_id:
            logger.warning("Telegram notification configured but token or chat_id missing")
            return
        
        # Create message
        severity_emoji = {
            ErrorSeverity.LOW: "ℹ️",
            ErrorSeverity.MEDIUM: "⚠️",
            ErrorSeverity.HIGH: "🚨",
            ErrorSeverity.CRITICAL: "💥"
        }
        
        emoji = severity_emoji.get(error_record['severity'], "⚠️")
        
        message = (
            f"{emoji} *ERROR ALERT* {emoji}\n"
            f"*Severity:* {error_record['severity']}\n"
            f"*Type:* {error_record['type']}\n"
            f"*Message:* {error_record['message']}\n"
            f"*Context:* {error_record['context'] or 'N/A'}\n"
            f"*Location:* {error_record['file']}:{error_record['line']} in {error_record['function']}\n"
            f"*Time:* {error_record['timestamp']}"
        )
        
        # Send message
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        logger.debug(f"Sent error notification to Telegram")
    except Exception as e:
        logger.error(f"Failed to send error notification: {e}")

def _retry_operation(func, args, max_retries, retry_delay, error_record):
    """
    Retry an operation with exponential backoff.
    
    Args:
        func (callable): Function to retry
        args (dict): Arguments for the function
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Base delay between retries in seconds
        error_record (dict): Original error record
        
    Returns:
        bool: True if retry succeeded, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Calculate exponential backoff
            backoff = retry_delay * (2 ** attempt)
            
            # Log retry attempt
            logger.info(f"Retrying operation after error ({attempt+1}/{max_retries}, delay: {backoff}s)")
            
            # Wait before retry
            time.sleep(backoff)
            
            # Retry the operation
            func(**args)
            
            # If we reach here, the retry succeeded
            logger.info(f"Retry attempt {attempt+1} succeeded")
            return True
        except Exception as retry_error:
            # Log retry failure
            logger.warning(f"Retry attempt {attempt+1} failed: {str(retry_error)}")
    
    # All retries failed
    logger.error(f"All {max_retries} retry attempts failed after error: {error_record['message']}")
    return False

def retry(max_retries=3, retry_delay=1, retry_exceptions=None):
    """
    Decorator to retry a function on exception.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Base delay between retries in seconds
        retry_exceptions (list): List of exception types to retry on
        
    Returns:
        callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if this exception should be retried
                    if retry_exceptions and not isinstance(e, tuple(retry_exceptions)):
                        raise
                    
                    # Log retry attempt
                    if attempt < max_retries:
                        backoff = retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Retry {attempt+1}/{max_retries} for {func.__name__} "
                            f"after {type(e).__name__}: {str(e)}. Waiting {backoff}s"
                        )
                        time.sleep(backoff)
            
            # If we get here, all retries failed
            logger.error(f"All {max_retries} retries failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    
    return decorator

def log_function_call(level=logging.DEBUG):
    """
    Decorator to log function calls.
    
    Args:
        level (int): Logging level
        
    Returns:
        callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = get_logger(func.__module__)
            func_name = func.__qualname__
            
            # Log function call with arguments
            arg_str = ', '.join([repr(arg) for arg in args])
            kwarg_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            all_args = ', '.join(filter(None, [arg_str, kwarg_str]))
            
            func_logger.log(level, f"Calling {func_name}({all_args})")
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Log result
                if result is not None:
                    # Truncate large results
                    result_str = repr(result)
                    if len(result_str) > 100:
                        result_str = result_str[:97] + "..."
                    
                    func_logger.log(level, f"{func_name} returned: {result_str}")
                else:
                    func_logger.log(level, f"{func_name} returned: None")
                
                return result
            except Exception as e:
                # Log exception
                func_logger.error(f"{func_name} raised: {type(e).__name__} - {str(e)}")
                raise
        
        return wrapper
    
    return decorator

def setup_exception_handling():
    """Set up global exception handling."""
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        """
        Global exception handler for unhandled exceptions.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default handler for KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Get traceback as string
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Log the exception
        logger.critical(f"Unhandled exception: {exc_type.__name__}: {exc_value}")
        logger.critical(f"Traceback:\n{tb_str}")
        
        # Send notification for critical errors
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'type': exc_type.__name__,
            'message': str(exc_value),
            'context': 'Unhandled exception',
            'severity': ErrorSeverity.CRITICAL,
            'file': 'unknown',
            'function': 'unknown',
            'line': 0,
            'traceback': tb_str
        }
        
        _send_error_notification(error_record)
    
    # Set the global exception handler
    sys.excepthook = global_exception_handler

def get_error_stats():
    """
    Get statistics about errors.
    
    Returns:
        dict: Error statistics
    """
    with error_lock:
        # Count errors by type and severity
        error_counts = {
            'total': len(error_registry),
            'by_type': {},
            'by_severity': {
                ErrorSeverity.LOW: 0,
                ErrorSeverity.MEDIUM: 0,
                ErrorSeverity.HIGH: 0,
                ErrorSeverity.CRITICAL: 0
            }
        }
        
        # Get most frequent errors
        sorted_errors = sorted(
            error_registry.values(),
            key=lambda e: e['count'],
            reverse=True
        )
        
        top_errors = []
        
        for error in sorted_errors[:10]:  # Top 10 errors
            top_errors.append({
                'type': error['type'],
                'message': error['message'],
                'context': error['context'],
                'count': error['count'],
                'first_occurred': error['first_occurred'],
                'last_occurred': error['last_occurred'],
                'severity': error['severity']
            })
            
            # Count by type
            if error['type'] in error_counts['by_type']:
                error_counts['by_type'][error['type']] += error['count']
            else:
                error_counts['by_type'][error['type']] = error['count']
            
            # Count by severity
            error_counts['by_severity'][error['severity']] += error['count']
        
        return {
            'error_counts': error_counts,
            'top_errors': top_errors,
            'total_error_count': sum(error['count'] for error in error_registry.values())
        }