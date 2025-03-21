"""
Time Utilities for Goalkick Ltd Trading Bot

Provides advanced time-related utilities for trading operations,
including time zone handling, scheduling, and time window calculations.
"""

import pytz
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional, Union, Dict, List

from config.logging_config import get_logger
from config.bot_config import SCHEDULE_CONFIG, MARKET_HOURS

logger = get_logger("time_utils")

class TimeUtils:
    """
    Comprehensive time utility class for trading operations.
    Handles time zone conversions, market hours, and scheduling.
    """
    
    @staticmethod
    def get_current_time(timezone: str = 'UTC') -> datetime:
        """
        Get current time in specified timezone.
        
        Args:
            timezone (str): Timezone name (default: UTC)
        
        Returns:
            datetime: Current time in specified timezone
        """
        try:
            tz = pytz.timezone(timezone)
            return datetime.now(tz)
        except Exception as e:
            logger.error(f"Error getting current time in {timezone}: {e}")
            return datetime.now(pytz.UTC)
    
    @staticmethod
    def convert_timezone(
        dt: datetime, 
        from_tz: str = 'UTC', 
        to_tz: str = 'UTC'
    ) -> datetime:
        """
        Convert datetime between timezones.
        
        Args:
            dt (datetime): Input datetime
            from_tz (str): Source timezone
            to_tz (str): Target timezone
        
        Returns:
            datetime: Converted datetime
        """
        try:
            from_timezone = pytz.timezone(from_tz)
            to_timezone = pytz.timezone(to_tz)
            
            # Localize the input datetime if not already localized
            if dt.tzinfo is None:
                dt = from_timezone.localize(dt)
            
            return dt.astimezone(to_timezone)
        except Exception as e:
            logger.error(f"Timezone conversion error: {e}")
            return dt
    
    @staticmethod
    def is_market_open(
        symbol: str, 
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if markets are open for a specific symbol.
        
        Args:
            symbol (str): Trading symbol
            current_time (datetime, optional): Time to check
        
        Returns:
            bool: Whether markets are open
        """
        try:
            if current_time is None:
                current_time = datetime.now(pytz.UTC)
            
            # Get market session for current time
            from config.bot_config import get_market_session
            session = get_market_session(current_time)
            
            # Check market hours configuration
            market_hours = MARKET_HOURS.get(session, {})
            
            return (
                market_hours.get('start') <= current_time.time() < 
                market_hours.get('end')
            )
        except Exception as e:
            logger.error(f"Market hours check failed: {e}")
            return True  # Default to open to prevent blocking trades
    
    @staticmethod
    def schedule_task(
        job: Callable, 
        interval: Union[int, str] = None, 
        time_of_day: str = None
    ) -> threading.Thread:
        """
        Schedule a recurring task with flexible configuration.
        
        Args:
            job (Callable): Function to be scheduled
            interval (int/str): Interval in seconds or predefined schedule
            time_of_day (str): Specific time to run the task
        
        Returns:
            threading.Thread: Scheduled task thread
        """
        def _run_scheduled_job():
            """Internal wrapper for scheduled job."""
            while True:
                try:
                    # Configure scheduling based on input
                    if interval and isinstance(interval, (int, float)):
                        schedule.every(interval).seconds.do(job)
                    elif interval:
                        getattr(schedule.every(), interval).do(job)
                    
                    if time_of_day:
                        schedule.every().day.at(time_of_day).do(job)
                    
                    # Run scheduled tasks
                    while True:
                        schedule.run_pending()
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"Scheduled job error: {e}")
                    time.sleep(SCHEDULE_CONFIG.get('error_retry_interval', 60))
        
        # Create and start thread
        thread = threading.Thread(target=_run_scheduled_job, daemon=True)
        thread.start()
        return thread
    
    @staticmethod
    def get_trading_windows(
        symbol: str, 
        lookback_hours: int = 24
    ) -> Dict[str, dict]:
        """
        Calculate optimal trading windows for a symbol.
        
        Args:
            symbol (str): Trading symbol
            lookback_hours (int): Hours to analyze for trading windows
        
        Returns:
            Dict of trading window details
        """
        try:
            # Get regional market weights for the symbol
            from config.bot_config import get_regional_weight
            
            current_time = datetime.now(pytz.UTC)
            windows = {}
            
            # Analyze each market session
            for session in ['Asian', 'European', 'US']:
                # Get market weight for this session
                weight = get_regional_weight(symbol, session)
                
                # Calculate trading window start and end
                session_hours = MARKET_HOURS.get(session, {})
                if session_hours:
                    start_time = datetime.combine(
                        current_time.date(), 
                        session_hours['start']
                    ).replace(tzinfo=pytz.UTC)
                    end_time = datetime.combine(
                        current_time.date(), 
                        session_hours['end']
                    ).replace(tzinfo=pytz.UTC)
                    
                    windows[session] = {
                        'start': start_time,
                        'end': end_time,
                        'weight': weight
                    }
            
            return windows
        except Exception as e:
            logger.error(f"Trading windows calculation failed: {e}")
            return {}

    @staticmethod
    def get_current_time_ms() -> int:
        """
        Get current time in milliseconds since epoch.
        
        Returns:
            int: Current time in milliseconds
        """
        return int(time.time() * 1000)

    @staticmethod
    def ms_to_datetime(timestamp_ms: int) -> datetime:
        """
        Convert millisecond timestamp to datetime.
        
        Args:
            timestamp_ms (int): Timestamp in milliseconds
            
        Returns:
            datetime: Datetime object
        """
        return datetime.fromtimestamp(timestamp_ms / 1000.0)

    @staticmethod
    def datetime_to_ms(dt: datetime) -> int:
        """
        Convert datetime to millisecond timestamp.
        
        Args:
            dt (datetime): Datetime object
            
        Returns:
            int: Timestamp in milliseconds
        """
        return int(dt.timestamp() * 1000)

    @staticmethod
    def get_timeframe_ms(timeframe: str) -> int:
        """
        Convert timeframe string to milliseconds.
        
        Args:
            timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
            
        Returns:
            int: Timeframe in milliseconds
        """
        return timeframe_to_milliseconds(timeframe)

    @staticmethod
    def add_timeframe(dt: datetime, timeframe: str, num_periods: int = 1) -> datetime:
        """
        Add timeframe periods to a datetime.
        
        Args:
            dt (datetime): Datetime object
            timeframe (str): Timeframe string
            num_periods (int): Number of periods to add
            
        Returns:
            datetime: New datetime
        """
        tf_ms = timeframe_to_milliseconds(timeframe)
        new_timestamp_ms = TimeUtils.datetime_to_ms(dt) + (tf_ms * num_periods)
        return TimeUtils.ms_to_datetime(new_timestamp_ms)

    @staticmethod
    def subtract_timeframe(dt: datetime, timeframe: str, num_periods: int = 1) -> datetime:
        """
        Subtract timeframe periods from a datetime.
        
        Args:
            dt (datetime): Datetime object
            timeframe (str): Timeframe string
            num_periods (int): Number of periods to subtract
            
        Returns:
            datetime: New datetime
        """
        return TimeUtils.add_timeframe(dt, timeframe, -num_periods)

    @staticmethod
    def get_timeframe_start(dt: datetime, timeframe: str) -> datetime:
        """
        Get the start of the current timeframe period.
        
        Args:
            dt (datetime): Datetime object
            timeframe (str): Timeframe string
            
        Returns:
            datetime: Start of current timeframe period
        """
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            minute_offset = dt.minute % minutes
            return dt.replace(minute=dt.minute - minute_offset, second=0, microsecond=0)
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            hour_offset = dt.hour % hours
            return dt.replace(hour=dt.hour - hour_offset, minute=0, second=0, microsecond=0)
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe.endswith('w'):
            weeks = int(timeframe[:-1])
            days_since_monday = dt.weekday()
            return (dt - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            logger.warning(f"Unsupported timeframe: {timeframe}")
            return dt

def timeframe_to_milliseconds(timeframe: str) -> int:
    """
    Convert a timeframe string to milliseconds.
    
    Args:
        timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        int: Timeframe in milliseconds
    """
    timeframe = timeframe.strip().lower()
    
    # Extract numeric value and unit
    if timeframe.endswith('m'):
        # Minutes
        minutes = int(timeframe[:-1])
        return minutes * 60 * 1000
    elif timeframe.endswith('h'):
        # Hours
        hours = int(timeframe[:-1])
        return hours * 60 * 60 * 1000
    elif timeframe.endswith('d'):
        # Days
        days = int(timeframe[:-1])
        return days * 24 * 60 * 60 * 1000
    elif timeframe.endswith('w'):
        # Weeks
        weeks = int(timeframe[:-1])
        return weeks * 7 * 24 * 60 * 60 * 1000
    else:
        # Default to 1 minute if unknown format
        logger.warning(f"Unknown timeframe format: {timeframe}, defaulting to 1 minute")
        return 60 * 1000

def get_candle_times(start_time: int, end_time: int, timeframe: str) -> List[int]:
    """
    Get a list of candle start times for a given time range and timeframe.
    
    Args:
        start_time (int): Start time in milliseconds
        end_time (int): End time in milliseconds
        timeframe (str): Timeframe string
        
    Returns:
        list: List of candle start times in milliseconds
    """
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    
    # Ensure start_time is aligned with timeframe
    start_time = start_time - (start_time % timeframe_ms)
    
    # Generate candle times
    candle_times = []
    current_time = start_time
    
    while current_time <= end_time:
        candle_times.append(current_time)
        current_time += timeframe_ms
    
    return candle_times

def get_market_session_for_time(dt: datetime) -> str:
    """
    Get the market session for a given datetime.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        str: Market session ('Asian', 'European', 'US')
    """
    time_obj = dt.time()
    
    # Check each session
    for session, session_hours in MARKET_HOURS.items():
        start_time = session_hours.get('start')
        end_time = session_hours.get('end')
        
        if start_time <= time_obj < end_time:
            return session
    
    # Default to 'US' if no session matches
    return 'US'

def format_time_difference(start_time: datetime, end_time: datetime) -> str:
    """
    Format time difference as a human-readable string.
    
    Args:
        start_time (datetime): Start time
        end_time (datetime): End time
        
    Returns:
        str: Formatted time difference
    """
    diff = end_time - start_time
    
    # Get total seconds
    total_seconds = diff.total_seconds()
    
    # Calculate days, hours, minutes, and seconds
    days = int(total_seconds // (24 * 3600))
    remaining_seconds = total_seconds % (24 * 3600)
    hours = int(remaining_seconds // 3600)
    remaining_seconds %= 3600
    minutes = int(remaining_seconds // 60)
    seconds = int(remaining_seconds % 60)
    
    # Format the difference
    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def wait_until(target_time: datetime, sleep_interval: int = 1) -> bool:
    """
    Wait until a specific time is reached.
    
    Args:
        target_time (datetime): Time to wait until
        sleep_interval (int): Sleep interval in seconds
        
    Returns:
        bool: True if successful, False if target time is in the past
    """
    now = datetime.now(target_time.tzinfo)
    
    if target_time < now:
        return False
    
    while now < target_time:
        sleep_duration = min(sleep_interval, (target_time - now).total_seconds())
        if sleep_duration <= 0:
            break
        
        time.sleep(sleep_duration)
        now = datetime.now(target_time.tzinfo)
    
    return True

def is_weekday(dt: datetime) -> bool:
    """
    Check if a datetime is a weekday (Monday-Friday).
    
    Args:
        dt (datetime): Datetime to check
        
    Returns:
        bool: True if weekday, False if weekend
    """
    return dt.weekday() < 5  # 0-4 = Monday-Friday, 5-6 = Saturday-Sunday

def is_within_time_range(dt: datetime, start_time: str, end_time: str) -> bool:
    """
    Check if a datetime is within a specific time range.
    
    Args:
        dt (datetime): Datetime to check
        start_time (str): Start time string (24-hour format, e.g., '09:30')
        end_time (str): End time string (24-hour format, e.g., '16:00')
        
    Returns:
        bool: True if within range, False otherwise
    """
    time_obj = dt.time()
    
    # Parse start and end times
    start_hour, start_minute = map(int, start_time.split(':'))
    end_hour, end_minute = map(int, end_time.split(':'))
    
    start_time_obj = datetime.time(start_hour, start_minute)
    end_time_obj = datetime.time(end_hour, end_minute)
    
    if start_time_obj <= end_time_obj:
        return start_time_obj <= time_obj <= end_time_obj
    else:
        # Handle overnight ranges (e.g., 22:00 to 04:00)
        return time_obj >= start_time_obj or time_obj <= end_time_obj
    
from datetime import datetime
import time

def get_current_time_ms():
    """Get current time in milliseconds."""
    return int(time.time() * 1000)

def get_timeframe_ms(timeframe):
    """
    Convert a timeframe string to milliseconds.
    
    Args:
        timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        int: Milliseconds for the timeframe
    """
    return timeframe_to_milliseconds(timeframe)

def timeframe_to_milliseconds(timeframe):
    """
    Convert a timeframe string to milliseconds.
    
    Args:
        timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        int: Number of milliseconds
    
    Raises:
        ValueError: If timeframe format is invalid
    """
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    if unit == 'm':
        return value * 60 * 1000
    elif unit == 'h':
        return value * 60 * 60 * 1000
    elif unit == 'd':
        return value * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Invalid timeframe unit: {unit}. Must be 'm', 'h', or 'd'")    

# Expose public methods for easy import
__all__ = ['TimeUtils', 'timeframe_to_milliseconds', 'get_candle_times',
           'get_market_session_for_time', 'format_time_difference',
           'wait_until', 'is_weekday', 'is_within_time_range','get_current_time_ms','get_timeframe_ms',
           'timeframe_to_milliseconds',
           ]