# """
# Time Utilities for Goalkick Ltd Trading Bot

# Provides advanced time-related utilities for trading operations,
# including time zone handling, scheduling, and time window calculations.
# """

# import pytz
# import schedule
# import time
# import threading
# from datetime import datetime, timedelta
# from typing import Callable, Optional, Union

# from config.logging_config import get_logger
# from config.bot_config import SCHEDULE_CONFIG, MARKET_HOURS

# logger = get_logger("time_utils")

# class TimeUtils:
#     """
#     Comprehensive time utility class for trading operations.
#     Handles time zone conversions, market hours, and scheduling.
#     """
    
#     @staticmethod
#     def get_current_time(timezone: str = 'UTC') -> datetime:
#         """
#         Get current time in specified timezone.
        
#         Args:
#             timezone (str): Timezone name (default: UTC)
        
#         Returns:
#             datetime: Current time in specified timezone
#         """
#         try:
#             tz = pytz.timezone(timezone)
#             return datetime.now(tz)
#         except Exception as e:
#             logger.error(f"Error getting current time in {timezone}: {e}")
#             return datetime.now(pytz.UTC)
    
#     @staticmethod
#     def convert_timezone(
#         dt: datetime, 
#         from_tz: str = 'UTC', 
#         to_tz: str = 'UTC'
#     ) -> datetime:
#         """
#         Convert datetime between timezones.
        
#         Args:
#             dt (datetime): Input datetime
#             from_tz (str): Source timezone
#             to_tz (str): Target timezone
        
#         Returns:
#             datetime: Converted datetime
#         """
#         try:
#             from_timezone = pytz.timezone(from_tz)
#             to_timezone = pytz.timezone(to_tz)
            
#             # Localize the input datetime if not already localized
#             if dt.tzinfo is None:
#                 dt = from_timezone.localize(dt)
            
#             return dt.astimezone(to_timezone)
#         except Exception as e:
#             logger.error(f"Timezone conversion error: {e}")
#             return dt
    
#     @staticmethod
#     def is_market_open(
#         symbol: str, 
#         current_time: Optional[datetime] = None
#     ) -> bool:
#         """
#         Check if markets are open for a specific symbol.
        
#         Args:
#             symbol (str): Trading symbol
#             current_time (datetime, optional): Time to check
        
#         Returns:
#             bool: Whether markets are open
#         """
#         try:
#             if current_time is None:
#                 current_time = datetime.now(pytz.UTC)
            
#             # Get market session for current time
#             from config.bot_config import get_market_session
#             session = get_market_session(current_time)
            
#             # Check market hours configuration
#             market_hours = MARKET_HOURS.get(session, {})
            
#             return (
#                 market_hours.get('start') <= current_time.time() < 
#                 market_hours.get('end')
#             )
#         except Exception as e:
#             logger.error(f"Market hours check failed: {e}")
#             return True  # Default to open to prevent blocking trades
    
#     @staticmethod
#     def schedule_task(
#         job: Callable, 
#         interval: Union[int, str] = None, 
#         time_of_day: str = None
#     ) -> threading.Thread:
#         """
#         Schedule a recurring task with flexible configuration.
        
#         Args:
#             job (Callable): Function to be scheduled
#             interval (int/str): Interval in seconds or predefined schedule
#             time_of_day (str): Specific time to run the task
        
#         Returns:
#             threading.Thread: Scheduled task thread
#         """
#         def _run_scheduled_job():
#             """Internal wrapper for scheduled job."""
#             while True:
#                 try:
#                     # Configure scheduling based on input
#                     if interval and isinstance(interval, (int, float)):
#                         schedule.every(interval).seconds.do(job)
#                     elif interval:
#                         getattr(schedule.every(), interval).do(job)
                    
#                     if time_of_day:
#                         schedule.every().day.at(time_of_day).do(job)
                    
#                     # Run scheduled tasks
#                     while True:
#                         schedule.run_pending()
#                         time.sleep(1)
#                 except Exception as e:
#                     logger.error(f"Scheduled job error: {e}")
#                     time.sleep(SCHEDULE_CONFIG.get('error_retry_interval', 60))
        
#         # Create and start thread
#         thread = threading.Thread(target=_run_scheduled_job, daemon=True)
#         thread.start()
#         return thread
    
#     @staticmethod
#     def get_trading_windows(
#         symbol: str, 
#         lookback_hours: int = 24
#     ) -> Dict[str, datetime]:
#         """
#         Calculate optimal trading windows for a symbol.
        
#         Args:
#             symbol (str): Trading symbol
#             lookback_hours (int): Hours to analyze for trading windows
        
#         Returns:
#             Dict of trading window details
#         """
#         try:
#             # Get regional market weights for the symbol
#             from config.bot_config import get_regional_weight
            
#             current_time = datetime.now(pytz.UTC)
#             windows = {}
            
#             # Analyze each market session
#             for session in ['Asian', 'European', 'US']:
#                 # Get market weight for this session
#                 weight = get_regional_weight(symbol, session)
                
#                 # Calculate trading window start and end
#                 session_hours = MARKET_HOURS.get(session, {})
#                 if session_hours:
#                     start_time = datetime.combine(
#                         current_time.date(), 
#                         session_hours['start']
#                     ).replace(tzinfo=pytz.UTC)
#                     end_time = datetime.combine(
#                         current_time.date(), 
#                         session_hours['end']
#                     ).replace(tzinfo=pytz.UTC)
                    
#                     windows[session] = {
#                         'start': start_time,
#                         'end': end_time,
#                         'weight': weight
#                     }
            
#             return windows
#         except Exception as e:
#             logger.error(f"Trading windows calculation failed: {e}")
#             return {}

# # Expose public methods for easy import
# __all__ = ['TimeUtils']