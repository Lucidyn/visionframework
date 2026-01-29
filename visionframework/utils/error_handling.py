"""
Error handling utilities for Vision Framework

This module provides utilities for consistent error handling across the Vision Framework.
"""

from typing import Optional, Dict, Any, Tuple
import traceback
from ..exceptions import VisionFrameworkError
from .monitoring.logger import get_logger

logger = get_logger(__name__)


class ErrorHandler:
    """
    Utility class for consistent error handling
    """

    @staticmethod
    def handle_error(
        error: Exception,
        error_type: type,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        raise_error: bool = False,
        log_traceback: bool = True
    ) -> Optional[Exception]:
        """
        Handle an error consistently

        Args:
            error: The original exception
            error_type: The type of exception to raise or log
            message: The error message
            context: Additional context information
            raise_error: Whether to raise the exception
            log_traceback: Whether to log the traceback

        Returns:
            The created exception if not raised, None otherwise
        """
        try:
            # Create the exception
            exception = error_type(
                message=message,
                context=context or {},
                original_error=error
            )

            # Log the error
            log_level = logger.error if log_traceback else logger.warning
            log_level(
                f"{exception}",
                exc_info=error if log_traceback else None
            )

            # Raise the exception if requested
            if raise_error:
                raise exception

            return exception
        except Exception as e:
            # If something goes wrong in error handling, just log it
            logger.error(f"Error in error handling: {e}", exc_info=True)
            return None

    @staticmethod
    def wrap_error(
        func,
        error_type: type,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        default_return=None,
        raise_error: bool = False
    ):
        """
        Decorator to wrap a function with error handling

        Args:
            func: The function to wrap
            error_type: The type of exception to raise or log
            message: The error message
            context: Additional context information
            default_return: The default return value if an error occurs
            raise_error: Whether to raise the exception

        Returns:
            The wrapped function
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(
                    error=e,
                    error_type=error_type,
                    message=message,
                    context=context,
                    raise_error=raise_error
                )
                return default_return
        return wrapper

    @staticmethod
    def validate_input(
        input_value: Any,
        expected_type: type,
        param_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input value

        Args:
            input_value: The input value to validate
            expected_type: The expected type
            param_name: The parameter name
            context: Additional context information

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not isinstance(input_value, expected_type):
            error_message = f"{param_name} must be {expected_type.__name__}, got {type(input_value).__name__}"
            logger.warning(error_message, extra=context)
            return False, error_message
        return True, None

    @staticmethod
    def format_error_message(
        message: str,
        error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format an error message with context

        Args:
            message: The base error message
            error: The original exception
            context: Additional context information

        Returns:
            Formatted error message
        """
        full_message = message

        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            full_message += f" [{context_str}]"

        if error:
            full_message += f" (Original error: {str(error)})"

        return full_message


# Global error handler instance
error_handler = ErrorHandler()
