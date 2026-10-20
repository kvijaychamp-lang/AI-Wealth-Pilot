"""
Secure environment variable loader with validation.

SECURITY BEST PRACTICES:
1. Never log or print sensitive environment variables
2. Validate required variables at startup
3. Use type-safe loading with proper defaults
4. Handle missing variables gracefully in development
5. Fail fast in production for missing critical variables
6. Use environment-specific validation rules
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv


class EnvironmentValidationError(Exception):
    """Raised when environment variable validation fails."""
    pass


def load_env() -> Dict[str, str]:
    """
    Load environment variables from .env file.

    SECURITY: This function loads all environment variables into memory.
    Sensitive values should never be logged or exposed in error messages.
    """
    env_path = Path('.') / '.env'

    # Load .env file if it exists
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    # Return a copy of environment variables (don't expose the actual os.environ)
    return dict(os.environ)


def get_required_env_var(key: str, env_vars: Dict[str, str]) -> str:
    """
    Get a required environment variable with validation.

    Args:
        key: Environment variable name
        env_vars: Dictionary of environment variables

    Returns:
        The environment variable value

    Raises:
        EnvironmentValidationError: If variable is missing or empty
    """
    value = env_vars.get(key, '').strip()
    if not value:
        raise EnvironmentValidationError(
            f"Required environment variable '{key}' is missing or empty. "
            f"Please set it in your .env file or environment."
        )
    return value


def get_optional_env_var(
    key: str,
    env_vars: Dict[str, str],
    default: str = ''
) -> str:
    """
    Get an optional environment variable with default fallback.

    Args:
        key: Environment variable name
        env_vars: Dictionary of environment variables
        default: Default value if variable is not set

    Returns:
        The environment variable value or default
    """
    return env_vars.get(key, default).strip()


def parse_comma_separated_list(value: str) -> List[str]:
    """
    Parse a comma-separated string into a list of stripped values.

    Args:
        value: Comma-separated string

    Returns:
        List of stripped string values
    """
    if not value.strip():
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def validate_environment(env_vars: Dict[str, str], app_env: str = 'development') -> None:
    """
    Validate environment variables based on application environment.

    SECURITY: This function ensures all required variables are present
    and have valid values before the application starts.

    Args:
        env_vars: Dictionary of environment variables
        app_env: Application environment (development/staging/production)

    Raises:
        EnvironmentValidationError: If validation fails
    """
    errors = []

    # Critical variables required in all environments
    critical_vars = ['APP_ENV']
    for var in critical_vars:
        try:
            get_required_env_var(var, env_vars)
        except EnvironmentValidationError as e:
            errors.append(str(e))

    # Production-specific validations
    if app_env == 'production':
        # API keys should be set in production
        prod_required_vars = ['NSE_API_KEY']
        for var in prod_required_vars:
            try:
                get_required_env_var(var, env_vars)
            except EnvironmentValidationError as e:
                errors.append(str(e))

        # Validate that sensitive keys are not default/empty values
        sensitive_vars = ['NSE_API_KEY', 'DATA_API_KEY', 'OPENAI_API_KEY']
        for var in sensitive_vars:
            value = env_vars.get(var, '').strip()
            if value and len(value) < 10:  # Basic length check for API keys
                errors.append(
                    f"Suspiciously short value for '{var}'. "
                    f"API keys should be longer than 10 characters."
                )

    # Validate numeric values
    numeric_vars = {
        'STREAMLIT_PORT': (1, 65535),
        'MAX_LOG_SIZE_MB': (1, 100),
        'LOG_BACKUP_COUNT': (1, 20),
        'TRAIN_TEST_SPLIT': (0.1, 0.9),
        'BATCH_SIZE': (1, 1000),
        'EPOCHS': (1, 1000),
        'LIVE_UPDATE_INTERVAL': (1, 3600),
        'MAX_API_RETRIES': (1, 10),
        'API_TIMEOUT': (1, 300),
        'PREDICTION_THRESHOLD': (0.0, 1.0),
        'MAX_DAILY_PREDICTIONS': (1, 1000),
        'PREDICTION_CACHE_TTL': (60, 86400)
    }

    for var, (min_val, max_val) in numeric_vars.items():
        value_str = env_vars.get(var, '').strip()
        if value_str:
            try:
                value = float(value_str) if '.' in value_str else int(value_str)
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"'{var}' value {value} is outside valid range "
                        f"[{min_val}, {max_val}]"
                    )
            except ValueError:
                errors.append(f"'{var}' must be a valid number, got: {value_str}")

    # Validate paths exist (if specified)
    path_vars = ['MODEL_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR']
    for var in path_vars:
        path_str = env_vars.get(var, '').strip()
        if path_str:
            path = Path(path_str)
            if path.is_absolute() and not path.exists():
                # Only warn for absolute paths that don't exist
                print(f"Warning: Path '{path}' specified in '{var}' does not exist")

    if errors:
        error_msg = "Environment validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise EnvironmentValidationError(error_msg)


def safe_get_env_var(
    key: str,
    env_vars: Dict[str, str],
    default: Any = None,
    var_type: type = str
) -> Any:
    """
    Safely get and convert an environment variable.

    SECURITY: This function provides type-safe access to environment variables
    with proper conversion and validation.

    Args:
        key: Environment variable name
        env_vars: Dictionary of environment variables
        default: Default value if variable is not set
        var_type: Expected type (str, int, float, bool, list)

    Returns:
        Converted value or default
    """
    value_str = env_vars.get(key, '').strip()

    if not value_str:
        return default

    try:
        if var_type == bool:
            # Handle boolean conversion
            return value_str.lower() in ('true', '1', 'yes', 'on')
        elif var_type == int:
            return int(value_str)
        elif var_type == float:
            return float(value_str)
        elif var_type == list:
            return parse_comma_separated_list(value_str)
        else:
            return value_str
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert '{key}' value '{value_str}' to {var_type.__name__}: {e}")
        return default
