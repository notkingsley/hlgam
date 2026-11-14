from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping

from dotenv import dotenv_values, load_dotenv


type TypeFunc[EnvType] = type[EnvType] | Callable[[str], EnvType]


class EnvError(Exception):
    """
    Base class for environment errors.
    """


class EnvNotFoundError(EnvError):
    """
    Raised when an environment variable is not found.
    """


class EnvTypeError(EnvError):
    """
    Raised when an environment variable has an invalid type.
    """


class EnvFileNotFoundError(EnvError):
    """
    Raised when a .env file is not found.
    """


def get_env_dir() -> Path:
    """
    Get the directory of the environment file.
    """
    DEFAULT_ENV_DIR = "./"
    return Path(optional("ENV_DIR", DEFAULT_ENV_DIR)).expanduser()


def resolve(file: str | Path, base_dir: str | Path | None = None) -> Path:
    """
    Resolve an environment file path.
    """
    file_path = Path(file).expanduser()
    if file_path.is_absolute():
        return file_path

    root = Path(base_dir).expanduser() if base_dir is not None else get_env_dir()
    return root / file_path


def load(
    file: str | Path,
    *,
    required: bool = True,
    override: bool = True,
    base_dir: str | Path | None = None,
) -> Path | None:
    """
    Load environment variables from a .env file.
    """
    file_path = resolve(file, base_dir)
    if not file_path.is_file():
        if required:
            raise EnvFileNotFoundError(f"File {file_path.name} not found in {file_path.parent}.")
        return None

    load_dotenv(file_path, override=override)
    return file_path


def load_values(
    file: str | Path,
    *,
    required: bool = False,
    base_dir: str | Path | None = None,
) -> dict[str, str]:
    """
    Read environment variables from a .env file without modifying os.environ.
    """
    file_path = resolve(file, base_dir)
    if not file_path.is_file():
        if required:
            raise EnvFileNotFoundError(f"File {file_path.name} not found in {file_path.parent}.")
        return {}

    return {
        key: value
        for key, value in dotenv_values(file_path).items()
        if key and value is not None
    }


def environ_values(prefix: str | None = None) -> dict[str, str]:
    """
    Return current process environment values, optionally filtered by prefix.
    """
    if prefix is None:
        return dict(os.environ)
    return {key: value for key, value in os.environ.items() if key.startswith(prefix)}


def cast[EnvType](value: str, type_func: TypeFunc[EnvType] = str):
    """
    Cast a string to a given type.
    """
    try:
        return type_func(value)
    except ValueError:
        raise EnvTypeError(f"Cannot convert {value} to {type_func}.")


def require[EnvType](name: str, type_func: TypeFunc[EnvType] = str):
    """
    Get an environment variable or raise an exception if it is not set.
    """
    if value := os.getenv(name):
        return cast(value, type_func)

    raise EnvNotFoundError(f"Environment variable {name} is not set.")


def require_from[EnvType](
    values: Mapping[str, str],
    name: str,
    type_func: TypeFunc[EnvType] = str,
):
    """
    Get a value from an environment mapping or raise an exception if it is not set.
    """
    if value := values.get(name):
        return cast(value, type_func)

    raise EnvNotFoundError(f"Environment variable {name} is not set.")


def optional[EnvType, DefaultEnvType](
    name: str, default: DefaultEnvType, type_func: TypeFunc[EnvType] = str
):
    """
    Get an environment variable or return a default value if it is not set.
    """
    if value := os.getenv(name):
        return cast(value, type_func)

    return default


def optional_from[EnvType, DefaultEnvType](
    values: Mapping[str, str],
    name: str,
    default: DefaultEnvType,
    type_func: TypeFunc[EnvType] = str,
):
    """
    Get a value from an environment mapping or return a default value if it is not set.
    """
    if value := values.get(name):
        return cast(value, type_func)

    return default


def blank_to_none(value: str | None) -> str | None:
    """
    Convert blank environment values to None.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
