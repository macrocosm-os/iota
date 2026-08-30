import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Callable

from common import settings as common_settings
from common.models.api_models import ValidationTaskResponse, ValidatorTask
from loguru import logger

available_functions: list[Callable] = []


def load_functions_from_directory(validator_code_dir: str) -> list[Callable]:
    """
    Dynamically load all callable functions from Python modules in the validator_code_dir.

    Args:
        validator_code_dir: Path to the directory containing validation function modules

    Returns:
        List of callable functions found in the directory
    """
    functions = []
    validator_code_path = Path(validator_code_dir)

    if not validator_code_path.exists():
        logger.warning(f"Validator code directory does not exist: {validator_code_dir}")
        return functions

    # Add the validator_code_dir to sys.path to handle relative imports
    validator_code_dir_str = str(validator_code_path.resolve())
    if validator_code_dir_str not in sys.path:
        sys.path.insert(0, validator_code_dir_str)
        path_added = True
    else:
        path_added = False

    try:
        # Walk through all Python files in the directory
        for py_file in validator_code_path.rglob("*.py"):
            # Skip __init__.py files and __pycache__ directories
            try:
                # Calculate the module name based on the relative path from validator_code_dir
                relative_path = py_file.relative_to(validator_code_path)
                # Convert path to module name (e.g., orchestrator/validator/validation_functions/scoring_functions/activation_intra_cosine_similarity.py
                # becomes orchestrator.validator.validation_functions.scoring_functions.activation_intra_cosine_similarity)
                module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
                module_name = ".".join(module_parts)

                # Create a module spec from the file
                spec = importlib.util.spec_from_file_location(module_name, py_file)

                if spec is None or spec.loader is None:
                    logger.warning(f"Could not create spec for {py_file}")
                    continue

                # Load the module
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all callable functions in the module
                for name, obj in inspect.getmembers(module, inspect.isroutine):
                    logger.info(f"Found function: {name} in {py_file}")
                    # Only include functions defined in this module (not imported)
                    if obj not in functions:
                        functions.append(obj)
                        logger.debug(f"Loaded function: {name} from {py_file}")

            except Exception as e:
                logger.exception(f"Error loading module {py_file}: {e}")
                continue

    finally:
        # Remove the path if we added it
        if path_added and validator_code_dir_str in sys.path:
            sys.path.remove(validator_code_dir_str)

    logger.info(f"Loaded {len(functions)} functions from {validator_code_dir}")
    logger.info(f"Available functions: {[f.__name__ for f in functions]}")
    return functions


async def execute_task(task: ValidatorTask, validator_code_dir: str):
    """
    Execute a task.
    """
    global available_functions

    logger.info(f"Executing task: {task.model_dump()}")
    logger.info(f"Validator code directory: {validator_code_dir}")

    # Reshape activations to the task's run mini batch size (matches what miners produced).
    # Safe as a process-wide set: execute_task runs one task at a time.
    common_settings.set_mini_batch_size(task.mini_batch_size)

    # Load functions from the validator_code_dir if not already loaded
    if not available_functions:
        available_functions = load_functions_from_directory(validator_code_dir)
        logger.info(f"Loaded {len(available_functions)} functions from {validator_code_dir}")
        logger.info(f"Available functions: {[f.__name__ for f in available_functions]}")

    function_to_execute = [f for f in available_functions if f.__name__ == task.function_name]
    if len(function_to_execute) != 1:
        raise ValueError(
            f"Unknown task function: {task.function_name}; available functions: {[f.__name__ for f in available_functions]}"
        )

    try:
        logger.info(f"Executing function: {function_to_execute[0].__name__} with inputs: {task.inputs}")
        result = await function_to_execute[0](task.inputs)

    except Exception as e:
        logger.exception(f"Error executing task {task.function_name}: {e}")
        raise RuntimeError(f"Error executing task {task.function_name}: {e}")

    return ValidationTaskResponse(
        task_result_id=task.task_result_id,
        task_type=task.task_type,
        function_name=task.function_name,
        inputs=task.inputs,
        outputs=result,
    )
