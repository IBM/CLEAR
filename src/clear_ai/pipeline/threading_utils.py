import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
logger = logging.getLogger(__name__)

def run_func_in_threads(func, input_list, max_workers=10, error_prefix="Error: ", progress_desc="Processing tasks",
                        task_timeout=300):  # Default 5 min timeout per task
    if not input_list:
        return []
    if len(input_list) == 1:
        item_args = input_list[0]
        try:
            # If item_args is not a tuple, func might expect a single argument, not *item_args
            if isinstance(item_args, tuple):
                return [func(*item_args)]
            else:
                return [func(item_args)]
        except Exception as e:
            return [f"{error_prefix}: {e}"]

    results = [None] * len(input_list)
    with ThreadPoolExecutor(max_workers) as executor:
        future_to_idx_and_id = {}
        for i, item_args in enumerate(input_list):
            item_identifier = f"task_index_{i}"

            if isinstance(item_args, tuple):
                future = executor.submit(func, *item_args)
            else:
                future = executor.submit(func, item_args)
            future_to_idx_and_id[future] = i

        for future in tqdm(as_completed(future_to_idx_and_id), total=len(input_list), desc=progress_desc):
            idx = future_to_idx_and_id[future]
            try:
                result_val = future.result(timeout=task_timeout)
            except TimeoutError:
                logger.error(f"Task for item {i} timed out after {task_timeout} seconds.")
                future.cancel()  # Attempt to cancel the timed-out task
                result_val = f"{error_prefix} item {i}: Task timed out after {task_timeout} seconds."
            except Exception as e:
                # This catches exceptions raised by the function `func` itself if not caught internally by func
                logger.error(f"Task for ID {i} raised an exception: {e}")
                result_val = f"{error_prefix}ID {i}: {e}"
            results[idx] = result_val
        return results
