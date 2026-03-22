def build_cli_overrides(args) -> dict:
    """
    Build CLI overrides dictionary from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of CLI overrides ready for load_config()
    """
    return {
        key: value
        for key, value in vars(args).items()
        if value is not None and key != 'agentic_config_path'
    }