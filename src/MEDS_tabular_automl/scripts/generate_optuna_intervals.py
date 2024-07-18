import sys


def get_intervals(name, list_of_params: str) -> list[str]:
    """Generates and prints optuna binary inclusion terms from a list of parameters to optimize.

    Args:
        name: The name of the parameter group.
        list_of_params: Comma-separated string of options.

    Examples:
        >>> get_intervals("param_name", "static,1d,full") # doctest: +NORMALIZE_WHITESPACE
        ['+param_name.param_static=int(interval(0,1))',
        '+param_name.param_1d=int(interval(0,1))',
        '+param_name.param_full=int(interval(0,1))']
        >>> get_intervals("other_param_name", "code/count,static/present") # doctest: +NORMALIZE_WHITESPACE
        ['+other_param_name.param_code_count=int(interval(0,1))',
        '+other_param_name.param_static_present=int(interval(0,1))']
    """
    params = [param.strip() for param in list_of_params.split(",")]
    kwarg_params = []
    for param in params:
        param = param.replace("/", "_")
        kwarg_params.append(f"+{name}.param_{param}=int(interval(0,1))")
    return kwarg_params


def main():
    """Generates all of the ranges of modality weights to iterate through."""
    if len(sys.argv) < 3:
        print("Usage: python generate_optuna_intervals.py <name> <param1,param2,param3>")
        sys.exit(1)

    name = sys.argv[1]
    list_of_params = sys.argv[2]
    print(" ".join(get_intervals(name, list_of_params)))


if __name__ == "__main__":
    main()
