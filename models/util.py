def extract_args(kwargs, names):
    result = {}
    for arg_name, arg_value in kwargs.items():
        if arg_name in names and arg_value is not None:
            result[arg_name] = arg_value

    return result