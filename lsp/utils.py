def snake_case_to_camel_case(snake_str: str) -> str:
    first, *rest = snake_str.split("_")
    return first.lower() + "".join(x.title() for x in rest)
