import sys
from itertools import combinations


def format_print(permutations: list[tuple[str, ...]]) -> None:
    """Prints all sets in a visually formatted string.

    Args:
        permutations: The list of all possible permutations of length >= 1.

    Examples:
        >>> format_print([('2',), ('2', '3'), ('2', '3', '4'), ('2', '4'), ('3',), ('3', '4'), ('4',)])
        [2],[2,3],[2,3,4],[2,4],[3],[3,4],[4]
    """
    out_str = ""
    for item in permutations:
        out_str += f"[{','.join(item)}],"
    out_str = out_str[:-1]
    print(out_str)


def get_subsets(list_of_options: list[str]) -> None:
    """Generates and prints all possible subsets of length >= 1 from a list of options.

    Args:
        list_of_options: The list of options.

    Examples:
        >>> get_subsets(['2', '3', '4'])
        [2],[2,3],[2,3,4],[2,4],[3],[3,4],[4]
    """
    sets = []
    for i in range(1, len(list_of_options) + 1):
        sets.extend(list(combinations(list_of_options, r=i)))
    format_print(sorted(sets))


def main():
    """Generates and prints all possible non-empty subsets from given list of options."""
    list_of_options = list(sys.argv[1].strip("[]").split(","))
    get_subsets(list_of_options)


if __name__ == "__main__":
    main()
