"""Main script for end-to-end task querying."""

import enum
import subprocess
import sys
from importlib.resources import files

CLI_SCRIPTS_DIR = files("MEDS_tabular_automl").parent.parent / "cli"


class Program(enum.Enum):
    DESCRIBE_CODES = "describe_codes.sh"
    TABULARIZATION = "tabularization.sh"
    TASK_SPECIFIC_CACHING = "task_specific_caching.sh"
    XGBOOST = "xgboost.sh"
    PROFILE_TABULARIZATION = "profile_tabularization.sh"

    @staticmethod
    def from_str(program_arg):
        match program_arg:
            case "describe_codes":
                return Program.DESCRIBE_CODES
            case "tabularization":
                return Program.TABULARIZATION
            case "task_specific_caching":
                return Program.TASK_SPECIFIC_CACHING
            case "xgboost":
                return Program.XGBOOST
            case "profile_tabularization":
                return Program.PROFILE_TABULARIZATION
            case _:
                raise ValueError(
                    f"Invalid program name {program_arg}, valid programs are {[p.name for p in Program]}"
                )

    @staticmethod
    def get_script(program):
        return CLI_SCRIPTS_DIR / program.value


def main():
    program = sys.argv[1]
    args = sys.argv[2:]
    program = Program.from_str(program)
    script_path = Program.get_script(program)
    command_parts = [str(script_path.resolve()), *args]
    subprocess.run(" ".join(command_parts), shell=True)


if __name__ == "__main__":
    main()
