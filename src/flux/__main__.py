from fire import Fire

from .cli import main as cli_main
from .cli_control import main as control_main
from .cli_fill import main as fill_main
from .cli_kontext import main as kontext_main
from .cli_redux import main as redux_main

if __name__ == "__main__":
    Fire(
        {
            "t2i": cli_main,
            "control": control_main,
            "fill": fill_main,
            "kontext": kontext_main,
            "redux": redux_main,
        }
    )
