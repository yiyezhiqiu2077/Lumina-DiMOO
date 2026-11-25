"""Editing entrypoint; retain original Lumina non-edit task interfaces."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    legacy = any(
        arg in sys.argv
        for arg in (
            "--output_dir",
            "--ref_image_path",
            "--prefix_mode",
            "--condition_cache",
        )
    )
    if legacy:
        from inference.tasks_i2i import main

        main()
        raise SystemExit
    if "--edit_type" in sys.argv:
        i = sys.argv.index("--edit_type")
        if sys.argv[i + 1] != "edit":
            from inference.tasks_i2i import main

            main()
            raise SystemExit
        del sys.argv[i : i + 2]
    from inference.edit import cli

    cli()
