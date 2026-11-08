#!/usr/bin/env python3
"""turbo launch entry for native swift stages: ``run_swift.py [sft|pt] ...`` — activates the msswift adapter before the stage's ``main()`` (swift's ``--external_plugins`` fires too late), then dispatches."""
from __future__ import annotations

import sys


def _extract_stage(argv):
    """Return the selected stage and arguments without its token."""
    from torchturbo.adapters.msswift.stages import iter_stages

    stage_names = {spec.name for spec in iter_stages()}
    if argv and argv[0] in stage_names:
        return argv[0], argv[1:]
    return "sft", argv


if __name__ == '__main__':
    # Match swift's cli worker preamble for device/dist-init parity.
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()

    # Optional leading positional stage token (swift's subcommand form):
    # ``run_swift.py pt --model ...``. Anything else (a flag) means sft.
    from torchturbo.adapters.msswift.stages import stage_wiring

    argv = sys.argv[1:]
    stage, argv = _extract_stage(argv)

    # Shared tt parser strips all --tt_*; swift sees only its own args.
    from torchturbo.engine.args import parse_torchturbo_args, set_torchturbo_args

    tt_args, swift_argv = parse_torchturbo_args(argv)
    set_torchturbo_args(tt_args)

    sys.argv = [sys.argv[0]] + swift_argv

    main = stage_wiring(stage)[1]

    # Activate msswift (registers models/trainers, swaps pipeline globals) before main() resolves its pipeline global.
    import torchturbo

    torchturbo.activate("msswift")

    main()
