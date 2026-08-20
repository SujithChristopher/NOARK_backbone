"""Verify the mocap sync input on the Radxa Dragon Q6A 40-pin header.

The Raspberry Pi version (gpio_test.py) hardcodes gpiochip4 line 17 and the
libgpiod v1 API. Neither carries over: on the Q6A line 17 is claimed by a
kernel driver, header pin numbers are not TLMM line numbers, and the installed
Python binding is libgpiod v2. So this resolves header names ("PIN_11") from
the device tree, and reports edges rather than spamming the level.

    uv run python helper_programs/gpio_test_q6a.py             # watch PIN_11
    uv run python helper_programs/gpio_test_q6a.py --pin 29    # by line number
    uv run python helper_programs/gpio_test_q6a.py --list      # header pin map
    uv run python helper_programs/gpio_test_q6a.py --drive PIN_13
        # loopback self-test: jumper header pin 13 to pin 11, and the driven
        # square wave should show up as edges on the watched pin

Header GPIO is 3.3V (3.63V tolerant) - never feed it 5V. PIN_11 is the default
because it idles low and is the same physical hole the Pi used for GPIO17, so
existing trigger wiring does not have to move.
"""

from __future__ import annotations

import argparse
import glob
import sys
import threading
import time
from datetime import datetime, timedelta

import gpiod
from gpiod.line import Bias, Direction, Edge, Value

CHIP = "gpiochip4"  # the SoC TLMM; the 40-pin header lives here
DEFAULT_PIN = "PIN_11"  # = TLMM line 29


def header_map() -> dict[str, int]:
    """Header pin name -> TLMM line, from the device tree's gpio-line-names."""
    for path in glob.glob("/proc/device-tree/soc@0/pinctrl@*/gpio-line-names"):
        with open(path, "rb") as fh:
            names = fh.read().split(b"\x00")
        found = {n.decode(): i for i, n in enumerate(names) if n}
        if found:
            return found
    return {}


def resolve(pin: str | int) -> int:
    """Accept a TLMM line number or a device-tree header name such as PIN_11."""
    text = str(pin)
    if text.lstrip("-").isdigit():
        return int(text)
    with gpiod.Chip(f"/dev/{CHIP}") as chip:
        try:
            return chip.line_offset_from_id(text)
        except OSError as exc:
            known = ", ".join(sorted(header_map())) or "none"
            raise SystemExit(
                f"no line named {text!r} on {CHIP} ({exc}). Known names: {known}"
            ) from exc


def print_table() -> None:
    """List every header pin with its line number and whether it is claimed."""
    names = header_map()
    if not names:
        raise SystemExit("no gpio-line-names in the device tree")
    print(f"{CHIP} - 40-pin header (3.3V, 3.63V tolerant)\n")
    print(f"{'header pin':>10}  {'line':>4}  {'state':<9}  claimed by")
    with gpiod.Chip(f"/dev/{CHIP}") as chip:
        for name, line in sorted(
            names.items(), key=lambda kv: int(kv[0].split("_")[1])
        ):
            info = chip.get_line_info(line)
            level = "?"
            if not info.used:
                try:
                    req = gpiod.request_lines(
                        f"/dev/{CHIP}",
                        consumer="gpio_test",
                        config={line: gpiod.LineSettings(direction=Direction.INPUT)},
                    )
                    level = "1" if req.get_value(line) == Value.ACTIVE else "0"
                    req.release()
                except (OSError, ValueError):
                    level = "?"
            state = "USED" if info.used else f"free ({level})"
            who = info.consumer or ("a kernel driver" if info.used else "-")
            print(f"{name:>10}  {line:>4}  {state:<9}  {who}")
    print(
        "\n'free (N)' is the level the pin idles at right now. A sync input "
        "wants one that idles 0."
    )


def drive_square_wave(pin: str | int, hz: float, stop: threading.Event) -> None:
    """Toggle another header pin, for a jumper-wire loopback test."""
    line = resolve(pin)
    try:
        req = gpiod.request_lines(
            f"/dev/{CHIP}",
            consumer="gpio_test_drive",
            config={line: gpiod.LineSettings(direction=Direction.OUTPUT)},
        )
    except (OSError, ValueError) as exc:
        print(f"cannot drive {pin} (line {line}): {exc} - is it claimed? see --list")
        return
    print(f"driving {pin} (line {line}) at {hz} Hz - jumper it to the watched pin")
    level = Value.INACTIVE
    try:
        while not stop.wait(0.5 / hz):
            level = Value.ACTIVE if level == Value.INACTIVE else Value.INACTIVE
            req.set_value(line, level)
    finally:
        req.set_value(line, Value.INACTIVE)
        req.release()


def watch(pin: str | int, debounce_us: int) -> None:
    """Report every edge with its timing, plus a heartbeat when the line is idle."""
    line = resolve(pin)
    settings = gpiod.LineSettings(
        direction=Direction.INPUT,
        edge_detection=Edge.BOTH,
        bias=Bias.AS_IS,  # the TLMM ignores bias here; use an external pull-down
        debounce_period=timedelta(microseconds=debounce_us),
    )
    try:
        req = gpiod.request_lines(
            f"/dev/{CHIP}", consumer="gpio_test", config={line: settings}
        )
    except (OSError, ValueError) as exc:  # EBUSY raises OSError, EINVAL ValueError
        with gpiod.Chip(f"/dev/{CHIP}") as chip:
            info = chip.get_line_info(line)
        who = info.consumer or "a kernel driver"
        extra = f" - already claimed by {who}" if info.used else ""
        raise SystemExit(f"cannot watch line {line}: {exc}{extra}") from exc

    label = f"{pin} (line {line})" if str(pin) != str(line) else f"line {line}"
    level = 1 if req.get_value(line) == Value.ACTIVE else 0
    print(f"watching {CHIP} {label}, idle level = {level}. Ctrl-C to stop.")

    rising = falling = 0
    highs: list[float] = []
    last_ns = time.monotonic_ns()
    last_beat = time.monotonic()
    try:
        while True:
            if req.wait_edge_events(timedelta(milliseconds=200)):
                for event in req.read_edge_events():
                    now_ns = time.monotonic_ns()
                    held = (now_ns - last_ns) / 1e6
                    last_ns = now_ns
                    up = event.event_type == event.Type.RISING_EDGE
                    if up:
                        rising += 1
                    else:
                        falling += 1
                        highs.append(held)
                    # local wall clock, to line up with the mocap operator's watch
                    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # noqa: DTZ005
                    print(
                        f"  {stamp}  {'RISING ' if up else 'FALLING'}  "
                        f"after {held:8.1f} ms {'low' if up else 'high'}"
                    )
                    level = 1 if up else 0
                    last_beat = time.monotonic()
            elif time.monotonic() - last_beat >= 2.0:
                last_beat = time.monotonic()
                idle = (time.monotonic_ns() - last_ns) / 1e9
                print(
                    f"  ... level {level} for {idle:.1f} s "
                    f"({rising} rising, {falling} falling so far)"
                )
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        req.release()
        print(f"summary: {rising} rising, {falling} falling", end="")
        if highs:
            print(f", mean high {sum(highs) / len(highs):.1f} ms", end="")
        print()
        if rising == falling == 0:
            print(
                "no edges seen - is the trigger wired to this pin? "
                "Try --list to pick another, or --drive for a loopback test."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the GPIO sync input on the Dragon Q6A 40-pin header"
    )
    parser.add_argument(
        "-p",
        "--pin",
        default=DEFAULT_PIN,
        help=f"header name or TLMM line to watch (default {DEFAULT_PIN})",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="print the header pin -> line map and exit",
    )
    parser.add_argument(
        "--drive",
        metavar="PIN",
        help="also drive this pin as a square wave (jumper it to "
        "--pin for a loopback self-test)",
    )
    parser.add_argument(
        "--hz", type=float, default=1.0, help="square wave rate for --drive (default 1)"
    )
    parser.add_argument(
        "--debounce-us",
        type=int,
        default=0,
        help="debounce period in microseconds (default 0, none)",
    )
    args = parser.parse_args()

    if args.list:
        print_table()
        return

    stop = threading.Event()
    driver = None
    if args.drive:
        if str(resolve(args.drive)) == str(resolve(args.pin)):
            sys.exit("--drive and --pin must be different lines")
        driver = threading.Thread(
            target=drive_square_wave, args=(args.drive, args.hz, stop), daemon=True
        )
        driver.start()
        time.sleep(0.2)
    try:
        watch(args.pin, args.debounce_us)
    finally:
        stop.set()
        if driver is not None:
            driver.join(timeout=2)


if __name__ == "__main__":
    main()
