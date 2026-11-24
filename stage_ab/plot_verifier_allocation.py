"""Render the verification budget experiments from their saved measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(directory):
    initial = json.loads((directory / "evaluation.json").read_text())["summary"]["fresh"]
    confirm = json.loads((directory / "confirmation.json").read_text())["summary"]["confirmation"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.1), layout="constrained")
    colors = ["#59697b", "#275986", "#258072", "#142f46"]
    names = ["Box gradients, 10", "Signed endpoints, 10", "Anchor lookup, 10", "Anchor lookup, 128"]
    keys = ["selective/block/10", "selective_point/block/10", "selective_anchor/block/10",
            "selective_anchor/all/128"]
    times = [1000 * initial[k]["certificate_s"] for k in keys]
    bars = left.barh(names, times, color=colors, height=.57)
    left.bar_label(bars, labels=[f"{x:.0f}" for x in times], padding=4)
    left.set_xlim(0, max(times)*1.18)
    left.invert_yaxis()
    left.set_title("Cost of the verifier", loc="left", weight="bold")
    left.set_xlabel("Verification time per 60 steps (ms)")
    def fraction(key):
        return 100 * confirm[key]["neural_fraction"]
    counts = [0, 4, 10, 20]
    shares = [fraction("selective_anchor/all/128")] + [
        fraction(f"anchor_then_point/block/{k}") for k in counts[1:]]
    right.plot(counts, shares, marker="o", color=colors[1], label="Neighboring faces")
    for name, rule, color, marker in [
        ("Largest bound", "top", colors[0], "s"),
        ("Remaining uncertainty", "uncertainty", colors[2], "D")
    ]:
        right.scatter([10], [fraction(f"anchor_then_point/{rule}/10")],
                      label=name, color=color, marker=marker, zorder=3)
    right.set_xticks(counts)
    right.set_ylim(min(shares)-1, max(shares)+1.5)
    right.set_title("Additional point checks after all-cell lookup", loc="left", weight="bold")
    right.set_xlabel("Cells receiving a point check per step")
    right.set_ylabel("Certified neural faces (%)")
    right.legend(loc="upper left", frameon=False, fontsize=9)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x" if axis is left else "y", alpha=.18)
        axis.set_axisbelow(True)
    fig.savefig(directory / "allocation.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
