import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils import load_video, get_all_paths


def classify_by_path(path):
    pstr = path.as_posix() if hasattr(path, "as_posix") else str(path)
    if "strong_movement" in pstr:
        return "strong_movement"
    if "low_movement" in pstr:
        return "low_movement"
    return "other"


def sample_gradient(cmap_name, n, vmin=0.3, vmax=0.9):
    """Pick n evenly spaced colors from a sequential colormap in [vmin, vmax]."""
    cmap = plt.get_cmap(cmap_name)
    if n <= 0:
        return []
    if n == 1:
        return [cmap((vmin + vmax) / 2.0)]
    vals = np.linspace(vmin, vmax, n)
    return [cmap(v) for v in vals]


def nanpad_stack(list_of_1d_arrays):
    """Pad sequences with NaN to the max length and stack -> (N_series, T_max)."""
    if not list_of_1d_arrays:
        return None
    max_len = max(len(a) for a in list_of_1d_arrays)
    M = np.full((len(list_of_1d_arrays), max_len), np.nan, dtype=float)
    for i, a in enumerate(list_of_1d_arrays):
        L = len(a)
        if L > 0:
            M[i, :L] = a
    return M


def series_mean(list_of_1d_arrays):
    M = nanpad_stack(list_of_1d_arrays)
    return None if M is None else np.nanmean(M, axis=0)


def series_median(list_of_1d_arrays):
    M = nanpad_stack(list_of_1d_arrays)
    return None if M is None else np.nanmedian(M, axis=0)


def percentile_bands(list_of_1d_arrays, q_low=25, q_high=75):
    M = nanpad_stack(list_of_1d_arrays)
    if M is None:
        return None, None
    low = np.nanpercentile(M, q_low, axis=0)
    high = np.nanpercentile(M, q_high, axis=0)
    return low, high


def main():
    paths = get_all_paths("../../data/input")

    # Collect per class
    series_by_group = {"low_movement": [], "strong_movement": [], "other": []}
    for path in paths:
        group = classify_by_path(path)
        # Adjust 'length' as you like; using 400 to keep sequences comparable
        video, frames, filename = load_video(str(path), length=400, order="CTHW")
        corrs = [np.corrcoef(frames[i].ravel(), frames[i + 1].ravel())[0, 1]
                 for i in range(len(frames) - 1)]
        series_by_group[group].append((filename, corrs))

    # Colorblind-friendly sequential palettes (two hues)
    colors_low    = sample_gradient("Blues",   len(series_by_group["low_movement"]))
    colors_strong = sample_gradient("Oranges", len(series_by_group["strong_movement"]))
    colors_other  = sample_gradient("Greys",   len(series_by_group["other"]))

    # Full-color hues for summary lines/bands
    med_color_low    = plt.get_cmap("Blues")(0.80)
    med_color_strong = plt.get_cmap("Oranges")(0.80)
    med_color_other  = plt.get_cmap("Greys")(0.65)

    plt.figure(figsize=(12, 7))

    # Transparent individual lines behind everything
    for i, (_, corrs) in enumerate(series_by_group["low_movement"]):
        plt.plot(corrs, color=colors_low[i], linewidth=1.3, alpha=0.22, zorder=1)
    for i, (_, corrs) in enumerate(series_by_group["strong_movement"]):
        plt.plot(corrs, color=colors_strong[i], linewidth=1.3, alpha=0.22, zorder=1)
    for i, (_, corrs) in enumerate(series_by_group["other"]):
        plt.plot(corrs, color=colors_other[i], linewidth=1.1, alpha=0.18, zorder=1)

    # Gather arrays
    low_arrays    = [np.asarray(c) for _, c in series_by_group["low_movement"]]
    strong_arrays = [np.asarray(c) for _, c in series_by_group["strong_movement"]]
    other_arrays  = [np.asarray(c) for _, c in series_by_group["other"]]

    # Summary stats
    low_med    = series_median(low_arrays)
    strong_med = series_median(strong_arrays)
    other_med  = series_median(other_arrays)

    low_mean    = series_mean(low_arrays)
    strong_mean = series_mean(strong_arrays)
    other_mean  = series_mean(other_arrays)

    # IQR bands
    low_q25, low_q75         = percentile_bands(low_arrays, 25, 75)
    strong_q25, strong_q75   = percentile_bands(strong_arrays, 25, 75)
    other_q25, other_q75     = percentile_bands(other_arrays, 25, 75)

    band_handles = []
    if low_q25 is not None:
        x = np.arange(len(low_q25))
        plt.fill_between(x, low_q25, low_q75, color=med_color_low, alpha=0.22, linewidth=0, zorder=2)
        band_handles.append(Patch(facecolor=med_color_low, alpha=0.22, label="low IQR (25–75%)"))
    if strong_q25 is not None:
        x = np.arange(len(strong_q25))
        plt.fill_between(x, strong_q25, strong_q75, color=med_color_strong, alpha=0.22, linewidth=0, zorder=2)
        band_handles.append(Patch(facecolor=med_color_strong, alpha=0.22, label="strong IQR (25–75%)"))
    if other_q25 is not None and len(series_by_group["other"]) > 0:
        x = np.arange(len(other_q25))
        plt.fill_between(x, other_q25, other_q75, color=med_color_other, alpha=0.18, linewidth=0, zorder=2)
        band_handles.append(Patch(facecolor=med_color_other, alpha=0.18, label="other IQR (25–75%)"))

    # Bold medians on top (center of IQR by definition)
    avg_handles = []
    if low_med is not None:
        plt.plot(low_med, color=med_color_low, linewidth=3.0, zorder=3)
        avg_handles.append(Line2D([0], [0], color=med_color_low, lw=3, label="low median"))
    if strong_med is not None:
        plt.plot(strong_med, color=med_color_strong, linewidth=3.0, zorder=3)
        avg_handles.append(Line2D([0], [0], color=med_color_strong, lw=3, label="strong median"))
    if other_med is not None and len(series_by_group["other"]) > 0:
        plt.plot(other_med, color=med_color_other, linewidth=2.4, zorder=3)
        avg_handles.append(Line2D([0], [0], color=med_color_other, lw=2.4, label="other median"))

    # OPTIONAL: dashed means to visualize skew vs. median (comment out if not needed)
    mean_handles = []
    if low_mean is not None:
        plt.plot(low_mean, color=med_color_low, linewidth=1.4, linestyle="--", alpha=0.9, zorder=3)
        mean_handles.append(Line2D([0], [0], color=med_color_low, lw=1.4, ls="--", label="low mean"))
    if strong_mean is not None:
        plt.plot(strong_mean, color=med_color_strong, linewidth=1.4, linestyle="--", alpha=0.9, zorder=3)
        mean_handles.append(Line2D([0], [0], color=med_color_strong, lw=1.4, ls="--", label="strong mean"))
    if other_mean is not None and len(series_by_group["other"]) > 0:
        plt.plot(other_mean, color=med_color_other, linewidth=1.2, linestyle="--", alpha=0.9, zorder=3)
        mean_handles.append(Line2D([0], [0], color=med_color_other, lw=1.2, ls="--", label="other mean"))

    # Legend: medians + IQR bands (+ means if kept)
    handles = avg_handles + band_handles + mean_handles
    if handles:
        plt.legend(handles=handles, loc="lower right", frameon=True)

    plt.xlabel("Frame Index")
    plt.ylabel("Correlation with Next Frame")
    plt.title("Frame-to-Frame Correlation")
    plt.tight_layout()
    plt.savefig("correlation_across_frames.pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
