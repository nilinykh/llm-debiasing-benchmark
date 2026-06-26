import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser


# This file keeps the original plot layout, but narrows it to one dataset,
# one annotation, and one condition at a time.


###############
# Gather data #
###############


def gather(base_dir: Path, dataset: str, annotation: str):

    print(f" - {dataset}/{annotation}")
    data_path = base_dir / "data" / dataset / annotation

    if not data_path.exists():
        raise Exception(f"Error: {data_path} not found")

    try:
        next(data_path.iterdir())
    except StopIteration:
        raise Exception(f"Error: {data_path} is empty")

    coeffs_all = []
    coeffs_exp = []
    coeffs_dsl = []
    coeffs_ppi = []

    for file in data_path.iterdir():
        data = np.load(file)
        coeffs_all.append(data["coeffs_all"])
        coeffs_exp.append(data["coeffs_exp"])
        coeffs_dsl.append(data["coeffs_dsl"])
        coeffs_ppi.append(data["coeffs_ppi"])

    return {
        "num_expert_samples": data["num_expert_samples"],
        "all": np.stack(coeffs_all, axis=0),
        "exp": np.stack(coeffs_exp, axis=0),
        "dsl": np.stack(coeffs_dsl, axis=0),
        "ppi": np.stack(coeffs_ppi, axis=0),
    }


def gather_hf(base_dir: Path, dataset: str, annotation: str):

    print(f" - {dataset}/{annotation}")
    data_path = base_dir / dataset / annotation

    if not data_path.exists():
        raise Exception(f"Error: {data_path} not found")

    required = {
        "all": "all.npy",
        "exp": "expert_only.npy",
        "dsl": "dsl.npy",
        "ppi": "ppi.npy",
        "num_expert_samples": "num_expert_samples.npy",
    }
    missing = [filename for filename in required.values() if not (data_path / filename).exists()]
    if missing:
        raise Exception(f"Error: missing files in {data_path}: {', '.join(missing)}")

    # HF baseline stores each metric as a .npy array instead of the per-run .npz
    # files used by the surprisal-text pipeline.
    return {
        "num_expert_samples": np.load(data_path / required["num_expert_samples"]),
        "all": np.load(data_path / required["all"]),
        "exp": np.load(data_path / required["exp"]),
        "dsl": np.load(data_path / required["dsl"]),
        "ppi": np.load(data_path / required["ppi"]),
    }


###########
# Metrics #
###########


def compute_rmse(coeffs_true, coeffs_pred):
    assert len(coeffs_true.shape) == 3
    assert coeffs_true.shape == coeffs_pred.shape
    error = (coeffs_true - coeffs_pred) / coeffs_true  # standardize per coeff
    rmse = np.sqrt(np.mean(error ** 2, axis=(0, 2)))
    num_repetitions = coeffs_true.shape[0]
    sd = np.sqrt(np.mean(error ** 2, axis=2)).std(axis=0)
    std_err = sd / np.sqrt(num_repetitions)
    upper = rmse + 2 * std_err
    lower = rmse - 2 * std_err
    return {
        "rmse": rmse,
        "upper": upper,
        "lower": lower,
    }


########################
# Axis Transformations #
########################


def forward(x, N):
    """
    Transformation from linspace(0,1) to logspace(log(200),log(N))/N
    """
    return N ** (x - 1) * 200 ** (1 - x)


#############
# Plot RMSE #
#############


def plot_rmse(ax, exp, dsl, ppi):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    X = np.linspace(0, 1, 10)

    ax.fill_between(
        X,
        exp["lower"],
        exp["upper"],
        color=colors[0],
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        X,
        exp["rmse"],
        "o-",
        color=colors[0],
        label=r"$\theta_\dagger$",
    )

    ax.fill_between(
        X,
        dsl["lower"],
        dsl["upper"],
        color=colors[1],
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        X,
        dsl["rmse"],
        "o-",
        color=colors[1],
        label="DSL",
    )

    ax.fill_between(
        X,
        ppi["lower"],
        ppi["upper"],
        color=colors[2],
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        X,
        ppi["rmse"],
        "o-",
        color=colors[2],
        label="PPI",
    )

    xticklabels = [f"{x:.2f}" for x in forward(X, 10000)]
    for i in [1, 2, 4, 5, 7, 8]:
        xticklabels[i] = ""
    ax.set_xticks(ticks=X, labels=xticklabels)
    ax.legend()


def plot_all(ax, data, datasets, annotations, num_reps):

    R_max = min(data[d][a]["all"].shape[0] for d in datasets for a in annotations)
    if num_reps is not None and R_max < num_reps:
        print(" - WARNING: not enough repetitions, using max available")
        R = R_max
    elif num_reps is not None:
        R = num_reps
    else:
        R = R_max

    coeffs_all = []
    coeffs_exp = []
    coeffs_dsl = []
    coeffs_ppi = []

    for d in datasets:
        for a in annotations:
            N = data[d][a]["all"].shape[0]
            subsample = np.random.choice(N, R, replace=False)
            coeffs_all.append(data[d][a]["all"][subsample, :, :])
            coeffs_exp.append(data[d][a]["exp"][subsample, :, :])
            coeffs_dsl.append(data[d][a]["dsl"][subsample, :, :])
            coeffs_ppi.append(data[d][a]["ppi"][subsample, :, :])

    coeffs_all = np.concatenate(coeffs_all, axis=-1)
    coeffs_exp = np.concatenate(coeffs_exp, axis=-1)
    coeffs_dsl = np.concatenate(coeffs_dsl, axis=-1)
    coeffs_ppi = np.concatenate(coeffs_ppi, axis=-1)

    rmse_exp = compute_rmse(coeffs_all, coeffs_exp)
    rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl)
    rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi)
    plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi)

    return R


def plot_dataset(ax, data, annotations, num_reps):

    R_max = min(data[a]["all"].shape[0] for a in annotations)
    if num_reps is not None and R_max < num_reps:
        print(" - WARNING: not enough repetitions, using max available")
        R = R_max
    elif num_reps is not None:
        R = num_reps
    else:
        R = R_max

    coeffs_all = []
    coeffs_exp = []
    coeffs_dsl = []
    coeffs_ppi = []

    for a in annotations:
        N = data[a]["all"].shape[0]
        subsample = np.random.choice(N, R, replace=False)
        coeffs_all.append(data[a]["all"][subsample, :, :])
        coeffs_exp.append(data[a]["exp"][subsample, :, :])
        coeffs_dsl.append(data[a]["dsl"][subsample, :, :])
        coeffs_ppi.append(data[a]["ppi"][subsample, :, :])

    coeffs_all = np.concatenate(coeffs_all, axis=-1)
    coeffs_exp = np.concatenate(coeffs_exp, axis=-1)
    coeffs_dsl = np.concatenate(coeffs_dsl, axis=-1)
    coeffs_ppi = np.concatenate(coeffs_ppi, axis=-1)

    rmse_exp = compute_rmse(coeffs_all, coeffs_exp)
    rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl)
    rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi)
    plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi)

    return R


def plot_annotation(ax, data, title):

    R = data["all"].shape[0]

    coeffs_all = data["all"]
    coeffs_exp = data["exp"]
    coeffs_dsl = data["dsl"]
    coeffs_ppi = data["ppi"]

    ax.set_title(title)
    rmse_exp = compute_rmse(coeffs_all, coeffs_exp)
    rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl)
    rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi)
    plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi)

    return R


def make_output_tag(condition, datasets, annotations):
    # Keep output names explicit so baseline and surprisal runs do not collide.
    tag_parts = [condition]
    if len(datasets) == 1:
        tag_parts.append(datasets[0])
    if len(annotations) == 1:
        tag_parts.append(annotations[0])
    return "_".join(tag_parts)


def run_original_style_plots(
    plot_dir,
    data,
    datasets,
    annotations,
    num_reps,
    xlabel,
    ylabel,
    rowsize,
    colsize,
    output_tag,
):
    # The plotting flow below intentionally stays close to the original script.
    fig, ax = plt.subplots(figsize=(colsize, rowsize))
    R = plot_all(ax, data, datasets, annotations, num_reps)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(plot_dir / f"rmse_all_{output_tag}.png")
    fig.savefig(plot_dir / f"rmse_all_{output_tag}.pdf")
    print("")
    print(f"Plot for all datasets (repetitions: {R})")

    print("")
    print("Plots for dataset:")
    rows = 2
    cols = 2
    titles = {
        "amazon": "Multi-domain Sentiment",
        "misinfo": "Misinfo-general",
        "biobias": "Bias in Biographies",
        "germeval": "Germeval18",
    }
    figsize = (cols * colsize, rows * rowsize)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, dataset in enumerate(datasets):
        ax = axs[i // rows, i % cols]
        ax.set_title(titles.get(dataset, dataset))
        R = plot_dataset(
            ax,
            data[dataset],
            annotations,
            num_reps,
        )
        print(f" - {dataset} (repetitions: {R})")
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(plot_dir / f"rmse_datasets_{output_tag}.png")
    fig.savefig(plot_dir / f"rmse_datasets_{output_tag}.pdf")

    print("")
    print("Plots for dataset/annotation:")
    rows = len(datasets)
    cols = len(annotations)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * colsize, rows * rowsize), squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, annotation in enumerate(annotations):
            R = plot_annotation(
                axs[i, j],
                data[dataset][annotation],
                f"{dataset}/{annotation}",
            )
            print(f" - {dataset}/{annotation} (repetitions: {R})")
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(plot_dir / f"rmse_annotations_{output_tag}.png")
    fig.savefig(plot_dir / f"rmse_annotations_{output_tag}.pdf")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--num_reps", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--annotation", type=str, default=None)
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "surprisal-text"],
        default="surprisal-text",
    )
    parser.add_argument("--output_root", type=Path, default=None)
    args = parser.parse_args()

    # Default to a central results folder so baseline and surprisal runs land together.
    if args.output_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
        output_root = workspace_root / "results" / "vary-expert-single"
    else:
        output_root = args.output_root

    plot_dir = output_root / args.condition
    plot_dir.mkdir(exist_ok=True, parents=True)

    datasets = ["amazon", "misinfo", "biobias", "germeval"]
    annotations = ["bert", "deepseek", "phi4", "claude"]

    if args.dataset is not None:
        datasets = [args.dataset]
    if args.annotation is not None:
        annotations = [args.annotation]

    xlabel = "Proportion of expert samples (log)"
    ylabel = "sRMSE"

    rowsize = 3
    colsize = 5

    print("")
    print("Gathering the data")
    gather_fn = gather_hf if args.condition == "baseline" else gather
    print(f"Condition: {args.condition}")
    output_tag = make_output_tag(args.condition, datasets, annotations)

    data = {
        d: {a: gather_fn(args.base_dir, d, a) for a in annotations}
        for d in datasets
    }
    run_original_style_plots(
        plot_dir,
        data,
        datasets,
        annotations,
        args.num_reps,
        xlabel,
        ylabel,
        rowsize,
        colsize,
        output_tag,
    )

    print("")
