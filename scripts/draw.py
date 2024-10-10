import os, sys
import json
import pickle
import numpy as np
import traceback
import matplotlib.pyplot as plt
from matplotlib import gridspec

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def load_jsonl(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data


def draw_iterations():
    data = load_jsonl("experiments/llama-guard-3-8b/Experiment 3_ universal_no_restriction_suffix/Universal_no_restriction_50_200_suffix.jsonl")
    length = np.asarray([len(adv) for adv in data[0]["result_adv_list"]])
    success_rate = np.asarray([0.99135417, 0.99166667, 0.98854167, 0.99583333, 0.99479167,
       0.99114583, 0.97447917, 0.98385417, 0.99010417, 0.98177083,
       0.9859375 , 0.9921875 , 0.97708333, 0.97552083, 0.97395833,
       0.97395833, 0.97395833, 0.97395833, 0.9828125 , 0.96666667,
       0.97135417, 0.9734375 , 0.96875   , 0.97083333, 0.96927083,
       0.97395833, 0.99270833, 0.99375   , 0.99427083, 0.98958333,
       0.98958333, 0.97395833, 0.97552083, 0.9828125 , 0.96041667,
       0.97604167, 0.96770833, 0.959375  , 0.95833333, 0.98072917,
       0.98333333, 0.98541667, 0.98177083, 0.97760417, 0.96770833,
       0.95729167, 0.96927083, 0.971875  , 0.99114583, 0.9955    ])
    loss = -np.asarray(data[0]["train_loss_list"])
    loss = (loss - loss.min()) / (loss.max() - loss.min())
    iterations = np.arange(len(data[0]["result_adv_list"]))

    # Creating a figure with two subplots (one above the other)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))

    # Upper subplot: Plotting the loss curve
    ax1.plot(iterations, loss, 'r-', label="Fitness score")  # 'r-' is a red solid line
    ax1.set_ylabel('Fitness score')
    ax1.legend()

    # Lower subplot with two y-axes
    ax2a = ax2  # First y-axis for the success rate
    ax2b = ax2.twinx()  # Second y-axis for the length

    # Plotting the success rate on the first y-axis
    line1, = ax2a.plot(iterations, success_rate, 'b-', label="Success rate")  # 'b-' is a blue solid line
    ax2a.set_xlabel('Iterations')
    ax2a.set_ylabel('Success rate')
    ax2a.set_ylim([0, 1])
    ax2a.tick_params(axis='y')

    # Plotting the length on the second y-axis
    line2, = ax2b.plot(iterations, length, 'g:', label="Length")  # 'g-' is a green solid line
    ax2b.set_ylabel('Length')
    ax2b.set_ylim([0, 75])
    ax2b.tick_params(axis='y')

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="lower left")  # Adjust loc as needed

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=0.4, h_pad=0.1)

    # Show the plot
    plt.savefig("figures/iterations.pdf")


def draw_length():
    data_files = [
        "experiments/llama-guard-3-8b/Experiment 8_universal_vocab_restriction_random/Universal_vocab_restriction_random_test_all.npy",
        "experiments/llama-guard-3-8b/Experiment 7_ universal_no_restriction_random/Universal_no_restriction_random_test_all.npy",
        "experiments/llama-guard-2-8b/Experiment 7_ universal_no_restriction_random/Universal_no_restriction_random2_test_all.npy",
        "experiments/vicuna-7b/Experiment 8_ Universal_random_vocab_restriction/universal_vocab_restriction_random_test_all.npy",
    ]

    data = []
    for data_file in data_files:
        data.append(np.load(data_file))
    data = np.hstack(data).T

    x1_labels = ["<100", "100-500", ">500"]
    y1 = [data[data[:,1] < 100, 0].mean(),
          data[np.logical_and(data[:,1] >= 100, data[:,1] <= 500), 0].mean(),
          data[data[:,1] > 500, 0].mean()]
    y1 = (np.asarray(y1) + 2) / 3
    x2_labels = ["Reasoning","Math","Programming","Comprehension","General"]
    y2 = [data[data[:,2] == i, 0].mean() for i in range(5)]
    y2 = (np.asarray(y2) + 2) / 3

    fig = plt.figure(figsize=(6, 1.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # First subplot
    ax1.bar(x1_labels, y1, width=0.5, color='blue', edgecolor='black')
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Success rate')
    ax1.set_xticks(x1_labels)
    ax1.set_ylim([0.8, 1])

    # Second subplot
    ax2.bar(x2_labels, y2, width=0.5, color='green', edgecolor='black')
    ax2.set_ylabel('Success rate')
    ax2.set_xticks(x2_labels)
    ax2.set_ylim([0.8, 1])

    ax2.set_xticklabels(x2_labels, rotation=15, ha="right")

    plt.tight_layout(pad=0.1)
    plt.savefig("figures/length.pdf")


def draw_mitigation():
    model_names = {
        "llama-guard-7b": "meta-llama/LlamaGuard-7b",
        "llama-guard-2-8b": "meta-llama/Meta-Llama-Guard-2-8B",
        "llama-guard-3-8b": "meta-llama/Llama-Guard-3-8B",
        "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    }

    experiment_dir = "experiments"

    data = {
        model_key: {"attack": {}, "normal": {}} for model_key in list(model_names.keys())
    }

    for root, dirs, files in os.walk(experiment_dir):
        for file in files:
            model_key = ""
            for k in list(model_names.keys()):
                if k in root:
                    model_key = k
                    break
            
            if file.endswith('.npy') and "Experiment 3" in root:
                if "smooth_llm" in file:
                    defense_name = file[:-4][file.index("smooth_llm"):]
                    data[model_key]["attack"][defense_name] = np.load(os.path.join(root, file)).reshape(-1).mean()
                elif "resilient_optimization" in file:
                    defense_name = "resilient_optimization"
                    data[model_key]["attack"][defense_name] = np.load(os.path.join(root, file)).reshape(-1).mean()
            if file.endswith('.npy') and "test_all" in file:
                if "smooth_llm" in file:
                    defense_name = file[:file.index("_test_all")]
                    tmp_data = np.load(os.path.join(root, file))
                    data[model_key]["normal"][defense_name] = (tmp_data[0, tmp_data[1, :] == 1].mean(),
                                                               tmp_data[0, tmp_data[1, :] == 0].mean())
                elif "resilient_optimization" in file:
                    defense_name = "resilient_optimization"
                    tmp_data = np.load(os.path.join(root, file))
                    data[model_key]["normal"][defense_name] = (tmp_data[1, tmp_data[2, :] == 1].mean(),
                                                               tmp_data[1, tmp_data[2, :] == 0].mean())
                    data[model_key]["normal"]["original"] = (tmp_data[0, tmp_data[2, :] == 1].mean(),
                                                             tmp_data[0, tmp_data[2, :] == 0].mean())

    data["llama-guard-7b"]["attack"]["original"] = 0.963
    data["llama-guard-2-8b"]["attack"]["original"] = 0.972
    data["llama-guard-3-8b"]["attack"]["original"] = 0.988
    data["vicuna-7b"]["attack"]["original"] = 0.998

    models = ["llama-guard-3-8b", "llama-guard-2-8b", "llama-guard-7b", "vicuna-7b"]
    model_labels = ["Llama-Guard-3", "Llama-Guard-2", "Llama-Guard", "Vicuna"]
    groups = ["original", "smooth_llm_insert_0.1", "smooth_llm_replace_0.1", "smooth_llm_patch_0.1", "resilient_optimization"]
    group_labels = ["Original", "Random perturbation\n(insert,p=0.1)", "Random perturbation\n(replace,p=0.1)", "Random perturbation\n(patch,p=0.1)", "Resilient optimization"]
    hatches = ['', '/', '\\', '+', '-']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 2))

    width = 0.1
    x = np.arange(len(models))
    for group_id, group in enumerate(groups):
        ax2.bar(x + (1 - len(groups)) * width * 0.5 + group_id * width, [data[model]["normal"][group][0] for model in models], width=width, edgecolor='black', hatch=hatches[group_id], label=group_labels[group_id])
        ax3.bar(x + (1 - len(groups)) * width * 0.5 + group_id * width, [data[model]["normal"][group][1] for model in models], width=width, edgecolor='black', hatch=hatches[group_id], label=group_labels[group_id])
        ax1.bar(x + (1 - len(groups)) * width * 0.5 + group_id * width, [data[model]["attack"][group] for model in models], width=width, edgecolor='black', hatch=hatches[group_id], label=group_labels[group_id])
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x, labels=model_labels)
        ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax2.set_title("True positive rate without attacks")
    ax3.set_title("False positive rate without attacks")
    ax1.set_title("Attack success rate")

    handles, labels = [], []
    for ax in [ax3]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.91, 0.89), ncol=1)
    plt.tight_layout(pad=0.2)
    fig.subplots_adjust(right=0.82, wspace=0.2)
    plt.savefig("figures/mitigation.pdf")


def draw_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))

    models = ["Llama-Guard-3", "Vicuna-v1.5-7b"]
    groups = ["Baseline (GCG)", "+ Token subsitution", "+ Fitness function"]

    success_rate = np.asarray([[0.99, 0.97, 0.97],
                               [1, 0.99, 0.998]])
    length = np.asarray([[203.2, 175.9, 40.3],
                         [125.6, 104.95, 27.8]])

    width = 0.3
    ax1.bar([0-width, 1-width], success_rate[:, 0], width=width, edgecolor='black', label=groups[0])
    ax1.bar([0, 1], success_rate[:, 1], width=width, hatch='/', edgecolor='black', label=groups[1])
    ax1.bar([0+width, 1+width], success_rate[:, 2], width=width, hatch='\\', edgecolor='black', label=groups[2])
    ax1.set_xticks([0, 1], labels=models)
    ax1.set_ylabel("Success rate")

    ax2.bar([0-width, 1-width], length[:, 0], width=width, edgecolor='black', label=groups[0])
    ax2.bar([0, 1], length[:, 1], width=width, hatch='/', edgecolor='black', label=groups[1])
    ax2.bar([0+width, 1+width], length[:, 2], width=width, hatch='\\', edgecolor='black', label=groups[2])
    ax2.set_xticks([0, 1], labels=models)
    ax2.set_ylabel("Length of adv. prompts")
    
    handles, labels = [], []
    for ax in [ax1]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.05, 0.1), ncol=3)
    plt.tight_layout(pad=0.2)
    fig.subplots_adjust(bottom=0.3)

    plt.savefig("figures/ablation.pdf")


if __name__ == "__main__":
    draw_mitigation()