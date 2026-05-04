import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_results(file_path):
    results = {
        "Baseline Model": {},
        "Augmented Model": {}
    }

    current_exp = None
    current_centre = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("=== Experiment:"):
                current_exp = line.split("Experiment:")[1].split("|")[0].strip()

                if current_exp not in results:
                    results[current_exp] = {}

            elif line.startswith("Test on"):
                current_centre = line.split("Test on")[1].strip()
            
            elif line.startswith("Accuracy:"):
                if current_exp is not None and current_centre is not None:
                    try:
                        acc = float(line.split("Accuracy:")[1].strip())
                        results[current_exp][current_centre] = acc
                    except ValueError:
                        print(f"Warning: Could not parse accuracy from line: {line}")
                        pass

    return results

def safe_get(results, exp, centre):
    return results.get(exp, {}).get(centre, None)

def build_table(results):
    rows = []

    for centre in ["centre_A", "centre_B", "centre_C"]:
        base_acc = safe_get(results, "Baseline Model", centre)
        aug_acc  = safe_get(results, "Augmented Model", centre)

        base_A = safe_get(results, "Baseline Model", "centre_A")
        aug_A  = safe_get(results, "Augmented Model", "centre_A")

        drop_base = base_A - base_acc if base_A is not None and base_acc is not None else None
        drop_aug  = aug_A - aug_acc if aug_A is not None and aug_acc is not None else None

        rows.append({
            "Centre": centre,
            "Baseline Acc": base_acc,
            "Augmented Acc": aug_acc,
            "Baseline Drop": drop_base,
            "Augmented Drop": drop_aug
        })

    df = pd.DataFrame(rows)
    return df

def plot_acc(df, title, save_path):
    centres = ["A", "B", "C"]
    x = range(len(centres))

    plt.figure(figsize=(8, 5), dpi=300)
    baseline_colour = "#2D6CDF"
    augmented_colour = "#E9A93F"

    plt.bar(x, df["Baseline Acc"], width=0.4, label="Baseline", color=baseline_colour)
    plt.bar([i + 0.4 for i in x], df["Augmented Acc"], width=0.4, label="Augmented", color=augmented_colour)

    plt.xticks([i + 0.2 for i in x], centres)
    plt.xlabel("Test Site")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.10)
    plt.title(title, fontsize=13, weight="bold")

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Value labels
    for i, v in enumerate(df["Baseline Acc"]):
        if pd.notna(v):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    
    for i, v in enumerate(df["Augmented Acc"]):
        if pd.notna(v):
            plt.text(i + 0.4, v, f"{v:.3f}", ha="center", va="bottom")
    
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_drop(df, save_path, title):
    centres = ["B", "C"]  # Exclude Centre A

    baseline_drops = df["Baseline Drop"][1:]
    augmented_drops = df["Augmented Drop"][1:]

    x = range(len(centres))

    plt.figure(figsize=(8, 5), dpi=300)
    baseline_colour = "#2D6CDF"
    augmented_colour = "#E9A93F"

    plt.bar(x, baseline_drops, width=0.4, label="Baseline", color=baseline_colour)
    plt.bar([i + 0.4 for i in x], augmented_drops, width=0.4, label="Augmented", color=augmented_colour)

    plt.xticks([i + 0.2 for i in x], centres)
    plt.xlabel("Test Site")
    plt.ylabel("Performance Drop")
    plt.title(title, fontsize=13, weight="bold")

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Value labels
    for i, v in enumerate(baseline_drops):
        if pd.notna(v):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    for i, v in enumerate(augmented_drops):
        if pd.notna(v):
            plt.text(i + 0.4, v, f"{v:.3f}", ha="center", va="bottom")

    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    results_dir = "results"
    csv_dir = os.path.join(results_dir, "csv")
    figure_dir = os.path.join(results_dir, "figures")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    for file in os.listdir(results_dir):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(results_dir, file)
        print(f"\nProcessing: {file}")

        results = parse_results(file_path)
        df = build_table(results)

        print(df)

        name = os.path.splitext(file)[0]

        # Save CSV
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)

        # Save figures
        plot_acc(
            df,
            title=f"Cross-Site Accuracy Comparison - {name}",
            save_path=os.path.join(figure_dir, f"{name}_acc.png")
        )
        plot_drop(
            df,
            title=f"Performance Drop from Source Domain (Site A) - {name}",
            save_path=os.path.join(figure_dir, f"{name}_drop.png")
        )

    print("\nAll results processed and saved.")