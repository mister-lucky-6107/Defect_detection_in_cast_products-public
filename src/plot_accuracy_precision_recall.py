import matplotlib.pyplot as plt

from set_seed import set_seed
set_seed(42)

def plot_accuracy_precision_recall(num_epochs, all_accuracy_val, all_precision_val, all_recall_val):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for i in range(len(all_accuracy_val)):
        plt.plot(
            range(num_epochs),
            all_accuracy_val[i],
            label=f"Fold {i+1}"
            )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Fold")

    plt.subplot(1, 2, 2)
    for i in range(len(all_precision_val)):
        plt.plot(
            range(num_epochs),
            all_precision_val[i],
            label=f"Fold {i+1}"
            )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.title("Validation Precision per Fold")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for i in range(len(all_recall_val)):
        plt.plot(
            range(num_epochs),
            all_recall_val[i],
            label=f"Fold {i+1}"
            )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.title("Validation Recall per Fold")

    # F1スコアのグラフ
    plt.subplot(1, 2, 2)
    for i in range(len(all_precision_val)):
        f1_scores = [
            2 * (p * r) / (p + r + 1e-8)
            for p, r in zip(all_precision_val[i], all_recall_val[i])
        ]
        plt.plot(
            range(num_epochs),
            f1_scores,
            label=f"Fold {i+1}"
        )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score per Fold")

    plt.figure(figsize=(6, 4))
    avg_f1_scores = []
    for epoch in range(num_epochs):
        epoch_f1s = []
        for i in range(len(all_precision_val)):
            p = all_precision_val[i][epoch]
            r = all_recall_val[i][epoch]
            f1 = 2 * (p * r) / (p + r + 1e-8)
            epoch_f1s.append(f1)
        avg_f1_scores.append(sum(epoch_f1s) / len(epoch_f1s))

    plt.plot(range(num_epochs), avg_f1_scores, label="Average F1", color='black')
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Average Validation F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.show()