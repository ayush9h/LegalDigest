import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("training_log_seq2seq.csv")

plt.figure(figsize=(15, 8))
plt.plot(
    df["epoch"],
    df["loss"],
    "bo",
)

plt.title("Loss computation for Flan-t5-small model")

plt.xlabel("Number of epochs", loc="center")
plt.ylabel("Loss", loc="center")
plt.savefig("training_loss_seq2seq.png")


df2 = pd.read_csv("training_log_causal.csv")

plt.figure(figsize=(15, 8))
plt.plot(
    df2["epoch"],
    df2["loss"],
    "bo",
)

plt.title("Loss computation for Llama-1b instruct model")

plt.xlabel("Number of epochs", loc="center")
plt.ylabel("Loss", loc="center")
plt.savefig("training_loss_causal.png")
