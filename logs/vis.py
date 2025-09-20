import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("training_log.csv")

plt.figure(figsize=(15, 8))
plt.plot(df["epoch"], df["loss"], "bo")
plt.xlabel("Number of epochs", loc="center")
plt.ylabel("Loss", loc="center")
plt.savefig("training_loss.png")
