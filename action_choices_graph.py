import matplotlib.pyplot as plt
import pandas as pd

# Updated dataset with three models
data_updated = {
    "Action": [
        "ROLL", "END_TURN", "MOVE_ROBBER", "DISCARD", "BUILD_ROAD",
        "BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD",
        "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY",
        "PLAY_ROAD_BUILDING", "MARITIME_TRADE"
    ],
    "Basic Reward Function": [
        36376, 36283, 6900, 7571, 3997, 1501, 640, 1461, 785,
        101, 92, 117, 5761
    ],
    "End Turn Penalty 2.5": [
        32890, 32705, 6654, 2941, 5216, 1787, 657, 2127, 1143,
        174, 141, 129, 11896
    ],
    "End Turn Penalty 10": [
        31870, 31630, 6668, 1255, 5622, 1798, 633, 2896, 1619,
        211, 218, 205, 14765
    ]
}

# Create a DataFrame
df_updated = pd.DataFrame(data_updated)

# Convert counts to percentages for each model
total_actions = {
    "Basic Reward Function": sum(df_updated["Basic Reward Function"]),
    "End Turn Penalty 2.5": sum(df_updated["End Turn Penalty 2.5"]),
    "End Turn Penalty 10": sum(df_updated["End Turn Penalty 10"])
}
df_updated["Basic Reward Function (%)"] = (df_updated["Basic Reward Function"] / total_actions["Basic Reward Function"]) * 100
df_updated["End Turn Penalty 2.5 (%)"] = (df_updated["End Turn Penalty 2.5"] / total_actions["End Turn Penalty 2.5"]) * 100
df_updated["End Turn Penalty 10 (%)"] = (df_updated["End Turn Penalty 10"] / total_actions["End Turn Penalty 10"]) * 100

# Plotting the percentage data for three models
plt.figure(figsize=(16, 9))
bar_width = 0.25
indices = range(len(df_updated["Action"]))

plt.bar(indices, df_updated["Basic Reward Function (%)"], width=bar_width, label="Basic Reward Function (%)", alpha=0.7)
plt.bar([i + bar_width for i in indices], df_updated["End Turn Penalty 2.5 (%)"], width=bar_width, label="End Turn Penalty 2.5 (%)", alpha=0.7)
plt.bar([i + 2 * bar_width for i in indices], df_updated["End Turn Penalty 10 (%)"], width=bar_width, label="End Turn Penalty 10 (%)", alpha=0.7)

# Customizing the plot
plt.xticks([i + bar_width for i in indices], df_updated["Action"], rotation=45, ha='right')
plt.xlabel("Action")
plt.ylabel("Percentage of Actions (%)")
plt.title("Percentage of Actions Chosen by Three Models")
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
