import matplotlib.pyplot as plt

# Data for the sensitivity analysis
current_speeds = [0.5, 0.4, 0.25, 0.6, 0.75]  # Ocean current speeds in m/s
search_probabilities = [95, 90, 85, 95, 60]  # Corresponding search probabilities (%)
labels = ['Standard', '-10% decrease', '-25% decrease', '+10% increase', '+25% increase']  # Labels for the bars

# Create bar chart
plt.figure(figsize=(8, 6))
plt.bar(labels, search_probabilities, color='#81B2DF')

# Title and labels
plt.title('Sensitivity Analysis of Ocean Current Speed and Search Probability', fontsize=14, fontweight='bold', family='Times New Roman')
plt.xlabel('Ocean Current Speed Change', fontsize=12, family='Times New Roman')
plt.ylabel('Search Probability (%)', fontsize=12, family='Times New Roman')

# Display the plot
plt.tight_layout()
plt.show()
