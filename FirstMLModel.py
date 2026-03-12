
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Prepare the Data
# X = Features (Hours Studied), y = Target (Test Score)
X = np.array([[1], [2], [3], [4], [5]]) 
y = np.array([10, 25, 35, 50, 65])

# 2. Initialize and Train the Model
model = LinearRegression()
model.fit(X, y)

# 3. Make Predictions for the line
y_pred = model.predict(X)

# 4. Visualizing the Results
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Scores') # Plot real data
plt.plot(X, y_pred, color='red', linewidth=2, label='ML Prediction') # Plot the line

# Add metadata to the graph
plt.title('Study Hours vs Test Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 5. Save the result for GitHub README
plt.savefig('results_plot.png')
print("Model trained successfully. Graph saved as 'results_plot.png'")

# Optional: Show the plot if running locally
# plt.show()