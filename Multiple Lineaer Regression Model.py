# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Attendance_Rate': [85, 90, 95, 85, 90],
    'Practice_Tests_Taken': [1, 2, 3, 4, 5],
    'Final_Score': [52, 58, 64, 67, 73]
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Define independent variables (X) and dependent variable (y)
X = df[['Hours_Studied', 'Attendance_Rate', 'Practice_Tests_Taken']]
y = df['Final_Score']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Get model coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Predict the final score for a student with 6 hours of study, 95% attendance, and 6 practice tests
predicted_score = model.predict([[6, 95, 6]])

# Output model details and prediction
print(f"Intercept (b₀): {intercept}")
print(f"Coefficients (b₁, b₂, b₃): {coefficients}")
print(f"Predicted Final Score: {predicted_score[0]}")

# ----------- 3D Visualization (only using two features for simplicity) ------------

# Create a grid of values for 'Hours_Studied' and 'Attendance_Rate'
x_surf, y_surf = np.meshgrid(
    np.linspace(X['Hours_Studied'].min(), X['Hours_Studied'].max(), 10),
    np.linspace(X['Attendance_Rate'].min(), X['Attendance_Rate'].max(), 10)
)

# Use the average of 'Practice_Tests_Taken' for all points in the surface
z_surf = model.predict(np.array([
    x_surf.ravel(), 
    y_surf.ravel(), 
    np.full_like(x_surf.ravel(), X['Practice_Tests_Taken'].mean())
]).T)

# Reshape prediction surface to match the grid
z_surf = z_surf.reshape(x_surf.shape)

# Create 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot original data points
ax.scatter(X['Hours_Studied'], X['Attendance_Rate'], y, color='red')

# Plot the regression surface
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='blue')

# Set axis labels
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Attendance Rate (%)')
ax.set_zlabel('Final Score')

# Set plot title
plt.title('Multiple Linear Regression Surface')
plt.show()
