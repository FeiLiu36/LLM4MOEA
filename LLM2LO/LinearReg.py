import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor



# Load the dataset from the text file
data = np.loadtxt('data.txt', delimiter=' ',dtype=str)[:, :-1]
data = data.astype(float)

print("data size = ", data.shape)

data = np.random.permutation(data)
# Separate the features (variables) and the target (objective)
X = data[:, :-1]
y = data[:, -1]

# Save the dataset to the text file
np.savetxt('datasave.txt', data, delimiter=' ')

# Create a Linear Regression model
model = LinearRegression(fit_intercept=False)

# Fit the model to the dataset
#model.fit(X_train, y_train)
model.fit(X, y)

# Print the weight vector
weight_vector = model.coef_
#scaled_weight_vector = weight_vector / np.sum(weight_vector)
print('Weight vector:', weight_vector )

y_predict = model.predict(X)

delta_y = y-y_predict

print("Mean delta y :", np.mean(delta_y))
print("Standard Deviation:", np.std(delta_y))
print("Theta: ",np.std(delta_y)/np.mean(X))



# # Predict the target values for new data points
# predicted_y = model.predict(X_test)

# # Calculate the R-squared score
# r2 = r2_score(y_test, predicted_y)
# print("R-squared:", r2)

