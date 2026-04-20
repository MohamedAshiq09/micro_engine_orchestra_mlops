import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

feature1 = np.random.randn(n_samples) * 10 + 50
feature2 = np.random.randn(n_samples) * 5 + 20
feature3 = np.random.randn(n_samples) * 15 + 100

# Create target with some relationship
target = (
    feature1 * 0.5 + 
    feature2 * 1.2 + 
    feature3 * 0.3 + 
    np.random.randn(n_samples) * 5
)

# Save data as CSV
data_array = np.column_stack([feature1, feature2, feature3, target])
np.savetxt('ml_model/data.csv', data_array, delimiter=',', 
           header='feature1,feature2,feature3,target', comments='')

# Train model
X = np.column_stack([feature1, feature2, feature3])
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model trained with R² score: {score:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save training statistics
train_stats = {
    'mean': {
        'feature1': float(X[:, 0].mean()),
        'feature2': float(X[:, 1].mean()),
        'feature3': float(X[:, 2].mean())
    },
    'std': {
        'feature1': float(X[:, 0].std()),
        'feature2': float(X[:, 1].std()),
        'feature3': float(X[:, 2].std())
    }
}

with open('models/train_stats.pkl', 'wb') as f:
    pickle.dump(train_stats, f)

print("Model and statistics saved successfully!")
