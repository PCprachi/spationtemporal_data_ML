
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


# Load data
data = pd.read_csv('/home/sumit/Downloads/prachi/Classification/merged_df.csv')  # Replace with your data

# Define bins and labels for visibility categories
bins = [-float('inf'), 0, 5, 10, float('inf')]
labels = ['very low visibility', 'low visibility', 'medium visibility', 'high visibility']

data['visibility_category'] = pd.cut(data['TARGETS'], bins=bins, labels=labels)
print(data['visibility_category'].value_counts())

# Split the data into features (X) and the target column (y)
X = data.drop(['TARGETS', 'visibility_category'], axis=1)
y = data['visibility_category']

# Split the data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the value counts of each category in the validation set before oversampling
print("Value Counts before oversampling:\n", y_valid.value_counts())

# Implement RandomOverSampler for balancing the classes on the validation set only
ros = RandomOverSampler(sampling_strategy='auto')  # Adjust 'sampling_strategy' as needed
X_valid_resampled, y_valid_resampled = ros.fit_resample(X_valid, y_valid)

# Display the value counts after oversampling
print("\nValue Counts after oversampling:\n", pd.Series(y_valid_resampled).value_counts())

# Define and train models
models = {
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid_resampled)

    print(f"\n{name}:\n")
    print("Classification Report:\n", classification_report(y_valid_resampled, y_pred))
    print("Accuracy Score:", accuracy_score(y_valid_resampled, y_pred))


# %%
from sklearn.model_selection import GridSearchCV

# Define the parameter grids for hyperparameter tuning
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features' :['sqrt']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

# Perform hyperparameter tuning for each model
for name, model in models.items():
    if name in param_grids:
        param_grid = param_grids[name]
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best Parameters for {name}:", best_params)
        model.set_params(**best_params)  # Set the best parameters for the model

# Evaluate models with tuned hyperparameters on the resampled validation set
for name, model in models.items():
    y_pred = model.predict(X_valid_resampled)
    print(f"\n{name} with Tuned Hyperparameters:\n")
    print("Classification Report:\n", classification_report(y_valid_resampled, y_pred))
    print("Accuracy Score:", accuracy_score(y_valid_resampled, y_pred))


# %%



