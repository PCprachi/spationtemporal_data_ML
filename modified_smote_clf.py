
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
data = pd.read_csv('/DATA2/MS_ppushpendra/ml/ML PROJECT/merged_df.csv')  

# Define bins and labels for visibility categories
bins = [-float('inf'), 0, 5, 10, float('inf')]
labels = ['very low visibility', 'low visibility', 'medium visibility', 'high visibility']
data['visibility_category'] = pd.cut(data['TARGETS'], bins=bins, labels=labels)

# Split data into features and target
X = data.drop(['TARGETS', 'visibility_category'] + [f'Unnamed: {i}' for i in range(20, 27)], axis=1)
y = data['visibility_category']

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to training data only
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define RandomForestClassifier with class weights
class_weights = {'low visibility': 1, 'medium visibility': 1, 'very low visibility': 5, 'high visibility': 5}
rf_model = RandomForestClassifier(class_weight=class_weights)

# GridSearchCV setup
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Create model with best parameters
best_model = RandomForestClassifier(**best_params, class_weight=class_weights)
best_model.fit(X_train_resampled, y_train_resampled)

# Evaluate using cross-validation
cv_results = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
print("Cross-Validated Accuracy:", cv_results.mean())

# Evaluate on validation set
y_pred = best_model.predict(X_valid)
print("Classification Report on Validation Set:\n", classification_report(y_valid, y_pred))
print("Accuracy Score on Validation Set:", accuracy_score(y_valid, y_pred))

# Save the model
joblib.dump(best_model, 'best_random_forest_model.joblib')
