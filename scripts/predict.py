import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from preprocessing_feature_engineering import get_processed_data

with open('results/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

test_path = 'data/test.csv'
X_test, y_test = get_processed_data(test_path, has_labels=True)

predictions = model.predict(X_test)

if y_test is not None:
    test_acc = accuracy_score(y_test, predictions)
    print(f"Test accuracy on test.csv: {test_acc}")
else:
    print("No labels in test.csv; accuracy not computed.")

# Save predictions
pred_df = pd.DataFrame({'Id': X_test.index, 'Cover_Type': predictions})  
pred_df.to_csv('results/test_predictions.csv', index=False)

print("Predictions saved to results/test_predictions.csv")