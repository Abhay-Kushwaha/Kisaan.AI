import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score, f1_score, root_mean_squared_error

def Train_Crop_Recommendation():
    data = pd.read_csv("dataset/Crop_recommendation.csv")
    features = ['N','P','K','temperature','humidity','ph','rainfall']
    target = 'label'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Crop Recommendation Accuracy: {accuracy:.2f}")
    f1_value = f1_score(y_test, y_pred, average='weighted')
    print(f"Crop Recommendation F1 Score: {f1_value:.2f}")
    with open("model/crop_predict_model.pkl", "wb") as f:
        pickle.dump(model, f)

def Train_Crop_Price():
    data = pd.read_csv("dataset/Crop_Price.csv")
    le_state = LabelEncoder()
    le_crop = LabelEncoder()
    data['State'] = le_state.fit_transform(data['State'])
    data['Crop'] = le_crop.fit_transform(data['Crop'])
    features = ['State', 'Crop', 'CostCultivation', 'CostCultivation2', 'Production', 'Yield', 'Temperature', 'RainFall Annual']
    target = 'Price'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Crop Prediction Mean Squared Error: {mse:.2f}")
    # rmse = root_mean_squared_error(y_test, y_pred)
    # print(f"Crop Prediction Root Mean Squared Error: {rmse:.2f}")
    r2 = r2_score(y_test, y_pred)
    print(f"Crop Prediction R² Score: {r2:.2f}")

    with open("model/crop_price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/state_encoder.pkl", "wb") as f:
        pickle.dump(le_state, f)
    with open("model/crop_encoder.pkl", "wb") as f:
        pickle.dump(le_crop, f)

def Train_Disease():
    df = pd.read_csv("dataset/symptoms_df.csv")
    df['Symptoms'] = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].values.tolist()
    df['Symptoms'] = df['Symptoms'].apply(lambda x: list(set(s.strip().lower() for s in x if pd.notnull(s) and s.strip())))
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['Symptoms'])
    y = df['Disease']
    model = DecisionTreeClassifier()
    model.fit(X, y)
    with open('model/disease_model.pkl', 'wb') as f:
        pickle.dump((model, mlb), f)

def Train_Breast_Cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    top_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension'
    ]
    top_indices = [list(feature_names).index(f) for f in top_features]
    X_top = X[:, top_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Breast Cancer Accuracy: {acc:.2f}")
    print(f"Breast Cancer F1 Score: {f1:.2f}")
    with open("model/breast_cancer_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved")

# call function
# Train_Crop_Recommendation()
# Train_Crop_Price()
# Train_Disease()
# Train_Breast_Cancer()