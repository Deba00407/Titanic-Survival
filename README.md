# Titanic Survival Prediction 🚢

A machine learning project that predicts passenger survival on the Titanic using a robust and modern preprocessing pipeline combined with a Decision Tree binary classifier. This repository includes the trained model, preprocessing pipeline, and a user-friendly Streamlit app for interactive predictions.

---

## 🚀 Project Overview

This project builds a predictive model using the famous Titanic dataset. It features a well-structured pipeline employing:

- **Data imputation** for missing values using `SimpleImputer` (with the most frequent strategy for categorical fields).
- **Categorical encoding** with a mix of **Ordinal Encoding** and **One-Hot Encoding** to properly handle categorical features.
- **Feature scaling** with `MinMaxScaler` where appropriate.
- **Feature selection** using `SelectKBest` with chi-squared scoring to identify the most relevant features.
- A **Decision Tree Classifier** for binary classification of survival outcome.
- Use of **`ColumnTransformer`** to apply transformations securely and efficiently within a pipeline.
- Packaging the entire preprocessing and model training steps into a single pipeline for streamlined training and prediction.
- A **Streamlit frontend** for easy interaction and prediction input.

---

## 🛠️ Technologies & Libraries

- Python 3.x
- scikit-learn (`sklearn`)
- pandas & numpy
- Streamlit (for the frontend)
- pickle (for model serialization)

---

## 🧰 Project Structure


---

## ⚙️ Data Preprocessing Pipeline

- **Imputation:**
  - Numeric columns: Missing values imputed with mean or median (configurable).
  - Categorical columns: Missing values imputed with the most frequent category using `SimpleImputer(strategy='most_frequent')`.

- **Encoding:**
  - **Ordinal Encoding** for ordinal categorical features (e.g., passenger class).
  - **One-Hot Encoding** for nominal categorical features (e.g., embark town, sex).

- **Scaling:**
  - Numeric features are scaled using `MinMaxScaler` to normalize their ranges.

- **Feature Selection:**
  - `SelectKBest` with `chi2` score function picks the top K features most predictive of survival.

- All transformations are implemented via **`ColumnTransformer`** ensuring modular, readable, and reproducible preprocessing.

---

## 🎯 Model Training

- The final classifier is a **Decision Tree** suited for binary classification (survived vs not survived).
- The full pipeline (imputation → encoding → scaling → feature selection → model training) ensures that the model can be saved and later loaded for inference without separate preprocessing steps.
- Model is saved using **pickle** (`model.pkl`).

---

## 📊 Using the Model

### Local Prediction

1. Load the model pipeline:
    ```python
    import pickle

    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    ```

2. Prepare your input data as a pandas DataFrame with columns in this order:

    ```python
    columns = ['sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'embark_town']
    ```

3. Run predictions:
    ```python
    prediction = model.predict(input_df)
    ```

### Interactive Frontend (Streamlit)

- Run the Streamlit app for an interactive UI:
    ```bash
    streamlit run app.py
    ```
- The app collects user inputs for all required fields, preprocesses inputs automatically, and shows survival predictions.

---

## 📦 Installation & Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Deba00407/Titanic-Survival
    cd Titanic-Survival
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

## 🔒 Security & Best Practices

- The pipeline encapsulates all data preprocessing and modeling steps, minimizing risks of data leakage.
- Using `ColumnTransformer` and pipelines ensures that the same transformations are consistently applied during training and inference.
- Missing values are handled gracefully with imputers avoiding runtime errors.
- Categorical encoding strategies are chosen to reflect feature semantics (ordinal vs nominal).
- The model file (`model.pkl`) is included in the repo for reproducible results.

---

## 📚 References

- [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic)
- [Scikit-learn Pipelines and ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [Streamlit Documentation](https://docs.streamlit.io)


---

## ⚖️ License

This project is licensed under the MIT License.

---

*Made with ❤️ and machine learning magic.*


