# 📈 Stock Market Prediction Project

This **README** serves as a guide to help you navigate through the project folder and understand the purpose of each file and component.
The project involves **comparing predictions made by three different models** — **Linear Regression**, **Random Forest**, and **LSTM (Long Short-Term Memory)** — on time series stock market data.

---

## 🗂️ Project Structure

### **1. Code Files**

- **`main.py`**Central script for the project.Loads test data from NumPy files, preprocesses the data, implements and utilizes the three machine learning models, and evaluates their performance.It also generates visualizations comparing **actual vs. predicted prices** using Matplotlib.
- **`linear_regression.ipynb`**Jupyter Notebook focused on implementing and analyzing the **Linear Regression** model.Includes **exploratory data analysis (EDA)** and visualizations of Linear Regression results.
- **`lstm.ipynb`**Jupyter Notebook focused on implementing and analyzing the **LSTM** model.Provides **in-depth visualizations** and analysis of LSTM performance on time series data.
- **`random_forest.ipynb`**
  Jupyter Notebook focused on implementing and analyzing the **Random Forest** model.
  Used for EDA and visualization of Random Forest results.

---

### **2. Data Files**

- **`tf_X_test.npy`** – Test features for TensorFlow-based models.
- **`tf_y_test.npy`** – Test labels corresponding to `tf_X_test.npy`.
- **`sk_X_test.npy`** – Test features for Scikit-learn-based models.
- **`sk_y_test.npy`** – Test labels corresponding to `sk_X_test.npy`.
- **`closedf.npy`** – Scaled dataset used as input for model training and evaluation.
- **`close_stock.npy`** – Stock data used for visualization and comparison of predictions with actual stock prices.

---

### **3. Model Files**

- **`model_feature.h5`** – TensorFlow model trained on specific features, used for stock price prediction.
- **`model_close.h5`** – TensorFlow model trained on closing prices, used for predicting closing stock prices.
- **`model_close_history.pkl`** – Pickle file containing training history of `model_close.h5`, used to analyze training progress and performance.
- **`model_feature_history.pkl`** – Pickle file containing training history of `model_feature.h5`, used for performance analysis.

---

### **4. Dataset Files**

- **`TSLA_dataset.xlsx`** – Excel file containing **Tesla stock data and features** used for model training and testing.

---

### **5. Other Files**

- **`demonstration.png`** – Image showing the result or output visualization of the model predictions.
- **`.DS_Store`** and **`__MACOSX/`** – System-generated metadata files (can be ignored).

---

### **6. Dependencies Used**

| Library                | Purpose                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| **TensorFlow**   | Implementing and training deep learning models, including Linear Regression and LSTM |
| **Scikit-learn** | Building machine learning models (e.g., Random Forest) and preprocessing data        |
| **Pandas**       | Data manipulation and analysis                                                       |
| **NumPy**        | Numerical computations and array handling                                            |
| **Matplotlib**   | Data visualization and comparison plots                                              |
| **Plotly**       | Interactive visualizations                                                           |
| **Joblib**       | Saving and loading Scikit-learn models and objects                                   |

---

### 📝 Notes

- The **`.DS_Store`** file and **`__MACOSX`** folder can be safely ignored; they are **system metadata** created by macOS.
- Model files (`.h5`, `.pkl`) are **large binary files** used for inference and training history, not for direct editing.
