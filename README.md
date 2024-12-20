# Energy-Consumption-Forecasting


## Project Overview
This project focuses on building a **forecasting model** to predict energy consumption for each hour of the next 24 hours. The model leverages time-series data, external features like weather and holidays, and an **LSTM** (Long Short-Term Memory) network for forecasting. The model is evaluated based on **Mean Absolute Percentage Error (MAPE)** to assess forecast accuracy. 

### Key Deliverables:
- **Forecasting Model**: Predict energy consumption for each of the next 24 hours.
- **Documentation**: Detailed report of preprocessing steps, feature engineering, model selection, and evaluation.
- **Output File**: A CSV file containing predicted energy values for the test dataset.

### Evaluation Metrics:
- **MAPE (Mean Absolute Percentage Error)**: The primary metric to evaluate forecast accuracy.
- **Bonus Points** for using **external features** (like weather data) or implementing **advanced forecasting models** (such as LSTM or ARIMA).

---

## 1. **Preprocessing**

The preprocessing stage is crucial for preparing the data before feeding it into the forecasting model. The steps included are:

### 1.1 Data Cleaning
- **Time Extrapolation**: Ensuring that the time series data is continuous without any gaps, allowing for smooth predictions.
- **Handling Missing Data**: Removing or imputing any NaN (Not a Number) values to prevent errors during training. This step is done using forward/backward filling, interpolation, or using models to predict missing values.

### 1.2 Feature Engineering and Selection
- **External Features**:
  - **Weather**: Weather data (such as temperature, humidity, etc.) is appended as external features, which are often correlated with energy consumption patterns.
  - **Holidays**: Information about public holidays or special days is included as a feature since energy usage patterns change during such times.
  
- **Correlation Analysis**: Correlations between features are analyzed to select the most relevant ones for training the model. This helps in reducing dimensionality and improving model performance and subsequent removal of redundant features.

### 1.3 Data Normalization
- The features are normalized to bring all variables to a similar scale, which helps in the faster convergence of the model. Normalization is done using Min-Max scaling or Standardization based on the characteristics of the dataset.

### 1.4 Window Creation
- Time-series data is split into **windows** of historical data (e.g., 24 hours of past data) for predicting the energy consumption of the next 24 hours. This creates the structure needed to train the LSTM model.

### 1.5 Data Splitting
- The dataset is split into **training, validation, and test sets**:
  - **Training**: 70% of the data.
  - **Validation**: 15% of the data used to tune the model.
  - **Test**: 15% of the data used to evaluate model performance after training.

---

## 2. **Model Design**

The **LSTM** model is selected for time-series forecasting due to its ability to capture long-term dependencies in data.

### 2.1 LSTM Architecture
The architecture is defined as:
- **Input Layer**: Takes historical energy consumption, weather, occupancy, and calendar as input.
- **LSTM Layer**: A single LSTM layer with **32 memory units** is used. This layer captures the long-term dependencies in the sequence of data.
- **Dense Layers**: Two **Dense layers** with **5 units** each are added to capture higher-level relationships and patterns.
- **Output Layer**: A **single node** representing the predicted energy consumption.

### 2.2 Loss Function and Optimizer
- **Loss Function**: **Mean Squared Error (MSE)** is used to measure the difference between the predicted and actual values.
- **Optimizer**: **RMSProp** is used for optimization as it adapts the learning rate and leads to faster convergence.

### 2.3 Hyperparameter Tuning
The performance of the LSTM model is optimized using the following methods:

#### 2.3.1 Random Search
Random Search is employed to explore the hyperparameter space. This involves:
- Randomly sampling hyperparameters (e.g., learning rate, LSTM units, batch size).
- Training multiple models with different hyperparameters.

#### 2.3.2 Grid Search
Grid Search is applied after Random Search to exhaustively search for the best set of hyperparameters in a defined grid. This step fine-tunes the model and leads to optimal performance.

---

## 3. **Training the Model**

The model is trained using the following parameters:
- **Epochs**: 20 epochs to ensure convergence.
- **Batch Size**: 64 is chosen based on the available hardware and experimentation.

During training, the model is validated on the validation set to ensure that it is not overfitting to the training data.

---

## 4. **Evaluation and Metrics**

### 4.1 Evaluation
The model is evaluated on the **test dataset** to check its generalization ability. The following metrics are used:
- **Mean Absolute Error (MAE)**:0.0595
- **Root Mean Squared Error (RMSE)**: 0.0876
- **Mean Absolute Percentage Error (MAPE)**: 5415912640867.0459

## 5. **Output**

After training and evaluation, the model generates predictions for the test dataset.




## 6. **Conclusion**

This forecasting model uses LSTM for energy consumption prediction, with key features like weather, holidays, and calendar data. The model is evaluated using standard metrics such as MAE, RMSE, and MAPE. By following this approach, you can predict future energy consumption with reasonable accuracy.

Here’s a suggestion for the last point on future advancements in your README:

---

### Future Advancements

As the field of time-series forecasting continues to evolve, there are several promising directions for enhancing the current LSTM-based model:

1. **Bayesian LSTM**: Future implementations could explore **Bayesian LSTM** networks, which introduce a probabilistic approach to model uncertainty. This method can enhance the model's ability to handle uncertainty and provide confidence intervals along with predictions, making it suitable for risk-sensitive applications.

2. **LSTM Optimized with Butterfly Optimization**: Another avenue for improvement could be the **Butterfly Optimization** algorithm, which is an advanced optimization technique. Integrating Butterfly Optimization with LSTM can help achieve better convergence rates and improve model performance, especially in large-scale time-series forecasting problems.

3. **Hybrid LSTM + KNN Model**: Combining LSTM with **K-Nearest Neighbors (KNN)** can create a hybrid model that leverages both the temporal dependencies captured by LSTM and the non-linear relationships modeled by KNN. This could potentially enhance prediction accuracy by combining the strengths of both models for time-series tasks with complex, non-linear patterns.

These advancements will enable the model to evolve with time, improving its robustness, efficiency, and adaptability in various forecasting challenges.

