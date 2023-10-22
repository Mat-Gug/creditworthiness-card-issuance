# ML Project: Creditworthiness Estimation Model :credit_card:

Hello and welcome! Thank you for taking the time to explore this project :blush:

This project is dedicated to the development of a machine learning model designed to estimate a customer's creditworthiness.

We begin with exploratory data analysis (EDA) of the `credit_record.csv` and `application_record.csv` datasets, which provide information about credit status and personal details of clients.

After a proper target estimation based on credit status information and data preprocessing, we continue with the Model Building phase, where multiple models are considered. We start with a Logistic Regression model, gradually progressing to explore more complex models, including Random Forest, Balanced Random Forest, and XGBoost. This comprehensive model selection process, which also includes the fine-tuning of the hyperparameters for each model, enables us to thoroughly assess their performance using a range of evaluation metrics, such as balanced accuracy and the F2 score, ensuring a meticulous examination of each model's predictive capabilities. The latter leads to the selection of the final model.

Finally, to gain a deeper understanding of the model's inner workings and to assess the significance of each feature in shaping the predictions for individual instances, we employ the SHAP technique, thereby shedding light on the hidden mechanisms at play within our machine learning model.

In summary, this project represents a comprehensive journey through the development of a creditworthiness estimation model, encompassing data exploration, model building, rigorous performance evaluation, and results' interpretability, all with the ultimate goal of providing valuable insights into the creditworthiness of our customers.

## Setting Up a Virtual Environment and Installing Dependencies

Before running the project, it's considered a best practice to create a virtual environment and install the required dependencies. This helps isolate project-specific dependencies from system-wide Python packages. To achieve this, follow these steps:

1. **Create a Virtual Environment:**
- For Windows:
  - To create a virtual environment with the default Python version:
    ```
    python -m venv credit_venv
    ```
  - To create a virtual environment with a specific Python version, such as Python 3.11, replace `python` with `py` followed by the desired Python version:
    ```
    py -3.11 -m venv credit_venv
    ```
- For macOS and Linux:
  ```
  python3 -m venv credit_venv
  ```
2. **Activate the Virtual Environment:**
- For Windows (Command Prompt):
  ```
  credit_venv\Scripts\activate
  ```
- For Windows (Git Bash):
  ```
  source credit_venv/Scripts/activate
  ```
- For macOS and Linux:
  ```
  source credit_venv/bin/activate
  ```
3. **Clone the Repository:**
```
git clone https://github.com/Mat-Gug/creditworthiness-card-issuance.git
```
4. **Navigate to the Project Directory:**
```
cd creditworthiness-card-issuance
```
5. **Install Required Dependencies:**
```
pip install -r requirements.txt
```
6. **Create an IPython Kernel for Jupyter Notebook:**

After activating your virtual environment, run the following command to create an IPython kernel for Jupyter Notebook:
```
python -m ipykernel install --user --name=credit_venv_kernel
```
If you don't have `ipykernel` installed, you can do it by running the following command:
```
pip install ipykernel
```
7. **Deactivate the Virtual Environment:**

Whenever you're done working on the project, you can deactivate the virtual environment:
```
deactivate
```
By following these steps, you'll have your project set up in an isolated virtual environment with all the required dependencies installed, and you'll be able to use Jupyter Notebook with your project-specific kernel. This is very helpful to ensure that the project runs consistently and without conflicts.

