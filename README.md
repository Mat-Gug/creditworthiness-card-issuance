# ML Project: Creditworthiness Estimation Model :credit_card:

Hello and welcome! Thank you for taking the time to explore this project :blush:

This project is dedicated to the development of a machine learning model designed to estimate a customer's creditworthiness.

## Setting Up a Virtual Environment and Installing Dependencies

Before running the project, it's considered a best practice to create a virtual environment and install the required dependencies. This helps isolate project-specific dependencies from system-wide Python packages. To achieve this, follow these steps:

1. **Clone the Repository:**
```
git clone https://github.com/Mat-Gug/creditworthiness-card-issuance.git
```
2. **Navigate to the Project Directory:**
```
cd creditworthiness-card-issuance
```
3. **Create a Virtual Environment:**
- For Windows:
  ```
  python -m venv credit_venv
  ```
- For macOS and Linux:
  ```
  python3 -m venv credit_venv
  ```
4. **Activate the Virtual Environment:**
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

