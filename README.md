# Spam Call Prediction Neural Network
Simple 3 layered neural network trained on a custom synthethic dataset.
- The solution provided has an average training accuracy of 86.67%
- The dataset provided is synthetic and does not represent real data.
> Important Disclaimer: The `spam_calls.csv` dataset available in this repository is a synthetic dataset created for illustrative purposes. All the information within, including phone numbers, country codes, and other data, is randomly generated and holds no real-world significance or correlation. It is crucial to understand that this dataset is entirely fictional, and any attempts to misuse, abuse, or misrepresent the data are strongly discouraged. Please exercise ethical and responsible use of the provided data.

### Screenshot:
![screenshot (8)](https://github.com/probablyliquid/spam-call-detection/assets/111677910/95430375-9bb4-43f2-a92d-d92a64c1510c)


## Instructions:

### Flask Application (Recommended)
1. Make sure you have Flask installed: `pip install flask`.
2. Run `python3 app.py`
3. Open your web browser and navigate to `http://127.0.0.1:5000/` or `http://localhost:5000/`

### Local Runtime using Python3
1. Clone/download the files.
2. Make sure you have Python3 and the following dependencies installed: `tensorflow`, `keras`, `pandas`, and `numpy`.
4. Run `spam_calls.py` using CLI command `python3 spam_calls.py`

### Website
1. Visit ![https://tamarind.onrender.com](https://tamarind.onrender.com/)
   > Please note that this is not an optimal or recommmended way to use this service as it is incredibly slow and unstable.

### Jupyter Notebook (or) Google Colab
1. Initialise Jupyter Notebook (or) Google Colab (for Colab, you need to connect to a runtime server)
2. Make sure that your runtime server has Python3 and the following dependencies installed: `tensorflow`, `pandas`, `numpy,`, and `keras`.
3. Upload all the files provided (you may skip the main.py file)
4. Make sure to set the correct path of the `spam_calls.csv` in `Spam_Calls.ipynb`  at `dataset = pd.read_csv(<correct_filepath>)`
5. Run the `Spam_Calls.ipynb` file, cell-by-cell.


Have a great day:)
##### - Priyanshu
