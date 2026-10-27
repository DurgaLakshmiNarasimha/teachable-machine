# Teachable Machine
Teach the app your own image classes: add classes, upload or capture examples with your webcam, train in one click, then test and download the model.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
python test_ml.py
```
Batch training from a folder (`dataset/<class>/*.jpg`):
```python
from ml import TeachableMachine, load_folder
tm = TeachableMachine(); print(tm.train(load_folder("dataset"))); tm.save("model.joblib")
```
## How it works
Features = downscaled pixels + colour histograms + edge histogram -> StandardScaler -> Logistic Regression (probabilities + cross-validated accuracy). Lightweight on purpose: trains in under a second, no GPU. To go further, swap `features()` for MobileNet embeddings.
