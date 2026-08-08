# 🐔 Chicken Disease Diagnosis App

A Streamlit web app that classifies chicken health/disease from an uploaded image, using a
transfer-learning CNN (EfficientNetB3) trained in `chicken-disease-detection.ipynb`.

---

## 📁 Project Structure

Your GitHub repo should look like this:

```
chicken-disease-app/
│
├── app.py                              # Streamlit app (entry point)
├── requirements.txt                    # Python dependencies for deployment
├── chicken_disease_pipeline.pkl        # Pickled pipeline (model path + class names + image size)
├── EfficientNetB3-Chicken Disease-XX.XX.h5   # Trained Keras model (referenced by the pickle)
├── README.md                           # This file
└── .gitignore                          # (optional) ignore venv/cache files
```

> **Important:** `app.py` looks for the `.h5` model file either at the exact path stored
> inside the pickle, or (as a fallback) in the **same folder as `app.py`**. Easiest and
> most reliable: keep the `.h5` file next to `app.py` in the repo root, as shown above.

---

## 🖥️ Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/chicken-disease-app.git
   cd chicken-disease-app
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## ☁️ Deploy on Streamlit Community Cloud

1. **Push everything to GitHub** — including the `.h5` model file and the `.pkl` pipeline.
   (GitHub allows files up to 100 MB without Git LFS; if your `.h5` is larger, use
   [Git LFS](https://git-lfs.com/) or host the model elsewhere and download it at
   startup — see "Large model files" below.)

   ```bash
   git init
   git add .
   git commit -m "Initial commit - chicken disease app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/chicken-disease-app.git
   git push -u origin main
   ```

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. Click **"New app"**, then select:
   - **Repository:** `<your-username>/chicken-disease-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. Click **Deploy**. Streamlit Cloud will install everything from `requirements.txt`
   and launch the app automatically. First build usually takes a few minutes because
   TensorFlow is a large package.

5. Any time you `git push` new changes, the deployed app auto-updates.

---

## 📦 Large Model Files (if `.h5` is over ~100 MB)

GitHub blocks files over 100 MB by default. If your model is that large, pick one:

**Option A — Git LFS**
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add "EfficientNetB3-Chicken Disease-XX.XX.h5"
git commit -m "Track model with LFS"
git push
```

**Option B — Download at startup**
Host the `.h5` file somewhere (Google Drive, Hugging Face Hub, S3, etc.), then in `app.py`
download it once if it's missing:
```python
import urllib.request

if not os.path.exists(MODEL_LOCAL_PATH):
    urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, MODEL_LOCAL_PATH)
```

---

## 🔧 requirements.txt

```
streamlit>=1.32
tensorflow-cpu>=2.16,<2.17
numpy>=1.26,<2.0
pandas>=2.0
h5py>=3.10
Pillow>=10.0
```

- `tensorflow-cpu` is used instead of `tensorflow` since Streamlit Cloud has no GPU —
  this keeps the build smaller and faster.
- Pin versions loosely to stay compatible with whatever Keras version was used to
  originally save the `.h5` file.

---

## 🩺 About the Model

- **Architecture:** EfficientNetB3 (ImageNet-pretrained base, frozen) + custom dense head
- **Input size:** stored inside `chicken_disease_pipeline.pkl` (matches training, typically 224×224)
- **Classes:** read dynamically from the pickle at runtime — no hardcoded list
- **Output:** predicted class + confidence score, plus a plain-language explanation and
  recommended action for the detected condition

---

## ⚠️ Disclaimer

This tool is for educational/assistive purposes only and is **not a substitute for
veterinary diagnosis**. Always consult a veterinarian for confirmed treatment decisions.
