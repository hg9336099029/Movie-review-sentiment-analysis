# Movie Sentiment analysis application

It is a web application that uses a **Simple RNN** model to predict the **sentiment of movie reviews** (positive or negative) from the IMDB dataset. Built with **TensorFlow/Keras** and **Streamlit**.

| Section | Details |
|---------|---------|
| **Features** | - Predicts positive/negative movie reviews<br>- Shows prediction confidence<br>- Interactive Streamlit interface<br>- Uses pre-trained Simple RNN on IMDB dataset |
| **Installation** | 1. Clone repo: `git clone https://github.com/your-username/MovieMood-RNN.git`<br>2. Create & activate virtual environment:<br> `python -m venv .venv`<br>Windows: `.venv\Scripts\activate`<br>macOS/Linux: `source .venv/bin/activate`<br>3. Install dependencies: `pip install -r requirements.txt` |
| **Usage** | 1. Run app: `streamlit run main.py`<br>2. Enter a movie review<br>3. Click **Predict Sentiment** to see result |
| **Project Structure** | ```<br>MovieMood-RNN/<br>├── main.py<br>├── simplernn_imdb_model.h5<br>├── requirements.txt<br>├── .gitignore<br>└── README.md<br>``` |
| **How it Works** | - **Preprocessing:** lowercasing, tokenizing, encoding with IMDB word index, padding sequences to 500<br>- **Prediction:** Simple RNN outputs score 0–1<br>- **Output:** score ≥0.5 → Positive, <0.5 → Negative |
| **Sample Reviews** | **Positive:** "Absolutely loved this movie! The storyline was gripping."<br>**Negative:** "This movie was a huge disappointment. The plot made no sense." |
| **Contributing** | - Improve preprocessing<br>- Add LSTM/GRU/Transformer models<br>- Enhance Streamlit UI |
| **License** | MIT License |
