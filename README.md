# Kindle Reviews Sentiment Analysis

## Overview

This project performs sentiment analysis on Amazon Kindle Store book reviews using multiple NLP techniques. The goal is to classify reviews as positive or negative based on their text content.

## Dataset

- **Source:** [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/)
- **Category:** Kindle Store
- **Size:** 982,619 reviews (each reviewer and product has at least 5 reviews)
- **Columns Used:** `reviewText`, `rating`

## Project Structure

- `main.ipynb` — Jupyter notebook containing all code and analysis
- `all_kindle_review.csv` — Dataset (not included, download from source)
- `README.md` — Project documentation

## Installation & Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/kindle-reviews-sentiment-analysis.git
   cd kindle-reviews-sentiment-analysis
   ```

2. **Install dependencies**
   ```sh
   pip install pandas numpy scikit-learn gensim nltk beautifulsoup4 lxml
   ```

3. **Download NLTK resources**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

4. **Place the dataset**
   - Download `all_kindle_review.csv` and place it in the project root.

## Workflow

1. **Data Preprocessing**
   - Lowercasing text
   - Removing special characters, URLs, HTML tags, and extra spaces
   - Removing stopwords
   - Lemmatization

2. **Feature Engineering**
   - Bag of Words (BOW)
   - TF-IDF
   - Word2Vec (average vectors)

3. **Model Training**
   - Gaussian Naive Bayes (for BOW and TF-IDF)
   - Random Forest Classifier (for Word2Vec)

4. **Evaluation**
   - Accuracy score
   - Classification report

## Usage

- Open `main.ipynb` in Jupyter Notebook or VS Code.
- Run all cells to reproduce the results.

## Results

- **BOW & TF-IDF:** Low accuracy
- **Word2Vec + Random Forest:** Achieved ~74% accuracy

## Visualizations

- Confusion matrix and classification reports are printed for each model.
- You can add more plots (e.g., ROC curve, feature importance) for deeper analysis.

## Improvements & Next Steps

- Use pre-trained embeddings (GloVe, Google News Word2Vec)
- Try other classifiers (XGBoost, LightGBM, Logistic Regression)
- Hyperparameter tuning with GridSearchCV
- Data augmentation for more robust models
- Ensemble methods for better performance
- Analyze misclassified samples for insights

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

Dataset license belongs to the original authors. Project code is MIT licensed.

## Acknowledgements

- Julian McAuley, UCSD for the dataset
- NLTK, Gensim, scikit-learn libraries

## Contact

For questions or suggestions, open an issue or contact [ayushmittal629@gmail.com](ayushmittal629@gmail.com).
