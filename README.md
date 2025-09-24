# Multilingual Text Classification System using NLP

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

This project implements a **Multilingual Text Classification System** using **Natural Language Processing (NLP)** and **Machine Learning models**. It classifies text across different languages into predefined categories by leveraging feature extraction techniques and classification algorithms.

## 🚀 Features

* **Multi-language support** for text classification across English, Spanish, French, German, and more
* **Comprehensive preprocessing pipeline**: tokenization, stopword removal, stemming/lemmatization
* **Advanced feature extraction** using **TF-IDF** and word embeddings
* **Multiple ML algorithms** for performance comparison:
  * Logistic Regression
  * Random Forest
  * Naïve Bayes
  * Support Vector Machine (SVM)
* **Detailed evaluation metrics**: accuracy, precision, recall, and F1-score
* **Language detection** capabilities for automatic text language identification
* **Easily extendable** to new datasets and languages
* **Performance visualization** with confusion matrices and classification reports

## ⚙️ Installation

### Prerequisites
* Python 3.8 or higher
* pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Shashivadhan1911/Multilingual-Text-Classification-System-NLP.git
cd Multilingual-Text-Classification-System-NLP
```

2. **Create and activate a virtual environment (recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Linux/Mac:
source venv/bin/activate
# For Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download required NLTK data:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📊 Dataset

### Dataset Format
The system works with multilingual datasets containing:
* **Text**: Raw text in various languages
* **Language**: Language code (en, es, fr, de, etc.)
* **Category**: Target classification label

### Expected CSV Structure
| text | language | category |
|------|----------|----------|
| "Breaking news about technology" | en | technology |
| "Noticias sobre deportes" | es | sports |
| "Actualités politiques" | fr | politics |
| "Wissenschaftliche Entdeckung" | de | science |

### Supported Languages
* **English** (en)
* **Spanish** (es)
* **French** (fr)
* **German** (de)
* **Italian** (it)
* **Portuguese** (pt)
* *Easily extendable to other languages*

### Dataset Sources
* Kaggle multilingual datasets
* Wikipedia article dumps
* News article collections
* Custom scraped datasets

**Note**: Place your dataset files in the `dataset/` folder before running the notebook.

## ▶️ Usage

### Running the Jupyter Notebook
```bash
jupyter notebook NLP_project.ipynb
```

### Step-by-Step Process
1. **Load Dataset**: Import and explore your multilingual dataset
2. **Data Preprocessing**: Clean text, remove noise, handle missing values
3. **Text Processing**: Tokenization, stopword removal, stemming/lemmatization
4. **Feature Extraction**: Generate TF-IDF vectors or word embeddings
5. **Model Training**: Train multiple ML algorithms
6. **Model Evaluation**: Compare performance metrics
7. **Results Analysis**: Visualize results and generate reports

### Quick Start Code Example
```python
# Import necessary libraries
from src.classifier import MultilingualClassifier
import pandas as pd

# Load your dataset
df = pd.read_csv('dataset/train.csv')

# Initialize classifier
classifier = MultilingualClassifier()

# Prepare data
X = df['text']
y = df['category']

# Train the model
classifier.fit(X, y)

# Make predictions
sample_texts = [
    "This is a technology article",
    "Este es un artículo sobre deportes",
    "Ceci est un article politique"
]
predictions = classifier.predict(sample_texts)
print("Predictions:", predictions)
```

## 📈 Results

### Performance Metrics
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **SVM** | **88.3%** | **88.1%** | **88.2%** | **88.2%** | 2.4s |
| **Random Forest** | 87.6% | 87.2% | 87.4% | 87.3% | 1.8s |
| **Logistic Regression** | 85.2% | 84.8% | 85.1% | 84.9% | 1.2s |
| **Naïve Bayes** | 82.1% | 81.8% | 82.0% | 81.9% | 0.8s |

### Key Insights
* **SVM** achieved the highest overall performance across all metrics
* **Random Forest** showed excellent generalization on unseen data
* **Logistic Regression** provided a good balance of speed and accuracy
* **Naïve Bayes** was fastest but with lower accuracy for complex multilingual tasks

### Language-Specific Performance
* **English**: 90.2% average accuracy
* **Spanish**: 87.8% average accuracy
* **French**: 86.4% average accuracy
* **German**: 85.1% average accuracy

*Detailed confusion matrices and classification reports are available in the notebook.*

## 🔧 Troubleshooting

### Common Issues and Solutions

**NLTK Download Error:**
```bash
python -c "import nltk; nltk.download('all')"
```

**Memory Issues with Large Datasets:**
* Reduce dataset size for initial testing
* Use batch processing for large datasets
* Consider using sparse matrices for feature extraction

**Language Detection Errors:**
```bash
pip install langdetect
```

**Encoding Issues:**
* Ensure your dataset is saved in UTF-8 encoding
* Use `pd.read_csv('file.csv', encoding='utf-8')` when loading data

**Low Performance on Specific Languages:**
* Ensure balanced representation of all languages in training data
* Consider language-specific preprocessing techniques
* Add more training samples for underrepresented languages

## 🔮 Future Enhancements

### Short-term Goals
* **Deep Learning Integration**: Implement LSTM and GRU models
* **Pre-trained Embeddings**: Integrate Word2Vec, GloVe, and FastText
* **Language Detection**: Automatic language identification for unknown texts
* **Cross-validation**: Implement k-fold cross-validation for robust evaluation

### Long-term Vision
* **Transformer Models**: Integration of BERT, RoBERTa, and multilingual BERT
* **Real-time API**: RESTful API for real-time text classification
* **Web Application**: User-friendly web interface using Flask/Django + React
* **Mobile App**: Cross-platform mobile application
* **Cloud Deployment**: AWS/Azure/GCP deployment with auto-scaling
* **Continuous Learning**: Online learning capabilities for model updates

## 🛠️ Technologies Used

### Core Libraries
* **Python 3.8+**: Primary programming language
* **scikit-learn**: Machine learning algorithms and evaluation
* **pandas**: Data manipulation and analysis
* **numpy**: Numerical computing
* **nltk**: Natural language processing toolkit
* **matplotlib & seaborn**: Data visualization

### Additional Tools
* **Jupyter Notebook**: Interactive development environment
* **langdetect**: Language identification
* **pickle**: Model serialization
* **re**: Regular expressions for text processing

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. **Fork the repository**
2. **Create your feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Contribution Guidelines
* Follow PEP 8 style guide for Python code
* Add unit tests for new features
* Update documentation for any changes
* Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{multilingual_text_classification,
  author = {Shashivadhan},
  title = {Multilingual Text Classification System using NLP},
  url = {https://github.com/Shashivadhan1911/Multilingual-Text-Classification-System-NLP},
  year = {2024}
}
```


## 🙏 Acknowledgments

* Thanks to the open-source community for providing excellent NLP libraries
* Inspired by multilingual NLP research and applications
* Special thanks to contributors and users who provide feedback and suggestions


---
⭐ **If you find this project helpful, please consider giving it a star!** ⭐
