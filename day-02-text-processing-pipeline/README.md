![streamlit-ui](https://github.com/user-attachments/assets/39cb944d-790a-4b85-ae83-2a83d747e820)# NLP Text Processing Pipeline 🧠📊

![image](https://github.com/user-attachments/assets/fb28151e-c117-407a-ba0d-771e22fe93fe)


[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/shanojpillai/day-02-text-processing-pipeline/blob/main/LICENSE)
[![NLP](https://img.shields.io/badge/NLP-Text%20Processing-green.svg)]()

## 📘 Project Overview

This interactive Streamlit application provides a comprehensive exploration of Natural Language Processing (NLP) text processing fundamentals. Designed for developers and students interested in understanding the core mechanics of how language models transform text.

### 🌟 Key Features

- **Text Preprocessing**: Clean and normalize raw text
- **Tokenization Techniques**: 
  - Basic Word Tokenization
  - Advanced Tokenization
  - Character-level Tokenization
- **Vocabulary Building**: 
  - Frequency-based token mapping
  - Special token handling
- **Interactive Exploration**: Real-time text processing and visualization

## 🛠 Technologies Used

<div align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit"/>
    <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib"/>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
</div>

## 📦 Project Structure

```
day-02-text-processing-pipeline/
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   └── tokenization.py
│   ├── vocabulary/
│   │   └── vocab_builder.py
│   └── models/
│       ├── embeddings.py
│       └── language_model.py
│
├── app.py
├── requirements.txt
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/shanojpillai/day-02-text-processing-pipeline.git
   cd day-02-text-processing-pipeline
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

```bash
streamlit run app.py
```
![Screenshot 2025-03-27 183745](https://github.com/user-attachments/assets/75bc449b-49a4-4bd8-a534-6c305db273a8)

## 🔍 Exploring the Pipeline

The application provides multiple tabs to explore NLP processing:

1. **Preprocessing**
   - Clean and normalize text
   - Remove special characters

2. **Tokenization**
   - Break text into tokens using different strategies
   - Compare tokenization approaches

3. **Vocabulary**
   - Build token vocabularies
   - Explore token frequencies
   - Manage vocabulary size

## 📚 Learning Objectives

- Understand text preprocessing techniques
- Explore different tokenization strategies
- Learn how vocabularies are constructed
- Gain insights into how language models process text

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

<div align="center">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License"/>
</div>

This project is open source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 👤 Author

**Shanoj**
- 🌐 GitHub: [shanojpillai](https://github.com/shanojpillai)

## 🙏 Acknowledgments

- Inspired by the journey of understanding Natural Language Processing
- Special thanks to the open-source community

---

**Happy NLP Exploring!** 🎉🧠

<div align="center">
    <sub>Built with ❤️ by Shanoj</sub>
</div>
