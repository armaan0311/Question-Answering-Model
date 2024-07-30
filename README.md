# Question-Answering-Model

**Project Title: Question-Answering System Using NLP Models**

**Overview**

This project aims to develop a state-of-the-art question-answering system using various NLP models, including BERT and T5. The system is trained and evaluated on a dataset derived from the Quora Question Answer Dataset, with preprocessing steps including normalization, tokenization, stop word removal, and lemmatization.

**Objectives**

Develop a question-answering model using BERT and T5.

Perform data preprocessing to prepare the dataset for training.

Evaluate model performance using metrics like ROUGE, BLEU, and F1-score.


**Dataset**

The dataset is based on the Quora Question Answer Dataset. The dataset can be accessed from the following link : https://huggingface.co/datasets/toughdata/quora-question-answer-dataset .

**Data Preprocessing**

Normalization: Convert text to lowercase and remove special characters.

Sentiment Analysis: Compute sentiment scores for questions, answers, and contexts using TextBlob.

Tokenization: Tokenize text into words using NLTK.

Stop Word Removal: Remove common stop words using NLTK.

Lemmatization: Reduce words to their base forms using WordNetLemmatizer.

Reassembly: Reassemble tokens into final processed strings.

**Model Development**

**BERT Model**

Tokenization: Utilize BertTokenizerFast for efficient tokenization.

Model Loading: Load the pre-trained BERT model (bert-large-uncased-whole-word-masking-finetuned-squad).

Data Processing: Define a function to process and encode data for BERT.

Training: Train the model using Trainer and TrainingArguments from Hugging Face's Transformers library.

**T5 and GPT Model**

Tokenization: Use T5Tokenizer and GPT2Tokenizer for tokenizing text.

Model Loading: Load the pre-trained T5 model (t5-base).

Data Processing: Encode data specifically for the T5 model.

Training: Train the model using similar techniques as BERT.

Evaluation: Assess performance using predefined metrics.

**Evaluation Metrics**

ROUGE: Measure the overlap between the generated and reference texts.

BLEU: Calculate precision for n-grams between generated and reference texts.

F1-Score: Evaluate the harmonic mean of precision and recall.

**Conclusion**

This project demonstrates the development and evaluation of a question-answering system using advanced NLP models. By leveraging BERT and T5, we explore different approaches to achieve high performance on the Quora Question Answer Dataset. The results are evaluated using robust metrics to ensure the effectiveness of the models.

**Future Work**

Explore further fine-tuning of models on larger datasets.

Implement more sophisticated data preprocessing techniques.

Experiment with other NLP models and architectures.

Optimize training and inference for production deployment.

Implement Frontend structure and framework.
