# Tokenizer (VB.NET)

The Tokenizer is a versatile text processing library written in Visual Basic (VB.NET). It provides functionalities for tokenizing text into words, sentences, characters, and n-grams. The library is designed to be flexible, customizable, and easy to integrate into your VB.NET projects.

## Introduction
This document provides an overview of the Tokenizer library, which is designed to break down text into individual tokens. The library offers various tokenization strategies and customization options to cater to different use cases. It aims to be an efficient and user-friendly tool for developers to process textual data.

The Tokenizer library is valuable in various natural language processing (NLP) and text analysis applications. Some associated use cases include:

1. Preprocessing for Text Classification: Tokenization is a crucial step in preparing text data for classification tasks, such as sentiment analysis, spam detection, or topic categorization.

2. Language Modeling: Tokenization is essential for training language models, including n-grams, recurrent neural networks (RNNs), and transformer-based models.

3. Information Retrieval: Tokenization is vital in information retrieval systems, such as search engines, to index and process documents efficiently.

4. Named Entity Recognition (NER): Tokenization is a critical component of NER systems that aim to identify entities like names, locations,

 and dates.

5. Machine Translation: Tokenization is used to preprocess text data for machine translation systems, enabling them to translate text at the word or subword level.

6. Text Generation: Tokenization is necessary for generating text using language models, chatbots, or text completion systems.



## Features

- Tokenize text into words
- Tokenize text into sentences
- Tokenize text into character-level tokens
- Generate n-grams from text
- Build vocabulary from tokenized text
- Remove stop words from tokenized text
- Customize tokenization behavior through various options.

**Pros:**
- Efficient tokenization of text data.
- Customizable tokenization strategies.
- Support for various languages and domain-specific patterns.
- Ability to tokenize at different levels (character, word, sentence).
- Open-source and extendable for future enhancements.
- Well-documented API and usage examples.

**Cons:**
- May require specific handling for domain-specific patterns not covered by default tokenization strategies.
- Memory usage may increase for large corpora, especially when using character-level tokenization.
- Limited tokenization support for highly complex languages or scripts.


# Introduction:
Tokenizer.vb is a Visual Basic (VB) project that provides functionalities for text tokenization, subword tokenization, and sentence generation. It includes various features to tokenize input text, split it into subwords, and generate random sentences using n-grams.

### How to Use:
To use the Tokenizer.vb project, follow these steps:

# Tokenization:

To tokenize a given text, create an instance of the Tokenizer class and call the Tokenize method with the input text and the desired token type.
The supported token types are _AlphaNumeric, _AlphaBet, _AlphaNumericPunctuation, _Word, _AlphaNumericWhiteSpace, _AlphaBetWhiteSpace, _Any, and _None.
### Example:
  ```vb

Dim tokenizer As New Tokenizer()
Dim text As String = "Hello, world! This is a sample text."
Dim tokens As List(Of String) = tokenizer.Tokenize(text, Tokenizer.Type._Word)
' Output: ["Hello", "world", "This", "is", "a", "sample", "text"]
  ```

# Subword Tokenization:

To tokenize a word into subwords based on a provided vocabulary, create an instance of the SubWord class and call the TokenizeToSubTokens method with the word and the vocabulary.
### Example:
```vb

Dim subwordTokenizer As New SubWord()
Dim word As String = "unbelievable"
Dim vocabulary As List(Of String) = New List(Of String) From {"un", "believable", "able"}
Dim subwords As List(Of String) = subwordTokenizer.TokenizeToSubTokens(word, vocabulary)
' Output: ["un", "believable"]
``` 

## Sentence Generation:

To generate random sentences using word n-grams and word n-gram probabilities, create an instance of the SubWord class and call the GenerateSentence method with the vocabulary and word n-gram probabilities.
### Example:

```vb
Copy code
Dim wordgramProbabilities As New Dictionary(Of List(Of String), Double)()
' ... Populate wordgramProbabilities with word n-gram probabilities ...
Dim sentence As String = SubWord.GenerateSentence(vocabulary, wordgramProbabilities)

' Output: "The quick brown fox jumps over the lazy dog."
```

# Tokenization:

The project allows tokenizing input text based on different token types, such as words, alphanumeric characters, punctuation, and more.
Users can easily obtain a list of tokens from the input text, which is useful for further analysis and processing.
Subword Tokenization:

Users can tokenize words into subwords using a provided vocabulary.
The method handles unknown words by replacing them with "[UNK]" (unknown token).
Subword tokenization is useful for various Natural Language Processing (NLP) tasks, especially when dealing with out-of-vocabulary words.
Sentence Generation:

The project provides a feature to generate random sentences using word n-grams and their probabilities.
Users can use this feature to generate sample sentences for testing or to create synthetic datasets for NLP models.
Text Processing Utilities:

The project offers extension methods to handle punctuations, encapsulated text extraction, and spacing of items.
Users can preprocess text data easily, add spaces around punctuations, and extract text enclosed in parentheses, brackets, curly braces, or angle brackets.
## Conclusion

The Tokenization Library provides a robust and efficient tool for breaking down natural language text into individual tokens. With its various tokenization options and support for multiple languages, it is a valuable asset for developers working on NLP tasks. While limited to tokenization, it serves as a crucial preprocessing step for various NLP applications, contributing to enhanced text understanding and analysis.