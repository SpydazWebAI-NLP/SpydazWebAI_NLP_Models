## SpydazWeb Corpus Creator

This NLP Library for Visual Basic is a comprehensive toolkit designed to empower developers with Natural Language Processing (NLP) capabilities within the .NET Framework. Authored by Leroy "Spydaz" Dyer (LeroySamuelDyer@hotmail.co.uk)
This library provides a wide range of functionalities to process, analyze, and manipulate text data using advanced techniques inspired by NLP, data science, machine learning, and language modeling.
The provided code represents a collection of classes and functions designed to create a comprehensive Natural Language Processing (NLP) framework in Visual Basic, with a focus on text preprocessing, vocabulary handling, encoding, decoding, and other NLP-related tasks. It aims to provide missing functionality available in Python's NLP libraries. This model can be classified as a **"Custom NLP Framework and Toolkit"**. Let's break down the classification and key features:

**Classification: Custom NLP Framework and Toolkit**

**Key Features and Components:**

1. **Document Preprocessing:** The framework includes methods for tokenizing sentences, words, and characters. It also provides methods to detect entities in text.

2. **Vocabulary Management:** The `Vocabulary` class handles vocabulary creation, word frequency, and TF-IDF calculations. It supports character-level, word-level, and sentence-level vocabularies.

3. **Encoding and Decoding:** The `Encode` and `Decode` classes facilitate the conversion between text and numeric representations using custom encodings. It supports character, word, and sentence encodings.

4. **Positional Encoding:** The `PositionalEncoding` and `PositionalDecoder` classes offer positional encoding and decoding for input embeddings, enhancing the model's understanding of word order.

5. **Text Similarity and Comparison:** The `CalculateCosineSimilarity` function computes cosine similarity between sentences, while the `CalculateWordOverlap` function measures the overlap of words between sentences.

6. **Entailment Detection:** The `DetermineEntailment` function determines entailment based on word overlap between sentences.

7. **Entity Detection:** The `Entity` structure and related methods support entity detection in text using a predefined list of entities.

8. **Serialization and Helper Functions:** The `Helper` module provides utility functions such as object-to-JSON conversion and other text-related tasks.

**Potential Use Cases:**

1. **Text Preprocessing:** The framework can be used to preprocess raw text data before feeding it into downstream NLP models or applications.

2. **Document Analysis:** It can be employed for extracting useful information from documents, performing text similarity analysis, and identifying entities.

3. **Sentiment Analysis:** By incorporating additional features for sentiment analysis, the framework could be extended to perform sentiment classification on text.

4. **Language Modeling:** The positional encoding functionality supports language modeling tasks, such as generating coherent and context-aware text.

5. **Information Retrieval:** The TF-IDF calculations and vocabulary management can aid in building search engines and information retrieval systems.

6. **Text Generation:** When integrated with appropriate modeling techniques, the framework can assist in text generation tasks like chatbots, article generation, etc.

7. **Text Classification:** Combined with machine learning algorithms, the framework could be used for text classification tasks such as topic categorization or spam detection.

8. **Educational Tool:** It can serve as a learning tool for students and practitioners interested in NLP, providing a clear implementation of various NLP concepts.

**Guidelines and Further Development:**


Overall, this custom NLP framework and toolkit provide a solid starting point for building NLP applications in Visual Basic, enabling developers to explore and experiment with a range of NLP techniques and use cases.


## Installation

To use this NLP library in your Visual Basic projects, you can follow these steps:

1. Clone or download the repository from [GitHub](.)
2. Include the necessary `.vb` files from the repository in your project.
3. Build and run your project to start using the library.

The library is also available on NuGet for easy installation in your projects.


## Components

### ChunkProcessor

The `ChunkProcessor` class provides functionalities to chunk and pad text data. It includes methods to split text into sentences, paragraphs, or treat a whole document as a chunk. Padding logic is also available to ensure consistent chunk sizes.

### TextCorpusChunker

The `TextCorpusChunker` class implements the `ICorpusChunker` interface and serves as the main entry point to the library. It offers methods to create punctuation and dictionary vocabularies, filter text data using vocabulary, generate classification and predictive datasets, load entity lists, and process text data.

### VocabularyGenerator

The `VocabularyGenerator` class contains methods for vocabulary creation and management. It can generate vocabulary from text data, including frequency and punctuation vocabularies. Exporting and importing vocabularies to/from files is also supported.

## Use Cases

### Text Data Preprocessing

The library enables text data preprocessing, such as chunking and padding, to prepare data for further analysis or modeling. It helps manage data granularity and ensure consistent input sizes.

### Vocabulary Generation

VocabularyGenerator assists in creating frequency and punctuation vocabularies from text data. These vocabularies can be used for various purposes, including feature extraction and filtering.

### Text Classification

With the TextCorpusChunker and VocabularyGenerator, you can generate classification datasets for training NLP models. Categorize documents into classes based on keyword associations.

### Predictive Modeling

The library aids in generating predictive modeling datasets by creating sliding windows of text data. This is useful for training sequence-based models like RNNs or transformers.


### Corpus Modeling 
Provides functionalities for working with text corpora. It includes features for vocabulary creation, text encoding/decoding, and generating positional encodings for input tokens.
The Corpus Model consists of the following components:

    1. **Vocabulary**: The `Vocabulary` class represents the vocabulary model for the corpus. It allows adding new terms, calculating term frequencies, TF-IDF scores, and checking if a term exists in the vocabulary.
    
    2. **Encode**: The `Encode` class provides methods for encoding text into integer representations based on a given vocabulary. It supports character-level, word-level, and sentence-level encoding.
    
    3. **Decode**: The `Decode` class allows decoding integer representations back into text using a provided vocabulary. It enables the conversion from encoded inputs to their original text forms.
    
    4. **Positional Encoding**: The `PositionalEncoding` class generates positional encodings for input tokens. It adds positional information to the input embeddings to capture the order of tokens in the sequence.
    
    5. **Positional Decoder**: The `PositionalDecoder` class performs decoding of encoded inputs by retrieving the original tokens based on positional encodings. It allows converting encoded inputs back into their original tokenized forms.

## Usage tips

To use the Corpus Model in your .NET VB project, follow these steps:

1. Create an instance of the `Corpus` class.

2. Initialize the `Vocabulary` with the desired vocabulary type (character, word, or sentence).

3. Optionally, add terms to the vocabulary using the `Vocabulary` class's methods.

4. Use the `Encode` class to encode text into integer representations based on the vocabulary. Choose the appropriate encoding method based on the desired granularity (character, word, or sentence).

5. Utilize the `Decode` class to decode the encoded inputs back into their original text forms using the vocabulary.

6. If positional encodings are required, create an instance of the `PositionalEncoding` class and provide the maximum length and embedding size. Use the `Encode` method to generate positional encodings for input tokens.

7. For decoding positional encodings, create an instance of the `PositionalDecoder` class and provide the maximum length and embedding size. Use the `Decode` method to retrieve the original tokens based on the positional encodings.

# Examples
Here are a few examples to illustrate the usage of the Model:

### Example 1: Text Data Preprocessing

```vb
Dim chunkProcessor As New ChunkProcessor(ChunkType.Paragraph, 100)
Dim textChunks As List(Of String) = chunkProcessor.Chunk(rawText, ChunkType.Paragraph)
Dim paddedChunks As List(Of String) = chunkProcessor.ApplyPadding(textChunks)
```

### Example 2: Vocabulary Generation

```vb
Dim vocabulary As HashSet(Of String) = VocabularyGenerator.CreateDictionaryVocabulary(textChunks)
VocabularyGenerator.ExportVocabulary("dictionary_vocabulary.txt", vocabulary)
```

### Example 3: Text Classification

```vb
Dim categories As List(Of String) = New List(Of String) From {"Sports", "Technology", "Politics"}
Dim classificationDataset As List(Of Tuple(Of String, String)) = TextCorpusChunker.GenerateClassificationDataset(textChunks, categories)
```

### Example 4: Predictive Modeling

```vb
Dim windowSize As Integer = 10
Dim predictiveDataset As List(Of String()) = TextCorpusChunker.GeneratePredictiveDataset(textChunks, windowSize)
```

## Creating a Vocabulary:
   ```vb
   Dim vocab As New Corpus.Vocabulary()
   vocab.AddNew("cat")
   vocab.AddNew("dog")
```

## Encoding and Decoding Text:
   
     ```vb
      Dim encodedText As List(Of Integer) = Corpus.Encode.Encode_Text("I have a cat", vocab, Corpus.VocabularyType.Word)
      Dim decodedText As String = Corpus.Decode.Decode_Text(encodedText, vocab)
     ```
## Generating Positional Encodings:


    ```vb
    Dim positionalEncoder As New Corpus.PositionalEncoding(maxLength:=10, embeddingSize:=16)
    Dim encodedTokens As List(Of List(Of Double)) = Corpus.Encode.Encode_Text("I have a cat", vocab, Corpus.VocabularyType.Word)
    Dim positionalEncodings As List(Of List(Of Double)) = positionalEncoder.Encode(encodedTokens)
    ```
## Decoding Positional Encodings:

    ```vb
    Dim positionalDecoder As New Corpus.PositionalDecoder(maxLength:=10, embeddingSize:=16)
    Dim decodedTokens As List(Of String) = positionalDecoder.Decode(positionalEncodings)
    ```

By utilizing this NLP library, you gain powerful tools for text data manipulation, processing, and analysis, making it an invaluable asset in your .NET projects. If you encounter any issues or have questions, please feel free to contact Leroy "Spydaz" Dyer at LeroySamuelDyer@hotmail.co.uk or visit [WWW.SpydazWeb.Co.UK](http://www.spydazweb.co.uk) for more information.

## License

This NLP Library for Visual Basic is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---