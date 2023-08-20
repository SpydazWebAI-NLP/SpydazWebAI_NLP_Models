# NLP Library for Visual Basic .NET

Welcome to the NLP Library for Visual Basic .NET! This comprehensive library offers a wide range of functionalities for natural language processing (NLP), text analysis, and text manipulation within the Visual Basic .NET environment. Whether you're working on sentiment analysis, text classification, information retrieval, or language modelling, this library provides powerful tools to streamline your tasks. The NLP Library provides a wide array of tools and methodologies for various NLP tasks, language modelling, and multimedia data analysis. From document search and word embeddings to vector storage and text classification, this library empowers you to explore, experiment, and innovate in the field of Natural Language Processing.

By using this library, you can create robust and efficient NLP pipelines, develop sophisticated recommendation systems, enhance sentiment analysis models, and unlock insights from textual and multimedia data. The modular structure and comprehensive documentation allow you to seamlessly integrate these functionalities into your Visual Basic projects, taking your NLP endeavours to the next level.

## Introduction

This NLP Library for Visual Basic .NET is designed to assist developers in tackling complex NLP challenges while working in the .NET framework (version 4.7). With an emphasis on modularity and object-oriented design, this library provides a comprehensive set of classes and functions to process, analyse, and manipulate text data efficiently.

The Purpose of the library is to provide Input modelling functionality , such as Tokenization, Text/Image/Audio Vectorization/Embeddings , Positional Encoding, Vector Storage and Search and Text Analysis. It also provides some methods for Language modelling, Using N-gram models  , Bag of Words, Advanced Corpus analysis. as well as Training Dataset Generation and modelling. The library also provides Various Math, Masking, and Text Extensions for Context search, Word Matrixes, searching Text Pre-processing etc. This library is focused on the Input modelling stage of the NLP  Pipelines. 


 -**Text Pre-processing**: The library is suitable for preparing text data for various NLP applications, such as sentiment analysis, text classification, and language modelling.
- **Keyword Extraction**: Users can extract important keywords from documents for indexing or summarization purposes.
- **Chatbots and Conversational AI**: The library's tokenization and similarity calculation functionalities are valuable for building chatbots and AI systems that understand user input.
- **Sentiment Analysis**: Text - and tokenization aid in sentiment analysis, helping to analyse and classify sentiment in text data.
- **Language Modelling**: The library forms a foundation for developing language models by providing tokenization and self-attention capabilities.
- **Information Retrieval**: Users can pre-process and tokenize text queries and documents for effective information retrieval.
- **Programming-related Tasks**: The reserved keyword detection and alphanumeric filtering are helpful for processing code snippets and programming-related text.
- **Tokenization**: The library provides tokenization functions to break down input text into tokens based on characters, words, or sentences. This is a fundamental step for many NLP tasks.
- **Normalization**: The library offers a function to normalize input text by converting it to lowercase, removing punctuation, and extra whitespace. This ensures consistent processing of text data.
- **Token Types**: The library defines various token types, such as grammatical punctuation, alphabets, numbers, code punctuation, encapsulation punctuation, math punctuation, and more. This categorization allows users to work with specific types of tokens.
- **Similarity Calculation**: The library includes methods to calculate similarity between tokens based on embeddings. This is useful for measuring semantic relatedness between words or phrases.
- **Self-Attention**: The library provides functionality to compute self-attention values for tokens within a list. Self-attention is a key concept in transformer-based models.
- **Reserved Keyword Detection**: Users can determine whether a given word is a reserved VBScript keyword. This is essential for working with programming-related texts.
- **Counting Tokens**: The library offers a function to count the number of tokens in a given string, which can be useful for various analytical tasks.
- **Encapsulation Extraction**: The library can extract encapsulated items (within parentheses, brackets, curly braces, etc.) from a string.
- **Alphanumeric Filtering**: Users can filter out non-alphanumeric characters from a string, focusing on meaningful tokens.

The provided code offers a versatile set of tools that can be utilized in various natural language processing and text analysis applications. This NLP Library for Visual Basic .NET is a versatile tool that caters to a wide range of NLP use cases and applications:. Some potential use cases include:

- Text Classification: Utilize the vocabulary generation and dataset creation functions to train and evaluate text classification models.
- Text Generation: Apply processed text chunks as input for text generation algorithms, such as Markov chains or recurrent neural networks.
- Sentiment Analysis: Use cleaned text as input for sentiment analysis models by removing noise and irrelevant information.
- Information Retrieval: Index chunked and filtered text to build search engine indexes or implement efficient information retrieval systems.
- Language Modelling: Leverage processed text chunks for training and evaluating language models for different NLP tasks.
- Text pre-processing and normalization
- Tokenization by characters, words, and sentences
- Named entity recognition (NER)
- Similarity calculation between phrases
- Advanced search functionalities
- Chunking and processing of text data
- N-gram analysis for pattern detection
- JSON serialization for data interchange
- Support for various languages and character sets-
- Experiment with novel word embedding training methods.
- Create custom pipelines for word embeddings.
- Enhance word embeddings for specific domains.
- Convert documents into TF-IDF vectors for analysis.
- Perform document clustering based on TF-IDF vectors.
- Enhance machine learning models with numerical representations.
- Train word embeddings using Word2Vec for semantic analysis.
- Utilize word embeddings for text generation tasks.
- Enhance recommendation systems with word embeddings.
- Train word embeddings using Skip-gram with Negative Sampling.
- Analyse semantic relationships between words.
- Incorporate word embeddings into sentiment analysis.
- Capture semantic relationships using word embeddings.
- Enhance topic modelling with optimized embeddings.
- Train word embeddings for text analysis and generation.
- Utilize trained embeddings for recommendation systems.
- Improve named entity recognition using word embeddings.
- Develop a search engine for indexing and retrieving documents.
- Create a database of documents with metadata.
- Extract snippets for context-based document previews.
- Efficiently find similar documents in large datasets.
- Build recommendation systems based on document similarity.
- Cluster or group similar documents for analysis.
- Store and retrieve different types of vectors (audio, image, text).
- Identify similar audio, image, or text data based on user queries.
- Implement content-based recommendation systems.

# Installation

To start using the NLP Library for Visual Basic .NET, follow these simple installation steps:

1. Clone or download the repository to your local machine.
2. Reference the library in your Visual Basic .NET project.
3. Begin exploring the vast array of classes and functions to enhance your NLP projects.

# Some Key Features

### Storage Modelling
#### Search Engine & Document Storage Model

This module provides essential components for building a basic search engine and document storage. With classes like `SearchEngine` and `Document`, you can add documents, search for documents by keywords, and extract contextual snippets.
#### MinHash and Locality-Sensitive Hashing (LSH)

This module introduces MinHash and LSH techniques for efficient similarity estimation and document clustering. It includes methods for estimating Jaccard similarity, grouping similar items, and employing a signature matrix.
#### Vector Storage Model

In this module, you'll find classes for handling different types of vectors (audio, image, text) and methods to discover similar vectors using various distance metrics. It provides a foundation for multimedia data retrieval and analysis.
### Word Embeddings Model

This module revolves around training word embeddings, calculating word similarities, and displaying matrices. It offers a base class `WordEmbeddingsModel` and showcases various methodologies for NLP tasks.

#### Morphological Analysis with FastText:

Use FastText to capture morphological information by representing words as bags of character n-grams. This can be particularly useful for languages with rich morphology or agglutinative structures.

#### Syntactic Relationship Modelling with CBOW:

Employ CBOW to learn word embeddings that capture syntactic relationships between words, enabling the model to predict a target word based on its context. Both FastText and CBOW offer unique advantages and can be applied in various NLP tasks, including text classification, sentiment analysis, and language generation.

#### Word2Vec

The `Word2Vector` class represents the Word2Vec model, enabling training of word embeddings using the Continuous Bag of Words (CBOW) or Skip-gram approaches. It also facilitates similarity calculation and text generation.

#### Audio2Vector

The Audio2Vector class contains methods for converting audio signals to complex vectors (spectra) and vice versa. It allows you to work with audio data in the frequency domain, which can be useful for various audio processing tasks. Methods: AudioToVector: Converts an audio signal into a list of complex vectors (spectra) by applying a sliding window and calculating the spectrum for each window. VectorToAudio: Converts a list of complex vectors back into an audio signal. This method can handle cases where the length of the vectors does not match the hop size. LoadAudio: Loads an audio file and returns the audio data as an array of doubles. SaveAudio: Saves an array of doubles (audio signal) as an audio file. Images Namespace This namespace contains classes for encoding and decoding image data.

#### Image2Vector

The Image2Vector class provides methods for converting images to vectors and saving vectors to files. Methods: SaveVectorToFile: Saves an image vector (array of doubles) to a text file. ImageDecoder The ImageDecoder class is used to decode image vectors and reconstruct images from vectors. Methods: DecodeImage: Decodes an image vector (array of doubles) and reconstructs the image by assigning grayscale values to pixels. ImageEncoder The ImageEncoder class is used to encode images into vectors. Methods: EncodeImage: Encodes an image file into a vector of grayscale pixel values. Usage Note The provided classes in the Audio and Images namespaces allow you to perform audio signal processing and image manipulation tasks, respectively. They offer methods for converting data between different representations and can be used for tasks such as audio analysis, audio synthesis, image compression, and image reconstruction.


# TokenizerPositional

The TokenizerPositional class provides tokenization methods with positional information. Methods: TokenizeByCharacter(input As String): Tokenizes input text into characters with start and end positions. TokenizeBySentence(input As String): Tokenizes input text into sentences with start and end positions. TokenizeByWord(input As String): Tokenizes input text into words with start and end positions. TokenizeInput(ByRef Corpus As List(Of String), tokenizationOption As TokenizerType): Tokenizes a corpus using different tokenization options (character, word, sentence).

# TokenizerTokenID

The TokenizerTokenID class handles token IDs and provides methods for tokenizing text and detokenizing token IDs. Constructor: New(ByRef Vocabulary As Dictionary(Of String, Integer)): Initializes the class with a vocabulary. Methods: TokenizeToTokenIDs(text As String): Tokenizes input text into token IDs using the provided vocabulary. Detokenize(tokenIds As List(Of Integer)): Given a list of token IDs, returns the corresponding text. Tokenizer The Tokenizer class serves as a master tokenizer that integrates and manages various tokenization methods. Properties: Vocabulary: A dictionary representing the vocabulary of tokens. PairFrequencies: A dictionary of subword pairs and their frequencies. maxSubwordLen: The maximum length of subwords in the vocabulary. VocabularyPruneValue: Defines the maximum number of entries in the vocabulary before pruning rare words. Methods: GetVocabulary(): Returns the list of tokens in the vocabulary. This set of tokenization classes allows you to tokenize text data at different levels of granularity and provides various functionalities, such as handling unknown subwords, generating token IDs, detokenizing, and managing the vocabulary. By integrating these tokenization methods into your NLP library, you can preprocess and tokenize text data efficiently for different NLP tasks.

# TokenizerBitWord

The TokenizerBitWord class adds methods for tokenizing text into subword tokens using the WordPiece and Byte-Pair Encoding (BPE) techniques. 
##### Methods: 
- TokenizeWordPiece(text As String): Tokenizes text into subword tokens using the WordPiece technique, handling unknown subwords and updating the vocabulary.
- TokenizeBitWord(subword As String): Tokenizes subword text into smaller parts using BPE, handling unknown subwords and updating the vocabulary.
- TokenizeBitWord(subword As String, ByRef Vocab As Dictionary(Of String, Integer)): An overloaded version of the previous method that allows using a specified vocabulary for tokenization.
- TokenizeBPE(text As String): Tokenizes text into subword tokens using the BPE technique, handling unknown subwords and updating the vocabulary. The TokenizerBitWord class introduces a different approach to subword tokenization.
##### Properties: 
- Vocabulary: A dictionary of subwords and their frequencies.
##### Methods: 
- Tokenize(ByRef Corpus As List(Of String)): Tokenizes a corpus into subwords using a vocabulary. This tokenizer handles different levels of tokenization, including paragraphs, sentences, and words. Usage These tokenization strategies offer different ways to preprocess text data for NLP tasks. 

You can choose the tokenization method that best suits your project's needs and integrate it into your NLP library. The classes provide methods for training the tokenizers and performing tokenization on text data. When developing your NLP library, you can utilize these tokenization strategies as part of a preprocessing pipeline. The tokenized text can then be used as input for further NLP tasks, such as training language models, text classification, sentiment analysis, and more. It's important to customize and adapt these tokenization methods based on the specific requirements of your project and the nature of the text data you're working with.

### NgramTokenizer

The NgramTokenizer class introduces methods for generating n-grams at different levels of text granularity. 
Methods: 
- TokenizetoCharacter(Document As String, n As Integer): Generates character n-grams from the document.
- TokenizetoWord(ByRef text As String, n As Integer): Generates word n-grams from the text.
- TokenizetoParagraph(text As String, n As Integer): Generates paragraph n-grams from the text.
- TokenizetoSentence(text As String, n As Integer): Generates sentence n-grams from the text. BasicTokenizer

###  TokenizerWordPiece

The TokenizerWordPiece class implements tokenization using the WordPiece algorithm, a subword tokenization approach commonly used in NLP. 
##### Methods: 
- Train(): Trains the tokenizer by calculating subword occurrences in the provided corpus and creating a vocabulary based on the most frequent subwords.
- Tokenize(text As String): Tokenizes input text into subwords using the trained vocabulary.

### TokenizerBPE

The TokenizerBPE class implements tokenization using Byte-Pair Encoding (BPE), another subword tokenization approach. 
##### Methods: 
- TrainBpeModel(corpus As List(Of String), numMerges As Integer): Trains the BPE model by iteratively merging the most frequent pairs of subwords. 

The class contains internal helper methods for merging subword pairs, finding the most frequent pair, and calculating pair frequency.

### PMI Class (Pointwise Mutual Information)

The PMI class provides functionality to calculate the Pointwise Mutual Information (PMI) matrix for a trained model. It involves calculating PMI values for pairs

### BasicTokenizer

The `BasicTokenizer` class provides basic tokenization methods at different levels of granularity. It includes methods for tokenizing documents into characters, words, sentences, and paragraphs.

- `TokenizeToCharacter(Document As String)`: Tokenizes a document into characters.
- `TokenizeToWord(Document As String)`: Tokenizes a document into words.
- `TokenizeToSentence(Document As String)`: Tokenizes a document into sentences.
- `TokenizeToParagraph(Document As String)`: Tokenizes a document into paragraphs.

### Vocabulary Management

The `Tokenizer` class, serving as a master tokenizer, provides various methods for managing the vocabulary used in tokenization.

- `Add_Vocabulary(initialVocabulary As List(Of String))`: Adds tokens from an initial vocabulary to the main vocabulary.
- `Initialize_Vocabulary(initialVocabulary As List(Of String), n As Integer)`: Initializes the vocabulary by adding n-grams from an initial vocabulary.
- `ComputePairFrequencies()`: Computes frequencies of subword pairs in the vocabulary.
- `UpdateFrequencyDictionary(mergedSubword As String)`: Updates the frequency dictionary with merged subwords.
- `UpdateVocabulary(ByRef Term As String)`: Updates the vocabulary with a new term or increases the frequency of an existing term.
- `UpdateCorpusWithMergedToken(ByRef corpus As List(Of String), pair As String)`: Updates the corpus with a merged token for the next iteration.
- `Prune(pruningThreshold As Integer)`: Prunes the vocabulary by removing infrequent tokens.

### CorpusCreator Class

The CorpusCreator class provides methods for generating classification and predictive datasets from text data.

### Word2WordMatrix Class

The Word2WordMatrix class is designed to generate a word-word matrix based on a collection of documents and a specified context window. It calculates the co-occurrence of words within a given window size.

# CoOccurrenceMatrix Class

The Co_Occurrence_Matrix class provides methods to work with co-occurrence matrices and calculate PMI values.

### PositionalEncoderDecoder Class
This class contains methods for encoding and decoding tokens and embeddings with positional information. It works as follows: - - EncodeTokenStr(token As String) As List(Of Double): Encodes a string token and returns its positional embedding as a list of doubles. EncodeTokenEmbedding(tokenEmbedding As List(Of Double)) As List(Of Double): Encodes a token embedding (list of doubles) and returns its positional embedding as a list of doubles. EncodeSentenceStr(tokens As List(Of String)) As List(Of List(Of Double)): Encodes a list of string tokens and returns their positional embeddings as a list of lists of doubles. EncodeSentenceEmbedding(tokenEmbeddings As List(Of List(Of Double))) As List(Of List(Of Double)): Encodes a list of token embeddings and returns their positional embeddings as a list of lists of doubles. DecodeTokenStr(positionalEmbedding As List(Of Double)) As String: Decodes a positional embedding (list of doubles) and returns the corresponding string token. DecodeTokenEmbedding(positionalEmbedding As List(Of Double)) As List(Of Double): Decodes a positional embedding (list of doubles) and returns the corresponding token embedding as a list of doubles. DecodeSentenceStr(sequenceEmbeddings As List(Of List(Of Double))) As List(Of String): Decodes a list of positional embeddings and returns the corresponding string tokens as a list of strings. DecodeSentenceEmbedding(sequenceEmbeddings As List(Of List(Of Double))) As List(Of List(Of Double)): Decodes a list of positional embeddings and returns the corresponding token embeddings as a list of lists of doubles.
The class uses an encoding matrix and a vocabulary list to perform the encoding and decoding operations. It's designed to work with different embedding model sizes (Dmodel) and maximum sequence lengths (MaxSeqLength) based on the provided vocabulary. This class essentially provides tools for representing tokens and embeddings in a sequential context, which can be useful in various natural language processing tasks.

### Corpus Language Model

This section includes classes and methods related to processing and analysing text corpora using various similarity metrics.

### BeliefNetwork Class

The `BeliefNetwork` class represents a probabilistic graphical model (belief network) and provides methods for loading training data, creating evidence, predicting outcomes, exporting the network structure, and more. It enables you to perform probabilistic inference on the defined belief network.


## Contributing

We welcome contributions from the community to enhance and expand this NLP Library for Visual Basic .NET. If you have ideas, improvements, or new features to suggest, please check out our [contribution guidelines](https://markdownview.tinygoodies.com/contribution-guidelines-link-here).

## License

This NLP Library for Visual Basic .NET is provided under the [MIT License](https://markdownview.tinygoodies.com/license-link-here). Feel free to use and modify the library according to your project's needs.

---

By leveraging the power of the NLP Library for Visual Basic .NET, you can embark on a journey of text analysis, manipulation, and exploration, unlocking new possibilities in the world of natural language processing. Start integrating this library into your projects and elevate your NLP endeavours to new heights.
