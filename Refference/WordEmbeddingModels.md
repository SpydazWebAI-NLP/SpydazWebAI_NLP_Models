# Basic WordEmbeddings Model (VB.NET)

The WordEmbeddings model is a VB.NET implementation of the Word Embeddings technique, which represents words as dense vectors in a continuous vector space. This model allows you to train word embeddings on a given corpus and perform various operations such as calculating similarity between words, discovering collocations, and generating training data.

The Positional Encoder Decoder is a Visual Basic .NET class that provides functionality for encoding and decoding tokens and sentences using positional embeddings. It allows you to convert between string tokens and their corresponding embeddings, and vice versa.

## Features

- Train the WordEmbeddings model using a corpus of text
- Calculate Pointwise Mutual Information (PMI) matrix for the trained model
- Discover collocations (word pairs) based on trained word embeddings
- Find most similar words to a given word
- Generate training data for machine learning tasks using the trained model
- Save and load the trained model for future use
- Encode string tokens and get their positional embeddings as lists of doubles.
- Encode token embeddings and get their positional embeddings as lists of doubles.
- Encode lists of string tokens and get their positional embeddings as lists of lists of doubles.
- Encode lists of token embeddings and get their positional embeddings as lists of lists of doubles.
- Decode positional embeddings and get the corresponding string tokens.
- Decode positional embeddings and get the corresponding token embeddings as lists of doubles.

## Usage

1. Install the necessary dependencies (e.g., .NET framework) for running VB.NET applications.

2. Clone or download the repository to your local machine.

3. Import the WordEmbeddings class into your VB.NET project.

4. Initialize an instance of the WordEmbeddings model with the desired parameters:
   
```vb
   Dim model As New WordEmbeddings(embeddingSize:=100, learningRate:=0.01, windowSize:=5)
```

### Train the model using a corpus of text:


```vb

Dim corpus As String() = {
    "united states is a country.", "England is a country",
    "the united kingdom is in europe.",
    "united airlines is an airline company.",
    "doberman is a breed of dog.",
    "dogs have many breeds.",
    "dogs love eating pizza."
}
model.Train(corpus)
```


### Perform various operations on the trained model:

## Calculate PMI matrix:

```vb

Dim pmiMatrix As Dictionary(Of String, Dictionary(Of String, Double)) = model.CalculatePMI()
```

### Discover collocations:
```vb

Dim words As String() = {"united", "states", "kingdom", "airlines"}
Dim collocations As List(Of Tuple(Of String, String)) = model.DiscoverCollocations(words, threshold:=1)
```
##  most similar words:
```vb

Dim similarWords As List(Of String) = model.GetMostSimilarWords("dog", topK:=3)
```

### Save and load the trained model for future use:

```vb

' Save the model
model.SaveModel("path/to/save/model.json")

' Load the model
Dim loadedModel As WordEmbeddings = WordEmbeddings.LoadModel("path/to/saved/model.json")
```


### 1. Create an instance of the `PositionalEncoderDecoder` class 
by providing the necessary parameters: `Dmodel` (embedding model size), `MaxSeqLength` (maximum sentence length), and `vocabulary` (a list of known vocabulary).

```vb
Dim encoderDecoder As New PositionalEncoderDecoder(Dmodel, MaxSeqLength, vocabulary)
```

### 2. Encoding:

## Encode a string token and get its positional embedding:
```vb
Dim token As String = "example"
Dim embedding As List(Of Double) = encoderDecoder.EncodeTokenStr(token)
```

## Encode a token embedding and get its positional embedding:
```vb
Dim tokenEmbedding As List(Of Double) = GetTokenEmbedding() ' Get the token embedding from somewhere
Dim embedding As List(Of Double) = encoderDecoder.EncodeTokenEmbedding(tokenEmbedding)
```

## Encode a list of string tokens and get their positional embeddings:
```vb
Dim sentence As New List(Of String) From {"This", "is", "an", "example"}
Dim embeddings As List(Of List(Of Double)) = encoderDecoder.EncodeSentenceStr(sentence)
```

## Encode a list of token embeddings and get their positional embeddings:
```vb
Dim sentenceEmbeddings As List(Of List(Of Double)) = GetSentenceEmbeddings() ' Get the token embeddings from somewhere
Dim embeddings As List(Of List(Of Double)) = encoderDecoder.EncodeSentenceEmbedding(sentenceEmbeddings)
```

### 3. Decoding:

## Decode a list of positional embeddings and get the corresponding string tokens:
```vb
Dim embeddings As New List(Of List(Of Double)) From {embedding1, embedding2, embedding3}
Dim tokens As List(Of String) = encoderDecoder.DecodeSentenceStr(embeddings)
```

## Decode a list of positional embeddings and get the corresponding token embeddings:
```vb
Dim embeddings As New List(Of List(Of Double)) From {embedding1, embedding2, embedding3}
Dim tokenEmbeddings As List(Of List(Of Double)) = encoderDecoder.DecodeSentenceEmbedding(embeddings)
```

## Decode a positional embedding and get the corresponding string token:
```vb
Dim embedding As List(Of Double) = GetPositionalEmbedding() ' Get the positional embedding from somewhere
Dim token As String = encoderDecoder.DecodeTokenStr(embedding)
```

## Decode a positional embedding and get the corresponding token embedding:
```vb
Dim embedding As List(Of Double) = GetPositionalEmbedding() ' Get the positional embedding from somewhere
Dim tokenEmbedding As List(Of Double) = encoderDecoder.DecodeTokenEmbedding(embedding)
```
#### Refer to the code and function documentation for more advanced usage and customization options.

# Dependencies
The WordEmbeddings model requires the following dependencies:

# NET framework (version X.X or higher)
License
This project is licensed under the MIT License.

# Contributing
Contributions to the WordEmbeddings model are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
