

# Positional Encoder Decoder (VB.NET)

The Positional Encoder Decoder is a Visual Basic .NET class that provides functionality for encoding and decoding tokens and sentences using positional embeddings. It allows you to convert between string tokens and their corresponding embeddings, and vice versa.

## Features

- Encode string tokens and get their positional embeddings as lists of doubles.
- Encode token embeddings and get their positional embeddings as lists of doubles.
- Encode lists of string tokens and get their positional embeddings as lists of lists of doubles.
- Encode lists of token embeddings and get their positional embeddings as lists of lists of doubles.
- Decode positional embeddings and get the corresponding string tokens.
- Decode positional embeddings and get the corresponding token embeddings as lists of doubles.

## Usage

1. Create an instance of the `PositionalEncoderDecoder` class by providing the necessary parameters: `Dmodel` (embedding model size), `MaxSeqLength` (maximum sentence length), and `vocabulary` (a list of known vocabulary).

```vb
Dim encoderDecoder As New PositionalEncoderDecoder(Dmodel, MaxSeqLength, vocabulary)
```

2. Encoding:

- Encode a string token and get its positional embedding:
```vb
Dim token As String = "example"
Dim embedding As List(Of Double) = encoderDecoder.EncodeTokenStr(token)
```

- Encode a token embedding and get its positional embedding:
```vb
Dim tokenEmbedding As List(Of Double) = GetTokenEmbedding() ' Get the token embedding from somewhere
Dim embedding As List(Of Double) = encoderDecoder.EncodeTokenEmbedding(tokenEmbedding)
```

- Encode a list of string tokens and get their positional embeddings:
```vb
Dim sentence As New List(Of String) From {"This", "is", "an", "example"}
Dim embeddings As List(Of List(Of Double)) = encoderDecoder.EncodeSentenceStr(sentence)
```

- Encode a list of token embeddings and get their positional embeddings:
```vb
Dim sentenceEmbeddings As List(Of List(Of Double)) = GetSentenceEmbeddings() ' Get the token embeddings from somewhere
Dim embeddings As List(Of List(Of Double)) = encoderDecoder.EncodeSentenceEmbedding(sentenceEmbeddings)
```

3. Decoding:

- Decode a list of positional embeddings and get the corresponding string tokens:
```vb
Dim embeddings As New List(Of List(Of Double)) From {embedding1, embedding2, embedding3}
Dim tokens As List(Of String) = encoderDecoder.DecodeSentenceStr(embeddings)
```

- Decode a list of positional embeddings and get the corresponding token embeddings:
```vb
Dim embeddings As New List(Of List(Of Double)) From {embedding1, embedding2, embedding3}
Dim tokenEmbeddings As List(Of List(Of Double)) = encoderDecoder.DecodeSentenceEmbedding(embeddings)
```

- Decode a positional embedding and get the corresponding string token:
```vb
Dim embedding As List(Of Double) = GetPositionalEmbedding() ' Get the positional embedding from somewhere
Dim token As String = encoderDecoder.DecodeTokenStr(embedding)
```

- Decode a positional embedding and get the corresponding token embedding:
```vb
Dim embedding As List(Of Double) = GetPositionalEmbedding() ' Get the positional embedding from somewhere
Dim tokenEmbedding As List(Of Double) = encoderDecoder.DecodeTokenEmbedding(embedding)
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## About

This `PositionalEncoderDecoder` class was developed as part of a project to enable token and sentence encoding/decoding using positional embeddings.

## Credits

The `PositionalEncoderDecoder` class is developed by [LEroySamuelDyer@hotmail.co.uk].
