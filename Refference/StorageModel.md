# VectorDatabase

**Audio2Vec, Image2Vec, and VectorDatabase** are a set of classes and functions for working with audio, image, and text vectors in VB.NET. This project provides utilities for vectorization, encoding, decoding, similarity search, and vector storage.

## Features

- **Audio2Vec**: Convert audio signals to complex vectors (spectrum) and vice versa. Load and save audio data from/to files.
- **Image2Vec**: Encode and decode images into vectors. Perform similarity search on image vectors.
- **VectorDatabase**: Store and query audio, image, and text vectors using a dictionary-based database.

## Dependencies

This project makes use of the following dependencies:

- [MathNet.Numerics](https://numerics.mathdotnet.com/): A mathematics library for numerical computations.
- [NAudio](https://github.com/naudio/NAudio): An audio library for .NET that provides audio playback, capture, and audio file format support.

## Getting Started

1. Clone this repository to your local machine using:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Open the project in your preferred VB.NET development environment.

3. Explore the various classes (`Audio2Vec`, `Image2Vec`, `VectorDatabase`) and their functions to understand how to work with audio, image, and text vectors.

4. Modify and extend the code to suit your specific use case.

## Usage

### Audio2Vec

Convert audio signal to complex vectors and back:

```vb
' Load an audio file
Dim audioPath As String = "path/to/your/audio/file.wav"
Dim audioSignal As Double() = Audio2Vec.LoadAudio(audioPath)

' Convert audio signal to complex vectors
Dim windowSize As Integer = 1024
Dim hopSize As Integer = 256
Dim audioVectors As List(Of Complex()) = Audio2Vec.AudioToVector(audioSignal, windowSize, hopSize)

' Convert complex vectors back to audio signal
Dim reconstructedAudio As Double() = Audio2Vec.VectorToAudio(audioVectors, hopSize)
```

### Image2Vec

Encode and decode images using vectors:

```vb
' Encode an image into a vector
Dim imagePath As String = "path/to/your/image.jpg"
Dim imageWidth As Integer = 256
Dim imageHeight As Integer = 256
Dim imageVector As Double() = Image2Vec.ImageEncoder.EncodeImage(imagePath, imageWidth, imageHeight)

' Decode a vector into an image
Dim outputImagePath As String = "path/to/output/decoded_image.jpg"
Image2Vec.ImageDecoder.DecodeImage(imageVector, imageWidth, imageHeight, outputImagePath)
```

### VectorDatabase

Store and query vectors using the database:

```vb
' Create a new vector database
Dim vectorDB As New VectorDatabase()

' Add audio, image, and text vectors
Dim audioVector As List(Of Complex) = GetAudioVector() ' Your audio vector
vectorDB.AddAudioVector(1, audioVector)

Dim imageVector As List(Of Double) = GetImageVector() ' Your image vector
vectorDB.AddImageVector(2, imageVector)

Dim textVector As List(Of Double) = GetTextVector() ' Your text vector
vectorDB.AddTextVector(3, textVector)

' Find similar audio vectors
Dim queryAudioVector As List(Of Complex) = GetQueryAudioVector() ' Your query audio vector
Dim similarAudioIds As List(Of Integer) = vectorDB.FindSimilarAudioVectors(queryAudioVector, numNeighbors)

' Find similar image vectors
Dim queryImageVector As List(Of Double) = GetQueryImageVector() ' Your query image vector
Dim similarImageIds As List(Of Integer) = vectorDB.FindSimilarImageVectors(queryImageVector, numNeighbors)

' Find similar text vectors
Dim queryTextVector As List(Of Double) = GetQueryTextVector() ' Your query text vector
Dim similarTextIds As List(Of Integer) = vectorDB.FindSimilarTextVectors(queryTextVector, numNeighbors)
```

## License

This project is open source and available under the [MIT License](LICENSE). Feel free to use and modify this code for your own projects. Contributions are welcome!

## Contact

If you have any questions or suggestions, feel free to [open an issue](https://github.com/your-username/your-repo-name/issues) on GitHub.

