Imports System.Drawing
Imports System.Drawing.Imaging
Imports System.IO
Imports System.Numerics
Imports System.Text
Imports System.Text.RegularExpressions
Imports System.Web.Script.Serialization
Imports System.Windows.Forms
Imports InputModelling.Factory
Imports InputModelling.LanguageModels
Imports InputModelling.Models.Audio
Imports InputModelling.Models.Images
Imports InputModelling.Models.MatrixModels
Imports InputModelling.Models.Readers
Imports InputModelling.Models.Storage
Imports InputModelling.Models.Storage.MinHashAndLSH
Imports InputModelling.Models.Text
Imports InputModelling.Models.Text.Word2Vector
Imports InputModelling.Models.Tokenizers
Imports InputModelling.Models.VocabularyModelling
Imports InputModelling.Utilitys
Imports InputModelling.Utilitys.TEXT
Imports MathNet.Numerics.IntegralTransforms
Imports Newtonsoft.Json

Namespace Factory
    Public MustInherit Class WordEmbeddingsModel

        ' A simple vocabulary for demonstration purposes.
        Private iVocabulary As New List(Of String) From {"apple", "orange", "banana", "grape", "cherry"}
        Public Property Vocabulary As List(Of String)
            Get
                Return iVocabulary
            End Get
            Set(value As List(Of String))
                iVocabulary = value
            End Set
        End Property
        ' Word embeddings dictionary to store the learned word vectors.
        Public WordEmbeddings As New WordEmbedding

        ' Hyperparameters for training.
        Public EmbeddingSize As Integer = 50 ' Size of word vectors.
        Public WindowSize As Integer = 2 ' Context window size.

        Public LearningRate As Double = 0.01 ' Learning rate for gradient descent.
        Public NumEpochs As Integer = 1000 ' Number of training epochs.

        ' Random number generator for initialization.
        Public Shared ReadOnly Rand As New Random()
        Public MustOverride Sub Train()
        Public MustOverride Sub Train(corpus As List(Of List(Of String)))
        Public Sub New(ByRef model As WordEmbeddingsModel)
            iVocabulary = model.Vocabulary
            WordEmbeddings = model.WordEmbeddings
            EmbeddingSize = model.EmbeddingSize
            WindowSize = model.WindowSize
            LearningRate = model.LearningRate
            NumEpochs = model.NumEpochs
        End Sub
        Public Sub New(ByRef Vocabulary As List(Of String))
            iVocabulary = Vocabulary
        End Sub
        Public Function ExportModel() As WordEmbeddingsModel
            Return Me
        End Function
        Public Sub SetTrainingParameters(ByRef Embeddingsize As Integer,
                                     ByRef WindowSize As Integer,
                                     ByRef LearningRate As Double, ByRef Epochs As Integer)
            Me.EmbeddingSize = Embeddingsize
            Me.WindowSize = WindowSize
            Me.LearningRate = LearningRate
            Me.NumEpochs = Epochs
        End Sub
        ' WordEmbedding class to store word vectors and handle operations on word embeddings.
        Public Class WordEmbedding
            Public embeddings As Dictionary(Of String, Double())

            Public Sub New()
                Me.embeddings = New Dictionary(Of String, Double())()
            End Sub

            Public Sub Add(word As String, vector As Double())
                embeddings(word) = vector
            End Sub

            Public Function GetVector(word As String) As Double()
                Return embeddings(word)
            End Function

            ' Implement other operations as needed for word embeddings.
            ' E.g., similarity, word lookup, etc.
        End Class
        Public Function ComputeDotProduct(vec1 As Double(), vec2 As Double()) As Double
            Return Enumerable.Range(0, EmbeddingSize).Sum(Function(i) vec1(i) * vec2(i))
        End Function

        Public Function Sigmoid(x As Double) As Double
            Return 1.0 / (1.0 + Math.Exp(-x))
        End Function

        ''' <summary>
        ''' Cosine Similarity(A, B) = (A dot B) / (||A|| * ||B||)
        '''  where:
        '''  A And B are the word vectors of two words.
        '''  A dot B Is the dot product Of the two vectors.
        '''  ||A|| And ||B|| are the magnitudes (Euclidean norms) of the vectors.
        '''  The cosine similarity ranges from -1 To 1, where 1 indicates the highest similarity, 0 indicates no similarity, And -1 indicates the highest dissimilarity.
        ''' </summary>
        ''' <param name="word1"></param>
        ''' <param name="word2"></param>
        ''' <param name="wordEmbeddings"></param>
        ''' <returns></returns>
        Public Function CosineSimilarity(word1 As String, word2 As String, wordEmbeddings As WordEmbedding) As Double
            Dim vector1 As Double() = wordEmbeddings.GetVector(word1)
            Dim vector2 As Double() = wordEmbeddings.GetVector(word2)

            ' Calculate the dot product of the two vectors.
            Dim dotProduct As Double = 0
            For i As Integer = 0 To vector1.Length - 1
                dotProduct += vector1(i) * vector2(i)
            Next

            ' Calculate the magnitudes of the vectors.
            Dim magnitude1 As Double = Math.Sqrt(vector1.Sum(Function(x) x * x))
            Dim magnitude2 As Double = Math.Sqrt(vector2.Sum(Function(x) x * x))

            ' Calculate the cosine similarity.
            Dim similarity As Double = dotProduct / (magnitude1 * magnitude2)

            Return similarity
        End Function
        Public Function DisplayMatrix(matrix As Dictionary(Of String, Dictionary(Of String, Double))) As DataGridView
            Dim dataGridView As New DataGridView()
            dataGridView.Dock = DockStyle.Fill
            dataGridView.AutoGenerateColumns = False
            dataGridView.AllowUserToAddRows = False

            ' Add columns to the DataGridView
            Dim wordColumn As New DataGridViewTextBoxColumn()
            wordColumn.HeaderText = "Word"
            wordColumn.DataPropertyName = "Word"
            dataGridView.Columns.Add(wordColumn)

            For Each contextWord As String In matrix.Keys
                Dim contextColumn As New DataGridViewTextBoxColumn()
                contextColumn.HeaderText = contextWord
                contextColumn.DataPropertyName = contextWord
                dataGridView.Columns.Add(contextColumn)
            Next

            ' Populate the DataGridView with the matrix data
            For Each word As String In matrix.Keys
                Dim rowValues As New List(Of Object)
                rowValues.Add(word)

                For Each contextWord As String In matrix.Keys
                    Dim count As Object = If(matrix(word).ContainsKey(contextWord), matrix(word)(contextWord), 0)
                    rowValues.Add(count)
                Next

                dataGridView.Rows.Add(rowValues.ToArray())
            Next

            Return dataGridView
        End Function
        Public Sub DisplayEmbeddingsModel()
            Dim dgv = DisplayMatrix(WordEmbeddings.embeddings)

            ' Create a form and add the DataGridView to it
            Dim kform As New Form
            kform.Text = "Embedding Matrix"
            kform.Size = New Size(800, 600)
            kform.Controls.Add(dgv)

            ' Display the form
            Application.Run(kform)
        End Sub
        Public Function DisplayMatrix(matrix As Dictionary(Of String, Double())) As DataGridView
            Dim dataGridView As New DataGridView()
            dataGridView.Dock = DockStyle.Fill
            dataGridView.AutoGenerateColumns = False
            dataGridView.AllowUserToAddRows = False

            ' Add columns to the DataGridView
            Dim wordColumn As New DataGridViewTextBoxColumn()
            wordColumn.HeaderText = "Word"
            wordColumn.DataPropertyName = "Word"
            dataGridView.Columns.Add(wordColumn)

            For Each contextWord As String In matrix.Keys
                Dim contextColumn As New DataGridViewTextBoxColumn()
                contextColumn.HeaderText = contextWord
                contextColumn.DataPropertyName = contextWord
                dataGridView.Columns.Add(contextColumn)
            Next

            ' Populate the DataGridView with the matrix data
            For Each word As String In matrix.Keys
                Dim rowValues As New List(Of Object)()
                rowValues.Add(word)

                For Each contextWord As String In matrix.Keys
                    Dim count As Integer = If(matrix.ContainsKey(contextWord), matrix(word)(contextWord), 0)
                    rowValues.Add(count)
                Next

                dataGridView.Rows.Add(rowValues.ToArray())
            Next

            Return dataGridView
        End Function
        Public Class WordListReader
            Private wordList As List(Of String)

            Public Sub New(filePath As String)
                wordList = New List(Of String)()
                ReadWordList(filePath)
            End Sub

            Private Sub ReadWordList(filePath As String)
                Using reader As New StreamReader(filePath)
                    While Not reader.EndOfStream
                        Dim line As String = reader.ReadLine()
                        If Not String.IsNullOrEmpty(line) Then
                            wordList.Add(line.Trim.ToLower)
                        End If
                    End While
                End Using
            End Sub

            Public Function GetWords() As List(Of String)
                Return wordList
            End Function
            ' Usage Example:
            Public Shared Sub Main()
                ' Assuming you have a wordlist file named 'words.txt' in the same directory
                Dim corpusRoot As String = "."
                Dim wordlistPath As String = Path.Combine(corpusRoot, "wordlist.txt")

                Dim wordlistReader As New WordListReader(wordlistPath)
                Dim words As List(Of String) = wordlistReader.GetWords()

                For Each word As String In words
                    Console.WriteLine(word)
                Next
                Console.ReadLine()
                ' Rest of your code...
            End Sub


        End Class
        Public Class PMI
            Public cooccurrenceMatrix As Dictionary(Of String, Dictionary(Of String, Double))
            Public vocabulary As Dictionary(Of String, Integer)
            Private wordToIndex As Dictionary(Of String, Integer)
            Private indexToWord As Dictionary(Of Integer, String)
            Private embeddingSize As Integer = embeddingMatrix.Length
            Private embeddingMatrix As Double(,)

            Public Sub New(vocabulary As Dictionary(Of String, Integer))
                If vocabulary Is Nothing Then
                    Throw New ArgumentNullException(NameOf(vocabulary))
                End If

                Me.vocabulary = vocabulary
            End Sub

            ''' <summary>
            ''' Discovers collocations among the specified words based on the trained model.
            ''' </summary>
            ''' <param name="words">The words to discover collocations for.</param>
            ''' <param name="threshold">The similarity threshold for collocation discovery.</param>
            ''' <returns>A list of collocations (word pairs) that meet the threshold.</returns>
            Public Function DiscoverCollocations(words As String(), threshold As Double) As List(Of Tuple(Of String, String))
                Dim collocations As New List(Of Tuple(Of String, String))

                For i As Integer = 0 To words.Length - 2
                    For j As Integer = i + 1 To words.Length - 1
                        Dim word1 As String = words(i)
                        Dim word2 As String = words(j)

                        If vocabulary.Keys.Contains(word1) AndAlso vocabulary.Keys.Contains(word2) Then
                            Dim vector1 As Double() = GetEmbedding(wordToIndex(word1))
                            Dim vector2 As Double() = GetEmbedding(wordToIndex(word2))
                            Dim similarity As Double = CalculateSimilarity(vector1, vector2)

                            If similarity >= threshold Then
                                collocations.Add(Tuple.Create(word1, word2))
                            End If
                        End If
                    Next
                Next

                Return collocations
            End Function
            Private Function CalculateSimilarity(vectorA As Double(), vectorB As Double()) As Double
                Dim dotProduct As Double = 0
                Dim magnitudeA As Double = 0
                Dim magnitudeB As Double = 0

                For i As Integer = 0 To vectorA.Length - 1
                    dotProduct += vectorA(i) * vectorB(i)
                    magnitudeA += vectorA(i) * vectorA(i)
                    magnitudeB += vectorB(i) * vectorB(i)
                Next

                If magnitudeA <> 0 AndAlso magnitudeB <> 0 Then
                    Return dotProduct / (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB))
                Else
                    Return 0
                End If
            End Function

            Private Sub InitializeEmbeddings()
                Dim vocabSize As Integer = vocabulary.Count
                embeddingMatrix = New Double(vocabSize - 1, embeddingSize - 1) {}

                Dim random As New Random()
                For i As Integer = 0 To vocabSize - 1
                    For j As Integer = 0 To embeddingSize - 1
                        embeddingMatrix(i, j) = random.NextDouble()
                    Next
                Next
            End Sub
            Private Function GetEmbedding(index As Integer) As Double()
                If indexToWord.ContainsKey(index) Then
                    Dim vector(embeddingSize - 1) As Double
                    For i As Integer = 0 To embeddingSize - 1
                        vector(i) = embeddingMatrix(index, i)
                    Next
                    Return vector
                Else
                    Return Nothing
                End If
            End Function
            ''' <summary>
            ''' Calculates the Pointwise Mutual Information (PMI) matrix for the trained model.
            ''' </summary>
            ''' <returns>A dictionary representing the PMI matrix.</returns>
            Public Function CalculatePMI() As Dictionary(Of String, Dictionary(Of String, Double))
                Dim pmiMatrix As New Dictionary(Of String, Dictionary(Of String, Double))

                Dim totalCooccurrences As Double = GetTotalCooccurrences()

                For Each targetWord In cooccurrenceMatrix.Keys
                    Dim targetOccurrences As Double = GetTotalOccurrences(targetWord)

                    If Not pmiMatrix.ContainsKey(targetWord) Then
                        pmiMatrix(targetWord) = New Dictionary(Of String, Double)
                    End If

                    For Each contextWord In cooccurrenceMatrix(targetWord).Keys
                        Dim contextOccurrences As Double = GetTotalOccurrences(contextWord)
                        Dim cooccurrences As Double = cooccurrenceMatrix(targetWord)(contextWord)

                        Dim pmiValue As Double = Math.Log((cooccurrences * totalCooccurrences) / (targetOccurrences * contextOccurrences))
                        pmiMatrix(targetWord)(contextWord) = pmiValue
                    Next
                Next

                Return pmiMatrix
            End Function
            Private Function GetTotalCooccurrences() As Double
                Dim total As Double = 0

                For Each targetWord In cooccurrenceMatrix.Keys
                    For Each cooccurrenceValue In cooccurrenceMatrix(targetWord).Values
                        total += cooccurrenceValue
                    Next
                Next

                Return total
            End Function

            Private Function GetTotalOccurrences(word As String) As Double
                Dim total As Double = 0

                If cooccurrenceMatrix.ContainsKey(word) Then
                    total = cooccurrenceMatrix(word).Values.Sum()
                End If

                Return total
            End Function
            Private Function GenerateCooccurrenceMatrix(corpus As String(), windowSize As Integer) As Dictionary(Of String, Dictionary(Of String, Double))
                Dim matrix As New Dictionary(Of String, Dictionary(Of String, Double))

                For Each sentence In corpus
                    Dim words As String() = sentence.Split(" "c)
                    Dim length As Integer = words.Length

                    For i As Integer = 0 To length - 1
                        Dim targetWord As String = words(i)

                        If Not matrix.ContainsKey(targetWord) Then
                            matrix(targetWord) = New Dictionary(Of String, Double)
                        End If

                        For j As Integer = Math.Max(0, i - windowSize) To Math.Min(length - 1, i + windowSize)
                            If i = j Then
                                Continue For
                            End If

                            Dim contextWord As String = words(j)
                            Dim distance As Double = 1 / Math.Abs(i - j)

                            If matrix(targetWord).ContainsKey(contextWord) Then
                                matrix(targetWord)(contextWord) += distance
                            Else
                                matrix(targetWord)(contextWord) = distance
                            End If
                        Next
                    Next
                Next

                Return matrix
            End Function
            Public Function DisplayMatrix(matrix As Dictionary(Of String, Dictionary(Of String, Integer))) As DataGridView
                Dim dataGridView As New DataGridView()
                dataGridView.Dock = DockStyle.Fill
                dataGridView.AutoGenerateColumns = False
                dataGridView.AllowUserToAddRows = False

                ' Add columns to the DataGridView
                Dim wordColumn As New DataGridViewTextBoxColumn()
                wordColumn.HeaderText = "Word"
                wordColumn.DataPropertyName = "Word"
                dataGridView.Columns.Add(wordColumn)

                For Each contextWord As String In matrix.Keys
                    Dim contextColumn As New DataGridViewTextBoxColumn()
                    contextColumn.HeaderText = contextWord
                    contextColumn.DataPropertyName = contextWord
                    dataGridView.Columns.Add(contextColumn)
                Next

                ' Populate the DataGridView with the matrix data
                For Each word As String In matrix.Keys
                    Dim rowValues As New List(Of Integer)
                    rowValues.Add(word)

                    For Each contextWord As String In matrix.Keys
                        Dim count As Object = If(matrix(word).ContainsKey(contextWord), matrix(word)(contextWord), 0)
                        rowValues.Add(count)
                    Next

                    dataGridView.Rows.Add(rowValues.ToArray())
                Next

                Return dataGridView
            End Function

        End Class

    End Class

    ''' <summary>
    ''' One possible way to combine the approaches is by using a two-step training process:
    '''  Pre-training Using Skip-gram With Negative Sampling:
    '''   In this step, 
    '''    you can pre-train the word embeddings using the skip-gram model 
    '''    with negative sampling on a large dataset Or a diverse corpus. 
    '''    This step allows you to efficiently learn word embeddings 
    '''    in a computationally efficient 
    '''    manner while capturing semantic relationships between words.
    '''  Fine-tuning using Hierarchical Softmax:
    '''   After pre-training the word embeddings, 
    '''    you can perform fine-tuning Using the hierarchical softmax technique. 
    '''    During the fine-tuning Step, 
    '''    you can use a smaller dataset 
    '''   Or a more domain-specific corpus 
    '''    To train the model Using hierarchical softmax. 
    '''    This Step enables you To refine the word embeddings 
    '''    And make them more accurate And context-specific.
    ''' </summary>
    Public Class HybridWordEmbeddingsModel
        Inherits WordEmbeddingsModel

        Public Sub New(ByRef model As WordEmbeddingsModel)
            MyBase.New(model)
        End Sub

        Public Sub New(ByRef Vocabulary As List(Of String))
            MyBase.New(Vocabulary)
        End Sub
        Public Enum ModelType
            Skipgram
            Glove
            SoftMax
            CBOW
            FastText
        End Enum
        Public Function PreTrain(ByRef model As WordEmbeddingsModel, ByRef iModelType As ModelType) As WordEmbeddingsModel
            model.Train()
            Dim preTrainedModel As WordEmbeddingsModel


            Select Case iModelType
                Case ModelType.Skipgram
                    preTrainedModel = New WordEmbeddingsWithNegativeSampling(model.Vocabulary)
                    preTrainedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs)  ' Set appropriate parameters for pre-training

                    preTrainedModel.Train() ' Pre-train the word embeddings using Skip-gram with Negative Sampling

                    Return preTrainedModel
                Case ModelType.Glove
                    preTrainedModel = New WordEmbeddingsWithGloVe(model.Vocabulary)
                    preTrainedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs)  ' Set appropriate parameters for pre-training

                    preTrainedModel.Train() ' Pre-train the word embeddings using Skip-gram with Negative Sampling

                    Return preTrainedModel
                Case ModelType.FastText
                    preTrainedModel = New WordEmbeddingsWithFastText(model.Vocabulary)
                    preTrainedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs)  ' Set appropriate parameters for pre-training

                    preTrainedModel.Train() ' Pre-train the word embeddings using Skip-gram with Negative Sampling

                    Return preTrainedModel
                Case ModelType.CBOW
                    preTrainedModel = New WordEmbeddingsWithCBOW(model.Vocabulary)
                    preTrainedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs)  ' Set appropriate parameters for pre-training

                    preTrainedModel.Train() ' Pre-train the word embeddings using Skip-gram with Negative Sampling

                    Return preTrainedModel
                Case ModelType.SoftMax
                    preTrainedModel = New WordEmbeddingsWithHierarchicalSoftmax(model.Vocabulary)
                    preTrainedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs)  ' Set appropriate parameters for pre-training

                    preTrainedModel.Train() ' Pre-train the word embeddings using Skip-gram with Negative Sampling

                    Return preTrainedModel
            End Select
            Return model
        End Function

        Public Function FineTune(ByRef Model As WordEmbeddingsModel, ByRef iModelType As ModelType) As WordEmbeddingsModel

            Dim fineTunedModel As WordEmbeddingsModel

            Model.Train()

            Select Case iModelType
                Case ModelType.CBOW
                    fineTunedModel = New WordEmbeddingsWithCBOW(Model.Vocabulary)
                    fineTunedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for fine-tuning
                    fineTunedModel.Train() ' Fine-tune the word embeddings using Hierarchical Softmax
                    Return fineTunedModel
                Case ModelType.FastText
                    fineTunedModel = New WordEmbeddingsWithFastText(Model.Vocabulary)
                    fineTunedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for fine-tuning
                    fineTunedModel.Train() ' Fine-tune the word embeddings using Hierarchical Softmax
                    Return fineTunedModel
                Case ModelType.Glove
                    fineTunedModel = New WordEmbeddingsWithGloVe(Model.Vocabulary)
                    fineTunedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for fine-tuning
                    fineTunedModel.Train() ' Fine-tune the word embeddings using Hierarchical Softmax
                    Return fineTunedModel
                Case ModelType.Skipgram
                    fineTunedModel = New WordEmbeddingsWithNegativeSampling(Model.Vocabulary)
                    fineTunedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for fine-tuning
                    fineTunedModel.Train() ' Fine-tune the word embeddings using Hierarchical Softmax
                    Return fineTunedModel
                Case ModelType.SoftMax
                    fineTunedModel = New WordEmbeddingsWithHierarchicalSoftmax(Model.Vocabulary)
                    fineTunedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for fine-tuning
                    fineTunedModel.Train() ' Fine-tune the word embeddings using Hierarchical Softmax
                    Return fineTunedModel

            End Select


            Return Model

        End Function


        Public Overloads Sub Train(Optional PretrainModel As ModelType = ModelType.Skipgram, Optional FineTuneModel As ModelType = ModelType.Glove)
            Dim hybrid As New HybridWordEmbeddingsModel(Vocabulary)
            Dim preTrainedModel = PreTrain(hybrid, PretrainModel)
            Dim fineTunedModel = FineTune(preTrainedModel, FineTuneModel)
            'set model
            Me.WordEmbeddings = fineTunedModel.WordEmbeddings

        End Sub


        Public Overrides Sub Train(corpus As List(Of List(Of String)))
            ' Step 1: Pre-training using Skip-gram with Negative Sampling.
            Console.WriteLine("Pre-training using Skip-gram with Negative Sampling...")
            Dim preTrainedModel As New WordEmbeddingsWithNegativeSampling(Vocabulary)
            preTrainedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for pre-training
            preTrainedModel.Train(corpus) ' Pre-train the word embeddings using Skip-gram with Negative Sampling


            ' Step 3: Fine-tuning using Hierarchical Softmax.
            Console.WriteLine("Fine-tuning using Hierarchical Softmax...")
            Dim fineTunedModel As New WordEmbeddingsWithHierarchicalSoftmax(Vocabulary)
            fineTunedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for fine-tuning
            fineTunedModel.Train(corpus) ' Fine-tune the word embeddings using Hierarchical Softmax

            ' Step 4: Set the fine-tuned word embeddings as the model's word embeddings.
            WordEmbeddings = fineTunedModel.WordEmbeddings

            Console.WriteLine("Training completed!")
        End Sub

        Public Overrides Sub Train()
            ' Step 1: Pre-training using Skip-gram with Negative Sampling.
            Console.WriteLine("Pre-training using Skip-gram with Negative Sampling...")
            Dim preTrainedModel As New WordEmbeddingsWithNegativeSampling(Vocabulary)
            preTrainedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for pre-training
            preTrainedModel.train() ' Pre-train the word embeddings using Skip-gram with Negative Sampling


            ' Step 3: Fine-tuning using Hierarchical Softmax.
            Console.WriteLine("Fine-tuning using Hierarchical Softmax...")
            Dim fineTunedModel As New WordEmbeddingsWithHierarchicalSoftmax(Vocabulary)
            fineTunedModel.SetTrainingParameters(EmbeddingSize, WindowSize, LearningRate, NumEpochs) ' Set appropriate parameters for fine-tuning
            fineTunedModel.Train() ' Fine-tune the word embeddings using Hierarchical Softmax

            ' Step 4: Set the fine-tuned word embeddings as the model's word embeddings.
            WordEmbeddings = fineTunedModel.WordEmbeddings

            Console.WriteLine("Training completed!")
        End Sub


    End Class

End Namespace
Namespace Models
    Namespace Text
        ''' <summary>
        ''' This is a TFIDF Vectorizer For basic Embeddings
        ''' </summary>
        Public Class SentenceVectorizer
            Private ReadOnly documents As List(Of List(Of String))
            Private ReadOnly idf As Dictionary(Of String, Double)

            Public Sub New(documents As List(Of List(Of String)))
                Me.documents = documents
                Me.idf = CalculateIDF(documents)
            End Sub

            Public Sub New()
                documents = New List(Of List(Of String))
                idf = New Dictionary(Of String, Double)
            End Sub

            Public Function Vectorize(sentence As List(Of String)) As List(Of Double)
                Dim termFrequency = CalculateTermFrequency(sentence)
                Dim vector As New List(Of Double)

                For Each term In idf.Keys
                    Dim tfidf As Double = termFrequency(term) * idf(term)
                    vector.Add(tfidf)
                Next

                Return vector
            End Function

            Public Function CalculateIDF(documents As List(Of List(Of String))) As Dictionary(Of String, Double)
                Dim idf As New Dictionary(Of String, Double)
                Dim totalDocuments As Integer = documents.Count

                For Each document In documents
                    Dim uniqueTerms As List(Of String) = document.Distinct().ToList()

                    For Each term In uniqueTerms
                        If idf.ContainsKey(term) Then
                            idf(term) += 1
                        Else
                            idf(term) = 1
                        End If
                    Next
                Next

                For Each term In idf.Keys
                    idf(term) = Math.Log(totalDocuments / idf(term))
                Next

                Return idf
            End Function

            Public Function CalculateTermFrequency(sentence As List(Of String)) As Dictionary(Of String, Double)
                Dim termFrequency As New Dictionary(Of String, Double)

                For Each term In sentence
                    If termFrequency.ContainsKey(term) Then
                        termFrequency(term) += 1
                    Else
                        termFrequency(term) = 1
                    End If
                Next

                Return termFrequency
            End Function

        End Class
        Public Class Word2Vector
            Private embeddingMatrix As Double(,)
            Private embeddingSize As Integer
            Private indexToWord As Dictionary(Of Integer, String)
            Private learningRate As Double
            Private negativeSamples As Integer
            Private vocabulary As HashSet(Of String)
            Private weights As Double()
            Private windowSize As Integer
            Private wordToIndex As Dictionary(Of String, Integer)

            Public Sub New(embeddingSize As Integer, learningRate As Double, windowSize As Integer, negativeSamples As Integer)
                Me.embeddingSize = embeddingSize
                Me.learningRate = learningRate
                Me.windowSize = windowSize
                Me.negativeSamples = negativeSamples

                vocabulary = New HashSet(Of String)()
                wordToIndex = New Dictionary(Of String, Integer)()
                indexToWord = New Dictionary(Of Integer, String)()
                weights = New Double(vocabulary.Count - 1) {}
            End Sub

            Public Enum TrainingMethod
                CBOW
                SkipGram
            End Enum
            Private Shared Function RemoveStopwords(text As String, stopwords As List(Of String)) As String
                Dim cleanedText As String = ""
                Dim words As String() = text.Split()

                For Each word As String In words
                    If Not stopwords.Contains(word.ToLower()) Then
                        cleanedText += word + " "
                    End If
                Next

                Return cleanedText.Trim()
            End Function

            ' Helper function to get context words within the window
            Function GetContextWords(ByVal sentence As String(), ByVal targetIndex As Integer) As List(Of String)
                Dim contextWords As New List(Of String)

                For i = Math.Max(0, targetIndex - windowSize) To Math.Min(sentence.Length - 1, targetIndex + windowSize)
                    If i <> targetIndex Then
                        contextWords.Add(sentence(i))
                    End If
                Next

                Return contextWords
            End Function


            Public Function GetSimilarWords(word As String, topK As Integer) As List(Of String)
                Dim wordIndex As Integer = wordToIndex(word)

                Dim similarities As New Dictionary(Of String, Double)()
                For Each otherWord As String In vocabulary
                    If otherWord <> word Then
                        Dim otherWordIndex As Integer = wordToIndex(otherWord)
                        Dim similarity As Double = CalculateSimilarity(GetEmbedding(wordIndex), GetEmbedding(otherWordIndex))
                        similarities.Add(otherWord, similarity)
                    End If
                Next

                Dim orderedSimilarities = similarities.OrderByDescending(Function(x) x.Value)
                Dim similarWords As New List(Of String)()

                Dim count As Integer = 0
                For Each pair In orderedSimilarities
                    similarWords.Add(pair.Key)
                    count += 1
                    If count >= topK Then
                        Exit For
                    End If
                Next

                Return similarWords
            End Function

            Public Function GetEmbedding(word As String) As Double()
                If wordToIndex.ContainsKey(word) Then
                    Dim wordIndex As Integer = wordToIndex(word)
                    Dim vector(embeddingSize - 1) As Double
                    For i As Integer = 0 To embeddingSize - 1
                        vector(i) = embeddingMatrix(wordIndex, i)
                    Next
                    Return vector
                Else
                    Return Nothing
                End If
            End Function

            Public Function GetMostSimilarWords(word As String, topK As Integer) As List(Of String)
                If wordToIndex.ContainsKey(word) Then
                    Dim wordIndex As Integer = wordToIndex(word)
                    Dim wordVector As Double() = GetEmbedding(word)
                    Dim similarities As New List(Of Tuple(Of String, Double))()

                    For i As Integer = 0 To vocabulary.Count - 1
                        If i <> wordIndex Then
                            Dim currentWord As String = indexToWord(i)
                            Dim currentVector As Double() = GetEmbedding(currentWord)
                            Dim similarity As Double = CalculateSimilarity(wordVector, currentVector)
                            similarities.Add(New Tuple(Of String, Double)(currentWord, similarity))
                        End If
                    Next

                    similarities.Sort(Function(x, y) y.Item2.CompareTo(x.Item2))

                    Dim similarWords As New List(Of String)()
                    For i As Integer = 0 To topK - 1
                        similarWords.Add(similarities(i).Item1)
                    Next

                    Return similarWords
                Else
                    Return Nothing
                End If
            End Function

            Public Sub Train(corpus As List(Of String))
                BuildVocabulary(corpus)
                InitializeEmbeddings()
                Dim trainingData As List(Of List(Of Integer)) = GenerateTrainingData(corpus)
                For epoch = 1 To 10
                    For Each sentenceIndices As List(Of Integer) In trainingData
                        For i As Integer = windowSize To sentenceIndices.Count - windowSize - 1
                            Dim targetIndex As Integer = sentenceIndices(i)
                            Dim contextIndices As List(Of Integer) = GetContextIndices(sentenceIndices, i)

                            Dim contextVectors As List(Of Double()) = GetContextVectors(contextIndices)
                            Dim targetVector As Double() = GetTargetVector(targetIndex)

                            UpdateTargetVector(contextVectors, targetVector)
                        Next
                    Next
                Next
            End Sub

            Public Sub Train(corpus As List(Of String), method As TrainingMethod)
                BuildVocabulary(corpus)
                InitializeEmbeddings()
                InitializeWeights()
                Dim trainingData As List(Of List(Of Integer)) = GenerateTrainingData(corpus)
                For Epoch = 1 To 10
                    For Each sentenceIndices As List(Of Integer) In trainingData
                        For i As Integer = windowSize To sentenceIndices.Count - windowSize - 1
                            Dim targetIndex As Integer = sentenceIndices(i)
                            Dim contextIndices As List(Of Integer) = GetContextIndices(sentenceIndices, i)

                            If method = TrainingMethod.CBOW Then
                                TrainCBOW(contextIndices, targetIndex)
                            ElseIf method = TrainingMethod.SkipGram Then
                                TrainSkipGram(targetIndex, contextIndices)
                            End If
                        Next
                    Next
                Next
            End Sub
            Private Sub BuildVocabulary(corpus As List(Of String))
                Dim index As Integer = 0
                For Each sentence As String In corpus
                    Dim cleanedText As String = Regex.Replace(sentence, "[^\w\s]", "").ToLower()
                    Dim tokens As String() = cleanedText.Split()
                    For Each token As String In tokens
                        If Not wordToIndex.ContainsKey(token) Then
                            vocabulary.Add(token)
                            wordToIndex.Add(token, index)
                            indexToWord.Add(index, token)
                            index += 1
                        End If
                    Next
                Next
            End Sub

            Private Function CalculateSimilarity(vector1 As Double(), vector2 As Double()) As Double
                Dim dotProduct As Double = 0.0
                Dim magnitude1 As Double = 0.0
                Dim magnitude2 As Double = 0.0

                For i As Integer = 0 To embeddingSize - 1
                    dotProduct += vector1(i) * vector2(i)
                    magnitude1 += vector1(i) * vector1(i)
                    magnitude2 += vector2(i) * vector2(i)
                Next

                magnitude1 = Math.Sqrt(magnitude1)
                magnitude2 = Math.Sqrt(magnitude2)

                Return dotProduct / (magnitude1 * magnitude2)
            End Function

            Private Function GenerateTrainingData(corpus As List(Of String)) As List(Of List(Of Integer))
                Dim trainingData As New List(Of List(Of Integer))

                For Each sentence As String In corpus
                    Dim cleanedText As String = Regex.Replace(sentence, "[^\w\s]", "").ToLower()
                    Dim tokens As String() = cleanedText.Split()
                    Dim sentenceIndices As New List(Of Integer)()

                    For Each token As String In tokens
                        sentenceIndices.Add(wordToIndex(token))
                    Next

                    trainingData.Add(sentenceIndices)
                Next

                Return trainingData
            End Function

            Private Function GetContextIndices(sentenceIndices As List(Of Integer), targetIndex As Integer) As List(Of Integer)
                Dim contextIndices As New List(Of Integer)

                Dim startIndex As Integer = Math.Max(0, targetIndex - windowSize)
                Dim endIndex As Integer = Math.Min(sentenceIndices.Count - 1, targetIndex + windowSize)

                For i As Integer = startIndex To endIndex
                    If i <> targetIndex Then
                        contextIndices.Add(sentenceIndices(i))
                    End If
                Next

                Return contextIndices
            End Function

            Private Function GetContextVector(contextIndex As Integer) As Double()
                Dim vector(embeddingSize - 1) As Double
                For i As Integer = 0 To embeddingSize - 1
                    vector(i) = embeddingMatrix(contextIndex, i)
                Next
                Return vector
            End Function

            Private Function GetContextVectors(contextIndices As List(Of Integer)) As List(Of Double())
                Dim contextVectors As New List(Of Double())

                For Each contextIndex As Integer In contextIndices
                    Dim vector(embeddingSize - 1) As Double
                    For i As Integer = 0 To embeddingSize - 1
                        vector(i) = embeddingMatrix(contextIndex, i)
                    Next
                    contextVectors.Add(vector)
                Next

                Return contextVectors
            End Function

            Private Function GetTargetVector(targetIndex As Integer) As Double()
                Dim vector(embeddingSize - 1) As Double
                For i As Integer = 0 To embeddingSize - 1
                    vector(i) = embeddingMatrix(targetIndex, i)
                Next
                Return vector
            End Function

            Private Sub InitializeEmbeddings()
                Dim vocabSize As Integer = vocabulary.Count
                embeddingMatrix = New Double(vocabSize - 1, embeddingSize - 1) {}

                Dim random As New Random()
                For i As Integer = 0 To vocabSize - 1
                    For j As Integer = 0 To embeddingSize - 1
                        embeddingMatrix(i, j) = random.NextDouble()
                    Next
                Next
            End Sub

            Private Sub InitializeWeights()
                For i As Integer = 0 To weights.Length - 1
                    weights(i) = 1.0
                Next
            End Sub

            Private Function PredictCBOW(contextVectors As List(Of Double())) As Double()
                Dim contextSum As Double() = Enumerable.Repeat(0.0, embeddingSize).ToArray()

                For Each contextVector As Double() In contextVectors
                    For i As Integer = 0 To embeddingSize - 1
                        contextSum(i) += contextVector(i)
                    Next
                Next

                For i As Integer = 0 To embeddingSize - 1
                    contextSum(i) /= contextVectors.Count
                Next

                Return contextSum
            End Function

            Private Function PredictSkipGram(contextVector As Double()) As Double()
                Return contextVector
            End Function

            Private Sub TrainCBOW(contextIndices As List(Of Integer), targetIndex As Integer)
                Dim contextVectors As New List(Of Double())

                For Each contextIndex As Integer In contextIndices
                    Dim vector(embeddingSize - 1) As Double
                    For i As Integer = 1 To embeddingSize - 1
                        vector(i) = embeddingMatrix(contextIndex, i)
                    Next
                    contextVectors.Add(vector)
                Next

                Dim targetVector As Double() = GetTargetVector(targetIndex)
                Dim predictedVector As Double() = PredictCBOW(contextVectors)

                UpdateTargetVector(predictedVector, targetVector, contextVectors)
            End Sub

            Private Sub TrainSkipGram(targetIndex As Integer, contextIndices As List(Of Integer))
                Dim targetVector As Double() = GetTargetVector(targetIndex)

                For Each contextIndex As Integer In contextIndices
                    Dim contextVector As Double() = GetContextVector(contextIndex)

                    Dim predictedVector As Double() = PredictSkipGram(contextVector)

                    UpdateContextVector(predictedVector, contextVector, targetVector)
                Next
            End Sub

            Private Sub UpdateContextVector(predictedVector As Double(), contextVector As Double(), targetVector As Double())
                Dim errorGradient As Double() = Enumerable.Repeat(0.0, embeddingSize).ToArray()

                For i As Integer = 0 To embeddingSize - 1
                    errorGradient(i) = predictedVector(i) - targetVector(i)
                Next

                If contextVector.Length = embeddingSize Then
                    For i As Integer = 0 To embeddingSize - 1
                        Dim contextIndex As Integer = CInt(contextVector(i))
                        If contextIndex >= 0 AndAlso contextIndex < embeddingMatrix.GetLength(0) Then
                            embeddingMatrix(contextIndex, i) -= errorGradient(i) * learningRate
                        End If
                    Next
                End If
            End Sub

            Private Sub UpdateTargetVector(contextVectors As List(Of Double()), targetVector As Double())
                Dim contextSum As Double() = Enumerable.Repeat(0.0, embeddingSize).ToArray()

                For Each contextVector As Double() In contextVectors
                    For i As Integer = 0 To embeddingSize - 1
                        contextSum(i) += contextVector(i)
                    Next
                Next

                For i As Integer = 0 To embeddingSize - 1
                    targetVector(i) += (contextSum(i) / contextVectors.Count) * learningRate
                Next
            End Sub

            Private Sub UpdateTargetVector(predictedVector As Double(), targetVector As Double(), contextVectors As List(Of Double()))
                Dim errorGradient As Double() = Enumerable.Repeat(0.0, embeddingSize).ToArray()

                For i As Integer = 0 To embeddingSize - 1
                    errorGradient(i) = predictedVector(i) - targetVector(i)
                Next

                For Each contextVector As Double() In contextVectors
                    If contextVector.Length <> embeddingSize Then
                        Continue For ' Skip invalid context vectors
                    End If

                    For i As Integer = 0 To embeddingSize - 1
                        Dim contextIndex As Integer = CInt(contextVector(i))
                        If contextIndex >= 0 AndAlso contextIndex < embeddingMatrix.GetLength(0) Then
                            embeddingMatrix(contextIndex, i) -= errorGradient(i) * learningRate
                        End If
                    Next
                Next
            End Sub

        End Class
        ''' <summary>
        '''Skip-gram with Negative Sampling:
        ''' Pros:
        ''' More computationally efficient: Negative sampling reduces the computational cost by Using a small number Of negative samples For Each positive context pair during training.
        ''' Simpler to implement: It's relatively easier to implement skip-gram with negative sampling compared to hierarchical softmax.
        ''' Performs well With large vocabularies: Negative sampling Is well-suited For training word embeddings With large vocabularies As it scales well.
        ''' Cons:
        ''' May sacrifice quality: With negative sampling, some negative samples may Not be truly informative, potentially leading To a slight degradation In the quality Of learned word embeddings compared To hierarchical softmax.
        ''' Tuning hyperparameters: The effectiveness Of negative sampling depends On the selection Of the number Of negative samples And learning rate, which may require tuning. 
        ''' </summary>
        Public Class WordEmbeddingsWithNegativeSampling
            Inherits WordEmbeddingsModel
            Public NumNegativeSamples As Integer = 5 ' Number of negative samples per positive sample.

            Public Sub New(ByRef Vocabulary As List(Of String), Optional NumberOfNegativeSamples As Integer = 5)
                MyBase.New(Vocabulary)
                Me.NumNegativeSamples = NumberOfNegativeSamples
            End Sub
            Public Sub New(ByRef model As WordEmbeddingsModel)
                MyBase.New(model)
            End Sub
            Public Overrides Sub train()
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Simulate training data (context pairs).
                Dim trainingData As New List(Of (String, String))()
                For i As Integer = 0 To Vocabulary.Count - 1
                    For j As Integer = Math.Max(0, i - WindowSize) To Math.Min(Vocabulary.Count - 1, i + WindowSize)
                        If i <> j Then
                            trainingData.Add((Vocabulary(i), Vocabulary(j)))
                        End If
                    Next
                Next

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    trainingData = trainingData.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each context pair.
                    For Each item In trainingData
                        ' Generate negative samples.
                        Dim negativeSamples As New List(Of String)()
                        While negativeSamples.Count < NumNegativeSamples
                            Dim randomWord = Vocabulary(Rand.Next(Vocabulary.Count))
                            If randomWord <> item.Item1 AndAlso randomWord <> item.Item2 AndAlso Not negativeSamples.Contains(randomWord) Then
                                negativeSamples.Add(randomWord)
                            End If
                        End While

                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2, negativeSamples)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub

            Public Overrides Sub Train(corpus As List(Of List(Of String)))
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    Dim vector As Double() = Enumerable.Range(0, EmbeddingSize).Select(Function(i) Rand.NextDouble()).ToArray()
                    WordEmbeddings.Add(word, vector)
                Next

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    For Each document In corpus
                        For wordIndex As Integer = 0 To document.Count - 1
                            Dim targetWord As String = document(wordIndex)
                            Dim contextStart As Integer = Math.Max(0, wordIndex - WindowSize)
                            Dim contextEnd As Integer = Math.Min(document.Count - 1, wordIndex + WindowSize)

                            ' Skip-gram with negative sampling.
                            For contextIndex As Integer = contextStart To contextEnd
                                If contextIndex = wordIndex Then Continue For ' Skip the target word itself.

                                Dim contextWord As String = document(contextIndex)
                                Dim loss As Double = 0

                                ' Positive pair (target word, context word).
                                Dim targetVector As Double() = WordEmbeddings.GetVector(targetWord)
                                Dim contextVector As Double() = WordEmbeddings.GetVector(contextWord)
                                Dim dotProduct As Double = ComputeDotProduct(targetVector, contextVector)
                                Dim sigmoidDotProduct As Double = Sigmoid(dotProduct)
                                loss += -Math.Log(sigmoidDotProduct)

                                ' Negative sampling (sample k negative words).
                                Dim numNegativeSamples As Integer = 5
                                For i As Integer = 1 To numNegativeSamples
                                    Dim negativeWord As String = Vocabulary(Rand.Next(Vocabulary.Count))
                                    If negativeWord = targetWord OrElse negativeWord = contextWord Then Continue For ' Skip positive pairs.

                                    Dim negativeVector As Double() = WordEmbeddings.GetVector(negativeWord)
                                    Dim negDotProduct As Double = ComputeDotProduct(targetVector, negativeVector)
                                    Dim sigmoidNegDotProduct As Double = Sigmoid(negDotProduct)
                                    loss += -Math.Log(1 - sigmoidNegDotProduct)
                                Next

                                ' Update word vectors using gradient descent.
                                Dim learningRateFactor As Double = LearningRate * (1 - (epoch / NumEpochs)) ' Reduce learning rate over epochs.
                                Dim gradient As Double = sigmoidDotProduct - 1 ' Gradient for positive pair.
                                For i As Integer = 0 To EmbeddingSize - 1
                                    targetVector(i) -= learningRateFactor * gradient * contextVector(i)
                                    contextVector(i) -= learningRateFactor * gradient * targetVector(i)
                                Next

                                ' Update word vectors for negative samples.
                                For i As Integer = 1 To numNegativeSamples
                                    Dim negativeWord As String = Vocabulary(Rand.Next(Vocabulary.Count))
                                    If negativeWord = targetWord OrElse negativeWord = contextWord Then Continue For ' Skip positive pairs.

                                    Dim negativeVector As Double() = WordEmbeddings.GetVector(negativeWord)
                                    Dim negDotProduct As Double = ComputeDotProduct(targetVector, negativeVector)
                                    Dim sigmoidNegDotProduct As Double = Sigmoid(negDotProduct)
                                    Dim negGradient As Double = sigmoidNegDotProduct ' Gradient for negative pair.

                                    For j As Integer = 0 To EmbeddingSize - 1
                                        targetVector(j) -= learningRateFactor * negGradient * negativeVector(j)
                                        negativeVector(j) -= learningRateFactor * negGradient * targetVector(j)
                                    Next
                                Next

                                ' Update the embeddings for target and context words.
                                WordEmbeddings.Add(targetWord, targetVector)
                                WordEmbeddings.Add(contextWord, contextVector)
                            Next
                        Next
                    Next
                Next
            End Sub

            Private Sub Update(targetWord As String, contextWord As String, negativeSamples As List(Of String))
                Dim targetEmbedding = WordEmbeddings.GetVector(targetWord)
                Dim contextEmbedding = WordEmbeddings.GetVector(contextWord)

                Dim targetLoss As Double = 0
                Dim contextLoss As Double = 0

                ' Compute the loss for the positive context pair.
                Dim positiveScore As Double = ComputeDotProduct(targetEmbedding, contextEmbedding)
                Dim positiveSigmoid As Double = Sigmoid(positiveScore)
                targetLoss += -Math.Log(positiveSigmoid)
                contextLoss += -Math.Log(positiveSigmoid)

                ' Compute the loss for the negative samples.
                For Each negativeWord In negativeSamples
                    Dim negativeEmbedding = WordEmbeddings.GetVector(negativeWord)
                    Dim negativeScore As Double = ComputeDotProduct(targetEmbedding, negativeEmbedding)
                    Dim negativeSigmoid As Double = Sigmoid(negativeScore)
                    targetLoss += -Math.Log(1 - negativeSigmoid)
                Next

                ' Compute the gradients and update the word embeddings.
                Dim targetGradient = contextEmbedding.Clone()
                Dim contextGradient = targetEmbedding.Clone()

                targetGradient = targetGradient.Select(Function(g) g * (positiveSigmoid - 1)).ToArray()
                contextGradient = contextGradient.Select(Function(g) g * (positiveSigmoid - 1)).ToArray()

                For Each negativeWord In negativeSamples
                    Dim negativeEmbedding = WordEmbeddings.GetVector(negativeWord)
                    Dim negativeSigmoid As Double = Sigmoid(ComputeDotProduct(targetEmbedding, negativeEmbedding))

                    For i As Integer = 0 To EmbeddingSize - 1
                        targetGradient(i) += negativeSigmoid * negativeEmbedding(i)
                        negativeEmbedding(i) += negativeSigmoid * targetEmbedding(i)
                    Next
                Next

                ' Update the word embeddings using the computed gradients.
                For i As Integer = 0 To EmbeddingSize - 1
                    targetEmbedding(i) -= LearningRate * targetGradient(i)
                    contextEmbedding(i) -= LearningRate * contextGradient(i)
                Next
            End Sub



        End Class
        ''' <summary>
        '''Hierarchical Softmax
        ''' Pros:
        ''' Theoretically more accurate: Hierarchical softmax provides more accurate training by transforming the softmax operation into a binary tree-based probability calculation, ensuring that Each word Is considered during training.
        ''' Better performance With smaller datasets: Hierarchical softmax Is more suitable For smaller datasets, where negative sampling might Not perform As well due To a limited number Of contexts.
        ''' Cons:
        ''' Computationally expensive For large vocabularies: Hierarchical softmax can become computationally expensive With larger vocabularies, As it requires traversing a binary tree To compute probabilities For Each word during training.
        ''' More complex To implement: Implementing hierarchical softmax can be more complex compared To negative sampling.
        ''' </summary>
        Public Class WordEmbeddingsWithHierarchicalSoftmax
            Inherits WordEmbeddingsModel
            Public Sub New(ByRef model As WordEmbeddingsModel)
                MyBase.New(model)
            End Sub
            Public Sub New(ByRef Vocabulary As List(Of String))
                MyBase.New(Vocabulary)
            End Sub

            Public Overrides Sub Train()
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Simulate training data (context pairs).
                Dim trainingData As New List(Of (String, String))()
                For i As Integer = 0 To Vocabulary.Count - 1
                    For j As Integer = Math.Max(0, i - WindowSize) To Math.Min(Vocabulary.Count - 1, i + WindowSize)
                        If i <> j Then
                            trainingData.Add((Vocabulary(i), Vocabulary(j)))
                        End If
                    Next
                Next

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    trainingData = trainingData.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each context pair.
                    For Each item In trainingData
                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub
            Public Overrides Sub Train(corpus As List(Of List(Of String)))
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Build the hierarchical softmax binary tree.
                Dim rootNode As New Node(Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                For Each word In Vocabulary
                    Dim pathToWord As List(Of Node) = GetPathToWord(rootNode, word)
                    Dim currentNode As Node = rootNode
                    For Each node In pathToWord
                        If node Is Nothing Then
                            Dim newNode As New Node(Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                            newNode.Parent = currentNode
                            If currentNode.Left Is Nothing Then
                                currentNode.Left = newNode
                            Else
                                currentNode.Right = newNode
                            End If
                            currentNode = newNode
                        Else
                            currentNode = node
                        End If
                    Next
                    currentNode.Word = word
                Next

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    Dim trainingData As New List(Of (String, String))()
                    For Each document In corpus
                        For wordIndex As Integer = 0 To document.Count - 1
                            Dim targetWord As String = document(wordIndex)
                            Dim contextStart As Integer = Math.Max(0, wordIndex - WindowSize)
                            Dim contextEnd As Integer = Math.Min(document.Count - 1, wordIndex + WindowSize)

                            For contextIndex As Integer = contextStart To contextEnd
                                If contextIndex = wordIndex Then Continue For ' Skip the target word itself.

                                trainingData.Add((targetWord, document(contextIndex)))
                            Next
                        Next
                    Next

                    ' Shuffle the training data.
                    trainingData = trainingData.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each context pair.
                    For Each item In trainingData
                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2, rootNode)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub

            Private Sub Update(targetWord As String, contextWord As String, rootNode As Node)
                Dim targetEmbedding = WordEmbeddings.GetVector(targetWord)
                Dim contextEmbedding = WordEmbeddings.GetVector(contextWord)

                Dim pathToContext = GetPathToWord(rootNode, contextWord)

                Dim targetLoss As Double = 0
                Dim contextLoss As Double = 0

                ' Compute the loss for the positive context pair.
                Dim positiveScore As Double = 0
                For Each node In pathToContext
                    positiveScore += ComputeDotProduct(targetEmbedding, node.Vector)
                Next
                Dim positiveSigmoid As Double = Sigmoid(positiveScore)
                targetLoss += -Math.Log(positiveSigmoid)
                contextLoss += -Math.Log(positiveSigmoid)

                ' Update the gradients for the target word.
                For Each node In pathToContext
                    Dim sigmoidGradient As Double = (positiveSigmoid - 1.0) * LearningRate

                    For i As Integer = 0 To EmbeddingSize - 1
                        node.Vector(i) -= sigmoidGradient * targetEmbedding(i)
                        targetEmbedding(i) -= sigmoidGradient * node.Vector(i)
                    Next

                    ' Move to the parent node.
                    node = node.Parent
                Next

                ' Update the gradients for the context word.
                For Each node In pathToContext
                    Dim sigmoidGradient As Double = (positiveSigmoid - 1.0) * LearningRate

                    For i As Integer = 0 To EmbeddingSize - 1
                        node.Vector(i) -= sigmoidGradient * contextEmbedding(i)
                        contextEmbedding(i) -= sigmoidGradient * node.Vector(i)
                    Next
                Next
            End Sub

            Private Function GetPathToWord(startNode As Node, word As String) As List(Of Node)
                Dim path As New List(Of Node)()
                Dim currentNode As Node = startNode

                While currentNode IsNot Nothing
                    path.Add(currentNode)
                    If currentNode.Word = word Then
                        Exit While
                    ElseIf WordEmbeddings.GetVector(word)(0) < 0 Then
                        currentNode = currentNode.Left
                    Else
                        currentNode = currentNode.Right
                    End If
                End While

                Return path
            End Function

            Private Sub Update(targetWord As String, contextWord As String)
                Dim targetEmbedding = WordEmbeddings.GetVector(targetWord)
                Dim contextEmbedding = WordEmbeddings.GetVector(contextWord)

                Dim pathToTarget = GetPathToWord(targetWord)
                Dim pathToContext = GetPathToWord(contextWord)

                Dim targetLoss As Double = 0
                Dim contextLoss As Double = 0

                ' Compute the loss for the positive context pair.
                Dim positiveScore As Double = 0
                For Each node In pathToContext
                    positiveScore += ComputeDotProduct(targetEmbedding, node.Vector)
                Next
                Dim positiveSigmoid As Double = Sigmoid(positiveScore)
                targetLoss += -Math.Log(positiveSigmoid)
                contextLoss += -Math.Log(positiveSigmoid)

                ' Update the gradients for the target word.
                For Each node In pathToTarget
                    Dim sigmoidGradient As Double = (positiveSigmoid - 1.0) * LearningRate

                    For i As Integer = 0 To EmbeddingSize - 1
                        node.Vector(i) -= sigmoidGradient * contextEmbedding(i)
                        contextEmbedding(i) -= sigmoidGradient * node.Vector(i)
                    Next

                    ' Move to the parent node.
                    node = node.Parent
                Next

                ' Update the gradients for the context word.
                For Each node In pathToContext
                    Dim sigmoidGradient As Double = (positiveSigmoid - 1.0) * LearningRate

                    For i As Integer = 0 To EmbeddingSize - 1
                        node.Vector(i) -= sigmoidGradient * targetEmbedding(i)
                        targetEmbedding(i) -= sigmoidGradient * node.Vector(i)
                    Next
                Next
            End Sub

            Private Function GetPathToWord(word As String) As List(Of Node)
                Dim path As New List(Of Node)()
                Dim currentNode As Node = New Node(WordEmbeddings.GetVector(word))

                While currentNode IsNot Nothing
                    path.Add(currentNode)
                    currentNode = currentNode.Parent
                End While

                Return path
            End Function



            ' Helper class to represent nodes in the hierarchical softmax binary tree.
            Private Class Node
                Public Property Vector As Double()
                Public Property Left As Node
                Public Property Right As Node
                Public Property Parent As Node
                Public Property Word As String
                Public Sub New(vector As Double())
                    Me.Vector = vector
                End Sub
            End Class

        End Class
        Public Class WordEmbeddingsWithGloVe
            Inherits WordEmbeddingsModel

            Public Sub New(ByRef model As WordEmbeddingsModel)
                MyBase.New(model)
            End Sub

            Public Sub New(ByRef Vocabulary As List(Of String))
                MyBase.New(Vocabulary)
            End Sub

            Public Overrides Sub Train()
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Construct the global word co-occurrence matrix.
                Dim coOccurrenceMatrix = BuildCoOccurrenceMatrix()

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    coOccurrenceMatrix = coOccurrenceMatrix.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each word pair in the co-occurrence matrix.
                    For Each item In coOccurrenceMatrix
                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2, item.Item3)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub
            Public Overrides Sub Train(corpus As List(Of List(Of String)))
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Construct the global word co-occurrence matrix.
                Dim coOccurrenceMatrix = BuildCoOccurrenceMatrix(corpus)

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    coOccurrenceMatrix = coOccurrenceMatrix.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each word pair in the co-occurrence matrix.
                    For Each item In coOccurrenceMatrix
                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2, item.Item3)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub

            Private Function BuildCoOccurrenceMatrix(corpus As List(Of List(Of String))) As List(Of (String, String, Double))
                ' Construct a global word co-occurrence matrix.
                Dim coOccurrenceMatrix As New List(Of (String, String, Double))()

                ' Simulate training data (context pairs).
                For Each document In corpus
                    For wordIndex As Integer = 0 To document.Count - 1
                        Dim targetWord As String = document(wordIndex)
                        Dim contextStart As Integer = Math.Max(0, wordIndex - WindowSize)
                        Dim contextEnd As Integer = Math.Min(document.Count - 1, wordIndex + WindowSize)

                        For contextIndex As Integer = contextStart To contextEnd
                            If contextIndex = wordIndex Then Continue For ' Skip the target word itself.

                            ' Increment the co-occurrence count for the word pair (targetWord, contextWord).
                            Dim coOccurrenceCount = 1.0 / (Math.Abs(contextIndex - wordIndex))
                            coOccurrenceMatrix.Add((targetWord, document(contextIndex), coOccurrenceCount))
                        Next
                    Next
                Next

                Return coOccurrenceMatrix
            End Function
            Private Function BuildCoOccurrenceMatrix() As List(Of (String, String, Double))
                ' Construct a global word co-occurrence matrix.
                Dim coOccurrenceMatrix As New List(Of (String, String, Double))()

                ' Simulate training data (context pairs).
                For i As Integer = 0 To Vocabulary.Count - 1
                    For j As Integer = Math.Max(0, i - WindowSize) To Math.Min(Vocabulary.Count - 1, i + WindowSize)
                        If i <> j Then
                            ' Increment the co-occurrence count for the word pair (Vocabulary(i), Vocabulary(j)).
                            Dim coOccurrenceCount = 1.0 / (Math.Abs(i - j))
                            coOccurrenceMatrix.Add((Vocabulary(i), Vocabulary(j), coOccurrenceCount))
                        End If
                    Next
                Next

                Return coOccurrenceMatrix
            End Function

            Private Sub Update(word1 As String, word2 As String, coOccurrenceCount As Double)
                Dim vector1 = WordEmbeddings.GetVector(word1)
                Dim vector2 = WordEmbeddings.GetVector(word2)

                Dim dotProduct As Double = ComputeDotProduct(vector1, vector2)
                Dim loss As Double = (dotProduct - Math.Log(coOccurrenceCount)) ^ 2

                Dim gradient1 = New Double(EmbeddingSize - 1) {}
                Dim gradient2 = New Double(EmbeddingSize - 1) {}

                For i As Integer = 0 To EmbeddingSize - 1
                    gradient1(i) = 2.0 * (dotProduct - Math.Log(coOccurrenceCount)) * vector2(i)
                    gradient2(i) = 2.0 * (dotProduct - Math.Log(coOccurrenceCount)) * vector1(i)
                Next

                ' Update the word embeddings using the computed gradients.
                For i As Integer = 0 To EmbeddingSize - 1
                    vector1(i) -= LearningRate * gradient1(i)
                    vector2(i) -= LearningRate * gradient2(i)
                Next
            End Sub

        End Class
        Public Class WordEmbeddingsWithFastText
            Inherits WordEmbeddingsModel

            Public Sub New(ByRef model As WordEmbeddingsModel)
                MyBase.New(model)
            End Sub

            Public Sub New(ByRef vocabulary As List(Of String))
                MyBase.New(vocabulary)
            End Sub

            Public Overrides Sub Train()
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Simulate training data (context pairs).
                Dim trainingData As New List(Of (String, String))()
                For i As Integer = 0 To Vocabulary.Count - 1
                    For j As Integer = Math.Max(0, i - WindowSize) To Math.Min(Vocabulary.Count - 1, i + WindowSize)
                        If i <> j Then
                            trainingData.Add((Vocabulary(i), Vocabulary(j)))
                        End If
                    Next
                Next

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    trainingData = trainingData.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each context pair.
                    For Each item In trainingData
                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub
            Public Overrides Sub Train(corpus As List(Of List(Of String)))
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Simulate training data (context pairs).
                Dim trainingData As New List(Of (String, String))()
                For i As Integer = 0 To Vocabulary.Count - 1
                    For j As Integer = Math.Max(0, i - WindowSize) To Math.Min(Vocabulary.Count - 1, i + WindowSize)
                        If i <> j Then
                            trainingData.Add((Vocabulary(i), Vocabulary(j)))
                        End If
                    Next
                Next

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    trainingData = trainingData.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each context pair.
                    For Each item In trainingData
                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub

            Private Sub Update(targetWord As String, contextWord As String)
                Dim targetEmbedding = WordEmbeddings.GetVector(targetWord)
                Dim contextEmbedding = WordEmbeddings.GetVector(contextWord)

                Dim targetLoss As Double = 0
                Dim contextLoss As Double = 0

                ' Compute the loss for the positive context pair.
                Dim positiveScore As Double = ComputeDotProduct(targetEmbedding, contextEmbedding)
                Dim positiveSigmoid As Double = Sigmoid(positiveScore)
                targetLoss += -Math.Log(positiveSigmoid)
                contextLoss += -Math.Log(positiveSigmoid)

                ' Compute the gradients and update the word embeddings.
                Dim targetGradient = contextEmbedding.Clone()
                Dim contextGradient = targetEmbedding.Clone()

                targetGradient = targetGradient.Select(Function(g) g * (positiveSigmoid - 1)).ToArray()
                contextGradient = contextGradient.Select(Function(g) g * (positiveSigmoid - 1)).ToArray()

                ' Update the word embeddings using the computed gradients.
                For i As Integer = 0 To EmbeddingSize - 1
                    targetEmbedding(i) -= LearningRate * targetGradient(i)
                    contextEmbedding(i) -= LearningRate * contextGradient(i)
                Next
            End Sub
        End Class
        Public Class WordEmbeddingsWithCBOW
            Inherits WordEmbeddingsModel

            Public Sub New(ByRef model As WordEmbeddingsModel)
                MyBase.New(model)
            End Sub

            Public Sub New(ByRef Vocabulary As List(Of String))
                MyBase.New(Vocabulary)
            End Sub

            Public Overrides Sub Train()
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Simulate training data (context pairs).
                Dim trainingData As New List(Of (List(Of String), String))()
                For i As Integer = 0 To Vocabulary.Count - 1
                    Dim contextWords As New List(Of String)()
                    For j As Integer = Math.Max(0, i - WindowSize) To Math.Min(Vocabulary.Count - 1, i + WindowSize)
                        If i <> j Then
                            contextWords.Add(Vocabulary(j))
                        End If
                    Next
                    If contextWords.Count > 0 Then
                        trainingData.Add((contextWords, Vocabulary(i)))
                    End If
                Next

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    trainingData = trainingData.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each context pair.
                    For Each item In trainingData
                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub

            Public Overrides Sub Train(corpus As List(Of List(Of String)))
                ' Initialize word embeddings randomly.
                For Each word In Vocabulary
                    WordEmbeddings.Add(word, Enumerable.Range(0, EmbeddingSize).Select(Function(_i) Rand.NextDouble() - 0.5).ToArray())
                Next

                ' Simulate training data (context pairs).
                Dim trainingData As New List(Of (List(Of String), String))()
                For Each document In corpus
                    For wordIndex As Integer = 0 To document.Count - 1
                        Dim targetWord As String = document(wordIndex)
                        Dim contextStart As Integer = Math.Max(0, wordIndex - WindowSize)
                        Dim contextEnd As Integer = Math.Min(document.Count - 1, wordIndex + WindowSize)

                        Dim contextWords As New List(Of String)()
                        For contextIndex As Integer = contextStart To contextEnd
                            If contextIndex <> wordIndex Then
                                contextWords.Add(document(contextIndex))
                            End If
                        Next

                        trainingData.Add((contextWords, targetWord))
                    Next
                Next

                ' Training loop.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    trainingData = trainingData.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each context pair.
                    For Each item In trainingData
                        ' Compute the gradients and update the word embeddings.
                        Update(item.Item1, item.Item2)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary.
            End Sub

            Private Sub Update(contextWords As List(Of String), targetWord As String)
                Dim contextEmbeddings = contextWords.Select(Function(word) WordEmbeddings.GetVector(word)).ToList()
                Dim targetEmbedding = WordEmbeddings.GetVector(targetWord)

                ' Average the context embeddings.
                Dim averageContext = New Double(EmbeddingSize - 1) {}
                For Each context In contextEmbeddings
                    For i As Integer = 0 To EmbeddingSize - 1
                        averageContext(i) += context(i)
                    Next
                Next

                For i As Integer = 0 To EmbeddingSize - 1
                    averageContext(i) /= contextEmbeddings.Count
                Next

                ' Compute the loss for the target word.
                Dim targetLoss As Double = 0
                Dim positiveScore As Double = ComputeDotProduct(targetEmbedding, averageContext)
                Dim positiveSigmoid As Double = Sigmoid(positiveScore)
                targetLoss += -Math.Log(positiveSigmoid)

                ' Compute the gradient and update the word embeddings.
                Dim targetGradient = averageContext.Select(Function(g) g * (positiveSigmoid - 1)).ToArray()

                For Each context In contextEmbeddings
                    Dim sigmoidGradient As Double = (positiveSigmoid - 1.0) * LearningRate

                    For i As Integer = 0 To EmbeddingSize - 1
                        context(i) -= sigmoidGradient * targetEmbedding(i)
                        targetGradient(i) += sigmoidGradient * context(i)
                    Next
                Next

                ' Update the word embeddings using the computed gradients.
                For i As Integer = 0 To EmbeddingSize - 1
                    targetEmbedding(i) -= LearningRate * targetGradient(i)
                Next
            End Sub
        End Class
        Public Class WordEmbeddingWithTemplate
            Inherits WordEmbeddingsModel

            Public Sub New(ByRef model As WordEmbeddingsModel)
                MyBase.New(model)
            End Sub

            Public Sub New(ByRef Vocabulary As List(Of String))
                MyBase.New(Vocabulary)
            End Sub

            Public Overrides Sub Train()
                Throw New NotImplementedException()
            End Sub

            Public Overrides Sub Train(corpus As List(Of List(Of String)))
                Throw New NotImplementedException()
            End Sub
        End Class
        Public Class WordEmbeddingWithSentiment
            Inherits WordEmbeddingsModel

            Private SentimentDictionary As New Dictionary(Of String, SentimentLabel)
            Private Shared PositiveSentiments As New List(Of String)
            Private Shared NegativeSentiments As New List(Of String)
            Private SentimentEmbeddings As New Dictionary(Of String, Double())

            Private Enum SentimentLabel
                Neutral
                Positive
                Negative
            End Enum

            ' WordSentiment class to store sentiment information for words.
            Private Class WordSentiment
                Public Property Word As String
                Public Property Sentiment As SentimentLabel
            End Class

            Private Sub InitializeVocab()
                ' Initialize sentiment dictionary with neutral sentiment for all words in the vocabulary.
                SentimentDictionary = New Dictionary(Of String, SentimentLabel)
                For Each item In WordEmbeddings.embeddings
                    SentimentDictionary.Add(item.Key, GetSentiment(item.Key))
                Next
            End Sub

            Public Sub New(ByRef model As WordEmbeddingsModel)
                MyBase.New(model)
            End Sub

            Public Function GetLabel(ByRef Word As String) As String
                Select Case GetSentiment(Word)
                    Case SentimentLabel.Negative
                        Return "Negative"
                    Case SentimentLabel.Positive
                        Return "Positive"
                    Case SentimentLabel.Neutral
                        Return "Neutral"
                End Select
                Return "Unknown"
            End Function

            Public Sub New(ByRef Vocabulary As List(Of String))
                MyBase.New(Vocabulary)
            End Sub

            Public Sub Initialize()
                LoadLists()
                InitializeVocab()
            End Sub

            ''' <summary>
            ''' Encode Sentiment into the Embedding model (does not affect the positions or order of the model)
            ''' </summary>
            Public Overrides Sub Train()
                Initialize()
                CreateSentimentEmbeddings()
                UpdateWordEmbeddings()
            End Sub
            Public Overrides Sub Train(corpus As List(Of List(Of String)))
                Initialize()
                CreateSentimentEmbeddings()
                UpdateWordEmbeddings()

                ' Now you can continue training the word embeddings using the corpus data.
                ' You can add the training loop here based on your specific model requirements.
                ' The previous part of the code initializes the sentiment embeddings, which can be used
                ' in combination with the word embeddings during the training process.
                ' You can perform additional iterations or training steps to fine-tune the embeddings
                ' based on your specific training algorithm.

                ' Training loop on the corpus data.
                For epoch As Integer = 1 To NumEpochs
                    Console.WriteLine($"Training Epoch {epoch}/{NumEpochs}")

                    ' Shuffle the training data to avoid learning order biases.
                    ' Modify the training data to suit your specific corpus structure and context pairs.
                    Dim trainingData As New List(Of (List(Of String), String))()
                    For Each document In corpus
                        For wordIndex As Integer = 0 To document.Count - 1
                            Dim contextWords As New List(Of String)()
                            Dim targetWord As String = document(wordIndex)
                            Dim contextStart As Integer = Math.Max(0, wordIndex - WindowSize)
                            Dim contextEnd As Integer = Math.Min(document.Count - 1, wordIndex + WindowSize)

                            For contextIndex As Integer = contextStart To contextEnd
                                If contextIndex = wordIndex Then Continue For ' Skip the target word itself.
                                contextWords.Add(document(contextIndex))
                            Next

                            If contextWords.Count > 0 Then
                                trainingData.Add((contextWords, targetWord))
                            End If
                        Next
                    Next

                    ' Shuffle the training data.
                    trainingData = trainingData.OrderBy(Function(_item) Rand.Next()).ToList()

                    ' Gradient descent for each context pair.
                    For Each item In trainingData
                        ' Compute the gradients and update the word embeddings.
                        UpdateWordEmbeddings(item.Item1, item.Item2)
                    Next
                Next

                ' Print the learned word embeddings.
                For Each word In Vocabulary
                    Console.WriteLine($"{word}: {String.Join(", ", WordEmbeddings.GetVector(word))}")
                Next

                ' Now you have learned word embeddings for the given vocabulary and trained them on the corpus data.
            End Sub

            Private Sub UpdateWordEmbeddings(contextWords As List(Of String), targetWord As String)
                Dim contextEmbeddings = contextWords.Select(Function(word) WordEmbeddings.GetVector(word)).ToList()
                Dim targetEmbedding = WordEmbeddings.GetVector(targetWord)

                ' Average the context embeddings.
                Dim averageContext = New Double(EmbeddingSize - 1) {}
                For Each context In contextEmbeddings
                    For i As Integer = 0 To EmbeddingSize - 1
                        averageContext(i) += context(i)
                    Next
                Next

                For i As Integer = 0 To EmbeddingSize - 1
                    averageContext(i) /= contextEmbeddings.Count
                Next

                ' Compute the loss for the target word.
                Dim targetLoss As Double = 0
                Dim positiveScore As Double = ComputeDotProduct(targetEmbedding, averageContext)
                Dim positiveSigmoid As Double = Sigmoid(positiveScore)
                targetLoss += -Math.Log(positiveSigmoid)

                ' Compute the gradient and update the word embeddings.
                Dim targetGradient = averageContext.Select(Function(g) g * (positiveSigmoid - 1)).ToArray()

                For Each context In contextEmbeddings
                    Dim sigmoidGradient As Double = (positiveSigmoid - 1.0) * LearningRate

                    For i As Integer = 0 To EmbeddingSize - 1
                        context(i) -= sigmoidGradient * targetEmbedding(i)
                        targetGradient(i) += sigmoidGradient * context(i)
                    Next
                Next

                ' Update the word embeddings using the computed gradients.
                For i As Integer = 0 To EmbeddingSize - 1
                    targetEmbedding(i) -= LearningRate * targetGradient(i)
                Next
            End Sub

            Public Shared Function CombineVectors(vector1 As Double(), vector2 As Double()) As Double()
                ' Combine two vectors element-wise
                Dim combinedVector As Double() = New Double(vector1.Length - 1) {}
                For i As Integer = 0 To vector1.Length - 1
                    combinedVector(i) = vector1(i) + vector2(i)
                Next
                Return combinedVector
            End Function

            Private Sub UpdateWordEmbeddings()
                Dim CurrentEmbeddings = WordEmbeddings.embeddings
                Dim NewEmbeddings As New Dictionary(Of String, Double())

                For Each item In SentimentEmbeddings
                    Dim NewVector = CombineVectors(item.Value, WordEmbeddings.GetVector(item.Key))
                    NewEmbeddings.Add(item.Key, NewVector)
                Next
                WordEmbeddings.embeddings = NewEmbeddings
            End Sub

            Private Shared Function GetSentiment(ByRef Word As String) As SentimentLabel
                For Each item In PositiveSentiments
                    If item = Word Then
                        Return SentimentLabel.Positive
                    End If
                Next

                For Each item In NegativeSentiments
                    If item = Word Then
                        Return SentimentLabel.Negative
                    End If
                Next

                Return SentimentLabel.Neutral
            End Function

            Private Function OneShotEncode(ByVal label As SentimentLabel) As Double()
                ' One-shot encode the sentiment label into a binary vector
                ' In this example, we'll represent the label with a 3-bit binary code
                Dim encodedVector As Double() = New Double(2) {} ' 3-bit binary code (0, 0, 0)

                Select Case label
                    Case SentimentLabel.Positive
                        encodedVector(0) = 1 ' 3-bit binary code (1, 0, 0)
                    Case SentimentLabel.Negative
                        encodedVector(1) = 1 ' 3-bit binary code (0, 1, 0)
                    Case SentimentLabel.Neutral
                        encodedVector(2) = 1 ' 3-bit binary code (0, 0, 1)
                End Select

                Return encodedVector
            End Function

            Private Sub CreateSentimentEmbeddings()
                For Each item In SentimentDictionary
                    SentimentEmbeddings.Add(item.Key, OneShotEncode(item.Value))
                Next
            End Sub

            Private Sub LoadLists()
                PositiveSentiments = LoadList("PositiveSent.txt")
                NegativeSentiments = LoadList("NegativeSent.txt")
            End Sub

            Private Function LoadList(ByRef FileName As String) As List(Of String)
                Dim corpusRoot As String = Application.StartupPath & "\data\"
                Dim wordlistPath As String = Path.Combine(corpusRoot, FileName)
                Dim wordlistReader As New WordListReader(wordlistPath)
                Dim Lst = wordlistReader.GetWords()
                Return Lst
            End Function

        End Class

        Public Class WordEmbeddingWithTfIdf
            Inherits WordEmbeddingsModel

            Public Sub New(ByRef model As WordEmbeddingsModel)
                MyBase.New(model)
            End Sub

            Public Sub New(ByRef Vocabulary As List(Of String))
                MyBase.New(Vocabulary)
            End Sub
            Public Overrides Sub Train(corpus As List(Of List(Of String)))
                ' Assuming you have pre-trained word embeddings stored in the 'WordEmbeddings.embeddings' variable.

                ' Step 1: Calculate term frequency (TF) for each term in the vocabulary.
                Dim termFrequency As New Dictionary(Of String, Integer)()
                For Each sentence In corpus
                    For Each term In sentence
                        If termFrequency.ContainsKey(term) Then
                            termFrequency(term) += 1
                        Else
                            termFrequency(term) = 1
                        End If
                    Next
                Next

                ' Step 2: Sort the vocabulary based on term frequency in descending order (highest ranked terms first).
                Vocabulary = termFrequency.OrderByDescending(Function(entry) entry.Value).Select(Function(entry) entry.Key).ToList()

                ' Step 3: Create a SentenceVectorizer using the sorted vocabulary to calculate TF-IDF scores.
                Dim sentenceVectorizer As New SentenceVectorizer(corpus)

                ' Step 4: Calculate TF-IDF vectors for each term in the sorted vocabulary.
                Dim tfidfWeightedEmbeddings As New Dictionary(Of String, List(Of Double))
                For Each term In Vocabulary
                    Dim tfidfVector As List(Of Double) = sentenceVectorizer.Vectorize(New List(Of String) From {term})
                    Dim wordEmbedding() As Double = WordEmbeddings.embeddings(term)

                    ' Multiply the word embedding by the corresponding TF-IDF score to get the weighted word embedding.
                    Dim weightedEmbedding As List(Of Double) = wordEmbedding.Select(Function(val, idx) val * tfidfVector(idx)).ToList()

                    ' Store the weighted embedding in the dictionary.
                    tfidfWeightedEmbeddings(term) = weightedEmbedding
                Next

                ' Step 5: Store the TF-IDF weighted word embeddings in the WordEmbedding class.
                For Each term In tfidfWeightedEmbeddings.Keys
                    WordEmbeddings.Add(term, tfidfWeightedEmbeddings(term).ToArray())
                Next
            End Sub


            Public Overrides Sub Train()
                ' Assuming you have pre-trained word embeddings stored in the 'WordEmbeddings.embeddings' variable.

                ' Step 1: Calculate term frequency (TF) for each term in the vocabulary.
                Dim termFrequency As New Dictionary(Of String, Integer)()
                For Each term In Vocabulary
                    termFrequency(term) = 0
                Next

                ' Count the occurrences of each term in the vocabulary.
                For Each sentence In Vocabulary
                    For Each term In sentence
                        If termFrequency.ContainsKey(term) Then
                            termFrequency(term) += 1
                        End If
                    Next
                Next

                ' Step 2: Sort the vocabulary based on term frequency in descending order (highest ranked terms first).
                Vocabulary = Vocabulary.OrderByDescending(Function(term) termFrequency(term)).ToList()

                ' Step 3: Create a SentenceVectorizer using the sorted vocabulary to calculate TF-IDF scores.
                Dim sentenceVectorizer As New SentenceVectorizer(New List(Of List(Of String)) From {Vocabulary})

                ' Step 4: Calculate TF-IDF vectors for each term in the sorted vocabulary.
                Dim tfidfWeightedEmbeddings As New Dictionary(Of String, List(Of Double))
                For Each term In Vocabulary
                    Dim tfidfVector As List(Of Double) = sentenceVectorizer.Vectorize(New List(Of String) From {term})
                    Dim wordEmbedding() As Double = WordEmbeddings.embeddings(term)

                    ' Multiply the word embedding by the corresponding TF-IDF score to get the weighted word embedding.
                    Dim weightedEmbedding As List(Of Double) = wordEmbedding.Select(Function(val, idx) val * tfidfVector(idx)).ToList()

                    ' Store the weighted embedding in the dictionary.
                    tfidfWeightedEmbeddings(term) = weightedEmbedding
                Next

                ' Step 5: Store the TF-IDF weighted word embeddings in the WordEmbedding class.
                For Each term In tfidfWeightedEmbeddings.Keys
                    WordEmbeddings.Add(term, tfidfWeightedEmbeddings(term).ToArray())
                Next
            End Sub

            ''' <summary>
            ''' This is a TFIDF Vectorizer For basic Embeddings
            ''' </summary>
            Public Class SentenceVectorizer
                Private ReadOnly documents As List(Of List(Of String))
                Private ReadOnly idf As Dictionary(Of String, Double)

                Public Sub New(documents As List(Of List(Of String)))
                    Me.documents = documents
                    Me.idf = CalculateIDF(documents)
                End Sub

                Public Sub New()
                    documents = New List(Of List(Of String))
                    idf = New Dictionary(Of String, Double)
                End Sub

                Public Function Vectorize(sentence As List(Of String)) As List(Of Double)
                    Dim termFrequency = CalculateTermFrequency(sentence)
                    Dim vector As New List(Of Double)

                    For Each term In idf.Keys
                        Dim tfidf As Double = termFrequency(term) * idf(term)
                        vector.Add(tfidf)
                    Next

                    Return vector
                End Function

                Public Function CalculateIDF(documents As List(Of List(Of String))) As Dictionary(Of String, Double)
                    Dim idf As New Dictionary(Of String, Double)
                    Dim totalDocuments As Integer = documents.Count

                    For Each document In documents
                        Dim uniqueTerms As List(Of String) = document.Distinct().ToList()

                        For Each term In uniqueTerms
                            If idf.ContainsKey(term) Then
                                idf(term) += 1
                            Else
                                idf(term) = 1
                            End If
                        Next
                    Next

                    For Each term In idf.Keys
                        idf(term) = Math.Log(totalDocuments / idf(term))
                    Next

                    Return idf
                End Function

                Public Function CalculateTermFrequency(sentence As List(Of String)) As Dictionary(Of String, Double)
                    Dim termFrequency As New Dictionary(Of String, Double)

                    For Each term In sentence
                        If termFrequency.ContainsKey(term) Then
                            termFrequency(term) += 1
                        Else
                            termFrequency(term) = 1
                        End If
                    Next

                    Return termFrequency
                End Function

            End Class

        End Class

    End Namespace
    Namespace Audio
        Public Class Audio2Vector
            Public Shared Function AudioToVector(audioSignal As Double(), windowSize As Integer, hopSize As Integer) As List(Of Complex())
                Dim vectors As New List(Of Complex())

                For i As Integer = 0 To audioSignal.Length - windowSize Step hopSize
                    Dim window(windowSize - 1) As Complex

                    For j As Integer = 0 To windowSize - 1
                        window(j) = audioSignal(i + j)
                    Next

                    Dim spectrum As Complex() = CalculateSpectrum(window)
                    vectors.Add(spectrum)
                Next

                Return vectors
            End Function
            Public Shared Function VectorToAudio(vectors As List(Of Complex()), hopSize As Integer) As Double()
                Dim audioSignal As New List(Of Double)()

                For Each spectrum As Complex() In vectors
                    Dim windowSignal As Double() = CalculateInverseSpectrum(spectrum)

                    For i As Integer = 0 To hopSize - 1
                        If audioSignal.Count + i < windowSignal.Length Then
                            audioSignal.Add(windowSignal(i))
                        End If
                    Next
                Next

                Return audioSignal.ToArray()
            End Function
            Public Shared Function VectorToAudio(vectors As List(Of Complex()), windowSize As Integer, hopSize As Integer) As Double()
                Dim audioSignal As New List(Of Double)()

                ' Initialize a buffer to accumulate audio segments
                Dim buffer(audioSignal.Count + windowSize - 1) As Double

                For Each spectrum As Complex() In vectors
                    Dim windowSignal As Double() = CalculateInverseSpectrum(spectrum)

                    For i As Integer = 0 To hopSize - 1
                        If audioSignal.Count + i < windowSignal.Length Then
                            ' Add the current window to the buffer
                            buffer(audioSignal.Count + i) += windowSignal(i)
                        End If
                    Next

                    ' Check if there's enough data in the buffer to generate a full segment
                    While buffer.Length >= hopSize
                        ' Extract a segment from the buffer and add to the audio signal
                        Dim segment(hopSize - 1) As Double
                        Array.Copy(buffer, segment, hopSize)
                        audioSignal.AddRange(segment)

                        ' Shift the buffer by hopSize
                        Dim newBuffer(buffer.Length - hopSize - 1) As Double
                        Array.Copy(buffer, hopSize, newBuffer, 0, newBuffer.Length)
                        buffer = newBuffer
                    End While
                Next

                ' Convert the remaining buffer to audio
                audioSignal.AddRange(buffer)

                Return audioSignal.ToArray()
            End Function
            Public Shared Function LoadAudio(audioPath As String) As Double()
                Dim audioData As New List(Of Double)()

                ' Use NAudio or other library to load audio data from the specified audioPath
                Using reader As New NAudio.Wave.AudioFileReader(audioPath)
                    Dim buffer(reader.WaveFormat.SampleRate * reader.WaveFormat.Channels - 1) As Single
                    While reader.Read(buffer, 0, buffer.Length) > 0
                        For Each sample As Single In buffer
                            audioData.Add(CDbl(sample))
                        Next
                    End While
                End Using

                Return audioData.ToArray()
            End Function
            Public Shared Sub SaveAudio(audioSignal As Double(), outputPath As String)
                ' Convert the array of doubles to an array of singles for NAudio
                Dim audioData(audioSignal.Length - 1) As Single
                For i As Integer = 0 To audioSignal.Length - 1
                    audioData(i) = CSng(audioSignal(i))
                Next

                ' Use NAudio or other library to save audio data to the specified outputPath
                Using writer As New NAudio.Wave.WaveFileWriter(outputPath, New NAudio.Wave.WaveFormat())
                    writer.WriteSamples(audioData, 0, audioData.Length)
                End Using
            End Sub
            Private Shared Function CalculateInverseSpectrum(spectrum As Complex()) As Double()
                ' Perform inverse FFT using MathNet.Numerics library
                Fourier.Inverse(spectrum, FourierOptions.Default)

                ' Return the real part of the inverse spectrum as the reconstructed signal
                Dim reconstructedSignal(spectrum.Length - 1) As Double
                For i As Integer = 0 To spectrum.Length - 1
                    reconstructedSignal(i) = spectrum(i).Real
                Next
                Return reconstructedSignal
            End Function
            Private Shared Function CalculateSpectrum(window As Complex()) As Complex()
                ' Perform FFT using MathNet.Numerics library
                Fourier.Forward(window, FourierOptions.Default)

                Return window
            End Function
        End Class
    End Namespace
    Namespace Images
        Public Class Image2Vector
            Public Shared Sub SaveVectorToFile(imgVector As Double(), outputPath As String)
                Using writer As New System.IO.StreamWriter(outputPath)
                    For Each value As Double In imgVector
                        writer.WriteLine(value)
                    Next
                End Using
            End Sub

            Public Class ImageDecoder
                Public Sub DecodeImage(imgVector As Double(), width As Integer, height As Integer, outputPath As String)
                    Dim decodedImage As New Bitmap(width, height)

                    Dim index As Integer = 0
                    For y As Integer = 0 To height - 1
                        For x As Integer = 0 To width - 1
                            Dim grayscaleValue As Integer = CInt(Math.Floor(imgVector(index) * 255))
                            Dim pixelColor As Color = Color.FromArgb(grayscaleValue, grayscaleValue, grayscaleValue)
                            decodedImage.SetPixel(x, y, pixelColor)
                            index += 1
                        Next
                    Next

                    decodedImage.Save(outputPath, Imaging.ImageFormat.Jpeg)
                End Sub
            End Class

            Public Class ImageEncoder
                Public Function EncodeImage(imagePath As String, width As Integer, height As Integer) As Double()
                    Dim resizedImage As Bitmap = ResizeImage(imagePath, width, height)
                    Dim grayscaleImage As Bitmap = ConvertToGrayscale(resizedImage)
                    Dim pixelValues As Double() = GetPixelValues(grayscaleImage)
                    Return pixelValues
                End Function

                Private Function ConvertToGrayscale(image As Bitmap) As Bitmap
                    Dim grayscaleImage As New Bitmap(image.Width, image.Height, PixelFormat.Format8bppIndexed)
                    Using g As Graphics = Graphics.FromImage(grayscaleImage)
                        Dim colorMatrix As ColorMatrix = New ColorMatrix(New Single()() {
                New Single() {0.299F, 0.299F, 0.299F, 0, 0},
                New Single() {0.587F, 0.587F, 0.587F, 0, 0},
                New Single() {0.114F, 0.114F, 0.114F, 0, 0},
                New Single() {0, 0, 0, 1, 0},
                New Single() {0, 0, 0, 0, 1}
            })
                        Dim attributes As ImageAttributes = New ImageAttributes()
                        attributes.SetColorMatrix(colorMatrix)
                        g.DrawImage(image, New Rectangle(0, 0, image.Width, image.Height), 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, attributes)
                    End Using
                    Return grayscaleImage
                End Function

                Private Function GetPixelValues(image As Bitmap) As Double()
                    Dim pixelValues As New List(Of Double)()
                    For y As Integer = 0 To image.Height - 1
                        For x As Integer = 0 To image.Width - 1
                            Dim pixelColor As Color = image.GetPixel(x, y)
                            Dim grayscaleValue As Double = pixelColor.R / 255.0
                            pixelValues.Add(grayscaleValue)
                        Next
                    Next
                    Return pixelValues.ToArray()
                End Function

                Private Function ResizeImage(imagePath As String, width As Integer, height As Integer) As Bitmap
                    Dim originalImage As Bitmap = New Bitmap(imagePath)
                    Dim resizedImage As Bitmap = New Bitmap(width, height)
                    Using g As Graphics = Graphics.FromImage(resizedImage)
                        g.DrawImage(originalImage, 0, 0, width, height)
                    End Using
                    Return resizedImage
                End Function
            End Class

            Public Class ImageSearch
                Public Function FindSimilarImages(queryVector As Double(), imageVectors As List(Of Tuple(Of String, Double())), numResults As Integer) As List(Of String)
                    Dim similarImages As New List(Of String)()

                    For Each imageVectorPair As Tuple(Of String, Double()) In imageVectors
                        Dim imageName As String = imageVectorPair.Item1
                        Dim imageVector As Double() = imageVectorPair.Item2

                        Dim distance As Double = CalculateEuclideanDistance(queryVector, imageVector)
                        similarImages.Add(imageName)
                    Next

                    similarImages.Sort() ' Sort the list of similar image names

                    Return similarImages.Take(numResults).ToList()
                End Function



                Private Function CalculateEuclideanDistance(vector1 As Double(), vector2 As Double()) As Double
                    Dim sumSquaredDifferences As Double = 0
                    For i As Integer = 0 To vector1.Length - 1
                        Dim difference As Double = vector1(i) - vector2(i)
                        sumSquaredDifferences += difference * difference
                    Next
                    Return Math.Sqrt(sumSquaredDifferences)
                End Function
            End Class
        End Class
    End Namespace
    Namespace Tokenizers
        Public Class TokenizerWordPiece
            Private ReadOnly corpus As List(Of String)
            Private vocabulary As Dictionary(Of String, Integer)
            Private maxVocabSize As Integer
            Private ReadOnly maxSubwordLength As Integer



            Public Sub New()
            End Sub
            Public Sub New(corpus As List(Of String))
                Me.corpus = corpus
                Me.vocabulary = New Dictionary(Of String, Integer)
                Me.maxVocabSize = 1000000
                Me.maxSubwordLength = 20
            End Sub
            Public Sub New(corpus As List(Of String), vocabulary As Dictionary(Of String, Integer), maxVocabSize As Integer, maxSubwordLength As Integer)
                If corpus Is Nothing Then
                    Throw New ArgumentNullException(NameOf(corpus))
                End If

                If vocabulary Is Nothing Then
                    Throw New ArgumentNullException(NameOf(vocabulary))
                End If

                Me.corpus = corpus
                Me.vocabulary = vocabulary
                Me.maxVocabSize = maxVocabSize
                Me.maxSubwordLength = maxSubwordLength
            End Sub
            Public Sub Train()
                Dim subwordCounts As New Dictionary(Of String, Integer)

                ' Count subword occurrences in the corpus
                For Each sentence As String In corpus
                    Dim tokens As List(Of String) = Tokenize(sentence)

                    For Each token As String In tokens
                        If subwordCounts.ContainsKey(token) Then
                            subwordCounts(token) += 1
                        Else
                            subwordCounts.Add(token, 1)
                        End If
                    Next
                Next

                ' Sort subwords by frequency and add them to the vocabulary
                Dim sortedSubwords = subwordCounts.OrderByDescending(Function(pair) pair.Value)

                For Each pair In sortedSubwords.Take(maxVocabSize)
                    vocabulary.Add(pair.Key, vocabulary.Count)
                Next
            End Sub


            Public Function GetVocabulary() As Dictionary(Of String, Integer)
                Return vocabulary
            End Function
            Public Function Tokenize(text As String) As List(Of String)
                Dim tokens As New List(Of String)
                Dim index As Integer = 0

                While index < text.Length
                    Dim subwordLength As Integer = Math.Min(maxSubwordLength, text.Length - index)
                    Dim subword As String = text.Substring(index, subwordLength)

                    While subwordLength > 0 AndAlso Not vocabulary.ContainsKey(subword)
                        subwordLength -= 1
                        subword = text.Substring(index, subwordLength)
                    End While

                    tokens.Add(subword)
                    index += subwordLength
                End While

                Return tokens
            End Function
            Public Shared Function CalculateWordPieceFrequency(ByVal subword As String, ByVal mergedWord As String) As Integer
                Dim occurrences As Integer = 0
                Dim index As Integer = -1

                While True
                    index = mergedWord.IndexOf(subword, index + 1)
                    If index = -1 Then
                        Exit While
                    End If

                    ' Check if the found index is part of a valid word (not a subword of another word)
                    If index = 0 OrElse mergedWord(index - 1) = " "c Then
                        Dim endIndex As Integer = index + subword.Length
                        If endIndex = mergedWord.Length OrElse mergedWord(endIndex) = " "c Then
                            occurrences += 1
                        End If
                    End If
                End While

                Return occurrences
            End Function


        End Class
        Public Class TokenizerBPE
            Public Class BpeSubwordPair
                Public Property Subword1 As String
                Public Property Subword2 As String
                Public Property Frequency As Integer

                Public Sub New(subword1 As String, subword2 As String, frequency As Integer)
                    Me.Subword1 = subword1
                    Me.Subword2 = subword2
                    Me.Frequency = frequency
                End Sub
            End Class
            Public Class BpeVocabulary
                Inherits Dictionary(Of String, Integer)
            End Class
            Private Sub New()
                ' Private constructor to prevent instantiation without parameters
            End Sub



            Public Shared Function TrainBpeModel(corpus As List(Of String), numMerges As Integer) As BpeVocabulary
                ' Tokenize the corpus at the character level to get the initial vocabulary
                Dim characterLevelVocabulary As BpeVocabulary = TokenizeCorpusToCharacterLevel(corpus)

                ' Merge the most frequent pairs of subwords iteratively
                For i As Integer = 0 To numMerges - 1
                    Dim mostFrequentPair As BpeSubwordPair = FindMostFrequentPair(characterLevelVocabulary)
                    If mostFrequentPair Is Nothing Then
                        Exit For
                    End If

                    Dim newSubword As String = mostFrequentPair.Subword1 + mostFrequentPair.Subword2
                    characterLevelVocabulary = MergeSubwordPair(characterLevelVocabulary, mostFrequentPair, newSubword)
                Next

                Return characterLevelVocabulary
            End Function

            Private Shared Function TokenizeCorpusToCharacterLevel(corpus As List(Of String)) As BpeVocabulary
                Dim characterLevelVocabulary As New BpeVocabulary()

                For Each document As String In corpus
                    For Each character As Char In document
                        Dim subword As String = character.ToString()

                        If characterLevelVocabulary.ContainsKey(subword) Then
                            characterLevelVocabulary(subword) += 1
                        Else
                            characterLevelVocabulary.Add(subword, 1)
                        End If
                    Next
                Next

                Return characterLevelVocabulary
            End Function

            Private Shared Function FindMostFrequentPair(vocabulary As BpeVocabulary) As BpeSubwordPair
                Dim mostFrequentPair As BpeSubwordPair = Nothing
                Dim maxFrequency As Integer = 0

                For Each subword1 As String In vocabulary.Keys
                    For Each subword2 As String In vocabulary.Keys
                        If subword1 <> subword2 Then
                            Dim pairFrequency As Integer = CalculatePairFrequency(vocabulary, subword1, subword2)
                            If pairFrequency > maxFrequency Then
                                maxFrequency = pairFrequency
                                mostFrequentPair = New BpeSubwordPair(subword1, subword2, pairFrequency)
                            End If
                        End If
                    Next
                Next

                Return mostFrequentPair
            End Function

            Private Shared Function CalculatePairFrequency(vocabulary As BpeVocabulary, subword1 As String, subword2 As String) As Integer
                Dim pairFrequency As Integer = 0

                For Each word As String In vocabulary.Keys
                    Dim mergedWord As String = word.Replace(subword1 + subword2, subword1 + subword2.ToLower())
                    Dim occurrences As Integer = 0
                    Dim index As Integer = -1

                    While True
                        index = mergedWord.IndexOf(subword1 + subword2.ToLower(), index + 1)
                        If index = -1 Then
                            Exit While
                        End If
                        occurrences += 1
                    End While


                    pairFrequency += occurrences * vocabulary(word)
                Next

                Return pairFrequency
            End Function

            Private Shared Function MergeSubwordPair(vocabulary As BpeVocabulary, pairToMerge As BpeSubwordPair, newSubword As String) As BpeVocabulary
                Dim newVocabulary As New BpeVocabulary()

                For Each subword As String In vocabulary.Keys
                    Dim mergedSubword As String = subword.Replace(pairToMerge.Subword1 + pairToMerge.Subword2, newSubword)
                    newVocabulary(mergedSubword) = vocabulary(subword)
                Next

                Return newVocabulary
            End Function
        End Class
        Public Class TokenizerBitWord
            Public Property Vocabulary As Dictionary(Of String, Integer)
            Public Sub New()
                Vocabulary = New Dictionary(Of String, Integer)
            End Sub
            Public Function Tokenize(ByRef Corpus As List(Of String)) As List(Of String)
                Dim tokens As New List(Of String)
                Dim Subword As String = ""

                Dim UnknownDocs As New List(Of String)
                'SubDoc Vocabulary Tokenizer
                For Each doc In Corpus
                    For i = 0 To doc.Count - 1
                        Subword &= doc(i)
                        If Vocabulary.ContainsKey(Subword.ToLower()) Then
                            tokens.Add(Subword)
                            Subword = ""
                        End If

                    Next
                    'Save unknowns
                    If Subword <> "" Then
                        UnknownDocs.Add(Subword)
                    End If
                Next
                'Unknown paragraphs
                Dim UnknownParagraphs As New List(Of String)
                If UnknownDocs.Count > 0 Then
                    For Each doc In UnknownDocs
                        Dim Para As List(Of String) = BasicTokenizer.TokenizeToParagraph(doc)
                        For Each item In Para
                            Subword = ""

                            Subword += item
                            If Vocabulary.ContainsKey(Subword.ToLower) Then
                                ' If the subword is in the Vocabulary, add it to the list of subwords
                                tokens.Add(Subword.ToLower)
                                ' Reset the subword for the next iteration
                                Subword = ""
                            End If
                            'Save unknowns
                            If Subword <> "" Then
                                UnknownParagraphs.Add(Subword)
                            End If
                        Next

                    Next
                End If
                'Unknown Sentences
                Dim UnknownSents As New List(Of String)
                If UnknownParagraphs.Count > 0 Then
                    For Each sent In UnknownParagraphs
                        Dim Sents As List(Of String) = BasicTokenizer.TokenizeToSentence(sent)


                        For Each item In Sents
                            Subword = ""

                            Subword += item
                            If Vocabulary.ContainsKey(Subword.ToLower) Then
                                ' If the subword is in the Vocabulary, add it to the list of subwords
                                tokens.Add(Subword.ToLower)
                                ' Reset the subword for the next iteration
                                Subword = ""
                            End If
                            'Save unknowns
                            If Subword <> "" Then
                                UnknownSents.Add(Subword)
                            End If
                        Next
                    Next
                End If
                'Unknown Words
                Dim UnknownWords As New List(Of String)
                If UnknownSents.Count > 0 Then
                    For Each Word In UnknownSents
                        Dim Words As List(Of String) = BasicTokenizer.TokenizeToWord(Word)
                        For Each item In Words
                            Subword = ""

                            Subword += item
                            If Vocabulary.ContainsKey(Subword.ToLower) Then
                                ' If the subword is in the Vocabulary, add it to the list of subwords
                                tokens.Add(Subword.ToLower)
                                ' Reset the subword for the next iteration
                                Subword = ""
                            End If
                            'Save unknowns
                            If Subword <> "" Then
                                UnknownWords.Add(Subword)
                            End If
                        Next

                    Next

                End If
                'Unknown Words
                Dim UnknownChars As New List(Of String)
                If UnknownWords.Count > 0 Then
                    For Each iChar In UnknownWords
                        Dim Chars As List(Of String) = BasicTokenizer.TokenizeToCharacter(iChar)
                        For Each item In Chars
                            Subword = ""

                            Subword += item
                            If Vocabulary.ContainsKey(Subword.ToLower) Then
                                ' If the subword is in the Vocabulary, add it to the list of subwords
                                tokens.Add(Subword.ToLower)
                                ' Reset the subword for the next iteration
                                Subword = ""
                            End If
                            'Save unknowns
                            If Subword <> "" Then
                                UnknownChars.Add(Subword)
                            End If
                        Next

                    Next

                End If

                For Each unkChar In UnknownChars
                    Vocabulary.Add(unkChar, 1)
                Next

                Console.WriteLine("Recognized Tokens")
                For Each tok In tokens
                    Console.WriteLine("Token =" & tok)
                Next

                Console.WriteLine("UnRecognized Tokens")
                For Each tok In UnknownChars
                    Console.WriteLine("Token =" & tok)
                Next
                Return tokens
            End Function

            Public Sub Train(corpus As List(Of String), MaxMergeOperations As Integer)
                ' Initialize the vocabulary with word-level subword units
                Tokenize(corpus)
                Dim mergeOperationsCount As Integer = 0

                While mergeOperationsCount < MaxMergeOperations
                    ' Compute the frequency of subword units in the vocabulary
                    Dim subwordFrequencies As New Dictionary(Of String, Integer)

                    For Each subword In Vocabulary.Keys
                        Dim subwordUnits = BasicTokenizer.TokenizeToCharacter(subword)
                        For Each unit In subwordUnits
                            If subwordFrequencies.ContainsKey(unit) Then
                                subwordFrequencies(unit) += Vocabulary(subword)
                            Else
                                subwordFrequencies.Add(unit, Vocabulary(subword))
                            End If
                        Next
                    Next

                    ' Find the most frequent pair of subword units
                    Dim mostFrequentPair As KeyValuePair(Of String, Integer) = subwordFrequencies.OrderByDescending(Function(pair) pair.Value).FirstOrDefault()

                    If mostFrequentPair.Value < 2 Then
                        ' Stop merging if the frequency of the most frequent pair is less than 2
                        Exit While
                    End If

                    ' Merge the most frequent pair into a new subword unit
                    Dim newSubwordUnit = mostFrequentPair.Key

                    ' Update the vocabulary by replacing occurrences of the merged subword pair with the new subword unit
                    Dim updatedVocabulary As New Dictionary(Of String, Integer)

                    For Each subword In Vocabulary.Keys
                        Dim mergedSubword = subword.Replace(mostFrequentPair.Key, newSubwordUnit)
                        updatedVocabulary(mergedSubword) = Vocabulary(subword)
                    Next

                    Vocabulary = updatedVocabulary
                    mergeOperationsCount += 1

                End While

            End Sub

            Public Class BasicTokenizer

                Public Shared Function TokenizeToCharacter(Document As String) As List(Of String)
                    Document = Document.ToLower()
                    Dim characters As Char() = Document.ToCharArray()
                    TokenizeToCharacter = New List(Of String)
                    For Each item In characters
                        TokenizeToCharacter.Add(item)
                    Next
                End Function

                Public Shared Function TokenizeToWord(Document As String) As List(Of String)
                    Document = Document.ToLower()
                    Document = Document.SpacePunctuation
                    Return Document.Split({" ", ".", ",", ";", ":", "!", "?"}, StringSplitOptions.RemoveEmptyEntries).ToList
                End Function

                Public Shared Function TokenizeToSentence(Document As String) As List(Of String)
                    Document = Document.ToLower()
                    Document = Document.SpacePunctuation
                    Return Split(Document, ".").ToList
                End Function

                Public Shared Function TokenizeToParagraph(Document As String) As List(Of String)
                    Document = Document.ToLower()

                    Return Split(Document, vbNewLine).ToList
                End Function

            End Class

        End Class
        Public Class TokenizerPositional
            Private iStopWords As List(Of String)

            Private Function RemoveStopWords(ByVal tokens As List(Of Token)) As List(Of Token)
                Return tokens.Where(Function(token) Not StopWords.Contains(token.Value)).ToList()
            End Function
            Public Property StopWordRemovalEnabled As Boolean

            Public Property StopWords As List(Of String)
                Get
                    Return iStopWords
                End Get
                Set(value As List(Of String))
                    iStopWords = value
                End Set
            End Property
            Public Structure Token
                ''' <summary>
                ''' Initializes a new instance of the Token structure.
                ''' </summary>
                ''' <param name="type">The type of the token.</param>
                ''' <param name="value">The string value of the token.</param>
                Public Sub New(ByVal type As String, ByVal value As String)
                    Me.Type = type
                    Me.Value = value
                End Sub

                Public Sub New(ByVal type As TokenType, ByVal value As String, ByVal startPosition As Integer, ByVal endPosition As Integer)
                    Me.Type = type
                    Me.Value = value
                    Me.StartPosition = startPosition
                    Me.EndPosition = endPosition
                End Sub

                Public Property EndPosition As Integer
                Public Property StartPosition As Integer
                Public Property Type As TokenType
                Public Property Value As String
            End Structure

            ''' <summary>
            ''' Returns Tokens With Positions
            ''' </summary>
            ''' <param name="input"></param>
            ''' <returns></returns>
            Public Shared Function TokenizeByCharacter(ByVal input As String) As List(Of Token)
                Dim characters As Char() = input.ToCharArray()
                Dim tokens As New List(Of Token)
                Dim currentPosition As Integer = 0

                For Each character As Char In characters
                    Dim startPosition As Integer = currentPosition
                    Dim endPosition As Integer = currentPosition
                    Dim token As New Token(TokenType.Character, character.ToString(), startPosition, endPosition)
                    tokens.Add(token)
                    currentPosition += 1
                Next

                Return tokens
            End Function

            ''' <summary>
            ''' Returns Tokens With Positions
            ''' </summary>
            ''' <param name="input"></param>
            ''' <returns></returns>
            Public Shared Function TokenizeBySentence(ByVal input As String) As List(Of Token)
                Dim sentences As String() = input.Split("."c)
                Dim tokens As New List(Of Token)
                Dim currentPosition As Integer = 0

                For Each sentence As String In sentences
                    Dim startPosition As Integer = currentPosition
                    Dim endPosition As Integer = currentPosition + sentence.Length - 1
                    Dim token As New Token(TokenType.Sentence, sentence, startPosition, endPosition)
                    tokens.Add(token)
                    currentPosition = endPosition + 2 ' Account for the period and the space after the sentence
                Next

                Return tokens
            End Function

            ''' <summary>
            ''' Returns Tokens With Positions
            ''' </summary>
            ''' <param name="input"></param>
            ''' <returns></returns>
            Public Shared Function TokenizeByWord(ByVal input As String) As List(Of Token)
                Dim words As String() = input.Split(" "c)
                Dim tokens As New List(Of Token)
                Dim currentPosition As Integer = 0

                For Each word As String In words
                    Dim startPosition As Integer = currentPosition
                    Dim endPosition As Integer = currentPosition + word.Length - 1
                    Dim token As New Token(TokenType.Word, word, startPosition, endPosition)
                    tokens.Add(token)
                    currentPosition = endPosition + 2 ' Account for the space between words
                Next

                Return tokens
            End Function

            ''' <summary>
            ''' Pure basic Tokenizer to Tokens
            ''' </summary>
            ''' <param name="Corpus"></param>
            ''' <param name="tokenizationOption">Type Of Tokenization</param>
            ''' <returns></returns>
            Public Shared Function TokenizeInput(ByRef Corpus As List(Of String), tokenizationOption As TokenizerType) As List(Of Token)
                Dim ivocabulary As New List(Of Token)

                For Each Doc In Corpus
                    Select Case tokenizationOption
                        Case TokenizerType._Char
                            ivocabulary.AddRange(TokenizeByCharacter(Doc.ToLower))
                        Case TokenizerType._Word
                            ivocabulary.AddRange(TokenizeByWord(Doc.ToLower))
                        Case TokenizerType._Sentence
                            ivocabulary.AddRange(TokenizeBySentence(Doc.ToLower))


                    End Select
                Next

                Return ivocabulary
            End Function

        End Class
        Public Class TokenizerTokenID
            Public TokenToId As New Dictionary(Of String, Integer)
            Private idToToken As New Dictionary(Of Integer, String)
            Private nextId As Integer = 0

            Private vocab As New Dictionary(Of String, Integer)
            Public Sub New(ByRef Vocabulary As Dictionary(Of String, Integer))
                vocab = Vocabulary
                TokenToId = New Dictionary(Of String, Integer)
                idToToken = New Dictionary(Of Integer, String)
            End Sub

            ''' <summary>
            ''' Pure Tokenizer (will tokenize based on the Tokenizer model settings)
            ''' </summary>
            ''' <param name="text"></param>
            ''' <returns></returns>
            Public Function TokenizeToTokenIDs(text As String) As List(Of Integer)
                Dim tokens = TokenizerPositional.TokenizeByWord(text)
                Dim tokenIds As New List(Of Integer)

                For Each itoken In tokens
                    Dim tokenId As Integer
                    If TokenToId.ContainsKey(itoken.Value) Then
                        tokenId = TokenToId(itoken.Value)
                    Else
                        'Not registered

                        tokenId = TokenToId(itoken.Value)

                    End If
                    tokenIds.Add(tokenId)

                Next

                Return tokenIds
            End Function

            Private Sub AddTokenID(text As String)

                If Not vocab.ContainsKey(text) Then
                    vocab(text) = nextId
                    nextId += 1
                    TokenToId = vocab.ToDictionary(Function(x) x.Key, Function(x) x.Value)
                    idToToken = TokenToId.ToDictionary(Function(x) x.Value, Function(x) x.Key)
                End If


            End Sub

            ''' <summary>
            ''' Given  a Set of Token ID Decode the Tokens 
            ''' </summary>
            ''' <param name="tokenIds"></param>
            ''' <returns></returns>
            Public Function Detokenize(tokenIds As List(Of Integer)) As String
                Dim tokens As New List(Of String)

                For Each tokenId As Integer In tokenIds
                    tokens.Add(idToToken(tokenId))
                Next

                Return String.Join(" ", tokens)
            End Function
        End Class
        Public Class Tokenizer
            Public Property Vocabulary As Dictionary(Of String, Integer)
            Public ReadOnly Property PairFrequencies As Dictionary(Of String, Integer) = ComputePairFrequencies()
            Public ReadOnly Property maxSubwordLen As Integer = Me.Vocabulary.Max(Function(token) token.Key.Length)
            Private ReadOnly unkToken As String = "<Unk>"
            ''' <summary>
            ''' Defines max entries in vocabulary before Pruning Rare Words
            ''' </summary>
            ''' <returns></returns>
            Public Property VocabularyPruneValue As Integer = 100000

            Public Sub New()
                Vocabulary = New Dictionary(Of String, Integer)

            End Sub
            Public Function GetVocabulary() As List(Of String)
                Return Vocabulary.Keys.ToList()
            End Function

            Public Sub New(vocabulary As Dictionary(Of String, Integer), Optional vocabularyPruneValue As Integer = 1000000)
                Me.Vocabulary = vocabulary
                Me.VocabularyPruneValue = vocabularyPruneValue
            End Sub

            Private Function TokenizeWordPiece(text As String) As List(Of String)
                Dim tokens As New List(Of String)
                Dim pos As Integer = 0

                While pos < text.Length
                    Dim foundSubword As Boolean = False
                    Dim subword As String = ""

                    For subwordLen As Integer = Math.Min(Me.maxSubwordLen, text.Length - pos) To 1 Step -1
                        subword = text.Substring(pos, subwordLen)

                        If Vocabulary.Keys.Contains(subword) Then
                            tokens.Add(subword)
                            pos += subwordLen
                            foundSubword = True
                            Exit For
                        End If
                    Next

                    ' If no subword from the vocabulary matches, split into WordPiece tokens
                    If Not foundSubword Then
                        Dim wordPieceTokens As List(Of String) = TokenizeBitWord(subword)
                        tokens.AddRange(wordPieceTokens)
                        UpdateVocabulary(subword)
                        pos += subword.Length
                    End If
                End While

                Return tokens
            End Function
            Private Function TokenizeBitWord(subword As String) As List(Of String)
                Dim wordPieceTokens As New List(Of String)
                Dim startIdx As Integer = 0

                While startIdx < subword.Length
                    Dim endIdx As Integer = subword.Length
                    Dim foundSubword As Boolean = False

                    While startIdx < endIdx
                        Dim candidate As String = subword.Substring(startIdx, endIdx - startIdx)
                        Dim isLast = endIdx = subword.Length

                        If Vocabulary.Keys.Contains(candidate) OrElse isLast Then
                            wordPieceTokens.Add(candidate)
                            startIdx = endIdx
                            foundSubword = True
                            Exit While
                        End If

                        endIdx -= 1
                    End While

                    ' If no subword from the vocabulary matches, break the subword into smaller parts
                    If Not foundSubword Then
                        wordPieceTokens.Add("<unk>")
                        startIdx += 1
                    End If
                End While

                Return wordPieceTokens
            End Function
            Private Shared Function TokenizeBitWord(subword As String, ByRef Vocab As Dictionary(Of String, Integer)) As List(Of String)

                Dim wordPieceTokens As New List(Of String)
                Dim startIdx As Integer = 0

                While startIdx < subword.Length
                    Dim endIdx As Integer = subword.Length
                    Dim foundSubword As Boolean = False

                    While startIdx < endIdx
                        Dim candidate As String = subword.Substring(startIdx, endIdx - startIdx)
                        Dim isLast = endIdx = subword.Length

                        If Vocab.Keys.Contains(candidate) OrElse isLast Then
                            wordPieceTokens.Add(candidate)
                            startIdx = endIdx
                            foundSubword = True
                            Exit While
                        End If

                        endIdx -= 1
                    End While

                    ' If no subword from the vocabulary matches, break the subword into smaller parts
                    If Not foundSubword Then
                        wordPieceTokens.Add("<unk>")
                        startIdx += 1
                    End If
                End While

                Return wordPieceTokens
            End Function

            Private Function TokenizeBPE(ByVal text As String) As List(Of String)
                Dim tokens As New List(Of String)

                While text.Length > 0
                    Dim foundToken As Boolean = False

                    ' Find the longest token in the vocabulary that matches the start of the text
                    For Each subword In Vocabulary.OrderByDescending(Function(x) x.Key.Length)
                        If text.StartsWith(subword.Key) Then
                            tokens.Add(subword.Key)
                            text = text.Substring(subword.Key.Length)
                            foundToken = True
                            Exit For
                        End If
                    Next

                    ' If no token from the vocabulary matches, break the text into subwords
                    If Not foundToken Then
                        Dim subwordFound As Boolean = False
                        Dim subword As String = ""
                        ' Divide the text into subwords starting from the longest possible length
                        For length = Math.Min(text.Length, 20) To 1 Step -1
                            subword = text.Substring(0, length)

                            ' Check if the subword is in the vocabulary
                            If Vocabulary.Keys(subword) Then
                                tokens.Add(subword)
                                text = text.Substring(length)
                                subwordFound = True
                                Exit For
                            End If
                        Next

                        ' If no subword from the vocabulary matches,
                        'Learn On the fly, But 
                        If Not subwordFound Then
                            '    Throw New Exception("Unrecognized subword in the text.")
                            tokens.AddRange(TokenizeBitWord(unkToken & subword))
                            UpdateVocabulary(subword)

                        End If
                    End If
                End While

                Return tokens
            End Function
            Private Class NgramTokenizer

                Public Shared Function TokenizetoCharacter(Document As String, n As Integer) As List(Of String)
                    TokenizetoCharacter = New List(Of String)
                    Document = Document.ToLower()
                    Document = Document.SpacePunctuation

                    ' Generate character n-grams
                    For i As Integer = 0 To Document.Length - n
                        Dim ngram As String = Document.Substring(i, n)
                        TokenizetoCharacter.Add(ngram)
                    Next

                End Function

                Public Shared Function TokenizetoWord(ByRef text As String, n As Integer) As List(Of String)
                    TokenizetoWord = New List(Of String)
                    text = text.ToLower()
                    text = text.SpacePunctuation

                    ' Split the clean text into individual words
                    Dim words() As String = text.Split({" ", ".", ",", ";", ":", "!", "?"}, StringSplitOptions.RemoveEmptyEntries)

                    ' Generate n-grams from the words
                    For i As Integer = 0 To words.Length - n
                        Dim ngram As String = String.Join(" ", words.Skip(i).Take(n))
                        TokenizetoWord.Add(ngram)
                    Next

                End Function

                Public Shared Function TokenizetoParagraph(text As String, n As Integer) As List(Of String)
                    TokenizetoParagraph = New List(Of String)

                    ' Split the text into paragraphs
                    Dim paragraphs() As String = text.Split({Environment.NewLine & Environment.NewLine}, StringSplitOptions.RemoveEmptyEntries)

                    ' Generate paragraph n-grams
                    For i As Integer = 0 To paragraphs.Length - n
                        Dim ngram As String = String.Join(Environment.NewLine & Environment.NewLine, paragraphs.Skip(i).Take(n))
                        TokenizetoParagraph.Add(ngram)
                    Next

                    Return TokenizetoParagraph
                End Function

                Public Shared Function TokenizetoSentence(text As String, n As Integer) As List(Of String)
                    Dim tokens As New List(Of String)

                    ' Split the text into Clauses
                    Dim Clauses() As String = text.Split({".", ",", ";", ":", "!", "?"}, StringSplitOptions.RemoveEmptyEntries)

                    ' Generate sentence n-grams
                    For i As Integer = 0 To Clauses.Length - n
                        Dim ngram As String = String.Join(" ", Clauses.Skip(i).Take(n))
                        tokens.Add(ngram)
                    Next

                    Return tokens
                End Function

            End Class
            Private Class BasicTokenizer

                Public Shared Function TokenizeToCharacter(Document As String) As List(Of String)
                    TokenizeToCharacter = New List(Of String)
                    Document = Document.ToLower()
                    For i = 0 To Document.Length - 1
                        TokenizeToCharacter.Add(Document(i))
                    Next
                End Function

                Public Shared Function TokenizeToWord(Document As String) As List(Of String)
                    Document = Document.ToLower()
                    Document = Document.SpacePunctuation
                    Return Document.Split({" ", ".", ",", ";", ":", "!", "?"}, StringSplitOptions.RemoveEmptyEntries).ToList
                End Function

                Public Shared Function TokenizeToSentence(Document As String) As List(Of String)
                    Document = Document.ToLower()
                    Document = Document.SpacePunctuation
                    Return Split(Document, ".").ToList
                    Return Document.Split({".", ",", ";", ":", "!", "?"}, StringSplitOptions.RemoveEmptyEntries).ToList
                End Function

                Public Shared Function TokenizeToParagraph(Document As String) As List(Of String)
                    Document = Document.ToLower()
                    Return Split(Document, vbNewLine).ToList
                End Function

            End Class
            Public Sub Add_Vocabulary(initialVocabulary As List(Of String))

                For Each word In initialVocabulary

                    UpdateVocabulary(word)

                Next

            End Sub
            Public Sub Initialize_Vocabulary(initialVocabulary As List(Of String), n As Integer)

                For Each word In initialVocabulary
                    For i As Integer = 0 To word.Length - n
                        UpdateVocabulary(word.Substring(i, n))
                    Next
                Next

            End Sub
            Private Function ComputePairFrequencies() As Dictionary(Of String, Integer)
                Dim pairFrequencies As Dictionary(Of String, Integer) = New Dictionary(Of String, Integer)

                For Each token As String In Vocabulary.Keys
                    Dim tokenChars As List(Of Char) = token.ToList()

                    For i As Integer = 0 To tokenChars.Count - 2
                        Dim pair As String = tokenChars(i) & tokenChars(i + 1)

                        If Not pairFrequencies.ContainsKey(pair) Then
                            pairFrequencies.Add(pair, Vocabulary(token))
                        Else
                            Dim value = pairFrequencies(pair)
                            value += Vocabulary(token)
                            pairFrequencies.Remove(pair)
                            pairFrequencies.Add(pair, value)


                        End If
                    Next
                Next

                Return pairFrequencies
            End Function

            Private Sub UpdateFrequencyDictionary(mergedSubword As String)
                PairFrequencies.Remove("")
                For i As Integer = 0 To mergedSubword.Length - 2
                    Dim bigram As String = mergedSubword.Substring(i, 2)
                    If PairFrequencies.ContainsKey(bigram) Then
                        PairFrequencies(bigram) += 1
                    Else
                        PairFrequencies.Add(bigram, 1)
                    End If
                Next
            End Sub
            Public Sub UpdateVocabulary(ByRef Term As String)
                If Vocabulary.Keys.Contains(Term) = True Then
                    Dim value = Vocabulary(Term)
                    value += 1
                    Vocabulary.Remove(Term)
                    Vocabulary.Add(Term, value)
                Else
                    Vocabulary.Add(Term, 1)
                End If

            End Sub
            Public Shared Function UpdateCorpusWithMergedToken(ByRef corpus As List(Of String), pair As String) As List(Of String)
                ' Update the text corpus with the merged token for the next iteration.
                Return corpus.ConvertAll(Function(text) text.Replace(pair, pair.Replace(" ", "_")))
            End Function
            Public Sub Prune(pruningThreshold As Integer)

                Dim minimumVocabularySize As Integer = VocabularyPruneValue
                If Vocabulary.Count > minimumVocabularySize Then
                    PruneVocabulary(pruningThreshold)
                End If

            End Sub
            Private Sub PruneVocabulary(threshold As Integer)
                ' Create a list to store tokens to be removed.
                Dim tokensToRemove As New List(Of String)

                ' Iterate through the vocabulary and identify tokens to prune.
                For Each token In Vocabulary
                    Dim tokenId As Integer = token.Value
                    Dim tokenFrequency As Integer = Vocabulary(token.Key)

                    ' Prune the token if it has frequency below the threshold (1) and is not recent (has a lower ID).
                    If tokenFrequency <= threshold AndAlso tokenId < Vocabulary.Count - 1 Then
                        tokensToRemove.Add(token.Key)
                    End If
                Next

                ' Remove the identified tokens from the vocabulary.
                For Each tokenToRemove In tokensToRemove
                    Vocabulary.Remove(tokenToRemove)
                Next

                Console.WriteLine("Pruning completed. Vocabulary size after pruning: " & Vocabulary.Count)
                Console.ReadLine()
            End Sub
            Public Sub Train(text As String, Epochs As Integer)
                ' Tokenize the text into individual characters

                Dim Bits As List(Of String) = TokenizeBitWord(text)
                For Each bit As String In Bits
                    UpdateVocabulary(bit)
                Next


                ' Train BPE using merging strategy
                Dim numMerges As Integer = Epochs ' Define the number of merges, you can adjust it as needed
                For mergeIndex As Integer = 0 To numMerges - 1
                    MergeMostFrequentBigram()
                    MergeMostFrequentPair(FindMostFrequentPair.Key)
                Next

                Prune(1)
            End Sub
            Public Function Tokenize(singleDocument As String, isWordPiece As Boolean) As List(Of String)
                ' Tokenize the document using the current vocabulary.
                Dim tokens As List(Of String) = If(isWordPiece, Tokenize(singleDocument, True), Tokenize(singleDocument, False))
                If tokens.Contains(unkToken) = True Then
                    tokens = TrainAndTokenize(singleDocument, isWordPiece, 1)
                End If
                Return tokens
            End Function
            Private Function TrainAndTokenize(singleDocument As String, isWordPiece As Boolean, Epochs As Integer) As List(Of String)
                ' Tokenize the document using the current vocabulary.
                Dim tokens As List(Of String) = If(isWordPiece, Tokenize(singleDocument, True), Tokenize(singleDocument, False))

                ' Train the tokenizer using the same document.
                If isWordPiece Then
                    TrainWordPiece(singleDocument, Epochs)
                Else
                    TrainBPE(singleDocument, Epochs)
                End If

                ' Re-tokenize the document with the updated vocabulary.
                Return If(isWordPiece, TokenizeWordPiece(singleDocument), TokenizeBPE(singleDocument))
            End Function
            Public Sub Train(text As String, isWordPiece As Boolean, Epochs As Integer)
                If isWordPiece Then
                    TrainWordPiece(text, Epochs)
                Else
                    TrainBPE(text, Epochs)
                End If
                Prune(1)
            End Sub
            Private Sub TrainWordPiece(text As String, Epochs As Integer)
                ' Tokenize the text into individual characters
                Dim Bits As List(Of String) = TokenizeWordPiece(text)
                For Each bit As String In Bits
                    UpdateVocabulary(bit)
                Next

                ' Train WordPiece using merging strategy
                Dim numMerges As Integer = Epochs ' Define the number of merges, you can adjust it as needed
                For mergeIndex As Integer = 0 To numMerges - 1
                    MergeMostFrequentBigram()
                    MergeMostFrequentPair(FindMostFrequentPair.Key)
                Next
            End Sub
            Private Sub TrainBPE(text As String, Epochs As Integer)
                ' Tokenize the text into individual characters
                Dim Bits As List(Of String) = TokenizeBPE(text)
                For Each bit As String In Bits
                    UpdateVocabulary(bit)
                Next

                ' Train BPE using merging strategy
                Dim numMerges As Integer = Epochs ' Define the number of merges, you can adjust it as needed
                For mergeIndex As Integer = 0 To numMerges - 1
                    MergeMostFrequentBigram()
                    MergeMostFrequentPair(FindMostFrequentPair.Key)
                Next
            End Sub
            Private Function FindMostFrequentPair() As KeyValuePair(Of String, Integer)
                ' Find the most frequent character pair from the frequency counts.
                Return PairFrequencies.Aggregate(Function(x, y) If(x.Value > y.Value, x, y))
            End Function
            Private Sub MergeMostFrequentPair(pair As String)
                ' Merge the most frequent character pair into a new subword unit.
                Dim mergedToken As String = pair.Replace(" ", "_")
                UpdateVocabulary(mergedToken)

            End Sub
            Private Sub MergeMostFrequentBigram()
                Dim mostFrequentBigram As String = GetMostFrequentBigram()
                If mostFrequentBigram IsNot Nothing Then
                    Dim mergedSubword As String = mostFrequentBigram.Replace("", " ")

                    UpdateVocabulary(mergedSubword)

                End If
            End Sub
            Private Function GetMostFrequentBigram() As String
                Dim mostFrequentBigram As String = Nothing
                Dim maxFrequency As Integer = 0

                For Each bigram In PairFrequencies.Keys
                    If PairFrequencies(bigram) > maxFrequency Then
                        mostFrequentBigram = bigram
                        maxFrequency = PairFrequencies(bigram)
                    End If
                Next

                Return mostFrequentBigram
            End Function

            Public Shared Function FindFrequentCharacterBigrams(Vocab As List(Of String), ByRef Freq_Threshold As Integer) As List(Of String)
                Dim bigramCounts As New Dictionary(Of String, Integer)

                For Each word In Vocab
                    Dim characters As Char() = word.ToCharArray()

                    For i As Integer = 0 To characters.Length - 2
                        Dim bigram As String = characters(i) & characters(i + 1)

                        If bigramCounts.ContainsKey(bigram) Then
                            bigramCounts(bigram) += 1
                        Else
                            bigramCounts.Add(bigram, 1)
                        End If
                    Next
                Next

                Dim frequentCharacterBigrams As New List(Of String)

                For Each pair In bigramCounts
                    If pair.Value > Freq_Threshold Then ' Adjust the threshold as needed
                        frequentCharacterBigrams.Add(pair.Key)
                    End If
                Next

                Return frequentCharacterBigrams
            End Function
            Public Shared Function GetHighFreq(ByRef Vocabulary As Dictionary(Of String, Integer), ByRef Threshold As Integer) As List(Of String)
                Dim HighFreq As New List(Of String)
                For Each item In Vocabulary
                    If item.Value > Threshold Then
                        HighFreq.Add(item.Key)
                    End If
                Next
                Return HighFreq
            End Function
            Public Shared Function TokenizeToCharacter(text As String) As List(Of String)
                Return BasicTokenizer.TokenizeToCharacter(text)
            End Function
            Public Shared Function TokenizeToWord(text As String) As List(Of String)
                Return BasicTokenizer.TokenizeToWord(text)
            End Function
            Public Shared Function TokenizeToSentence(text As String) As List(Of String)
                Return BasicTokenizer.TokenizeToSentence(text)
            End Function
            Public Shared Function TokenizeToSentenceGram(text As String, ByRef n As Integer) As List(Of String)
                Return NgramTokenizer.TokenizetoSentence(text, n)
            End Function
            Public Shared Function TokenizeToWordGram(text As String, ByRef n As Integer) As List(Of String)
                Return NgramTokenizer.TokenizetoWord(text, n)
            End Function
            Public Shared Function TokenizeToNGram(text As String, ByRef n As Integer) As List(Of String)
                Return NgramTokenizer.TokenizetoCharacter(text, n)
            End Function
            Public Shared Function TokenizeToBitWord(text As String, ByRef Vocab As Dictionary(Of String, Integer)) As List(Of String)
                Dim Words = Tokenizer.TokenizeToWord(text)
                Dim Tokens As New List(Of String)
                For Each item In Words
                    Tokens.AddRange(TokenizeBitWord(item, Vocab))
                Next
                Return Tokens
            End Function
        End Class
    End Namespace
    Namespace MatrixModels
        Public Class PMI
            ''' <summary>
            ''' Calculates the Pointwise Mutual Information (PMI) matrix for the trained model.
            ''' </summary>
            ''' <returns>A dictionary representing the PMI matrix.</returns>
            Public Shared Function CalculatePMI(ByRef model As Dictionary(Of String, Dictionary(Of String, Double))) As Dictionary(Of String, Dictionary(Of String, Double))
                Dim pmiMatrix As New Dictionary(Of String, Dictionary(Of String, Double))

                Dim totalCooccurrences As Double = GetTotalCooccurrences(model)

                For Each targetWord In model.Keys
                    Dim targetOccurrences As Double = GetTotalOccurrences(targetWord, model)

                    If Not pmiMatrix.ContainsKey(targetWord) Then
                        pmiMatrix(targetWord) = New Dictionary(Of String, Double)
                    End If

                    For Each contextWord In model(targetWord).Keys
                        Dim contextOccurrences As Double = GetTotalOccurrences(contextWord, model)
                        Dim cooccurrences As Double = model(targetWord)(contextWord)

                        Dim pmiValue As Double = Math.Log((cooccurrences * totalCooccurrences) / (targetOccurrences * contextOccurrences))
                        pmiMatrix(targetWord)(contextWord) = pmiValue
                    Next
                Next

                Return pmiMatrix
            End Function
            Public Shared Function GetTotalCooccurrences(ByRef Model As Dictionary(Of String, Dictionary(Of String, Double))) As Double
                Dim total As Double = 0

                For Each targetWord In Model.Keys
                    For Each cooccurrenceValue In Model(targetWord).Values
                        total += cooccurrenceValue
                    Next
                Next

                Return total
            End Function
            Public Shared Function GetTotalOccurrences(word As String, ByRef Model As Dictionary(Of String, Dictionary(Of String, Double))) As Double
                Dim total As Double = 0

                If Model.ContainsKey(word) Then
                    total = Model(word).Values.Sum()
                End If

                Return total
            End Function
            Public Shared Function CalculateCosineSimilarity(vectorA As Double(), vectorB As Double()) As Double
                Dim dotProduct As Double = 0
                Dim magnitudeA As Double = 0
                Dim magnitudeB As Double = 0

                For i As Integer = 0 To vectorA.Length - 1
                    dotProduct += vectorA(i) * vectorB(i)
                    magnitudeA += vectorA(i) * vectorA(i)
                    magnitudeB += vectorB(i) * vectorB(i)
                Next

                If magnitudeA <> 0 AndAlso magnitudeB <> 0 Then
                    Return dotProduct / (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB))
                Else
                    Return 0
                End If
            End Function
            Public Shared Function GenerateCooccurrenceMatrix(corpus As String(), windowSize As Integer) As Dictionary(Of String, Dictionary(Of String, Double))
                Dim matrix As New Dictionary(Of String, Dictionary(Of String, Double))

                For Each sentence In corpus
                    Dim words As String() = sentence.Split(" "c)
                    Dim length As Integer = words.Length

                    For i As Integer = 0 To length - 1
                        Dim targetWord As String = words(i)

                        If Not matrix.ContainsKey(targetWord) Then
                            matrix(targetWord) = New Dictionary(Of String, Double)
                        End If

                        For j As Integer = Math.Max(0, i - windowSize) To Math.Min(length - 1, i + windowSize)
                            If i = j Then
                                Continue For
                            End If

                            Dim contextWord As String = words(j)
                            Dim distance As Double = 1 / Math.Abs(i - j)

                            If matrix(targetWord).ContainsKey(contextWord) Then
                                matrix(targetWord)(contextWord) += distance
                            Else
                                matrix(targetWord)(contextWord) = distance
                            End If
                        Next
                    Next
                Next

                Return matrix
            End Function

        End Class

        ''' <summary>
        ''' Returns a list WordGram Probability Given a Sequence of Tokens 
        ''' </summary>
        Public Class Wordgram
            Private n As Integer
            Public Shared Sub Main()
                ' Train the wordgram model
                Dim trainingData As New List(Of String) From {"I love cats and dogs.", "Dogs are loyal companions."}
                Dim words As New List(Of String) From {
                    "apple", "banana", "orange", "apple", "pear", "kiwi", "orange", "mango", "kiwi", "guava", "kiwi", "orange", "orange", "apple", "banana"
                }
                Dim sentences As New List(Of String) From {
                    "I love apples.",
                    "Bananas are tasty.",
                    "I love apples.",
                    "I enjoy eating bananas.",
                    "mango is a delicious fruit.", "Bananas are tasty.",
                    "I love apples.", "I enjoy eating bananas.",
                    "Kiwi is a delicious fruit.", "I love apples.",
                    "I enjoy eating bananas.",
                    "orange is a delicious fruit.", "I love apples.",
                    "I enjoy eating bananas.",
                    "Kiwi is a delicious fruit.", "Bananas are tasty."
                }
                Dim Corpus As New List(Of String)
                Corpus.AddRange(sentences)
                Corpus.AddRange(words)


                ' Generate a sentence using the wordgram model
                For I = 1 To 5
                    Dim wordgramModel As New Wordgram(Corpus, I)
                    Dim generatedSentence As String = wordgramModel.GenerateSentence()
                    Console.WriteLine(generatedSentence)
                Next I
                Console.ReadLine()
            End Sub

            Public wordgramCounts As New Dictionary(Of List(Of String), Integer)
            Public wordgramProbabilities As New Dictionary(Of List(Of String), Double)
            Public Sub New(trainingData As List(Of String), n As Integer)
                Me.n = n
                TrainModel(trainingData)
            End Sub
            Private Sub TrainModel(trainingData As List(Of String))
                ' Preprocess training data and tokenize into wordgrams
                Dim wordgrams As New List(Of List(Of String))
                For Each sentence As String In trainingData
                    Dim tokens() As String = sentence.Split(" "c)
                    For i As Integer = 0 To tokens.Length - n
                        Dim wordgram As List(Of String) = tokens.Skip(i).Take(n).ToList()
                        wordgrams.Add(wordgram)
                    Next
                Next

                ' Count wordgrams
                For Each wordgram As List(Of String) In wordgrams
                    If wordgramCounts.ContainsKey(wordgram) Then
                        wordgramCounts(wordgram) += 1
                    Else
                        wordgramCounts.Add(wordgram, 1)
                    End If
                Next

                ' Calculate wordgram probabilities
                Dim totalCount As Integer = wordgramCounts.Values.Sum()
                For Each wordgram As List(Of String) In wordgramCounts.Keys
                    Dim count As Integer = wordgramCounts(wordgram)
                    Dim probability As Double = count / totalCount
                    wordgramProbabilities.Add(wordgram, probability)
                Next
            End Sub
            Private Function GenerateNextWord(wordgram As List(Of String)) As String
                Dim random As New Random()
                Dim candidates As New List(Of String)
                Dim probabilities As New List(Of Double)

                ' Collect candidate words and their probabilities
                For Each candidateWordgram As List(Of String) In wordgramCounts.Keys
                    If candidateWordgram.GetRange(0, n - 1).SequenceEqual(wordgram) Then
                        Dim candidateWord As String = candidateWordgram.Last()
                        Dim probability As Double = wordgramProbabilities(candidateWordgram)
                        candidates.Add(candidateWord)
                        probabilities.Add(probability)
                    End If
                Next

                ' Randomly select a candidate word based on probabilities
                Dim totalProbability As Double = probabilities.Sum()
                Dim randomValue As Double = random.NextDouble() * totalProbability
                Dim cumulativeProbability As Double = 0

                For i As Integer = 0 To candidates.Count - 1
                    cumulativeProbability += probabilities(i)
                    If randomValue <= cumulativeProbability Then
                        Return candidates(i)
                    End If
                Next

                Return ""
            End Function
            Public Function GenerateSentence() As String
                Dim sentence As New List(Of String)
                Dim random As New Random()

                ' Start the sentence with a random wordgram
                Dim randomIndex As Integer = random.Next(0, wordgramCounts.Count)
                Dim currentWordgram As List(Of String) = wordgramCounts.Keys(randomIndex)
                sentence.AddRange(currentWordgram)

                ' Generate subsequent words based on wordgram probabilities
                While wordgramCounts.ContainsKey(currentWordgram)
                    Dim nextWord As String = GenerateNextWord(currentWordgram)
                    If nextWord = "" Then
                        Exit While
                    End If
                    sentence.Add(nextWord)

                    ' Backoff to lower-order wordgrams if necessary
                    If currentWordgram.Count > 1 Then
                        currentWordgram.RemoveAt(0)
                    Else
                        Exit While
                    End If
                    currentWordgram.Add(nextWord)
                End While

                Return String.Join(" ", sentence)
            End Function
            Private Sub Train(trainingData As List(Of String))
                ' Preprocess training data and tokenize into wordgrams
                Dim wordgrams As New List(Of List(Of String))
                For Each sentence As String In trainingData
                    Dim tokens() As String = sentence.Split(" "c)
                    For i As Integer = 0 To tokens.Length - n
                        Dim wordgram As List(Of String) = tokens.Skip(i).Take(n).ToList()
                        wordgrams.Add(wordgram)
                    Next
                Next

                ' Count wordgrams
                For Each wordgram As List(Of String) In wordgrams
                    If wordgramCounts.ContainsKey(wordgram) Then
                        wordgramCounts(wordgram) += 1
                    Else
                        wordgramCounts.Add(wordgram, 1)
                    End If
                Next

                ' Calculate wordgram probabilities based on frequency-based distribution
                For Each wordgram As List(Of String) In wordgramCounts.Keys
                    Dim count As Integer = wordgramCounts(wordgram)
                    Dim order As Integer = wordgram.Count

                    ' Calculate the frequency threshold for higher-order n-grams
                    Dim frequencyThreshold As Integer = 5 ' Set your desired threshold
                    If order = n AndAlso count >= frequencyThreshold Then
                        wordgramProbabilities.Add(wordgram, count)
                    ElseIf order < n AndAlso count >= frequencyThreshold Then
                        ' Assign the frequency to lower-order n-grams
                        Dim lowerOrderWordgram As List(Of String) = wordgram.Skip(1).ToList()
                        If wordgramProbabilities.ContainsKey(lowerOrderWordgram) Then
                            wordgramProbabilities(lowerOrderWordgram) += count
                        Else
                            wordgramProbabilities.Add(lowerOrderWordgram, count)
                        End If
                    End If
                Next

                ' Normalize probabilities within each order
                For order As Integer = 1 To n
                    Dim totalProbability As Double = 0
                    For Each wordgram As List(Of String) In wordgramProbabilities.Keys.ToList()
                        If wordgram.Count = order Then
                            totalProbability += wordgramProbabilities(wordgram)
                        End If
                    Next
                    For Each wordgram As List(Of String) In wordgramProbabilities.Keys.ToList()
                        If wordgram.Count = order Then
                            wordgramProbabilities(wordgram) /= totalProbability
                        End If
                    Next
                Next
            End Sub


        End Class
        Public Class Co_Occurrence_Matrix
            Public Shared Function PrintOccurrenceMatrix(ByRef coOccurrenceMatrix As Dictionary(Of String, Dictionary(Of String, Integer)), entityList As List(Of String)) As String
                ' Prepare the header row
                Dim headerRow As String = "|               |"

                For Each entity As String In entityList
                    If coOccurrenceMatrix.ContainsKey(entity) Then
                        headerRow &= $" [{entity}] ({coOccurrenceMatrix(entity).Count}) |"
                    End If
                Next

                Dim str As String = ""
                ' Print the header row
                Console.WriteLine(headerRow)

                str &= headerRow & vbNewLine
                ' Print the co-occurrence matrix
                For Each entity As String In coOccurrenceMatrix.Keys
                    Dim rowString As String = $"| [{entity}] ({coOccurrenceMatrix(entity).Count})        |"

                    For Each coOccurringEntity As String In entityList
                        Dim count As Integer = 0
                        If coOccurrenceMatrix(entity).ContainsKey(coOccurringEntity) Then
                            count = coOccurrenceMatrix(entity)(coOccurringEntity)
                        End If
                        rowString &= $"{count.ToString().PadLeft(7)} "
                    Next

                    Console.WriteLine(rowString)
                    str &= rowString & vbNewLine
                Next
                Return str
            End Function

            ''' <summary>
            ''' The co-occurrence matrix shows the frequency of co-occurrences between different entities in the given text. Each row represents an entity, and each column represents another entity. The values in the matrix indicate how many times each entity appeared within the specified window size of the other entities. A value of 0 means that the two entities did not co-occur within the given window size.
            ''' </summary>
            ''' <param name="text"></param>
            ''' <param name="entityList"></param>
            ''' <param name="windowSize"></param>
            ''' <returns></returns>
            Public Shared Function iCoOccurrenceMatrix(text As String, entityList As List(Of String), windowSize As Integer) As Dictionary(Of String, Dictionary(Of String, Integer))
                Dim coOccurrenceMatrix As New Dictionary(Of String, Dictionary(Of String, Integer))

                Dim words() As String = text.ToLower().Split(" "c) ' Convert the text to lowercase here
                For i As Integer = 0 To words.Length - 1
                    If entityList.Contains(words(i)) Then
                        Dim entity As String = words(i)
                        If Not coOccurrenceMatrix.ContainsKey(entity) Then
                            coOccurrenceMatrix(entity) = New Dictionary(Of String, Integer)()
                        End If

                        For j As Integer = i - windowSize To i + windowSize
                            If j >= 0 AndAlso j < words.Length AndAlso i <> j AndAlso entityList.Contains(words(j)) Then
                                Dim coOccurringEntity As String = words(j)
                                If Not coOccurrenceMatrix(entity).ContainsKey(coOccurringEntity) Then
                                    coOccurrenceMatrix(entity)(coOccurringEntity) = 0
                                End If

                                coOccurrenceMatrix(entity)(coOccurringEntity) += 1
                            End If
                        Next
                    End If
                Next

                Return coOccurrenceMatrix
            End Function

            ''' <summary>
            ''' The PMI matrix measures the statistical association or co-occurrence patterns between different entities in the text. It is calculated based on the co-occurrence matrix. PMI values are used to assess how much more likely two entities are to co-occur together than they would be if their occurrences were independent of each other.
            '''
            '''  positive PMI value indicates that the two entities are likely To co-occur more often than expected by chance, suggesting a positive association between them.
            '''  PMI value Of 0 means that the two entities co-occur As often As expected by chance, suggesting no significant association.
            '''  negative PMI value indicates that the two entities are less likely To co-occur than expected by chance, suggesting a negative association Or avoidance.
            ''' </summary>
            ''' <param name="coOccurrenceMatrix"></param>
            ''' <returns></returns>
            Public Shared Function CalculatePMI(coOccurrenceMatrix As Dictionary(Of String, Dictionary(Of String, Integer))) As Dictionary(Of String, Dictionary(Of String, Double))
                Dim pmiMatrix As New Dictionary(Of String, Dictionary(Of String, Double))

                For Each entity As String In coOccurrenceMatrix.Keys
                    Dim entityOccurrences As Integer = coOccurrenceMatrix(entity).Sum(Function(kv) kv.Value)

                    If Not pmiMatrix.ContainsKey(entity) Then
                        pmiMatrix(entity) = New Dictionary(Of String, Double)()
                    End If

                    For Each coOccurringEntity As String In coOccurrenceMatrix(entity).Keys
                        Dim coOccurringEntityOccurrences As Integer = coOccurrenceMatrix(entity)(coOccurringEntity)

                        Dim pmi As Double = Math.Log((coOccurringEntityOccurrences * coOccurrenceMatrix.Count) / (entityOccurrences * coOccurrenceMatrix(coOccurringEntity).Sum(Function(kv) kv.Value)), 2)
                        pmiMatrix(entity)(coOccurringEntity) = pmi
                    Next
                Next

                Return pmiMatrix
            End Function
            Public Shared Function PrintOccurrenceMatrix(ByRef coOccurrenceMatrix As Dictionary(Of String, Dictionary(Of String, Double)), entityList As List(Of String)) As String
                ' Prepare the header row
                Dim headerRow As String = "|               |"

                For Each entity As String In entityList
                    If coOccurrenceMatrix.ContainsKey(entity) Then
                        headerRow &= $" [{entity}] ({coOccurrenceMatrix(entity).Count}) |"
                    End If
                Next

                Dim str As String = ""
                ' Print the header row
                Console.WriteLine(headerRow)

                str &= headerRow & vbNewLine
                ' Print the co-occurrence matrix
                For Each entity As String In coOccurrenceMatrix.Keys
                    Dim rowString As String = $"| [{entity}] ({coOccurrenceMatrix(entity).Count})        |"

                    For Each coOccurringEntity As String In entityList
                        Dim count As Integer = 0
                        If coOccurrenceMatrix(entity).ContainsKey(coOccurringEntity) Then
                            count = coOccurrenceMatrix(entity)(coOccurringEntity)
                        End If
                        rowString &= $"{count.ToString().PadLeft(7)} "
                    Next

                    Console.WriteLine(rowString)
                    str &= rowString & vbNewLine
                Next
                Return str
            End Function
            ''' <summary>
            ''' The PMI matrix measures the statistical association or co-occurrence patterns between different entities in the text. It is calculated based on the co-occurrence matrix. PMI values are used to assess how much more likely two entities are to co-occur together than they would be if their occurrences were independent of each other.
            '''
            '''  positive PMI value indicates that the two entities are likely To co-occur more often than expected by chance, suggesting a positive association between them.
            '''  PMI value Of 0 means that the two entities co-occur As often As expected by chance, suggesting no significant association.
            '''  negative PMI value indicates that the two entities are less likely To co-occur than expected by chance, suggesting a negative association Or avoidance.
            ''' </summary>
            ''' <param name="coOccurrenceMatrix"></param>
            ''' <returns></returns>
            Public Shared Function GetPM_Matrix(ByRef coOccurrenceMatrix As Dictionary(Of String, Dictionary(Of String, Integer))) As Dictionary(Of String, Dictionary(Of String, Double))

                Dim pmiMatrix As Dictionary(Of String, Dictionary(Of String, Double)) = CalculatePMI(coOccurrenceMatrix)
                Return pmiMatrix

            End Function


        End Class
        Public Class Word2WordMatrix
            Private matrix As Dictionary(Of String, Dictionary(Of String, Integer))

            Public Sub New()
                matrix = New Dictionary(Of String, Dictionary(Of String, Integer))
            End Sub
            Public Shared Function CreateDataGridView(matrix As Dictionary(Of String, Dictionary(Of String, Double))) As DataGridView
                Dim dataGridView As New DataGridView()
                dataGridView.Dock = DockStyle.Fill
                dataGridView.AutoGenerateColumns = False
                dataGridView.AllowUserToAddRows = False

                ' Add columns to the DataGridView
                Dim wordColumn As New DataGridViewTextBoxColumn()
                wordColumn.HeaderText = "Word"
                wordColumn.DataPropertyName = "Word"
                dataGridView.Columns.Add(wordColumn)

                For Each contextWord As String In matrix.Keys
                    Dim contextColumn As New DataGridViewTextBoxColumn()
                    contextColumn.HeaderText = contextWord
                    contextColumn.DataPropertyName = contextWord
                    dataGridView.Columns.Add(contextColumn)
                Next

                ' Populate the DataGridView with the matrix data
                For Each word As String In matrix.Keys
                    Dim rowValues As New List(Of Object)
                    rowValues.Add(word)

                    For Each contextWord As String In matrix.Keys
                        Dim count As Object = If(matrix(word).ContainsKey(contextWord), matrix(word)(contextWord), 0)
                        rowValues.Add(count)
                    Next

                    dataGridView.Rows.Add(rowValues.ToArray())
                Next

                Return dataGridView
            End Function

            Public Shared Function CreateDataGridView(matrix As Dictionary(Of String, Dictionary(Of String, Integer))) As DataGridView
                Dim dataGridView As New DataGridView()
                dataGridView.Dock = DockStyle.Fill
                dataGridView.AutoGenerateColumns = False
                dataGridView.AllowUserToAddRows = False

                ' Add columns to the DataGridView
                Dim wordColumn As New DataGridViewTextBoxColumn()
                wordColumn.HeaderText = "Word"
                wordColumn.DataPropertyName = "Word"
                dataGridView.Columns.Add(wordColumn)

                For Each contextWord As String In matrix.Keys
                    Dim contextColumn As New DataGridViewTextBoxColumn()
                    contextColumn.HeaderText = contextWord
                    contextColumn.DataPropertyName = contextWord
                    dataGridView.Columns.Add(contextColumn)
                Next

                ' Populate the DataGridView with the matrix data
                For Each word As String In matrix.Keys
                    Dim rowValues As New List(Of Object)()
                    rowValues.Add(word)

                    For Each contextWord As String In matrix.Keys
                        Dim count As Integer = If(matrix(word).ContainsKey(contextWord), matrix(word)(contextWord), 0)
                        rowValues.Add(count)
                    Next

                    dataGridView.Rows.Add(rowValues.ToArray())
                Next

                Return dataGridView
            End Function

            Public Sub AddDocument(document As String, contextWindow As Integer)
                Dim words As String() = document.Split({" "c}, StringSplitOptions.RemoveEmptyEntries)

                For i As Integer = 0 To words.Length - 1
                    Dim currentWord As String = words(i)

                    If Not matrix.ContainsKey(currentWord) Then
                        matrix(currentWord) = New Dictionary(Of String, Integer)()
                    End If

                    For j As Integer = Math.Max(0, i - contextWindow) To Math.Min(words.Length - 1, i + contextWindow)
                        If i <> j Then
                            Dim contextWord As String = words(j)

                            If Not matrix(currentWord).ContainsKey(contextWord) Then
                                matrix(currentWord)(contextWord) = 0
                            End If

                            matrix(currentWord)(contextWord) += 1
                        End If
                    Next
                Next
            End Sub
            Public Shared Sub Main()
                ' Fill the matrix with your data
                Dim documents As List(Of String) = New List(Of String)()
                documents.Add("This is the first document.")
                documents.Add("The second document is here.")
                documents.Add("And this is the third document.")

                Dim contextWindow As Integer = 1
                Dim matrixBuilder As New Word2WordMatrix()

                For Each document As String In documents
                    matrixBuilder.AddDocument(document, contextWindow)
                Next

                Dim wordWordMatrix As Dictionary(Of String, Dictionary(Of String, Integer)) = matrixBuilder.GetWordWordMatrix()

                ' Create the DataGridView control
                Dim dataGridView As DataGridView = Word2WordMatrix.CreateDataGridView(wordWordMatrix)

                ' Create a form and add the DataGridView to it
                Dim form As New Form()
                form.Text = "Word-Word Matrix"
                form.Size = New Size(800, 600)
                form.Controls.Add(dataGridView)

                ' Display the form
                Application.Run(form)
            End Sub
            Public Function GetWordWordMatrix() As Dictionary(Of String, Dictionary(Of String, Integer))
                Return matrix
            End Function
        End Class

    End Namespace
    Namespace Readers

        Public Class CorpusCategorizer
            Private categoryMap As Dictionary(Of String, List(Of String))

            Public Sub New()
                categoryMap = New Dictionary(Of String, List(Of String))()
            End Sub

            Public Sub AddCategory(category As String, keywords As List(Of String))
                If Not categoryMap.ContainsKey(category) Then
                    categoryMap.Add(category, keywords)
                Else
                    categoryMap(category).AddRange(keywords)
                End If
            End Sub

            Public Function CategorizeDocument(document As String) As List(Of String)
                Dim categories As New List(Of String)()

                For Each category As KeyValuePair(Of String, List(Of String)) In categoryMap
                    Dim categoryKeywords As List(Of String) = category.Value
                    For Each keyword As String In categoryKeywords
                        If document.Contains(keyword) Then
                            categories.Add(category.Key)
                            Exit For
                        End If
                    Next
                Next

                Return categories
            End Function

        End Class

        Public Class CorpusCreator
            Public maxSequenceLength As Integer = 0
            Public Vocabulary As New List(Of String)

            Public Sub New(vocabulary As List(Of String), maxSeqLength As Integer)
                If vocabulary Is Nothing Then
                    Throw New ArgumentNullException(NameOf(vocabulary))
                End If

                Me.Vocabulary = vocabulary
                Me.maxSequenceLength = maxSeqLength
            End Sub

            ''' <summary>
            ''' Generates a classification dataset by labeling text data with classes.
            ''' </summary>
            ''' <param name="data">The list of processed text data chunks.</param>
            ''' <param name="classes">The list of class labels.</param>
            ''' <returns>A list of input-output pairs for classification.</returns>
            Public Shared Function GenerateClassificationDataset(data As List(Of String), classes As List(Of String)) As List(Of Tuple(Of String, String))
                Dim dataset As New List(Of Tuple(Of String, String))

                For Each chunk As String In data
                    For Each [class] As String In classes
                        If IsTermPresent([class], chunk) Then
                            dataset.Add(Tuple.Create(chunk, [class]))
                            Exit For
                        End If
                    Next
                Next

                Return dataset
            End Function

            ''' <summary>
            ''' Creates a predictive dataset for training machine learning models.
            ''' </summary>
            ''' <param name="data">The list of processed text data chunks.</param>
            ''' <param name="windowSize">The size of the input window for predictive modeling.</param>
            ''' <returns>A list of input-output pairs for predictive modeling.</returns>
            Public Shared Function GeneratePredictiveDataset(data As List(Of String), windowSize As Integer) As List(Of String())
                Dim dataset As New List(Of String())

                For Each chunk As String In data
                    Dim words As String() = chunk.Split({" "}, StringSplitOptions.RemoveEmptyEntries)
                    For i As Integer = 0 To words.Length - windowSize
                        Dim inputWords As String() = words.Skip(i).Take(windowSize).ToArray()
                        Dim targetWord As String = words(i + windowSize)
                        dataset.Add(New String() {String.Join(" ", inputWords), targetWord})
                    Next
                Next

                Return dataset
            End Function

            Public Shared Function GenerateTransformerBatches(data As List(Of String), batch_size As Integer, seq_length As Integer) As List(Of Tuple(Of List(Of String), List(Of String)))
                Dim batches As New List(Of Tuple(Of List(Of String), List(Of String)))

                For i As Integer = 0 To data.Count - batch_size Step batch_size
                    Dim batchInputs As New List(Of String)
                    Dim batchTargets As New List(Of String)

                    For j As Integer = i To i + batch_size - 1
                        Dim words As String() = data(j).Split({" "}, StringSplitOptions.RemoveEmptyEntries)
                        If words.Length > seq_length Then
                            batchInputs.Add(String.Join(" ", words.Take(seq_length)))
                            batchTargets.Add(String.Join(" ", words.Skip(1).Take(seq_length)))
                        End If
                    Next

                    If batchInputs.Count > 0 Then
                        batches.Add(Tuple.Create(batchInputs, batchTargets))
                    End If
                Next

                Return batches
            End Function

            ''' <summary>
            ''' Checks if a specific term (entity or keyword) is present in the processed text data.
            ''' </summary>
            ''' <param name="term">The term to check.</param>
            ''' <param name="data">The processed text data.</param>
            ''' <returns>True if the term is present; otherwise, false.</returns>
            Public Shared Function IsTermPresent(term As String, data As String) As Boolean
                Return data.ToLower().Contains(term.ToLower())
            End Function

            Public Function CreateClassificationDataset(data As List(Of String), classes As List(Of String)) As List(Of Tuple(Of String, String))
                Dim dataset As New List(Of Tuple(Of String, String))

                For Each chunk As String In data
                    For Each iclass As String In classes
                        If IsTermPresent(iclass, chunk) Then
                            dataset.Add(Tuple.Create(chunk, iclass))
                            Exit For
                        End If
                    Next
                Next

                Return dataset
            End Function

            ''' <summary>
            ''' Creates batches of data for training.
            ''' </summary>
            ''' <param name="Corpus">The training data as a list of string sequences.</param>
            ''' <param name="batchSize">The size of each batch.</param>
            Public Sub CreateData(ByRef Corpus As List(Of List(Of String)), ByRef batchSize As Integer)
                For batchStart As Integer = 0 To Corpus.Count - 1 Step batchSize
                    Dim batchEnd As Integer = Math.Min(batchStart + batchSize - 1, Corpus.Count - 1)
                    Dim batchInputs As List(Of List(Of Integer)) = GetBatchInputs(Corpus, batchStart, batchEnd)
                    Dim batchTargets As List(Of List(Of Integer)) = GetBatchTargets(Corpus, batchStart, batchEnd)

                    ' Perform further operations on the batches
                Next

            End Sub

            Public Function CreatePredictiveDataset(data As List(Of String), windowSize As Integer) As List(Of String())
                Dim dataset As New List(Of String())

                For Each chunk As String In data
                    Dim words As String() = chunk.Split({" "}, StringSplitOptions.RemoveEmptyEntries)
                    For i As Integer = 0 To words.Length - windowSize
                        Dim inputWords As String() = words.Skip(i).Take(windowSize).ToArray()
                        Dim targetWord As String = words(i + windowSize)
                        dataset.Add(New String() {String.Join(" ", inputWords), targetWord})
                    Next
                Next

                Return dataset
            End Function

            ''' <summary>
            ''' Converts a batch of data from a list of string sequences to a list of integer sequences.
            ''' </summary>
            ''' <param name="data">The input data as a list of string sequences.</param>
            ''' <param name="startIndex">The starting index of the batch.</param>
            ''' <param name="endIndex">The ending index of the batch.</param>
            ''' <returns>A list of integer sequences representing the batch inputs.</returns>
            Public Function GetBatchInputs(data As List(Of List(Of String)),
                                       startIndex As Integer,
                                       endIndex As Integer) As List(Of List(Of Integer))
                Dim batchInputs As New List(Of List(Of Integer))

                For i As Integer = startIndex To endIndex
                    Dim sequence As List(Of String) = data(i)

                    ' Convert words to corresponding indices
                    Dim indices As List(Of Integer) = ConvertWordsToIndices(sequence)

                    ' Pad or truncate sequence to the maximum length
                    indices = PadOrTruncateSequence(indices, maxSequenceLength)

                    ' Add the sequence to the batch
                    batchInputs.Add(indices)
                Next

                Return batchInputs
            End Function

            ''' <summary>
            ''' Converts a batch of data from a list of string sequences to a list of integer sequences as targets.
            ''' </summary>
            ''' <param name="data">The input data as a list of string sequences.</param>
            ''' <param name="startIndex">The starting index of the batch.</param>
            ''' <param name="endIndex">The ending index of the batch.</param>
            ''' <returns>A list of integer sequences representing the batch targets.</returns>
            Public Function GetBatchTargets(data As List(Of List(Of String)), startIndex As Integer, endIndex As Integer) As List(Of List(Of Integer))
                Dim batchTargets As New List(Of List(Of Integer))

                For i As Integer = startIndex To endIndex
                    Dim sequence As List(Of String) = data(i)

                    ' Convert words to corresponding indices
                    Dim indices As List(Of Integer) = ConvertWordsToIndices(sequence)

                    ' Shift the sequence to get the target sequence
                    Dim targetIndices As List(Of Integer) = ShiftSequence(indices)

                    ' Pad or truncate sequence to the maximum length
                    targetIndices = PadOrTruncateSequence(targetIndices, maxSequenceLength)

                    ' Add the target sequence to the batch
                    batchTargets.Add(targetIndices)
                Next

                Return batchTargets
            End Function

            ''' <summary>
            ''' Pads or truncates a sequence to a specified length.
            ''' </summary>
            ''' <param name="sequence">The input sequence.</param>
            ''' <param name="length">The desired length.</param>
            ''' <returns>The padded or truncated sequence.</returns>
            Public Function PadOrTruncateSequence(sequence As List(Of Integer), length As Integer) As List(Of Integer)
                If sequence.Count < length Then
                    ' Pad the sequence with a special padding token
                    sequence.AddRange(Enumerable.Repeat(Vocabulary.IndexOf("PAD"), length - sequence.Count))
                ElseIf sequence.Count > length Then
                    ' Truncate the sequence to the desired length
                    sequence = sequence.GetRange(0, length)
                End If

                Return sequence
            End Function

            ''' <summary>
            ''' Shifts a sequence to the right and adds a special token at the beginning.
            ''' </summary>
            ''' <param name="sequence">The input sequence.</param>
            ''' <returns>The shifted sequence.</returns>
            Public Function ShiftSequence(sequence As List(Of Integer)) As List(Of Integer)
                ' Shifts the sequence to the right and adds a special token at the beginning
                Dim shiftedSequence As New List(Of Integer) From {Vocabulary.IndexOf("START")}

                For i As Integer = 0 To sequence.Count - 1
                    shiftedSequence.Add(sequence(i))
                Next

                Return shiftedSequence
            End Function

            ''' <summary>
            ''' Converts a list of words to a list of corresponding indices based on the vocabulary.
            ''' </summary>
            ''' <param name="words">The list of words to convert.</param>
            ''' <returns>A list of corresponding indices.</returns>
            Private Function ConvertWordsToIndices(words As List(Of String)) As List(Of Integer)
                Dim indices As New List(Of Integer)

                For Each word As String In words
                    If Vocabulary.Contains(word) Then
                        indices.Add(Vocabulary.IndexOf(word))
                    Else
                    End If
                Next

                Return indices
            End Function

        End Class

        Public Class ModelCorpusReader
            Private categoryMap As Dictionary(Of String, List(Of String))
            Private corpusFiles As List(Of String)
            Private corpusRoot As String

            Public Sub New(corpusRootPath As String)
                corpusRoot = corpusRootPath
                corpusFiles = New List(Of String)()
                categoryMap = New Dictionary(Of String, List(Of String))
                LoadCorpusFiles()
            End Sub

            Public Sub AddCategory(category As String, keywords As List(Of String))
                If Not categoryMap.ContainsKey(category) Then
                    categoryMap.Add(category, keywords)
                Else
                    categoryMap(category).AddRange(keywords)
                End If
            End Sub

            Public Function CategorizeDocument(document As String) As List(Of String)
                Dim categories As New List(Of String)()

                For Each category As KeyValuePair(Of String, List(Of String)) In categoryMap
                    Dim categoryKeywords As List(Of String) = category.Value
                    For Each keyword As String In categoryKeywords
                        If document.Contains(keyword) Then
                            categories.Add(category.Key)
                            Exit For
                        End If
                    Next
                Next

                Return categories
            End Function

            Public Function GetWordsFromWordList(wordListFilePath As String) As List(Of String)
                Dim wordList As New List(Of String)()

                Using reader As New StreamReader(wordListFilePath)
                    While Not reader.EndOfStream
                        Dim line As String = reader.ReadLine()
                        If Not String.IsNullOrEmpty(line) Then
                            wordList.Add(line.Trim())
                        End If
                    End While
                End Using

                Return wordList
            End Function

            Public Function TaggedSentences() As List(Of List(Of Tuple(Of String, String)))
                Dim itaggedSentences As New List(Of List(Of Tuple(Of String, String)))()

                For Each file As String In corpusFiles
                    Dim taggedSentencesInFile As New List(Of Tuple(Of String, String))()

                    Using reader As New StreamReader(file)
                        While Not reader.EndOfStream
                            Dim line As String = reader.ReadLine()
                            Dim wordsTags As String() = line.Split(" ")

                            For Each wordTag As String In wordsTags
                                Dim parts As String() = wordTag.Split("/")
                                If parts.Length = 2 Then
                                    Dim word As String = parts(0)
                                    Dim tag As String = parts(1)
                                    taggedSentencesInFile.Add(New Tuple(Of String, String)(word, tag))
                                End If
                            Next
                        End While
                    End Using

                    itaggedSentences.Add(taggedSentencesInFile)
                Next

                Return itaggedSentences
            End Function

            Private Sub LoadCorpusFiles()
                corpusFiles.Clear()
                If Directory.Exists(corpusRoot) Then
                    corpusFiles.AddRange(Directory.GetFiles(corpusRoot))
                End If
            End Sub

        End Class

        Public Class TaggedCorpusReader
            Private corpusFiles As List(Of String)
            Private corpusRoot As String

            Public Sub New(corpusRootPath As String)
                corpusRoot = corpusRootPath
                corpusFiles = New List(Of String)
                LoadCorpusFiles()
            End Sub

            Public Function TaggedSentences() As List(Of List(Of Tuple(Of String, String)))
                Dim itaggedSentences As New List(Of List(Of Tuple(Of String, String)))()

                For Each file As String In corpusFiles
                    Dim taggedSentencesInFile As New List(Of Tuple(Of String, String))()

                    Using reader As New StreamReader(file)
                        While Not reader.EndOfStream
                            Dim line As String = reader.ReadLine()
                            Dim wordsTags As String() = line.Split(" ")

                            For Each wordTag As String In wordsTags
                                Dim parts As String() = wordTag.Split("/")
                                If parts.Length = 2 Then
                                    Dim word As String = parts(0)
                                    Dim tag As String = parts(1)
                                    taggedSentencesInFile.Add(New Tuple(Of String, String)(word, tag))
                                End If
                            Next
                        End While
                    End Using

                    itaggedSentences.Add(taggedSentencesInFile)
                Next

                Return itaggedSentences
            End Function

            Private Sub LoadCorpusFiles()
                corpusFiles.Clear()
                If Directory.Exists(corpusRoot) Then
                    corpusFiles.AddRange(Directory.GetFiles(corpusRoot))
                End If
            End Sub

        End Class
        Public Class WordListCorpusReader
            Private wordList As List(Of String)

            Public Sub New(filePath As String)
                wordList = New List(Of String)()
                ReadWordList(filePath)
            End Sub

            Public Function GetWords() As List(Of String)
                Return wordList
            End Function

            Private Sub ReadWordList(filePath As String)
                Using reader As New StreamReader(filePath)
                    While Not reader.EndOfStream
                        Dim line As String = reader.ReadLine()
                        If Not String.IsNullOrEmpty(line) Then
                            wordList.Add(line.Trim())
                        End If
                    End While
                End Using
            End Sub

        End Class
    End Namespace
    Namespace Trees
        Namespace BeliefTree
            Public Class Node
                Public Property Name As String
                Public Property States As List(Of String)
                Public Property Parents As List(Of Node)
                Public Property CPT As ConditionalProbabilityTable

                Public Sub New(name As String, states As List(Of String))
                    Me.Name = name
                    Me.States = states
                    Parents = New List(Of Node)()
                End Sub
            End Class
            Public Class ConditionalProbabilityTable
                Public Property Node As Node
                Public Property Values As Dictionary(Of List(Of String), Double)

                Public Sub New(node As Node)
                    Me.Node = node
                    Values = New Dictionary(Of List(Of String), Double)()
                End Sub

                Public Sub SetEntry(parentStates As List(Of String), value As Double)
                    Values(parentStates) = value
                End Sub
            End Class
            Public Class InferenceEngine
                Public Sub New(network As BeliefNetwork)
                    Me.Network = network
                End Sub

                Public Property Network As BeliefNetwork

                Public Function CalculateConditionalProbability(node As Node, state As String) As Double
                    Dim totalProbability As Double = 0.0
                    Dim parentNodes = node.Parents

                    For Each parentState In CartesianProduct(parentNodes.Select(Function(n) n.States))
                        Dim evidence As New Dictionary(Of Node, String)()
                        For i = 0 To parentNodes.Count - 1
                            evidence(parentNodes(i)) = parentState(i)
                        Next

                        Dim jointProbability As Double = CalculateJointProbability(evidence)
                        totalProbability += jointProbability
                    Next

                    Dim evidenceWithState As New Dictionary(Of Node, String)()
                    evidenceWithState(node) = state

                    Dim conditionalProbability = CalculateJointProbability(evidenceWithState) / totalProbability
                    Return conditionalProbability
                End Function

                Private Function CalculateJointProbability(evidence As Dictionary(Of Node, String)) As Double
                    Dim jointProbability As Double = 1.0

                    For Each node In Network.Nodes
                        Dim nodeProbability As Double

                        If evidence.ContainsKey(node) Then
                            Dim parentStates = node.Parents.Select(Function(parent) evidence(parent))
                            nodeProbability = node.CPT.Values(parentStates.ToList())
                        Else
                            Dim parentStates = node.Parents.Select(Function(parent) evidence(parent))
                            nodeProbability = node.CPT.Values(parentStates.ToList())
                        End If

                        jointProbability *= nodeProbability
                    Next

                    Return jointProbability
                End Function

                Private Iterator Function CartesianProduct(sequences As IEnumerable(Of IEnumerable(Of String))) As IEnumerable(Of List(Of String))
                    Dim enumerators = sequences.Select(Function(seq) seq.GetEnumerator()).ToArray()
                    Dim values = New List(Of String)(enumerators.Length)

                    While True
                        values.Clear()

                        For i = 0 To enumerators.Length - 1
                            Dim enumerator = enumerators(i)
                            If Not enumerator.MoveNext() Then
                                enumerator.Reset()
                                enumerator.MoveNext()
                            End If
                            values.Add(enumerator.Current)
                        Next

                        Yield values.ToList()
                    End While
                End Function
            End Class
            Public Class BeliefNetwork
                Public Property Nodes As List(Of Node)
                Public Sub LoadTrainingData(trainingData As Dictionary(Of String, Dictionary(Of List(Of String), Double)))
                    For Each entry In trainingData
                        Dim nodeName As String = entry.Key
                        Dim values As Dictionary(Of List(Of String), Double) = entry.Value

                        DefineCPT(nodeName, values)
                    Next
                End Sub

                Public Function CreateEvidence(nodeName As String, state As String) As Dictionary(Of Node, String)
                    Dim evidence As New Dictionary(Of Node, String)()
                    Dim node As Node = Nodes.Find(Function(n) n.Name = nodeName)
                    evidence.Add(node, state)
                    Return evidence
                End Function

                Public Function GetNodeByName(nodeName As String) As Node
                    Return Nodes.Find(Function(n) n.Name = nodeName)
                End Function

                Public Function PredictWithEvidence(targetNodeName As String, evidence As Dictionary(Of Node, String)) As String
                    Dim targetNode As Node = GetNodeByName(targetNodeName)
                    Return Predict(targetNode, evidence)
                End Function
                Public Sub LoadTrainingDataFromFile(filePath As String)
                    Dim trainingData As Dictionary(Of String, Dictionary(Of List(Of String), Double)) = LoadTrainingData(filePath)
                    For Each entry In trainingData
                        Dim nodeName As String = entry.Key
                        Dim values As Dictionary(Of List(Of String), Double) = entry.Value

                        DefineCPT(nodeName, values)
                    Next
                End Sub
                Public Sub ExportToFile(filePath As String)
                    Dim lines As New List(Of String)()

                    For Each node In Nodes
                        lines.Add(node.Name)
                        For Each state In node.States
                            If node.Parents.Count > 0 Then
                                Dim parentStates As New List(Of String)()
                                For Each parent In node.Parents
                                    parentStates.Add(parent.Name)
                                Next
                                parentStates.Add(state)
                                lines.Add(String.Join(" ", parentStates) & " " & node.CPT.Values(parentStates))
                            Else
                                lines.Add(state & " " & node.CPT.Values(New List(Of String)()))
                            End If
                        Next
                    Next

                    File.WriteAllLines(filePath, lines)
                    Console.WriteLine("Network exported to " & filePath)
                End Sub
                Public Sub DisplayAsTree()
                    Dim form As New Form()
                    Dim treeView As New TreeView()
                    treeView.Dock = DockStyle.Fill
                    form.Controls.Add(treeView)

                    For Each node In Nodes
                        Dim treeNode As New TreeNode(node.Name)

                        If node.Parents.Count > 0 Then
                            Dim parentNodes As New List(Of String)()
                            For Each parent In node.Parents
                                parentNodes.Add(parent.Name)
                            Next

                            Dim parentNode As TreeNode = FindOrCreateParentNode(treeView.Nodes, parentNodes)
                            parentNode.Nodes.Add(treeNode)
                        Else
                            treeView.Nodes.Add(treeNode)
                        End If

                        For Each state In node.States
                            Dim stateNode As New TreeNode(state & " (" & node.CPT.Values(New List(Of String) From {state}) & ")")
                            treeNode.Nodes.Add(stateNode)
                        Next
                    Next

                    Application.Run(form)
                End Sub

                Public Shared Sub DisplayAsTree(ByRef Network As BeliefNetwork)
                    Dim form As New Form()
                    Dim treeView As New TreeView()
                    treeView.Dock = DockStyle.Fill
                    form.Controls.Add(treeView)

                    For Each node In Network.Nodes
                        Dim treeNode As New TreeNode(node.Name)

                        If node.Parents.Count > 0 Then
                            Dim parentNodes As New List(Of String)()
                            For Each parent In node.Parents
                                parentNodes.Add(parent.Name)
                            Next

                            Dim parentNode As TreeNode = Network.FindOrCreateParentNode(treeView.Nodes, parentNodes)
                            parentNode.Nodes.Add(treeNode)
                        Else
                            treeView.Nodes.Add(treeNode)
                        End If

                        For Each state In node.States
                            Dim stateNode As New TreeNode(state & " (" & node.CPT.Values(New List(Of String) From {state}) & ")")
                            treeNode.Nodes.Add(stateNode)
                        Next
                    Next

                    Application.Run(form)
                End Sub


                Private Function FindOrCreateParentNode(collection As TreeNodeCollection, parentNodes As List(Of String)) As TreeNode
                    Dim parentNode As TreeNode = Nothing

                    For Each parentName In parentNodes
                        Dim node As TreeNode = collection.Find(parentName, False).FirstOrDefault()

                        If node IsNot Nothing Then
                            collection = node.Nodes
                            parentNode = node
                        Else
                            Dim newNode As New TreeNode(parentName)
                            If parentNode Is Nothing Then
                                collection.Add(newNode)
                            Else
                                parentNode.Nodes.Add(newNode)
                            End If
                            collection = newNode.Nodes
                            parentNode = newNode
                        End If
                    Next

                    Return parentNode
                End Function

                Public Shared Function LoadTrainingData(filePath As String) As Dictionary(Of String, Dictionary(Of List(Of String), Double))
                    Dim trainingData As New Dictionary(Of String, Dictionary(Of List(Of String), Double))()

                    If File.Exists(filePath) Then
                        Dim lines As String() = File.ReadAllLines(filePath)
                        Dim currentEntry As String = Nothing
                        Dim currentCPT As New Dictionary(Of List(Of String), Double)()

                        For Each line In lines
                            Dim parts As String() = Split(line, " "c, StringSplitOptions.RemoveEmptyEntries)

                            If parts.Length = 1 Then
                                ' Start of a new entry
                                If currentEntry IsNot Nothing Then
                                    trainingData.Add(currentEntry, currentCPT)
                                    currentCPT = New Dictionary(Of List(Of String), Double)()
                                End If

                                currentEntry = parts(0)
                            ElseIf parts.Length = 2 Then
                                ' CPT entry
                                Dim state As String = parts(0)
                                Dim probability As Double = Double.Parse(parts(1))
                                currentCPT.Add(New List(Of String) From {state}, probability)
                            ElseIf parts.Length > 2 Then
                                ' CPT entry with parent states
                                Dim states As New List(Of String)(parts.Length - 1)
                                For i As Integer = 0 To parts.Length - 2
                                    states.Add(parts(i))
                                Next
                                Dim probability As Double = Double.Parse(parts(parts.Length - 1))
                                currentCPT.Add(states, probability)
                            End If
                        Next

                        ' Add the last entry
                        If currentEntry IsNot Nothing Then
                            trainingData.Add(currentEntry, currentCPT)
                        End If
                    Else
                        Console.WriteLine("Training data file not found.")
                    End If

                    Return trainingData
                End Function
                Public Sub New()
                    Nodes = New List(Of Node)()
                End Sub

                Public Sub AddNode(node As Node)
                    Nodes.Add(node)
                End Sub
                Public Function Predict(targetNode As Node, evidence As Dictionary(Of Node, String)) As String
                    Dim engine As New InferenceEngine(Me)
                    Dim conditionalProbability As Double = engine.CalculateConditionalProbability(targetNode, evidence(targetNode))
                    Dim predictedState As String = If(conditionalProbability > 0.5, evidence(targetNode), GetOppositeState(targetNode, evidence(targetNode)))
                    Return predictedState
                End Function

                Private Function GetOppositeState(node As Node, state As String) As String
                    Return node.States.Find(Function(s) s <> state)
                End Function

                Public Sub DefineCPT(nodeName As String, values As Dictionary(Of List(Of String), Double))
                    Dim node = Nodes.Find(Function(n) n.Name = nodeName)
                    Dim cpt As New ConditionalProbabilityTable(node)
                    For Each entry In values
                        cpt.SetEntry(entry.Key, entry.Value)
                    Next
                    node.CPT = cpt
                End Sub
                Public Sub DisplayNetworkStructure()
                    Console.WriteLine("Network Structure:")
                    For Each node In Nodes
                        Console.WriteLine("Node: " & node.Name)
                        Console.WriteLine("Parents: " & String.Join(", ", node.Parents.Select(Function(parent) parent.Name)))
                        Console.WriteLine("CPT:")
                        For Each entry In node.CPT.Values
                            Console.WriteLine("  P(" & node.Name & " = " & String.Join(", ", entry.Key) & ") = " & entry.Value)
                        Next
                        Console.WriteLine()
                    Next
                End Sub
                Public Sub AddEdge(parentNode As Node, childNode As Node)
                    childNode.Parents.Add(parentNode)
                End Sub
            End Class
        End Namespace
    End Namespace
    Namespace VocabularyModelling
        Public Class VocabularyBuilder
            Private embeddingMatrix As Double(,)
            Private embeddingSize As Integer
            Private iterations As Integer
            Private Function GetEmbedding(index As Integer) As Double()
                If indexToWord.ContainsKey(index) Then
                    Dim vector(embeddingSize - 1) As Double
                    For i As Integer = 0 To embeddingSize - 1
                        vector(i) = embeddingMatrix(index, i)
                    Next
                    Return vector
                Else
                    Return Nothing
                End If
            End Function
            Public Function GenerateCooccurrenceMatrix(corpus As String(), windowSize As Integer) As Dictionary(Of String, Dictionary(Of String, Double))
                Dim matrix As New Dictionary(Of String, Dictionary(Of String, Double))

                For Each sentence In corpus
                    Dim words As String() = sentence.Split(" "c)
                    Dim length As Integer = words.Length

                    For i As Integer = 0 To length - 1
                        Dim targetWord As String = words(i)

                        If Not matrix.ContainsKey(targetWord) Then
                            matrix(targetWord) = New Dictionary(Of String, Double)
                        End If

                        For j As Integer = Math.Max(0, i - windowSize) To Math.Min(length - 1, i + windowSize)
                            If i = j Then
                                Continue For
                            End If

                            Dim contextWord As String = words(j)
                            Dim distance As Double = 1 / Math.Abs(i - j)

                            If matrix(targetWord).ContainsKey(contextWord) Then
                                matrix(targetWord)(contextWord) += distance
                            Else
                                matrix(targetWord)(contextWord) = distance
                            End If
                        Next
                    Next
                Next

                Return matrix
            End Function
            Public Model As New Dictionary(Of String, Dictionary(Of String, Double))
            Private windowSize As Integer
            Public Sub Train(corpus As String(), ByRef WindowSize As Integer, ByRef Iterations As Integer)
                BuildVocabulary(corpus.ToList)
                InitializeEmbeddings()
                Model = GenerateCooccurrenceMatrix(corpus, WindowSize)

                For iteration As Integer = 1 To Iterations
                    For Each targetWord In Model.Keys
                        Dim targetIndex As Integer = GetOrCreateWordIndex(targetWord)
                        Dim targetEmbedding As Double() = GetEmbedding(targetIndex)

                        For Each contextWord In Model(targetWord).Keys
                            Dim contextIndex As Integer = GetOrCreateWordIndex(contextWord)
                            Dim contextEmbedding As Double() = GetEmbedding(contextIndex)
                            Dim cooccurrenceValue As Double = Model(targetWord)(contextWord)
                            Dim weight As Double = Math.Log(cooccurrenceValue)

                            For i As Integer = 0 To embeddingSize - 1
                                targetEmbedding(i) += weight * contextEmbedding(i)
                            Next
                        Next
                    Next
                Next
                Model = PMI.CalculatePMI(Model)
            End Sub
            Public Sub Train(corpus As String(), iterations As Integer, ByRef LearningRate As Double)

                Me.iterations = iterations
                Model = GenerateCooccurrenceMatrix(corpus, windowSize)
                InitializeEmbeddings()

                For iteration As Integer = 1 To iterations
                    For Each targetWord In Model.Keys
                        If wordToIndex.ContainsKey(targetWord) Then
                            Dim targetIndex As Integer = wordToIndex(targetWord)
                            Dim targetEmbedding As Double() = GetEmbedding(targetIndex)

                            ' Initialize gradient accumulator for target embedding
                            Dim gradTarget As Double() = New Double(embeddingSize - 1) {}

                            For Each contextWord In Model(targetWord).Keys
                                If wordToIndex.ContainsKey(contextWord) Then
                                    Dim contextIndex As Integer = wordToIndex(contextWord)
                                    Dim contextEmbedding As Double() = GetEmbedding(contextIndex)
                                    Dim cooccurrenceValue As Double = Model(targetWord)(contextWord)
                                    Dim weight As Double = Math.Log(cooccurrenceValue)

                                    ' Initialize gradient accumulator for context embedding
                                    Dim gradContext As Double() = New Double(embeddingSize - 1) {}

                                    ' Calculate the gradients
                                    For i As Integer = 0 To embeddingSize - 1
                                        Dim gradCoefficient As Double = weight * targetEmbedding(i)

                                        gradTarget(i) = LearningRate * gradCoefficient
                                        gradContext(i) = LearningRate * gradCoefficient
                                    Next

                                    ' Update the target and context embeddings
                                    For i As Integer = 0 To embeddingSize - 1
                                        targetEmbedding(i) += gradTarget(i)
                                        contextEmbedding(i) += gradContext(i)
                                    Next
                                End If
                            Next
                        End If
                    Next
                Next
            End Sub


            Private Sub InitializeEmbeddings()
                Dim vocabSize As Integer = vocabulary.Count
                embeddingMatrix = New Double(vocabSize - 1, embeddingSize - 1) {}

                Dim random As New Random()
                For i As Integer = 0 To vocabSize - 1
                    For j As Integer = 0 To embeddingSize - 1
                        embeddingMatrix(i, j) = random.NextDouble()
                    Next
                Next
            End Sub
            Private Function CalculateSimilarity(vectorA As Double(), vectorB As Double()) As Double
                Dim dotProduct As Double = 0
                Dim magnitudeA As Double = 0
                Dim magnitudeB As Double = 0

                For i As Integer = 0 To vectorA.Length - 1
                    dotProduct += vectorA(i) * vectorB(i)
                    magnitudeA += vectorA(i) * vectorA(i)
                    magnitudeB += vectorB(i) * vectorB(i)
                Next

                If magnitudeA <> 0 AndAlso magnitudeB <> 0 Then
                    Return dotProduct / (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB))
                Else
                    Return 0
                End If
            End Function
            ''' <summary>
            ''' Discovers collocations among the specified words based on the trained model.
            ''' </summary>
            ''' <param name="words">The words to discover collocations for.</param>
            ''' <param name="threshold">The similarity threshold for collocation discovery.</param>
            ''' <returns>A list of collocations (word pairs) that meet the threshold.</returns>
            Public Function DiscoverCollocations(words As String(), threshold As Double) As List(Of Tuple(Of String, String))
                Dim collocations As New List(Of Tuple(Of String, String))

                For i As Integer = 0 To words.Length - 2
                    For j As Integer = i + 1 To words.Length - 1
                        Dim word1 As String = words(i)
                        Dim word2 As String = words(j)

                        If vocabulary.Contains(word1) AndAlso vocabulary.Contains(word2) Then
                            Dim vector1 As Double() = GetEmbedding(wordToIndex(word1))
                            Dim vector2 As Double() = GetEmbedding(wordToIndex(word2))
                            Dim similarity As Double = CalculateSimilarity(vector1, vector2)

                            If similarity >= threshold Then
                                collocations.Add(Tuple.Create(word1, word2))
                            End If
                        End If
                    Next
                Next

                Return collocations
            End Function
            ''' <summary>
            ''' Gets the most similar words to the specified word.
            ''' </summary>
            ''' <param name="word">The target word.</param>
            ''' <param name="topK">The number of similar words to retrieve.</param>
            ''' <returns>A list of the most similar words.</returns>
            Public Function GetMostSimilarWords(word As String, topK As Integer) As List(Of String)
                Dim wordIndex As Integer = wordToIndex(word)

                Dim similarities As New Dictionary(Of String, Double)()
                For Each otherWord As String In vocabulary
                    If otherWord <> word Then
                        Dim otherWordIndex As Integer = wordToIndex(otherWord)
                        Dim similarity As Double = CalculateSimilarity(GetEmbedding(wordIndex), GetEmbedding(otherWordIndex))
                        similarities.Add(otherWord, similarity)
                    End If
                Next

                Dim orderedSimilarities = similarities.OrderByDescending(Function(x) x.Value)
                Dim mostSimilarWords As New List(Of String)()

                Dim count As Integer = 0
                For Each pair In orderedSimilarities
                    mostSimilarWords.Add(pair.Key)
                    count += 1
                    If count >= topK Then
                        Exit For
                    End If
                Next

                Return mostSimilarWords
            End Function
            Private vocabulary As HashSet(Of String)
            Private wordToIndex As Dictionary(Of String, Integer)
            Private indexToWord As Dictionary(Of Integer, String)
            Public Function BuildVocabulary(corpus As List(Of String)) As HashSet(Of String)

                Dim index As Integer = 0
                For Each sentence As String In corpus
                    Dim cleanedText As String = Regex.Replace(sentence, "[^\w\s]", "").ToLower()
                    Dim tokens As String() = cleanedText.Split()
                    For Each token As String In tokens
                        If Not vocabulary.Contains(token) Then
                            vocabulary.Add(token)
                            wordToIndex.Add(token, index)
                            indexToWord.Add(index, token)
                            index += 1
                        End If
                    Next
                Next
                Return vocabulary
            End Function
            Public Function GetOrCreateWordIndex(word As String) As Integer
                If wordToIndex.ContainsKey(word) Then
                    Return wordToIndex(word)
                Else
                    Dim newIndex As Integer = vocabulary.Count
                    vocabulary.Add(word)
                    wordToIndex.Add(word, newIndex)
                    indexToWord.Add(newIndex, word)
                    Return newIndex
                End If
            End Function
            Public Function DisplayMatrix(matrix As Dictionary(Of String, Dictionary(Of String, Double))) As DataGridView
                Dim dataGridView As New DataGridView()
                dataGridView.Dock = DockStyle.Fill
                dataGridView.AutoGenerateColumns = False
                dataGridView.AllowUserToAddRows = False

                ' Add columns to the DataGridView
                Dim wordColumn As New DataGridViewTextBoxColumn()
                wordColumn.HeaderText = "Word"
                wordColumn.DataPropertyName = "Word"
                dataGridView.Columns.Add(wordColumn)

                For Each contextWord As String In matrix.Keys
                    Dim contextColumn As New DataGridViewTextBoxColumn()
                    contextColumn.HeaderText = contextWord
                    contextColumn.DataPropertyName = contextWord
                    dataGridView.Columns.Add(contextColumn)
                Next

                ' Populate the DataGridView with the matrix data
                For Each word As String In matrix.Keys
                    Dim rowValues As New List(Of Object)
                    rowValues.Add(word)

                    For Each contextWord As String In matrix.Keys
                        Dim count As Object = If(matrix(word).ContainsKey(contextWord), matrix(word)(contextWord), 0)
                        rowValues.Add(count)
                    Next

                    dataGridView.Rows.Add(rowValues.ToArray())
                Next

                Return dataGridView
            End Function
            Public Sub DisplayModel()
                DisplayMatrix(Model)
            End Sub
            Public Function DisplayMatrix(matrix As Dictionary(Of String, Dictionary(Of String, Integer))) As DataGridView
                Dim dataGridView As New DataGridView()
                dataGridView.Dock = DockStyle.Fill
                dataGridView.AutoGenerateColumns = False
                dataGridView.AllowUserToAddRows = False

                ' Add columns to the DataGridView
                Dim wordColumn As New DataGridViewTextBoxColumn()
                wordColumn.HeaderText = "Word"
                wordColumn.DataPropertyName = "Word"
                dataGridView.Columns.Add(wordColumn)

                For Each contextWord As String In matrix.Keys
                    Dim contextColumn As New DataGridViewTextBoxColumn()
                    contextColumn.HeaderText = contextWord
                    contextColumn.DataPropertyName = contextWord
                    dataGridView.Columns.Add(contextColumn)
                Next

                ' Populate the DataGridView with the matrix data
                For Each word As String In matrix.Keys
                    Dim rowValues As New List(Of Object)()
                    rowValues.Add(word)

                    For Each contextWord As String In matrix.Keys
                        Dim count As Integer = If(matrix(word).ContainsKey(contextWord), matrix(word)(contextWord), 0)
                        rowValues.Add(count)
                    Next

                    dataGridView.Rows.Add(rowValues.ToArray())
                Next

                Return dataGridView
            End Function

        End Class
        Public Class VocabularyGenerator

            Public Shared Function CreateDictionaryVocabulary(data As List(Of String)) As HashSet(Of String)
                Dim vocabulary As New HashSet(Of String)

                For Each chunk As String In data
                    Dim words As String() = chunk.Split({" ", ".", ",", "!", "?"}, StringSplitOptions.RemoveEmptyEntries)
                    For Each word As String In words
                        vocabulary.Add(word.ToLower())
                    Next
                Next

                Return vocabulary
            End Function

            Public Shared Function CreateFrequencyVocabulary(data As List(Of String)) As Dictionary(Of String, Integer)
                Dim frequencyVocabulary As New Dictionary(Of String, Integer)

                For Each chunk As String In data
                    Dim words As String() = chunk.Split({" ", ".", ",", "!", "?"}, StringSplitOptions.RemoveEmptyEntries)
                    For Each word As String In words
                        Dim cleanedWord As String = word.ToLower()

                        If frequencyVocabulary.ContainsKey(cleanedWord) Then
                            frequencyVocabulary(cleanedWord) += 1
                        Else
                            frequencyVocabulary(cleanedWord) = 1
                        End If
                    Next
                Next

                Return frequencyVocabulary
            End Function

            ''' <summary>
            ''' Creates a vocabulary of punctuation marks and symbols detected in the processed text.
            ''' </summary>
            ''' <param name="data">The list of processed text data chunks.</param>
            Public Shared Function CreatePunctuationVocabulary(data As List(Of String)) As HashSet(Of String)
                Dim PunctuationVocabulary = New HashSet(Of String)

                For Each chunk As String In data
                    Dim symbols As String() = chunk.Split().Where(Function(token) Not Char.IsLetterOrDigit(token(0))).ToArray()
                    For Each symbol As String In symbols
                        PunctuationVocabulary.Add(symbol)
                    Next
                Next
                Return PunctuationVocabulary
            End Function

            Public Shared Sub ExportFrequencyVocabularyToFile(vocabulary As Dictionary(Of String, Integer), outputPath As String)
                Using writer As New StreamWriter(outputPath)
                    For Each kvp As KeyValuePair(Of String, Integer) In vocabulary
                        writer.WriteLine($"{kvp.Key}: {kvp.Value}")
                    Next
                End Using
            End Sub

            Public Shared Sub ExportVocabulary(outputPath As String, ByRef Vocabulary As HashSet(Of String))
                File.WriteAllLines(outputPath, Vocabulary.OrderBy(Function(word) word))

            End Sub

            Public Shared Function ImportFrequencyVocabularyFromFile(filePath As String) As Dictionary(Of String, Integer)
                Dim vocabulary As New Dictionary(Of String, Integer)()

                Try
                    Dim lines As String() = File.ReadAllLines(filePath)
                    For Each line As String In lines
                        Dim parts As String() = line.Split(New String() {": "}, StringSplitOptions.None)
                        If parts.Length = 2 Then
                            Dim word As String = parts(0)
                            Dim frequency As Integer
                            If Integer.TryParse(parts(1), frequency) Then
                                vocabulary.Add(word, frequency)
                            End If
                        End If
                    Next
                Catch ex As Exception
                    ' Handle exceptions, such as file not found or incorrect format
                    Console.WriteLine("Error importing frequency vocabulary: " & ex.Message)
                End Try

                Return vocabulary
            End Function

            Public Shared Function ImportVocabularyFromFile(filePath As String) As HashSet(Of String)
                Dim punctuationVocabulary As New HashSet(Of String)()

                Try
                    Dim lines As String() = File.ReadAllLines(filePath)
                    For Each line As String In lines
                        punctuationVocabulary.Add(line)
                    Next
                Catch ex As Exception
                    ' Handle exceptions, such as file not found
                    Console.WriteLine("Error importing punctuation vocabulary: " & ex.Message)
                End Try

                Return punctuationVocabulary
            End Function

        End Class
        Public Module Modelling

            ''' <summary>
            ''' creates a list of words with thier positional encoding (cosine simularity)
            ''' </summary>
            ''' <param name="DocText">document</param>
            ''' <returns>tokens with positional encoding</returns>
            <Runtime.CompilerServices.Extension()>
            Public Function PositionalEncoder(ByRef DocText As String) As List(Of WordVector)
                Dim sequence As String = "The quick brown fox jumps over the lazy dog."
                Dim words As String() = DocText.Split(" ")

                ' Create a list to store the positional encoding for each word
                Dim encoding As New List(Of List(Of Double))

                ' Calculate the positional encoding for each word in the sequence
                For i As Integer = 0 To words.Length - 1
                    ' Create a list to store the encoding vector for this word
                    Dim encodingVector As New List(Of Double)

                    ' Calculate the encoding vector for each dimension (8 dimensions for positional encoding)
                    For j As Integer = 0 To 7
                        Dim exponent As Double = j / 2

                        ' Calculate the sine or cosine value based on whether j is even or odd
                        If j Mod 2 = 0 Then
                            encodingVector.Add(Math.Sin(i / (10000 ^ exponent)))
                        Else
                            encodingVector.Add(Math.Cos(i / (10000 ^ exponent)))
                        End If
                    Next

                    ' Add the encoding vector for this word to the list of encodings
                    encoding.Add(encodingVector)
                Next
                Dim WordVects As New List(Of WordVector)
                ' Print the positional encoding for each word in the sequence
                For i As Integer = 0 To words.Length - 1
                    Dim NVect As New WordVector
                    NVect.Token = words(i)
                    For Each item In encoding(i)
                        NVect.PositionalEncodingVector.Add(item)
                    Next

                    WordVects.Add(NVect)
                Next
                Return WordVects
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function Top_N_Words(ByRef corpus As String, ByRef Count As Integer)
                Dim words As String() = corpus.Split(" ")
                Dim wordCount As New Dictionary(Of String, Integer)

                ' Count the frequency of each word in the corpus
                For Each word As String In words
                    If wordCount.ContainsKey(word) Then
                        wordCount(word) += 1
                    Else
                        wordCount.Add(word, 1)
                    End If
                Next

                ' Sort the dictionary by value (frequency) in descending order
                Dim sortedDict = (From entry In wordCount Order By entry.Value Descending Select entry).Take(Count)
                Dim LSt As New List(Of String)
                ' Print the top ten words and their frequency
                For Each item In sortedDict
                    LSt.Add(item.Key)

                Next
                Return LSt
            End Function

            ''' <summary>
            ''' calculates the probability of a word in a corpus of documents
            ''' </summary>
            ''' <param name="Token">to be found</param>
            ''' <param name="corpus">collection of documents</param>
            <Runtime.CompilerServices.Extension()>
            Public Sub ProbablityOfWordInCorpus(ByRef Token As String, ByRef corpus As List(Of String))
                Dim word As String = Token
                Dim count As Integer = 0
                Dim totalWords As Integer = 0

                ' Count the number of times the word appears in the corpus
                For Each sentence As String In corpus
                    Dim words() As String = sentence.Split(" ")
                    For Each w As String In words
                        If w.Equals(word) Then
                            count += 1
                        End If
                        totalWords += 1
                    Next
                Next

                ' Calculate the probability of the word in the corpus
                Dim probability As Double = count / totalWords

                Console.WriteLine("The probability of the word '" & word & "' in the corpus is " & probability)
            End Sub

            <Runtime.CompilerServices.Extension()>
            Public Function TokenizeChars(ByRef Txt As String) As String

                Dim NewTxt As String = ""
                For Each chr As Char In Txt

                    NewTxt &= chr.ToString & ","
                Next

                Return NewTxt
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function TokenizeSentences(ByRef txt As String) As String
                Dim NewTxt As String = ""
                Dim Words() As String = txt.Split(".")
                For Each item In Words
                    NewTxt &= item & ","
                Next
                Return NewTxt
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function TokenizeWords(ByRef txt As String) As String
                Dim NewTxt As String = ""
                Dim Words() As String = txt.Split(" ")
                For Each item In Words
                    NewTxt &= item & ","
                Next
                Return NewTxt
            End Function

            ''' <summary>
            ''' Creates a vocabulary from the string presented
            ''' a dictionary of words from the text. containing word frequencys and sequence embeddings
            ''' this can be used to create word embeddings for the string
            ''' </summary>
            ''' <param name="InputString"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function CreateVocabulary(ByVal InputString As String) As List(Of WordVector)
                Return WordVector.CreateVocabulary(InputString.ToLower)
            End Function

            ''' <summary>
            ''' Creates embeddings by generating an internal vocabulary from the text provided
            ''' </summary>
            ''' <param name="InputString">document</param>
            ''' <returns>list of word vectors containing embeddings</returns>
            Public Function CreateWordEmbeddings(ByVal InputString As String) As List(Of WordVector)
                Return InputString.CreateWordEmbeddings(InputString.CreateVocabulary)
            End Function

            ''' <summary>
            ''' Creates a list of word-embeddings for string using a provided vocabulary
            ''' </summary>
            ''' <param name="InputString">document</param>
            ''' <param name="Vocabulary">Pretrained vocabulary</param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function CreateWordEmbeddings(ByVal InputString As String, ByRef Vocabulary As List(Of WordVector)) As List(Of WordVector)
                Return WordVector.EncodeWordsToVectors(InputString, Vocabulary)
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function OneHotEncoding(ByRef EncodedList As List(Of WordVector), KeyWords As List(Of String)) As List(Of WordVector)
                Return WordVector.OneShotEncoding(EncodedList, KeyWords)
            End Function

            ''' <summary>
            ''' looks up sequence encoding in vocabulary - used to encode a Document
            ''' </summary>
            ''' <param name="EncodedWordlist"></param>
            ''' <param name="Token"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Function LookUpSeqEncoding(ByRef EncodedWordlist As List(Of WordVector), ByRef Token As String) As Integer
                Return (WordVector.LookUpSeqEncoding(EncodedWordlist, Token))
            End Function

            ''' <summary>
            ''' used for decoding token by sequence encoding
            ''' </summary>
            ''' <param name="EncodedWordlist"></param>
            ''' <param name="EncodingValue"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function LookUpBySeqEncoding(ByRef EncodedWordlist As List(Of WordVector), ByRef EncodingValue As Integer) As String
                Return (WordVector.LookUpSeqEncoding(EncodedWordlist, EncodingValue))
            End Function

        End Module
    End Namespace
End Namespace
Namespace EncoderDecoders
    ''' <summary>
    ''' Encoding:
    ''' EncodeTokenStr: Encodes a String token And returns its positional embedding As a list Of doubles.
    '''    EncodeTokenEmbedding: Encodes a token embedding (list Of doubles) And returns its positional embedding As a list Of doubles.
    '''    EncodeSentenceStr: Encodes a list Of String tokens And returns their positional embeddings As a list Of lists Of doubles.
    '''    EncodeSentenceEmbedding: Encodes a list Of token embeddings And returns their positional embeddings As a list Of lists Of doubles.
    '''Decoding:
    '''DecodeTokenStr: Decodes a positional embedding (list Of doubles) And returns the corresponding String token.
    '''    DecodeTokenEmbedding: Decodes a positional embedding (list Of doubles) And returns the corresponding token embedding As a list Of doubles.
    '''    DecodeSentenceStr: Decodes a list Of positional embeddings And returns the corresponding String tokens As a list Of strings.
    '''    DecodeSentenceEmbedding: Decodes a list Of positional embeddings And returns the corresponding token embeddings As a list Of lists Of doubles.
    '''     </summary>
    Public Class PositionalEncoderDecoder
        Private encodingMatrix As List(Of List(Of Double))
        Private Vocabulary As New List(Of String)

        ''' <summary>
        ''' 
        ''' </summary>
        ''' <param name="Dmodel">Embedding Model Size 
        ''' (1. often best to use the Vocabulary D_model)
        ''' (2. often a Fixed 512 is used LLM)
        ''' (3: 64 SMall LLM) </param>
        ''' <param name="MaxSeqLength">Max Sentence Length</param>
        ''' <param name="vocabulary">Known VocabularyList</param>
        Public Sub New(ByRef Dmodel As Integer, MaxSeqLength As Integer, vocabulary As List(Of String))
            '1. Create Embedding Matrix  Dmodel * MaxSeqLength
            CreateEmbeddingMatrix(Dmodel, MaxSeqLength)
            '2. Set Reference Vocabulary
            Me.Vocabulary = vocabulary
        End Sub

        'Encode
        Public Function EncodeTokenStr(ByRef nToken As String) As List(Of Double)
            Dim positionID As Integer = GetTokenIndex(nToken)
            Return If(positionID <> -1, encodingMatrix(positionID), New List(Of Double)())
        End Function

        Public Function EncodeTokenEmbedding(ByRef TokenEmbedding As List(Of Double)) As List(Of Double)
            Dim positionID As Integer = GetTokenIndex(TokenEmbedding)
            Return If(positionID <> -1, encodingMatrix(positionID), New List(Of Double)())
        End Function
        Public Function EncodeSentenceStr(ByRef Sentence As List(Of String)) As List(Of List(Of Double))
            Dim EncodedSentence As New List(Of List(Of Double))
            For Each Word In Sentence

                EncodedSentence.Add(EncodeTokenStr(Word))
            Next
            Return EncodedSentence
        End Function
        Public Function EncodeSentenceEmbedding(ByRef SentenceEmbeddings As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim EncodedSentence As New List(Of List(Of Double))
            For Each Word In SentenceEmbeddings

                EncodedSentence.Add(EncodeTokenEmbedding(Word))
            Next
            Return EncodedSentence
        End Function

        'Decode
        Public Function DecodeSentenceStr(ByRef Sentence As List(Of List(Of Double))) As List(Of String)
            Dim DecodedSentence As New List(Of String)
            For Each Word In Sentence

                DecodedSentence.Add(DecodeTokenStr(Word))
            Next
            Return DecodedSentence
        End Function
        Public Function DecodeSentenceEmbedding(ByRef Sentence As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim DecodedSentence As New List(Of List(Of Double))
            For Each Word In Sentence

                DecodedSentence.Add(DecodeTokenEmbedding(Word))
            Next
            Return DecodedSentence
        End Function
        ''' <summary>
        ''' Used For String Tokens
        ''' </summary>
        ''' <param name="PositionalEmbeddingVector"></param>
        ''' <returns>String Token</returns>
        Public Function DecodeTokenStr(ByRef PositionalEmbeddingVector As List(Of Double)) As String
            Dim positionID As Integer = GetPositionID(PositionalEmbeddingVector)
            Return If(positionID <> -1, Vocabulary(positionID), "")
        End Function
        ''' <summary>
        ''' USed to decode WOrdEMbedding Vectors instead of strings
        ''' </summary>
        ''' <param name="PositionalEmbeddingVector"></param>
        ''' <returns>WOrdEMbedding Vector</returns>
        Public Function DecodeTokenEmbedding(ByRef PositionalEmbeddingVector As List(Of Double)) As List(Of Double)
            Dim positionID As Integer = GetPositionID(PositionalEmbeddingVector)
            Return If(positionID <> -1, encodingMatrix(positionID), New List(Of Double)())
        End Function



        Private Sub CreateEmbeddingMatrix(ByRef WidthLength As Integer, HeightLength As Integer)
            encodingMatrix = New List(Of List(Of Double))
            ' Create the encoding matrix
            For pos As Integer = 0 To HeightLength - 1
                Dim encodingRow As List(Of Double) = New List(Of Double)()

                For i As Integer = 0 To WidthLength - 1
                    Dim angle As Double = pos / Math.Pow(10000, (2 * i) / WidthLength)
                    encodingRow.Add(Math.Sin(angle))
                    encodingRow.Add(Math.Cos(angle))
                Next

                encodingMatrix.Add(encodingRow)
            Next
        End Sub
        'GetPos
        Private Function GetPositionID(PositionalEmbeddingVector As List(Of Double)) As Integer
            For i As Integer = 0 To encodingMatrix.Count - 1
                If PositionalEmbeddingVector.SequenceEqual(encodingMatrix(i)) Then
                    Return i
                End If
            Next

            Return -1 ' Position ID not found
        End Function
        Private Function GetTokenIndex(PositionalEncoding As List(Of Double)) As Integer

            For i As Integer = 0 To encodingMatrix.Count - 1
                If PositionalEncoding.SequenceEqual(encodingMatrix(i)) Then
                    Return i
                End If
            Next

            Return -1 ' Token not found
        End Function
        Private Function GetTokenIndex(token As String) As Integer

            Return Vocabulary.IndexOf(token)
        End Function
    End Class


End Namespace
Namespace Examples
    Public Class MinHashAndLSHExample
        Sub Main()
            ' Create an LSHIndex object with 3 hash tables and 2 hash functions per table
            Dim lshIndex As New LSHIndex(numHashTables:=3, numHashFunctionsPerTable:=2)

            ' Create some sample articles
            Dim article1 As New Document("The Art of Baking: Mastering the Perfect Sweet Chocolate Cake", 0)
            Dim article2 As New Document("A Journey Exploring Exotic Cuisines: A Culinary Adventure in Southeast Asia", 1)
            Dim article3 As New Document("Nutrition for Optimal Brain Health: Foods that Boost Cognitive Function", 2)
            Dim article4 As New Document("The Rise of Artificial Intelligence: A Game-Changer in the Tech World", 3)
            Dim article5 As New Document("Introduction to Quantum Computing: Unraveling the Power of Qubits", 4)

            ' Add the articles to the LSH index
            lshIndex.AddDocument(article1)
            lshIndex.AddDocument(article2)
            lshIndex.AddDocument(article3)
            lshIndex.AddDocument(article4)
            lshIndex.AddDocument(article5)

            ' Create a query article
            Dim queryArticle As New Document("Delicious Desserts: A Journey Through Sweet Delights", -1)

            ' Find similar articles using LSH
            Dim similarArticles As List(Of Document) = lshIndex.FindSimilarDocuments(queryArticle)

            ' Display the results
            Console.WriteLine("Query Article: " & queryArticle.Content)
            If similarArticles.Count = 0 Then
                Console.WriteLine("No similar articles found.")
            Else
                Console.WriteLine("Similar Articles:")
                For Each article As Document In similarArticles
                    Console.WriteLine(article.Content)
                Next
            End If

            Console.ReadLine()
        End Sub
    End Class
    Public Class VectorStorageExample
        Public Sub SaveVectorToFile(imgVector As Double(), outputPath As String)
            Using writer As New System.IO.StreamWriter(outputPath)
                For Each value As Double In imgVector
                    writer.WriteLine(value)
                Next
            End Using
        End Sub
        Sub Main()
            Dim db As VectorStorageModel = New VectorStorageModel()

            ' Adding sample vectors to the database
            db.AddVector(1, New List(Of Double)() From {1.0, 2.0}, VectorStorageModel.VectorType.Text)
            db.AddVector(2, New List(Of Double)() From {3.0, 4.0}, VectorStorageModel.VectorType.Text)
            db.AddVector(3, New List(Of Double)() From {5.0, 6.0}, VectorStorageModel.VectorType.Text)

            ' Query vector
            Dim queryVector As List(Of Double) = New List(Of Double)() From {2.5, 3.5}

            ' Find similar vectors
            Dim similarVectors As List(Of Integer) = db.FindSimilarTextVectors(queryVector, 2)

            ' Display results
            Console.WriteLine("Query Vector: " & String.Join(",", queryVector))
            Console.WriteLine("Similar Vectors: " & String.Join(",", similarVectors))

            Dim searchEngine As New SearchEngine()

            searchEngine.AddDocument(1, "This is an example document containing some sample text.")
            searchEngine.AddDocument(2, "Another document with additional content for demonstration.")

            Dim query As String = "example document"
            Dim results As List(Of Tuple(Of Integer, String)) = searchEngine.Search(query)

            For Each result As Tuple(Of Integer, String) In results
                Console.WriteLine($"Document ID: {result.Item1}")
                Console.WriteLine($"Snippet: {result.Item2}")
                Console.WriteLine()
            Next


            Dim imageEncoder As New Image2Vector.ImageEncoder()
            Dim imageDecoder As New Image2Vector.ImageDecoder()

            Dim imagePath As String = "path_to_your_image.jpg"
            Dim width As Integer = 224
            Dim height As Integer = 224
            Dim outputVectorPath As String = "encoded_image_vector.txt"
            Dim outputImagePath As String = "decoded_image.jpg"

            ' Encode image to vector
            Dim imgVector As Double() = imageEncoder.EncodeImage(imagePath, width, height)

            ' Save encoded vector to file
            SaveVectorToFile(imgVector, outputVectorPath)

            ' Decode vector back to image
            imageDecoder.DecodeImage(imgVector, width, height, outputImagePath)

            Console.WriteLine("Image encoded, vector saved, and image decoded.")

            Dim imageSearch As New Image2Vector.ImageSearch()

            Dim queryImagePath As String = "query_image.jpg"
            Dim ImagequeryVector As Double() = imageEncoder.EncodeImage(queryImagePath, 224, 224)

            Dim imageVectors As New List(Of Tuple(Of String, Double()))()
            ' Populate imageVectors with the names and vectors of your images

            Dim numResults As Integer = 5
            Dim similarImageNames As List(Of String) = imageSearch.FindSimilarImages(ImagequeryVector, imageVectors, numResults)

            Console.WriteLine("Similar images:")
            For Each imageName As String In similarImageNames
                Console.WriteLine(imageName)
            Next
            ' Load audio data (replace with your own audio loading code)
            Dim audioPath As String = "path_to_your_audio_file.wav"
            Dim audioSignal As Double() = Audio2Vector.LoadAudio(audioPath)

            ' Set window size and hop size for feature extraction
            Dim windowSize As Integer = 1024
            Dim hopSize As Integer = 256

            ' Convert audio to vectors
            Dim audioVectors As List(Of Complex()) = Audio2Vector.AudioToVector(audioSignal, windowSize, hopSize)

            ' Convert vectors back to audio
            Dim reconstructedAudio As Double() = Audio2Vector.VectorToAudio(audioVectors, hopSize)

            ' Save reconstructed audio (replace with your own saving code)
            Dim outputAudioPath As String = "reconstructed_audio.wav"
            Audio2Vector.SaveAudio(reconstructedAudio, outputAudioPath)

            Console.WriteLine("Audio-to-Vector and Vector-to-Audio Conversion Complete.")

        End Sub
    End Class
    Public Class WordEmbeddingsExample
        Public Shared Sub Run()
            ' Sample text corpus for training word embeddings.
            Dim textCorpus As List(Of String) = New List(Of String) From
            {
                "apple orange banana",
                "orange banana grape",
                "grape cherry apple",
                "apple apple apple",
                "cherry banana banana"
            }

            ' Create a custom vocabulary from the text corpus.
            Dim vocabulary As List(Of String) = textCorpus.SelectMany(Function(sentence) sentence.Split()).Distinct().ToList()

            ' Create a WordEmbeddingsModel and train it with the text corpus.
            Dim wordEmbeddingsModel As New HybridWordEmbeddingsModel(vocabulary)
            wordEmbeddingsModel.Train()

            ' Get the word vector for a specific word.
            Dim word As String = "apple"
            Dim wordVector As Double() = wordEmbeddingsModel.WordEmbeddings.GetVector(word)


            ' Calculate the cosine similarity between two words.
            Dim word1 As String = "apple"
            Dim word2 As String = "orange"
            Dim similarity As Double = wordEmbeddingsModel.CosineSimilarity(word1, word2, wordEmbeddingsModel.WordEmbeddings)

            ' Display the word vector and similarity result.
            Console.WriteLine($"Word Vector for '{word}': {String.Join(", ", wordVector)}")
            Console.WriteLine($"Cosine Similarity between '{word1}' and '{word2}': {similarity}")
        End Sub
    End Class

    Public Class HybridWordEmbeddingsExample
        Public Shared Sub Run()
            ' Sample text corpus for training word embeddings.
            Dim textCorpus As List(Of String) = New List(Of String) From
            {
                "apple orange banana",
                "orange banana grape",
                "grape cherry apple",
                "apple apple apple",
                "cherry banana banana"
            }

            ' Create a custom vocabulary from the text corpus.
            Dim vocabulary As List(Of String) = textCorpus.SelectMany(Function(sentence) sentence.Split()).Distinct().ToList()

            ' Create the hybrid word embeddings model.
            Dim hybridModel As New HybridWordEmbeddingsModel(vocabulary)

            ' Train the hybrid model using the two-step training process.
            hybridModel.Train()

            ' Get the word vector for a specific word after training.
            Dim word As String = "apple"
            Dim wordVector As Double() = hybridModel.WordEmbeddings.GetVector(word)

            ' Calculate the cosine similarity between two words after training.
            Dim word1 As String = "apple"
            Dim word2 As String = "orange"
            Dim similarity As Double = hybridModel.CosineSimilarity(word1, word2, hybridModel.WordEmbeddings)

            ' Display the word vector and similarity result after training.
            Console.WriteLine($"Word Vector for '{word}' after training: {String.Join(", ", wordVector)}")
            Console.WriteLine($"Cosine Similarity between '{word1}' and '{word2}' after training: {similarity}")
        End Sub
    End Class

    Public Module CorpusReaderExamples

        Public Sub ExampleCalculateSentenceSimilarity()
            ' Example sentences
            Dim sentence1 As String = "The cat is on the mat."
            Dim sentence2 As String = "The mat has a cat."

            ' Tokenize the sentences
            Dim tokens1 As String() = sentence1.Split(" "c)
            Dim tokens2 As String() = sentence2.Split(" "c)

            ' Calculate the word overlap
            Dim overlap As Integer = CalculateWordOverlap(tokens1, tokens2)

            ' Determine entailment based on overlap
            Dim entailment As Boolean = DetermineEntailment(overlap)

            ' Display the results
            Console.WriteLine("Sentence 1: " & sentence1)
            Console.WriteLine("Sentence 2: " & sentence2)
            Console.WriteLine("Word Overlap: " & overlap)
            Console.WriteLine("Entailment: " & entailment)
            Console.ReadLine()
        End Sub

        ' Usage Example:
        Public Sub ExampleCorpusCatagorizer()
            ' Create an instance of the CorpusCategorizer
            Dim categorizer As New CorpusCategorizer()

            ' Add categories and their associated keywords
            categorizer.AddCategory("Sports", New List(Of String) From {"football", "basketball", "tennis"})
            categorizer.AddCategory("Technology", New List(Of String) From {"computer", "software", "internet"})
            categorizer.AddCategory("Politics", New List(Of String) From {"government", "election", "policy"})

            ' Assuming you have a corpus with multiple documents
            Dim corpus As New List(Of String)()
            corpus.Add("I love playing basketball and football.")
            corpus.Add("Software engineering is my passion.")
            corpus.Add("The government announced new policies.")

            ' Categorize each document in the corpus
            For Each document As String In corpus
                Dim categories As List(Of String) = categorizer.CategorizeDocument(document.ToLower)
                Console.WriteLine("Categories for document: " & document)
                For Each category As String In categories
                    Console.WriteLine("- " & category)
                Next
                Console.WriteLine()
            Next
            Console.ReadLine()
            ' Rest of your code...
        End Sub

        Public Sub ExampleCorpusCreator()

            ' Load and preprocess your text data
            Dim rawData As New List(Of String)  ' Load your raw text data
            Dim processedData As New List(Of String) ' Preprocess your data using existing methods

            ' Generate batches of training data
            Dim batch_size As Integer = 32
            Dim seq_length As Integer = 50
            Dim batches As List(Of Tuple(Of List(Of String), List(Of String))) = CorpusCreator.GenerateTransformerBatches(processedData, batch_size, seq_length)

            ' Iterate through batches during training
            For Each batch As Tuple(Of List(Of String), List(Of String)) In batches
                Dim inputSequences As List(Of String) = batch.Item1
                Dim targetSequences As List(Of String) = batch.Item2

                ' Perform further processing, tokenization, and padding if needed
                ' Feed the batches to your transformer model for training
            Next

        End Sub

        Public Sub ExampleCreateFrequencyVocabularyDictionary()
            Dim frequencyVocabulary As New Dictionary(Of String, Integer)()
            ' Populate the frequencyVocabulary dictionary with word frequencies

            Dim outputFilePath As String = "frequency_vocabulary.txt"
            VocabularyGenerator.ExportFrequencyVocabularyToFile(frequencyVocabulary, outputFilePath)

            Console.WriteLine($"Frequency vocabulary exported to: {outputFilePath}")

        End Sub

        Public Sub ExampleCreateFrequencyVocabularyFromData()
            Dim textChunks As New List(Of String)()
            ' Populate the textChunks list with your text data

            Dim frequencyVocabulary As Dictionary(Of String, Integer) = VocabularyGenerator.CreateFrequencyVocabulary(textChunks)

            ' Print the frequency vocabulary
            For Each kvp As KeyValuePair(Of String, Integer) In frequencyVocabulary
                Console.WriteLine($"Word: {kvp.Key}, Frequency: {kvp.Value}")
            Next

        End Sub

        Public Sub ExampleLoadFrequencyVocabularyDictionary()
            Dim inputFilePath As String = "frequency_vocabulary.txt"
            Dim importedVocabulary As Dictionary(Of String, Integer) = VocabularyGenerator.ImportFrequencyVocabularyFromFile(inputFilePath)

            ' Use the importedVocabulary dictionary for further processing or analysis
            For Each kvp As KeyValuePair(Of String, Integer) In importedVocabulary
                Console.WriteLine($"Word: {kvp.Key}, Frequency: {kvp.Value}")
            Next

        End Sub

        Public Sub ExampleLoadPunctuationDictionary()
            Dim inputFilePath As String = "punctuation_vocabulary.txt"
            Dim importedPunctuationVocabulary As HashSet(Of String) = VocabularyGenerator.ImportVocabularyFromFile(inputFilePath)

            ' Use the importedPunctuationVocabulary HashSet for further processing or analysis
            For Each symbol As String In importedPunctuationVocabulary
                Console.WriteLine($"Punctuation Symbol: {symbol}")
            Next

        End Sub

        ' Usage Example:
        Public Sub ExampleModelCorpusReader()
            ' Assuming you have a corpus directory with tagged files and a wordlist file
            Dim corpusRootPath As String = "path/to/corpus"
            Dim wordlistFilePath As String = "path/to/wordlist.txt"

            ' Create an instance of the ModelCorpusReader
            Dim corpusReader As New ModelCorpusReader(corpusRootPath)

            ' Add categories and their associated keywords
            corpusReader.AddCategory("Sports", New List(Of String) From {"football", "basketball", "tennis"})
            corpusReader.AddCategory("Technology", New List(Of String) From {"computer", "software", "internet"})
            corpusReader.AddCategory("Politics", New List(Of String) From {"government", "election", "policy"})

            ' Retrieve tagged sentences from the corpus
            Dim taggedSentences As List(Of List(Of Tuple(Of String, String))) = corpusReader.TaggedSentences()

            ' Print the tagged sentences
            For Each sentence As List(Of Tuple(Of String, String)) In taggedSentences
                For Each wordTag As Tuple(Of String, String) In sentence
                    Console.WriteLine("Word: " & wordTag.Item1 & ", Tag: " & wordTag.Item2)
                Next
                Console.WriteLine()
            Next

            ' Retrieve words from the wordlist file
            Dim wordList As List(Of String) = corpusReader.GetWordsFromWordList(wordlistFilePath)

            ' Print the words
            For Each word As String In wordList
                Console.WriteLine(word)
            Next

            ' Assuming you have a document for categorization
            Dim document As String = "I love playing basketball and football."

            ' Categorize the document
            Dim categories As List(Of String) = corpusReader.CategorizeDocument(document)

            ' Print the categories
            For Each category As String In categories
                Console.WriteLine(category)
            Next

            ' Rest of your code...
        End Sub

        ' Usage Example:
        Public Sub ExampleRegexFilter()
            Dim regexFilter As New RegexFilter()

            ' Example data and patterns
            Dim data As New List(Of String)()
            data.Add("This is a sample sentence.")
            data.Add("1234567890")
            data.Add("Let's remove @special characters!")

            Dim patterns As New List(Of String)()
            patterns.Add("[0-9]+")
            patterns.Add("@\w+")

            ' Filter data using regex patterns
            Dim filteredData As List(Of String) = regexFilter.FilterUsingRegexPatterns(data, patterns)

            ' Print filtered data
            For Each chunk As String In filteredData
                Console.WriteLine(chunk)
            Next

            ' Rest of your code...
        End Sub

        ' Usage Example:
        Public Sub ExampleTaggedCorpusReader()
            ' Assuming you have a corpus directory with tagged files
            Dim corpusRootPath As String = "path/to/corpus"

            ' Create an instance of the TaggedCorpusReader
            Dim corpusReader As New TaggedCorpusReader(corpusRootPath)

            ' Retrieve tagged sentences from the corpus
            Dim taggedSentences As List(Of List(Of Tuple(Of String, String))) = corpusReader.TaggedSentences()

            ' Print the tagged sentences
            For Each sentence As List(Of Tuple(Of String, String)) In taggedSentences
                For Each wordTag As Tuple(Of String, String) In sentence
                    Console.WriteLine("Word: " & wordTag.Item1 & ", Tag: " & wordTag.Item2)
                Next
                Console.WriteLine()
            Next

            ' Rest of your code...
        End Sub

        ' Usage Example:
        Public Sub ExampleTextCorpusChunker()
            ' Assuming you have input data and a wordlist file
            Dim inputData As String = "This is a sample sentence. Another sentence follows."
            Dim wordlistFilePath As String = "path/to/wordlist.txt"

            ' Create an instance of the TextCorpusChunker
            Dim chunker As New TextCorpusChunker(ChunkType.Sentence, 0)

            ' Load entity list if needed
            chunker.LoadEntityListFromFile("path/to/entitylist.txt")

            ' Process and filter text data
            Dim processedData As List(Of String) = chunker.ProcessTextData(inputData, useFiltering:=True)

            ' Generate classification dataset
            Dim classes As New List(Of String) From {"Class1", "Class2", "Class3"}
            Dim classificationDataset As List(Of Tuple(Of String, String)) = chunker.GenerateClassificationDataset(processedData, classes)

            ' Generate predictive dataset
            Dim windowSize As Integer = 3
            Dim predictiveDataset As List(Of String()) = chunker.GeneratePredictiveDataset(processedData, windowSize)

            ' Rest of your code...
        End Sub

        ' Usage Example:
        Public Sub ExampleVocabularyGenerator()
            ' Example data
            Dim data As New List(Of String)()
            data.Add("This is a sample sentence.")
            data.Add("Another sentence follows.")

            ' Create a dictionary vocabulary
            Dim dictionaryVocabulary As HashSet(Of String) = VocabularyGenerator.CreateDictionaryVocabulary(data)

            ' Create a frequency vocabulary
            Dim frequencyVocabulary As Dictionary(Of String, Integer) = VocabularyGenerator.CreateFrequencyVocabulary(data)

            ' Create a punctuation vocabulary
            Dim punctuationVocabulary As HashSet(Of String) = VocabularyGenerator.CreatePunctuationVocabulary(data)

            ' Export vocabulary to a file
            VocabularyGenerator.ExportVocabulary("dictionary_vocabulary.txt", dictionaryVocabulary)

            ' Import vocabulary from a file
            Dim importedVocabulary As HashSet(Of String) = VocabularyGenerator.ImportVocabularyFromFile("dictionary_vocabulary.txt")

            ' Export frequency vocabulary to a file
            VocabularyGenerator.ExportFrequencyVocabularyToFile(frequencyVocabulary, "frequency_vocabulary.txt")

            ' Import frequency vocabulary from a file
            Dim importedFrequencyVocabulary As Dictionary(Of String, Integer) = VocabularyGenerator.ImportFrequencyVocabularyFromFile("frequency_vocabulary.txt")

            ' Rest of your code...
        End Sub

        ' Usage Example:
        Public Sub ExampleWordlistReader()
            ' Assuming you have a wordlist file named 'words.txt' in the same directory
            Dim corpusRoot As String = "."
            Dim wordlistPath As String = Path.Combine(corpusRoot, "wordlist.txt")

            Dim wordlistReader As New WordListCorpusReader(wordlistPath)
            Dim words As List(Of String) = wordlistReader.GetWords()

            For Each word As String In words
                Console.WriteLine(word)
            Next
            Console.ReadLine()
            ' Rest of your code...
        End Sub

    End Module
    Public Class Word2VectorExamples
        Public Sub Word2VectorExample()
            Dim stopwords As List(Of String) = New List(Of String) From {"this", "own", "to", "is", "a", "with", "on", "is", "at", "they", "and", "the", "are", "for"}


            ' Create an instance of the Word2VecModel
            Dim model As New Word2Vector(embeddingSize:=512, learningRate:=0.01, windowSize:=3, negativeSamples:=8)
            ' Define the sample article
            Dim article As New List(Of String)
            article.Add("Dog breeds and cat breeds are popular choices for pet owners. Dogs are known for their loyalty, companionship, and diverse characteristics. There are various dog breeds, such as Labrador Retrievers, German Shepherds, Golden Retrievers, and Bulldogs. Each breed has its unique traits and temperaments. Labrador Retrievers are friendly and energetic, while German Shepherds are intelligent and protective. Golden Retrievers are gentle and great with families, while Bulldogs are sturdy and have a distinct appearance.")
            article.Add("On the other hand, cat breeds also have their own charm. Cats are independent, agile, and make great companions. Some popular cat breeds include Maine Coons, Siamese cats, Persians, and Bengals. Maine Coons are large and known for their friendly nature. Siamese cats are vocal and have striking blue eyes. Persians are long-haired and have a calm demeanor, while Bengals have a wild appearance with their spotted coat patterns.")
            article.Add("Both dogs and cats bring joy and love to their owners. Whether you prefer dogs or cats, there's a breed out there for everyone's preferences and lifestyles.")
            Dim Cleaned As New List(Of String)
            For Each item In article
                item = RemoveToken.RemoveStopWords(item, stopwords)
                Cleaned.Add(item)
            Next
            article = Cleaned
            ' Define the sample articles for cats and dogs
            Dim catArticles As New List(Of String)
            catArticles.Add("Maine Coons are one of the largest domestic cat breeds. They have a gentle and friendly nature.")
            catArticles.Add("Siamese cats are known for their striking blue eyes and vocal behavior.")
            catArticles.Add("Persian cats have long-haired coats and a calm demeanor.")
            catArticles.Add("Bengal cats have a wild appearance with their spotted coat patterns.")

            Dim dogArticles As New List(Of String)
            dogArticles.Add("Labrador Retrievers are friendly and energetic dogs.")
            dogArticles.Add("German Shepherd dogs are intelligent and protective.")
            dogArticles.Add("Golden Retrievers are gentle and great with families.")
            dogArticles.Add("Bulldogs have a sturdy build and a distinct appearance.")
            dogArticles.Add("dogs have a sturdy build and a distinct appearance.")


            Cleaned = New List(Of String)
            For Each item In dogArticles
                item = RemoveToken.RemoveStopWords(item, stopwords)
                Cleaned.Add(item)
            Next
            dogArticles = Cleaned

            Cleaned = New List(Of String)
            For Each item In catArticles
                item = RemoveToken.RemoveStopWords(item, stopwords)
                Cleaned.Add(item)
            Next
            catArticles = Cleaned
            ' Train the model with the articles
            article.AddRange(dogArticles)
            article.AddRange(catArticles)

            For i = 1 To 100
                ' Train the model with cat articles
                model.Train(article)
            Next


            For i = 1 To 100
                ' Train the model with cat articles
                model.Train(article, TrainingMethod.CBOW)
            Next


            ' Get the most similar words to "dog" / "cat"
            Dim similarDogWords As List(Of String) = model.GetMostSimilarWords("dogs", topK:=5)
            Dim similarCatWords As List(Of String) = model.GetMostSimilarWords("cats", topK:=5)

            ' Display the output
            Console.WriteLine("Most similar words to 'dog':")
            For Each word As String In similarDogWords
                Console.WriteLine(word)
            Next
            Console.WriteLine()
            Console.WriteLine("Most similar words to 'cat':")
            For Each word As String In similarCatWords
                Console.WriteLine(word)
            Next

            Console.WriteLine()

            For i = 1 To 100
                ' Train the model with  articles
                model.Train(article, TrainingMethod.SkipGram)
            Next

            ' Get the most similar words to "dog"
            similarDogWords = model.GetMostSimilarWords("dogs", topK:=8)
            ' Get the most similar words to "cat"
            similarCatWords = model.GetMostSimilarWords("cats", topK:=8)

            ' Display the output
            Console.WriteLine("Most similar words to 'dog' using Skip-gram with negative sampling:")
            For Each word As String In similarDogWords
                Console.WriteLine(word)
            Next
            Console.WriteLine()
            Console.WriteLine("Most similar words to 'cat' using Skip-gram with negative sampling:")
            For Each word As String In similarCatWords
                Console.WriteLine(word)
            Next
            Console.WriteLine()
            ' Wait for user input to exit
            Console.ReadLine()
        End Sub

    End Class
    Public Class TokenizerExample
        Public Sub Main()
            Dim Corpus As List(Of String) = GetDocumentCorpus()
            Dim sentences As New List(Of String) From {
            "I love apples.",
            "Bananas are tasty."}
            Dim Tokenizer As New Tokenizer
            For Each item In Corpus
                Tokenizer.Train(item, 5)
            Next

            For Each item In sentences
                Console.WriteLine("Document =" & item)
                Dim Tokens = Tokenizer.Tokenize(item, True)

                For Each Tok In Tokens
                    Console.WriteLine("TOKEN =" & Tok)
                Next

            Next

        End Sub
        ''' <summary>
        ''' When no lists are available, A mixed corpus of documents 
        ''' </summary>
        ''' <returns></returns>
        Public Function GetDocumentCorpus() As List(Of String)
            ' Load paragraphs based on different topics
            Dim paragraphs As New List(Of String)
            Dim sentences As New List(Of String) From {
            "The quick brown fox jumped over the sly lazy dog",
            "Bananas are tasty.",
            "I love apples.",
            "I enjoy eating bananas.",
            "Kiwi is a delicious fruit.", "Bananas are tasty.",
            "I love apples.", "I enjoy eating bananas.",
            "Kiwi is a delicious fruit.", "I love apples.",
            "I enjoy eating bananas.",
            "Kiwi is a delicious fruit.", "I love apples.",
            "I enjoy eating bananas.",
            "Kiwi is a delicious fruit.", "Bananas are tasty.", "Fisherman, like to fish in the sea, every the Fisher has fished in every place he is fishing.",
        "the lowest of the lower of the lowered tempo of the music",
        "the higher and highest and",
        "I was running, she ran after me, he was run down, until he was finished",
        "it was the the end came and the party ended."
    }
            ' Computer Science Topics
            paragraphs.Add("Computer Science is the study of computation and information processing.")
            paragraphs.Add("Algorithms and data structures are fundamental concepts in computer science.")
            paragraphs.Add("Computer networks enable communication and data exchange between devices.")
            paragraphs.Add("Artificial Intelligence is a branch of computer science that focuses on creating intelligent machines.")
            paragraphs.Add("Software engineering is the discipline of designing, developing, and maintaining software systems.")

            ' NLP Topics
            paragraphs.Add("Natural Language Processing (NLP) is a subfield of artificial intelligence.")
            paragraphs.Add("NLP techniques enable computers to understand, interpret, and generate human language.")
            paragraphs.Add("Named Entity Recognition (NER) is a common task in NLP.")
            paragraphs.Add("Machine Translation is the task of translating text from one language to another.")
            paragraphs.Add("Sentiment analysis aims to determine the sentiment or opinion expressed in a piece of text.")

            paragraphs.Add("The quick brown fox jumps over the lazy dog.")
            paragraphs.Add("The cat and the dog are best friends.")
            paragraphs.Add("Programming languages are used to write computer programs.")
            paragraphs.Add("Natural Language Processing (NLP) is a subfield of artificial intelligence.")
            paragraphs.Add("Machine learning algorithms can be used for sentiment analysis.")
            ' Train the model on a corpus of text
            Dim trainingData As New List(Of String)
            trainingData.Add("Hello")
            trainingData.Add("Hi there")
            trainingData.Add("How are you?")
            trainingData.Add("What's up?")
            trainingData.Add("I'm doing well, thanks!")
            trainingData.Add("Not too bad, how about you?")
            trainingData.Add("Great! What can I help you with?")
            trainingData.Add("I need some assistance")
            trainingData.Add("Can you provide me with information?")
            trainingData.Add("Sure, what do you need?")
            trainingData.Add("Can you tell me about your services?")
            trainingData.Add("We offer a wide range of services to cater to your needs.")
            trainingData.Add("What are the payment options?")
            trainingData.Add("We accept all major credit cards and PayPal.")
            trainingData.Add("Do you have any ongoing promotions?")
            trainingData.Add("Yes, we have a special discount available for new customers.")
            trainingData.Add("How long does shipping take?")
            trainingData.Add("Shipping usually takes 3-5 business days.")
            trainingData.Add("What is your return policy?")
            trainingData.Add("We offer a 30-day return policy for unused items.")
            trainingData.Add("Can you recommend a good restaurant nearby?")
            trainingData.Add("Sure! There's a great Italian restaurant called 'La Bella Vita' just a few blocks away.")
            trainingData.Add("What movies are currently playing?")
            trainingData.Add("The latest releases include 'Avengers: Endgame' and 'The Lion King'.")
            trainingData.Add("What time does the museum open?")
            trainingData.Add("The museum opens at 9:00 AM.")
            trainingData.Add("How do I reset my password?")
            trainingData.Add("You can reset your password by clicking on the 'Forgot Password' link on the login page.")
            trainingData.Add("What are the system requirements for this software?")
            trainingData.Add("The system requirements are listed on our website under the 'Support' section.")
            trainingData.Add("Can you provide technical support?")
            trainingData.Add("Yes, we have a dedicated support team available 24/7 to assist you.")
            trainingData.Add("What is the best way to contact customer service?")
            trainingData.Add("You can reach our customer service team by phone, email, or live chat.")
            trainingData.Add("How do I cancel my subscription?")
            trainingData.Add("To cancel your subscription, please go to your account settings and follow the instructions.")
            trainingData.Add("What are the available colors for this product?")
            trainingData.Add("The available colors are red, blue, and green.")
            trainingData.Add("Do you offer international shipping?")
            trainingData.Add("Yes, we offer international shipping to select countries.")
            trainingData.Add("Can I track my order?")
            trainingData.Add("Yes, you can track your order by entering the tracking number on our website.")
            trainingData.Add("What is your privacy policy?")
            trainingData.Add("Our privacy policy can be found on our website under the 'Privacy' section.")
            trainingData.Add("How do I request a refund?")
            trainingData.Add("To request a refund, please contact our customer service team with your order details.")
            trainingData.Add("What are the opening hours?")
            trainingData.Add("We are open from Monday to Friday, 9:00 AM to 6:00 PM.")
            trainingData.Add("Is there a warranty for this product?")
            trainingData.Add("Yes, this product comes with a one-year warranty.")
            trainingData.Add("Can I schedule an appointment?")
            trainingData.Add("Yes, you can schedule an appointment by calling our office.")
            trainingData.Add("Do you have any vegetarian options?")
            trainingData.Add("Yes, we have a dedicated vegetarian menu.")
            trainingData.Add("What is your company's mission statement?")
            trainingData.Add("Our mission is to provide high-quality products and excellent customer service.")
            trainingData.Add("How can I apply for a job at your company?")
            trainingData.Add("You can apply for a job by submitting your resume through our online application form.")
            'movie dialogues
            trainingData.Add("Luke: I am your father.")
            trainingData.Add("Darth Vader: Noooo!")
            trainingData.Add("Han Solo: May the Force be with you.")
            trainingData.Add("Princess Leia: I love you.")
            trainingData.Add("Han Solo: I know.")
            trainingData.Add("Yoda: Do or do not. There is no try.")
            trainingData.Add("Obi-Wan Kenobi: You were the chosen one!")
            trainingData.Add("Anakin Skywalker: I hate you!")
            trainingData.Add("Marty McFly: Great Scott!")
            trainingData.Add("Doc Brown: Roads? Where we're going, we don't need roads.")
            trainingData.Add("Tony Stark: I am Iron Man.")
            trainingData.Add("Peter Parker: With great power comes great responsibility.")
            trainingData.Add("Bruce Wayne: I'm Batman.")
            trainingData.Add("Alfred Pennyworth: Why do we fall? So we can learn to pick ourselves up.")
            trainingData.Add("Sherlock Holmes: Elementary, my dear Watson.")
            trainingData.Add("Dr. John Watson: It is a capital mistake to theorize before one has data.")
            trainingData.Add("James Bond: The name's Bond. James Bond.")
            trainingData.Add("Harry Potter: I solemnly swear that I am up to no good.")
            trainingData.Add("Ron Weasley: Bloody hell!")
            trainingData.Add("Hermione Granger: It's LeviOsa, not LevioSA.")
            trainingData.Add("Gandalf: You shall not pass!")
            trainingData.Add("Frodo Baggins: I will take the ring, though I do not know the way.")
            trainingData.Add("Samwise Gamgee: I can't carry it for you, but I can carry you!")
            trainingData.Add("Dumbledore: Happiness can be found even in the darkest of times.")
            trainingData.Add("Severus Snape: Always.")


            paragraphs.AddRange(trainingData)

            Dim inputTexts As String() = {
                "John Doe is a software developer from New York. He specializes in Python programming.",
                "Mary Smith is an artist from Los Angeles. She loves to paint landscapes.",
                "Peter Johnson is a doctor from Chicago. He works at a local hospital.",
                "Sara Williams is a teacher from Boston. She teaches English literature.",
                "David Brown is a musician from Seattle. He plays the guitar in a band.",
                "I am a software developer with 5 years of experience. I have expertise in Python and Java.",
        "As a data scientist, I have a Ph.D. in Machine Learning and 8 years of experience.",
        "I am a web developer skilled in Java and Python. I have worked at Microsoft for 10 years.",
        "I am an electrical engineer with a Master's degree and 8 years of experience in power systems.",
        "As a nurse, I have a Bachelor's degree in Nursing and 5 years of experience in a hospital setting.",
        "I am a graphic designer with expertise in Adobe Photoshop and Illustrator. I have worked freelance for 5 years.",
        "As a teacher, I have a Bachelor's degree in Education and 8 years of experience in primary schools.",
        "I am a mechanical engineer with a Ph.D. in Robotics and 10 years of experience in autonomous systems.",
        "As a lawyer, I have a Juris Doctor degree and 5 years of experience in corporate law.",
        "I am a marketing specialist with expertise in digital marketing and social media management. I have worked at Google for 8 years.",
        "As a chef, I have culinary training and 5 years of experience in high-end restaurants.",
        "I am a financial analyst with a Master's degree in Finance and 8 years of experience in investment banking.",
        "I am a software developer with 5 years of experience. I have expertise in Python and Java.",
        "As a data scientist, I have a Ph.D. in Machine Learning and 8 years of experience.",
        "I am a web developer skilled in Java and Python. I have worked at Microsoft for 10 years.",
        "I am an electrical engineer with a Master's degree and 8 years of experience in power systems.",
        "As a nurse, I have a Bachelor's degree in Nursing and 5 years of experience in a hospital setting.",
        "I am a graphic designer with expertise in Adobe Photoshop and Illustrator. I have worked freelance for 5 years.",
        "As a teacher, I have a Bachelor's degree in Education and 8 years of experience in primary schools.",
        "I am a mechanical engineer with a Ph.D. in Robotics and 10 years of experience in autonomous systems.",
        "As a lawyer, I have a Juris Doctor degree and 5 years of experience in corporate law.",
        "I am a marketing specialist with expertise in digital marketing and social media management. I have worked at Google for 8 years.",
        "As a chef, I have culinary training and 5 years of experience in high-end restaurants.",
        "I am a financial analyst with a Master's degree in Finance and 8 years of experience in investment banking.",
        "I am a software developer with 5 years of experience. I have expertise in Python and Java.",
        "As a data scientist, I have a Ph.D. in Machine Learning and 8 years of experience.",
        "I am a web developer skilled in Java and Python. I have worked at Microsoft for 10 years.",
        "I am an electrical engineer with a Master's degree and 8 years of experience in power systems.",
        "As a nurse, I have a Bachelor's degree in Nursing and 5 years of experience in a hospital setting.",
        "I am a graphic designer with expertise in Adobe Photoshop and Illustrator. I have worked freelance for 5 years.",
        "As a teacher, I have a Bachelor's degree in Education and 8 years of experience in primary schools.",
        "I am a mechanical engineer with a Ph.D. in Robotics and 10 years of experience in autonomous systems.",
        "As a lawyer, I have a Juris Doctor degree and 5 years of experience in corporate law.",
        "I am a marketing specialist with expertise in digital marketing and social media management. I have worked at Google for 8 years.",
        "As a chef, I have culinary training and 5 years of experience in high-end restaurants.",
        "I am a financial analyst with a Master's degree in Finance and 8 years of experience in investment banking."
    }
            paragraphs.AddRange(inputTexts)
            Dim NLP As String = "Natural language processing (NLP) Is a field Of artificial intelligence that focuses On the interaction between computers And humans Using natural language. It combines linguistics, computer science, And machine learning To enable computers To understand, interpret, And generate human language.

Machine learning is a subset of artificial intelligence that deals with the development of algorithms and models that allow computers to learn and make predictions or decisions without being explicitly programmed. It plays a crucial role in various applications, including NLP.

In recent news, researchers at XYZ University have developed a new deep learning algorithm for sentiment analysis in NLP. The algorithm achieved state-of-the-art results on multiple datasets and has the potential to improve various NLP tasks.

Another significant development in the computer science industry is the introduction of GPT-3, a powerful language model developed by OpenAI. GPT-3 utilizes advanced machine learning techniques to generate human-like text and has shown promising results in various language-related tasks.

Key people in the data science and AI industry include Andrew Ng, the founder of deeplearning.ai and a prominent figure in the field of machine learning, and Yann LeCun, the director of AI Research at Facebook and a pioneer in deep learning.

These are just a few examples of the vast field of NLP, machine learning, and the latest developments in the computer science industry."
            paragraphs.Add(NLP)
            paragraphs.AddRange(sentences)
            Return paragraphs
        End Function

    End Class
End Namespace
Namespace LanguageModels
    ''' <summary>
    ''' Corpus Language Model
    ''' Used to HoldDocuments : a corpus of documents Calculating detecting the
    ''' known entitys and topics in the model;
    ''' A known list of Entitys and Topics are required to create this model
    ''' This language model is ideally suited for NER / and other corpus interogations
    '''
    ''' </summary>
    Public Class Corpus

        Public Class iCompare

            Public Shared Function GetDistinctWords(text As String) As HashSet(Of String)
                ' Split the text into words and return a HashSet of distinct words
                Dim words() As String = text.Split({" ", ".", ",", ";", ":", "!", "?"}, StringSplitOptions.RemoveEmptyEntries)
                Dim distinctWords As New HashSet(Of String)(words, StringComparer.OrdinalIgnoreCase)

                Return distinctWords
            End Function

            Public Shared Function BuildWordVector(words As HashSet(Of String)) As Dictionary(Of String, Integer)
                Dim wordVector As New Dictionary(Of String, Integer)

                For Each word As String In words
                    If wordVector.ContainsKey(word) Then
                        wordVector(word) += 1
                    Else
                        wordVector(word) = 1
                    End If
                Next

                Return wordVector
            End Function

            '1. Cosine Similarity Calculation:
            '```vb
            Public Shared Function ComputeCosineSimilarity(phrase1 As String, phrase2 As String) As Double
                Dim words1 As HashSet(Of String) = GetDistinctWords(phrase1)
                Dim words2 As HashSet(Of String) = GetDistinctWords(phrase2)

                Dim wordVector1 As Dictionary(Of String, Integer) = BuildWordVector(words1)
                Dim wordVector2 As Dictionary(Of String, Integer) = BuildWordVector(words2)

                Dim dotProduct As Integer = ComputeDotProduct(wordVector1, wordVector2)
                Dim magnitude1 As Double = ComputeVectorMagnitude(wordVector1)
                Dim magnitude2 As Double = ComputeVectorMagnitude(wordVector2)

                ' Compute the cosine similarity as the dot product divided by the product of magnitudes
                Dim similarityScore As Double = dotProduct / (magnitude1 * magnitude2)

                Return similarityScore
            End Function

            Public Shared Function ComputeDotProduct(vector1 As Dictionary(Of String, Integer), vector2 As Dictionary(Of String, Integer)) As Integer
                Dim dotProduct As Integer = 0

                For Each word As String In vector1.Keys
                    If vector2.ContainsKey(word) Then
                        dotProduct += vector1(word) * vector2(word)
                    End If
                Next

                Return dotProduct
            End Function

            '2. Jaccard Similarity Calculation:
            '```vb
            Public Shared Function ComputeJaccardSimilarity(phrase1 As String, phrase2 As String) As Double
                Dim words1 As HashSet(Of String) = GetDistinctWords(phrase1)
                Dim words2 As HashSet(Of String) = GetDistinctWords(phrase2)

                Dim intersectionCount As Integer = words1.Intersect(words2).Count()
                Dim unionCount As Integer = words1.Count + words2.Count - intersectionCount

                ' Compute the Jaccard Similarity as the ratio of intersection count to union count
                Dim similarityScore As Double = intersectionCount / unionCount

                Return similarityScore
            End Function

            Public Shared Function ComputeSimilarityScore(phrase As String, contextLine As String) As Double
                ' Here you can implement your own logic for computing the similarity score between the phrase and the context line.
                ' For simplicity, let's use a basic approach that counts the number of common words between them.

                Dim phraseWords As HashSet(Of String) = GetDistinctWords(phrase)
                Dim contextWords As HashSet(Of String) = GetDistinctWords(contextLine)

                Dim commonWordsCount As Integer = phraseWords.Intersect(contextWords).Count()

                Dim totalWordsCount As Integer = phraseWords.Count + contextWords.Count

                ' Compute the similarity score as the ratio of common words count to total words count
                Dim similarityScore As Double = commonWordsCount / totalWordsCount

                Return similarityScore
            End Function

            Public Shared Function ComputeVectorMagnitude(vector As Dictionary(Of String, Integer)) As Double
                Dim magnitude As Double = 0

                For Each count As Integer In vector.Values
                    magnitude += count * count
                Next

                magnitude = Math.Sqrt(magnitude)

                Return magnitude
            End Function

        End Class

        ''' <summary>
        ''' Used to create NewCorpus - With Or Without a Recognition template
        ''' </summary>
        Public Class ProcessInputAPI
            Private iCurrentOriginalText As String
            Private KnownEntitys As Corpus.Recognition_Data

            Public Sub New(ByRef KnownData As Corpus.Recognition_Data)
                Me.KnownEntitys = KnownData
            End Sub

            Public Sub New()
                KnownEntitys = New Corpus.Recognition_Data
            End Sub

            Public ReadOnly Property CurrentInput As String
                Get
                    Return iCurrentOriginalText
                End Get
            End Property

            Public Function ProcessDocument(ByRef InputText As String) As Corpus
                Dim iCorpus As New Corpus(KnownEntitys)
                iCorpus.AddDocument(InputText)
                Return iCorpus
            End Function

            Public Function ProcessCorpus(ByRef InputText As List(Of String)) As Corpus
                Dim iCorpus As New Corpus(KnownEntitys)
                iCorpus.AddCorpus(InputText)
                Return iCorpus
            End Function

        End Class

        Public Shared Function ExtractSimilarPhrases(text As String, searchPhrase As String, similarityThreshold As Double) As List(Of String)
            Dim result As New List(Of String)()

            Dim sentences() As String = text.Split({".", "!", "?"}, StringSplitOptions.RemoveEmptyEntries)

            For Each sentence As String In sentences
                Dim similarityScore As Double = iCompare.ComputeSimilarityScore(searchPhrase, sentence)

                If similarityScore >= similarityThreshold Then
                    result.Add(sentence)
                End If
            Next

            Return result
        End Function

        Public Shared Function QueryCorpus(question As String, corpus As List(Of String)) As String
            Dim maxScore As Double = Double.MinValue
            Dim bestAnswer As String = ""

            For Each document As String In corpus
                Dim score As Double = iCompare.ComputeSimilarityScore(question, document)

                If score > maxScore Then
                    maxScore = score
                    bestAnswer = document
                End If
            Next

            Return bestAnswer
        End Function

        ''' <summary>
        ''' Returns phrase and surrounding comments and position
        ''' </summary>
        ''' <param name="corpus"></param>
        ''' <param name="phrase"></param>
        ''' <returns></returns>
        Public Shared Function SearchPhraseInCorpus(corpus As List(Of String), phrase As String) As Dictionary(Of String, List(Of String))
            Dim result As New Dictionary(Of String, List(Of String))()

            For i As Integer = 0 To corpus.Count - 1
                Dim document As String = corpus(i)
                Dim lines() As String = document.Split(Environment.NewLine)

                For j As Integer = 0 To lines.Length - 1
                    Dim line As String = lines(j)
                    Dim index As Integer = line.IndexOf(phrase, StringComparison.OrdinalIgnoreCase)

                    While index >= 0
                        Dim context As New List(Of String)()

                        ' Get the surrounding context sentences
                        Dim startLine As Integer = Math.Max(0, j - 1)
                        Dim endLine As Integer = Math.Min(lines.Length - 1, j + 1)

                        For k As Integer = startLine To endLine
                            context.Add(lines(k))
                        Next

                        ' Add the result to the dictionary
                        Dim position As String = $"Document: {i + 1}, Line: {j + 1}, Character: {index + 1}"
                        result(position) = context

                        ' Continue searching for the phrase in the current line
                        index = line.IndexOf(phrase, index + 1, StringComparison.OrdinalIgnoreCase)
                    End While
                Next
            Next

            Return result
        End Function

        ''' <summary>
        ''' Searches for phrases based on simularity ie same words
        ''' </summary>
        ''' <param name="corpus"></param>
        ''' <param name="phrase"></param>
        ''' <param name="similarityThreshold"></param>
        ''' <returns></returns>
        Public Shared Function SearchPhraseInCorpus(corpus As List(Of String), phrase As String, similarityThreshold As Double) As Dictionary(Of String, List(Of String))
            Dim result As New Dictionary(Of String, List(Of String))()

            For i As Integer = 0 To corpus.Count - 1
                Dim document As String = corpus(i)
                Dim lines() As String = document.Split(Environment.NewLine)

                For j As Integer = 0 To lines.Length - 1
                    Dim line As String = lines(j)
                    Dim index As Integer = line.IndexOf(phrase, StringComparison.OrdinalIgnoreCase)

                    While index >= 0
                        Dim context As New List(Of String)()

                        ' Get the surrounding context sentences
                        Dim startLine As Integer = Math.Max(0, j - 1)
                        Dim endLine As Integer = Math.Min(lines.Length - 1, j + 1)

                        For k As Integer = startLine To endLine
                            Dim contextLine As String = lines(k)

                            ' Compute the similarity score between the context line and the phrase
                            Dim similarityScore As Double = iCompare.ComputeSimilarityScore(phrase, contextLine)

                            ' Add the context line only if its similarity score exceeds the threshold
                            If similarityScore >= similarityThreshold Then
                                context.Add(contextLine)
                            End If
                        Next

                        ' Add the result to the dictionary
                        Dim position As String = $"Document: {i + 1}, Line: {j + 1}, Character: {index + 1}"
                        result(position) = context

                        ' Continue searching for the phrase in the current line
                        index = line.IndexOf(phrase, index + 1, StringComparison.OrdinalIgnoreCase)
                    End While
                Next
            Next

            Return result
        End Function

        Public Function ToJson(ByRef iObject As Object) As String
            Dim Converter As New JavaScriptSerializer
            Return Converter.Serialize(iObject)
        End Function

        Public Class Tokenizer

            ''' <summary>
            ''' Normalizes the input string by converting it to lowercase and removing punctuation and extra whitespace.
            ''' </summary>
            ''' <param name="input">The input string.</param>
            ''' <returns>The normalized input string.</returns>
            Public Function NormalizeInput(input As String) As String
                ' Convert to lowercase
                Dim normalizedInput As String = input.ToLower()

                ' Remove punctuation
                normalizedInput = Regex.Replace(normalizedInput, "[^\w\s]", "")

                ' Remove extra whitespace
                normalizedInput = Regex.Replace(normalizedInput, "\s+", " ")

                Return normalizedInput
            End Function

            ''' <summary>
            ''' Tokenizes the input string by character.
            ''' </summary>
            ''' <param name="input">The input string.</param>
            ''' <returns>The list of character tokens.</returns>
            Public Shared Function TokenizeByCharacter(input As String) As List(Of Token)
                Dim tokens As New List(Of Token)

                For i As Integer = 0 To input.Length - 1
                    Dim token As New Token(input(i).ToString())
                    tokens.Add(token)
                Next

                Return tokens
            End Function

            ''' <summary>
            ''' Tokenizes the input string by word.
            ''' </summary>
            ''' <param name="input">The input string.</param>
            ''' <returns>The list of word tokens.</returns>
            Public Shared Function TokenizeByWord(input As String) As List(Of Token)
                Dim tokens As New List(Of Token)
                Dim words As String() = input.Split(" "c)

                For i As Integer = 0 To words.Length - 1
                    Dim token As New Token(words(i))
                    tokens.Add(token)
                Next

                Return tokens
            End Function

            ''' <summary>
            ''' Tokenizes the input string by sentence.
            ''' </summary>
            ''' <param name="input">The input string.</param>
            ''' <returns>The list of sentence tokens.</returns>
            Public Shared Function TokenizeBySentence(input As String) As List(Of Token)
                Dim tokens As New List(Of Token)
                Dim sentences As String() = input.Split("."c)

                For i As Integer = 0 To sentences.Length - 1
                    Dim token As New Token(sentences(i))
                    tokens.Add(token)
                Next

                Return tokens
            End Function

            ''' <summary>
            ''' Tokenizes the input string by whitespace.
            ''' </summary>
            ''' <param name="input">The input string.</param>
            ''' <returns>The list of tokens.</returns>
            Public Shared Function Tokenize(input As String) As List(Of String)
                ' Simple tokenization by splitting on whitespace
                Return New List(Of String)(input.Split({" "c}, StringSplitOptions.RemoveEmptyEntries))
            End Function

            Public Class Token

                ''' <summary>
                ''' Initializes a new instance of the Token class.
                ''' </summary>
                ''' <param name="value">The string value of the token.</param>
                Public Sub New(value As String)
                    If value Is Nothing Then
                        Throw New ArgumentNullException(NameOf(value))
                    End If

                    Me.Value = value
                End Sub

                ''' <summary>
                ''' Initializes a new instance of the Token class with sequence encoding.
                ''' </summary>
                ''' <param name="value">The string value of the token.</param>
                ''' <param name="sequenceEncoding">The sequence encoding value of the token.</param>
                Public Sub New(value As String, sequenceEncoding As Integer)
                    Me.New(value)
                    Me.SequenceEncoding = sequenceEncoding
                End Sub

                ''' <summary>
                ''' Gets or sets the embeddings of the token.
                ''' </summary>
                Public Property Embeddings As List(Of Double)

                ''' <summary>
                ''' Calculates the similarity between this token and the given token.
                ''' </summary>
                ''' <param name="token">The other token.</param>
                ''' <returns>The similarity value between the tokens.</returns>
                Private Function CalculateSimilarity(token As Token) As Double
                    If Embeddings IsNot Nothing AndAlso token.Embeddings IsNot Nothing Then
                        Dim dotProduct As Double = 0.0
                        Dim magnitudeA As Double = 0.0
                        Dim magnitudeB As Double = 0.0

                        For i As Integer = 0 To Embeddings.Count - 1
                            dotProduct += Embeddings(i) * token.Embeddings(i)
                            magnitudeA += Math.Pow(Embeddings(i), 2)
                            magnitudeB += Math.Pow(token.Embeddings(i), 2)
                        Next

                        magnitudeA = Math.Sqrt(magnitudeA)
                        magnitudeB = Math.Sqrt(magnitudeB)

                        If magnitudeA = 0.0 OrElse magnitudeB = 0.0 Then
                            Return 0.0
                        Else
                            Return dotProduct / (magnitudeA * magnitudeB)
                        End If
                    Else
                        Return 0.0
                    End If
                End Function

                ''' <summary>
                ''' Gets or sets the string value of the token.
                ''' </summary>
                Public Property Value As String

                ''' <summary>
                ''' Gets or sets the sequence encoding value of the token.
                ''' </summary>
                Public Property SequenceEncoding As Integer

                ''' <summary>
                ''' Gets or sets the positional encoding value of the token.
                ''' </summary>
                Public Property PositionalEncoding As Integer

                ''' <summary>
                ''' Gets or sets the frequency of the token in the language model corpus.
                ''' </summary>
                Public Property Frequency As Double

                ''' <summary>
                ''' Gets or sets the embedding vector of the token.
                ''' </summary>
                Public Property Embedding As Double

                Public Function CalculateSelfAttention(tokens As List(Of Token)) As Double
                    Dim total As Double = 0.0
                    For Each token As Token In tokens
                        total += CalcSimilarity(token)
                    Next
                    Return Math.Log(Math.Sqrt(total))
                End Function

                Private Function CalcSimilarity(token As Token) As Double
                    If Embeddings IsNot Nothing AndAlso token.Embeddings IsNot Nothing Then
                        Dim dotProduct As Double = 0.0
                        For i As Integer = 0 To Embeddings.Count - 1
                            dotProduct += Embeddings(i) * token.Embeddings(i)
                        Next
                        Return dotProduct
                    End If
                    Return 0.0
                End Function

                ''' <summary>
                ''' Calculates the self-attention of the token within the given list of tokens.
                ''' </summary>
                ''' <param name="tokens">The list of tokens.</param>
                ''' <returns>The self-attention value of the token.</returns>
                Public Function CalculateAttention(tokens As List(Of Token)) As Double
                    Dim qVector As List(Of Double) = Me.Embeddings
                    Dim kMatrix As New List(Of Double)
                    Dim vMatrix As New List(Of Double)

                    ' Create matrices K and V
                    For Each token In tokens
                        kMatrix.Add(token.Embedding)
                        vMatrix.Add(token.Embedding)
                    Next

                    ' Compute self-attention
                    Dim attention As Double = 0.0
                    Dim sqrtKLength As Double = Math.Sqrt(kMatrix(0))

                    For i As Integer = 0 To kMatrix.Count - 1
                        Dim kVector As List(Of Double) = kMatrix
                        Dim dotProduct As Double = 0.0

                        ' Check vector dimensions
                        If qVector.Count = kVector.Count Then
                            For j As Integer = 0 To qVector.Count - 1
                                dotProduct += qVector(j) * kVector(j)
                            Next

                            dotProduct /= sqrtKLength
                            attention += dotProduct * vMatrix(i) ' We consider only the first element of the value vector for simplicity
                        Else
                            ' Handle case when vector dimensions do not match
                            Console.WriteLine("Vector dimensions do not match.")
                        End If
                    Next

                    Return attention
                End Function

            End Class

        End Class

        ''' <summary>
        ''' An array of characters (. ! ?) used to tokenize sentences.
        ''' </summary>
        Public Shared ReadOnly SentenceEndMarkers As Char() = {".", "!", "?"}

        Public CorpusContext As List(Of Vocabulary.FeatureContext)

        ''' <summary>
        ''' A list of strings representing the documents in the corpus.
        ''' </summary>
        Public CorpusDocs As List(Of String)

        ''' <summary>
        ''' A string representing the concatenated text of all documents in the corpus.
        ''' </summary>
        Public CorpusText As String

        ''' <summary>
        '''  A list of unique words in the corpus.
        ''' </summary>
        Public CorpusUniqueWords As List(Of String)

        ''' <summary>
        ''' TotalWords in Corpus
        ''' </summary>
        Public ReadOnly Property CorpusWordcount As Integer
            Get
                Return GetWordCount()
            End Get
        End Property

        ''' <summary>
        '''  A list of Document objects representing individual documents in the corpus.
        ''' </summary>
        Public Documents As List(Of Document)

        ''' <summary>
        ''' A list of Entity structures representing detected entities in the corpus.
        ''' </summary>
        Public Entitys As List(Of Entity)

        ''' <summary>
        ''' A Vocabulary object representing the language model.
        ''' </summary>
        Public Langmodel As Vocabulary

        ''' <summary>
        ''' A Recognition_Data structure representing named entity recognition data.
        ''' </summary>
        Public NER As Recognition_Data

        ''' <summary>
        ''' A list of Topic structures representing detected topics in the corpus.
        ''' </summary>
        Public Topics As List(Of Topic)

        ''' <summary>
        ''' Initializes a new instance of the Corpus class.
        ''' </summary>
        ''' <param name="data">The recognition data for entity and topic detection.</param>
        Public Sub New(ByVal data As Recognition_Data)
            NER = data
            Documents = New List(Of Document)
            Entitys = New List(Of Entity)
            Topics = New List(Of Topic)
            CorpusDocs = New List(Of String)
            CorpusUniqueWords = New List(Of String)
            CorpusText = String.Empty

            Langmodel = New Vocabulary
        End Sub

        ''' <summary>
        ''' type of sentence
        ''' </summary>
        Public Enum SentenceType
            Unknown = 0
            Declaritive = 1
            Interogative = 2
            Exclamitory = 3
            Conditional = 4
            Inference = 5
            Imperitive = 6
        End Enum

        ''' <summary>
        ''' Processes the text by removing unwanted characters, converting to lowercase, and removing extra whitespace.
        ''' </summary>
        ''' <param name="text"></param>
        ''' <returns></returns>
        Public Shared Function ProcessText(ByRef text As String) As String
            ' Remove unwanted characters
            Dim processedText As String = Regex.Replace(text, "[^a-zA-Z0-9\s]", "")

            ' Convert to lowercase
            processedText = processedText.ToLower()

            ' Remove extra whitespace
            processedText = Regex.Replace(processedText, "\s+", " ")

            Return processedText
        End Function

        ''' <summary>
        ''' Adds a corpus of documents to the existing corpus.
        ''' </summary>
        ''' <param name="docs"></param>
        ''' <returns></returns>
        Public Function AddCorpus(ByRef docs As List(Of String)) As Corpus

            'Add aCorpus of documents to the corpus

            For Each item In docs

                AddDocument(item)

            Next
            UpdateContext()
            Return Me

        End Function

        ''' <summary>
        ''' Adds a document to the corpus and updates the corpus properties.
        ''' </summary>
        ''' <param name="Text"></param>
        Public Sub AddDocument(ByRef Text As String)
            Dim Doc As New Document(Text)
            Documents.Add(Doc.AddDocument(ProcessText(Text)))
            'Update Corpus
            CorpusDocs.Add(ProcessText(Text))

            CorpusUniqueWords = GetUniqueWords()

            Dim iText As String = ""
            For Each item In Documents
                iText &= item.ProcessedText & vbNewLine

            Next
            CorpusText = iText

            '' corpus entitys and topics
            Doc.Entitys = Entity.DetectEntitys(Doc.ProcessedText, NER.Entitys)
            Doc.Topics = Topic.DetectTopics(Doc.ProcessedText, NER.Topics)
            Entitys.AddRange(Doc.Entitys)
            Entitys = Entitys

            Topics.AddRange(Doc.Topics)
            Topics = Topics
            'Update VocabModel

            Dim Wrds = Text.Split(" ")

            For Each item In Wrds
                Langmodel.AddNew(item, CorpusDocs)
            Next
        End Sub

        ''' <summary>
        ''' Retrieves the list of unique words in the corpus.
        ''' </summary>
        ''' <returns></returns>
        Public Function GetUniqueWords() As List(Of String)
            Dim lst As New List(Of String)
            For Each item In Documents
                lst.AddRange(item.UniqueWords)
            Next
            Return lst
        End Function

        ''' <summary>
        ''' Retrieves the total word count in the corpus.
        ''' </summary>
        ''' <returns></returns>
        Public Function GetWordCount() As Integer
            Dim count As Integer = 0
            For Each item In Documents
                count += item.WordCount
            Next
            Return count
        End Function

        ''' <summary>
        ''' Updates the Features in the model (each document context)
        ''' by the topics discovered in the text, updating the individual documents and adding the
        ''' feature context to the corpus context
        ''' </summary>
        Private Sub UpdateContext()
            CorpusContext = New List(Of Vocabulary.FeatureContext)
            For Each Topic In Topics.Distinct
                For Each doc In Documents
                    Dim Context = Vocabulary.FeatureContext.GetDocumentContext(Langmodel, doc, Topic.Topic)
                    doc.Features.Add(Context)
                    CorpusContext.Add(Context)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Represents an individual document in the corpus. It contains properties such as word count, processed text, sentences, topics, etc.
        ''' </summary>
        Public Structure Document

            Public ReadOnly Property WordCount As Integer
                Get
                    Return GetWordCount()
                End Get
            End Property

            Private Function GetWordCount() As Integer
                Dim Str = Functs.TokenizeWords(OriginalText)
                Return Str.Count
            End Function

            '''' <summary>
            '''' COntains the Vocabulary for this document
            '''' </summary>
            Public DocumentVocabulary As Vocabulary

            Public Entitys As List(Of Entity)

            ''' <summary>
            ''' Context can be updated by the corpus owner as required, these contexts
            ''' can be used to score the document and provided higher embedding values
            ''' </summary>
            Public Features As List(Of Vocabulary.FeatureContext)

            ''' <summary>
            ''' Preserve original
            ''' </summary>
            Public OriginalText As String

            ''' <summary>
            ''' Cleaned Text
            ''' </summary>
            Public ProcessedText As String

            ''' <summary>
            ''' Sentences within Text
            ''' </summary>
            Public Sentences As List(Of Sentence)

            Public Topics As List(Of Topic)
            Public TopWords As List(Of String)
            Public UniqueWords As List(Of String)

            Public Sub New(ByRef originalText As String)

                Me.OriginalText = originalText
                Topics = New List(Of Topic)
                TopWords = New List(Of String)
                UniqueWords = New List(Of String)
                Sentences = New List(Of Sentence)
                DocumentVocabulary = New Vocabulary
                Entitys = New List(Of Entity)
            End Sub

            Public Function AddDocument(ByRef Text As String) As Document
                OriginalText = Text
                'Remove unwanted symbols
                ProcessedText = ProcessText(Text)

                Dim Sents As List(Of String) = Text.Split(".").ToList
                Dim Count As Integer = 0
                For Each item In Sents
                    Count += 1
                    Dim Sent As New Sentence(item)
                    Me.Sentences.Add(Sent.AddSentence(item, Count))
                Next
                UniqueWords = Corpus.Functs.GetUniqueWordsInText(ProcessedText)
                Dim IDocs As New List(Of String)
                'Adds only its-self to its own personal corpus vocabulary(document Specific)
                IDocs.Add(ProcessedText)
                For Each item In UniqueWords
                    DocumentVocabulary.AddNew(item, IDocs)
                Next
                TopWords = Corpus.Functs.GetTopWordsInText(ProcessedText)

                Return Me
            End Function

            Public Structure Sentence

                Public Clauses As List(Of Clause)

                Public Entitys As List(Of Entity)

                Public OriginalSentence As String

                Public Position As Integer

                Public ProcessedSentence As String

                Public UniqueWords As List(Of String)

                Private iSentencetype As SentenceType

                Public Sub New(originalSentence As String)
                    Me.New()
                    Me.OriginalSentence = originalSentence
                    Clauses = New List(Of Clause)
                    Entitys = New List(Of Entity)
                    UniqueWords = New List(Of String)
                End Sub

                Public ReadOnly Property ClauseCount As Integer
                    Get
                        Return Clauses.Count
                    End Get

                End Property

                Public ReadOnly Property SentenceType As String
                    Get
                        Select Case iSentencetype
                            Case Corpus.SentenceType.Conditional
                                Return "Conditional"
                            Case Corpus.SentenceType.Declaritive
                                Return "Declarative"
                            Case Corpus.SentenceType.Exclamitory
                                Return "exclamatory"
                            Case Corpus.SentenceType.Imperitive
                                Return "imperative"
                            Case Corpus.SentenceType.Inference
                                Return "inference"
                            Case Corpus.SentenceType.Interogative
                                Return "interrogative"
                            Case Corpus.SentenceType.Unknown
                                Return "unknown"
                            Case Else
                                Return "unknown"
                        End Select
                    End Get
                End Property

                Public ReadOnly Property WordCount As Integer
                    Get
                        Return GetWordCount(ProcessedSentence)
                    End Get
                End Property

                Public Shared Function GetClauses(ByRef Text As String) As List(Of Clause)
                    Dim clauses As New List(Of Clause)

                    '

                    If Text.Contains(",") Then
                        Dim iClauses As List(Of String) = Text.Split(",").ToList
                        For Each item In iClauses
                            Dim Iclause As New Clause
                            Iclause.Text = item
                            Iclause.ClauseSeperator = ","
                            Dim Words = Functs.TokenizeWords(Iclause.Text)
                            Dim count As Integer = 0
                            For Each wrd In Words
                                count += 1
                                Iclause.Words.Add(New Clause.Word(wrd, count))

                            Next

                            clauses.Add(Iclause)

                        Next
                    Else

                        'Add detect end punctuation use for

                        Dim Iclause As New Clause
                        Iclause.Words = New List(Of Clause.Word)
                        Iclause.Text = Text
                        'Use end punctuation
                        Iclause.ClauseSeperator = "."
                        Dim Words = Functs.TokenizeWords(Iclause.Text)
                        Dim count As Integer = 0
                        If Words.Count > 0 Then
                            For Each wrd In Words

                                count += 1
                                Iclause.Words.Add(New Clause.Word(wrd, count))

                            Next
                        End If
                        clauses.Add(Iclause)

                    End If
                    Return clauses
                End Function

                Public Function AddSentence(ByRef text As String, ByRef iPosition As Integer) As Sentence
                    OriginalSentence = text
                    ProcessedSentence = ProcessText(text)
                    Clauses = GetClauses(ProcessedSentence)
                    UniqueWords = Corpus.Functs.GetUniqueWordsInText(ProcessedSentence)

                    Position = iPosition
                    Return Me
                End Function

                Private Function GetWordCount(ByRef Text As String) As Integer
                    Dim Str = Functs.TokenizeWords(Text)
                    Return Str.Count
                End Function

                ''' <summary>
                ''' Represents a clause within a sentence. It contains properties such as text, word count, words, etc.
                ''' </summary>
                Public Structure Clause

                    ''' <summary>
                    ''' Independent Clause / Dependant Clause
                    ''' </summary>
                    Public Clause As String

                    Public ClauseSeperator As String
                    Public ClauseType As SentenceType

                    ''' <summary>
                    ''' Note: if = "." then declarative, = "?" Question = "!" Exclamitory
                    ''' </summary>
                    Public EndPunctuation As String

                    Public Text As String
                    Public Words As List(Of Clause.Word)
                    Private mLearningPattern As String

                    Private mPredicate As String

                    Private mSubjectA As String

                    Private mSubjectB As String

                    ''' <summary>
                    ''' the learning pattern locates the Subjects in the sentence A# sat on #b
                    ''' </summary>
                    ''' <returns></returns>
                    Public Property LearningPattern As String
                        Get
                            Return mLearningPattern
                        End Get
                        Set(value As String)
                            mLearningPattern = value
                        End Set
                    End Property

                    ''' <summary>
                    ''' Predicate / Linking verb / Concept (Sat on) (is sitting on) (AtLocation) this is the
                    ''' dividing content in the sentence
                    ''' </summary>
                    ''' <returns></returns>
                    Public Property Predicate As String
                        Get
                            Return mPredicate
                        End Get
                        Set(value As String)
                            mPredicate = value
                        End Set
                    End Property

                    ''' <summary>
                    ''' First detected subject (the Cat)
                    ''' </summary>
                    ''' <returns></returns>
                    Public Property SubjectA As String
                        Get
                            Return mSubjectA
                        End Get
                        Set(value As String)
                            mSubjectA = value
                        End Set
                    End Property

                    ''' <summary>
                    ''' Second detected subject / Object (the mat)
                    ''' </summary>
                    ''' <returns></returns>
                    Public Property SubjectB As String
                        Get
                            Return mSubjectB
                        End Get
                        Set(value As String)
                            mSubjectB = value
                        End Set
                    End Property

                    Public ReadOnly Property WordCount As Integer
                        Get
                            Return Words.Count
                        End Get

                    End Property

                    ''' <summary>
                    ''' Represents a word in the text
                    ''' </summary>
                    Public Structure Word

                        ''' <summary>
                        ''' Position of word in Sentence/Document
                        ''' </summary>
                        Public Position As Integer

                        ''' <summary>
                        ''' Word
                        ''' </summary>
                        Public text As String

                        Public Sub New(word As String, position As Integer)
                            If word Is Nothing Then
                                Throw New ArgumentNullException(NameOf(word))
                            End If

                            Me.text = word
                            Me.Position = position

                        End Sub

                    End Structure

                End Structure

            End Structure

        End Structure

        ''' <summary>
        ''' NER Data held(known) by the corpus
        ''' </summary>
        Public Class Recognition_Data
            Public Entitys As List(Of Entity)
            Public Topics As List(Of Topic)

            Public Sub New()
                Entitys = New List(Of Entity)
                Topics = New List(Of Topic)
            End Sub

        End Class

        Public Structure Term
            Public DocNumber As List(Of Integer)

            ''' <summary>
            ''' Term Frequency
            ''' </summary>
            Dim Freq As Integer

            ''' <summary>
            ''' Inverse Document Frequency
            ''' </summary>
            Dim IDF As Double

            ''' <summary>
            ''' Value
            ''' </summary>
            Dim Term As String

        End Structure

        ''' <summary>
        ''' Represents a topic detected in the text. It has properties for the keyword and topic itself.
        ''' </summary>
        Public Structure Topic
            Public Keyword As String
            Public Topic As String

            Public Shared Function DetectTopics(ByRef text As String, TopicList As List(Of Topic)) As List(Of Topic)
                Dim detectedTopics As New List(Of Topic)()
                For Each item In TopicList
                    If text.ToLower.Contains(item.Keyword) Then
                        detectedTopics.Add(item)
                    End If
                Next

                Return detectedTopics
            End Function

        End Structure

        Public Class Functs

            ''' <summary>
            ''' Returns the top words in a given text
            ''' </summary>
            ''' <param name="text"></param>
            ''' <returns></returns>
            Public Shared Function GetTopWordsInText(ByRef text As String) As List(Of String)
                Dim words As List(Of String) = Functs.TokenizeWords(text)
                Dim wordCounts As New Dictionary(Of String, Integer)()

                For Each word As String In words
                    If wordCounts.ContainsKey(word) Then
                        wordCounts(word) += 1
                    Else
                        wordCounts(word) = 1
                    End If
                Next

                ' Sort the words based on their counts in descending order
                Dim sortedWords As List(Of KeyValuePair(Of String, Integer)) = wordCounts.OrderByDescending(Function(x) x.Value).ToList()

                ' Get the top 10 words
                Dim topWords As List(Of String) = sortedWords.Take(10).Select(Function(x) x.Key).ToList()

                Return topWords
            End Function

            ''' <summary>
            ''' Returns a list of the unique words in the text
            ''' </summary>
            ''' <param name="text"></param>
            ''' <returns></returns>
            Public Shared Function GetUniqueWordsInText(ByRef text As String) As List(Of String)
                Dim words As List(Of String) = Functs.TokenizeWords(text)
                Dim uniqueWords As List(Of String) = words.Distinct().ToList()
                Return uniqueWords
            End Function

            Public Shared Sub PrintSentencesToConsole(ByRef iSentences As List(Of String))
                For Each sentence In iSentences
                    Console.WriteLine(sentence)
                Next
            End Sub

            ''' <summary>
            ''' Tokenizes the text into sentences based on punctuation end markers.
            ''' </summary>
            ''' <param name="text">The text to tokenize.</param>
            ''' <returns>A list of sentences.</returns>
            Public Shared Function TokenizeSentences(ByVal text As String) As List(Of Document.Sentence)
                Dim sentences As New List(Of Document.Sentence)()

                ' Split text into sentences based on punctuation end markers
                Dim pattern As String = $"(?<=[{String.Join("", SentenceEndMarkers)}])\s+"
                Dim sentenceTexts As String() = Regex.Split(text, pattern)

                For Each sentenceText As String In sentenceTexts
                    Dim isentence As New Document.Sentence()
                    isentence.OriginalSentence = sentenceText.Trim()

                    isentence.Clauses = Document.Sentence.GetClauses(text)
                    ' ... other sentence properties ...
                    sentences.Add(isentence)
                Next

                Return sentences
            End Function

            ''' <summary>
            ''' Tokenizes the sentence into words.
            ''' </summary>
            ''' <param name="sentenceText">The text of the sentence.</param>
            ''' <returns>A list of words.</returns>
            Public Shared Function TokenizeWords(ByVal sentenceText As String) As List(Of String)
                Dim words As New List(Of String)()

                ' Split sentence into words
                Dim wordPattern As String = "\b\w+\b"
                Dim wordMatches As MatchCollection = Regex.Matches(sentenceText, wordPattern)

                For Each match As Match In wordMatches
                    words.Add(match.Value.ToLower())
                Next

                Return words
            End Function

            Public Shared Function Top_N_Words(ByRef iDocContents As String, ByRef n As Integer) As List(Of String)
                Dim words As String() = iDocContents.Split(" ")
                Dim wordCount As New Dictionary(Of String, Integer)

                ' Count the frequency of each word in the corpus
                For Each word As String In words
                    If wordCount.ContainsKey(word) Then
                        wordCount(word) += 1
                    Else
                        wordCount.Add(word, 1)
                    End If
                Next

                ' Sort the dictionary by value (frequency) in descending order
                Dim sortedDict = (From entry In wordCount Order By entry.Value Descending Select entry).Take(n)
                Dim LSt As New List(Of String)
                ' Print the top ten words and their frequency
                For Each item In sortedDict
                    LSt.Add(item.Key)

                Next
                Return LSt
            End Function

        End Class

        ''' <summary>
        ''' Represents the vocabulary model for the corpus.
        ''' (a record of words which can be looked up in the corpus)
        ''' It includes methods for adding new terms, calculating frequencies, TF-IDF, etc.
        ''' </summary>
        Public Class Vocabulary
            Public Current As List(Of VocabularyEntry)

            ''' <summary>
            ''' Used for TDM Calc
            ''' </summary>
            Private Docs As List(Of String)

            ''' <summary>
            ''' Prepare vocabulary for use
            ''' </summary>
            Public Sub New()
                Current = New List(Of VocabularyEntry)
                Docs = New List(Of String)
            End Sub

            ''' <summary>
            ''' Used to add Words or update a word in the vocabulary language model
            ''' </summary>
            ''' <param name="Term"></param>
            ''' <param name="Docs"></param>
            Public Sub AddNew(ByRef Term As String, ByRef Docs As List(Of String))
                Me.Docs = Docs
                Current.Add(New VocabularyEntry(Term,
                          CalcSequenceEncoding(Term),
                          CalcFrequency(Term),
                          CalcTF_IDF(Term)))

            End Sub

            Private Function CalcFrequency(ByRef Word As String) As Double
                ' Calculate frequency of term in the corpus (current)
                Dim count As Integer = 0
                For Each entry In Current
                    If entry.Text = Word Then

                        count += 1 + entry.Frequency
                    Else
                        Return 1
                    End If
                Next
                Return count
            End Function

            Public Function GetEntry(ByRef Token As String) As VocabularyEntry
                For Each item In Current
                    If item.Text = Token Then Return item
                Next
                Return Nothing
            End Function

            Public Function GetEntry(ByRef SequenceEmbedding As Integer) As VocabularyEntry
                For Each item In Current
                    If item.SequenceEncoding = SequenceEmbedding Then Return item
                Next
                Return Nothing
            End Function

            Public Function CheckEntry(ByRef Token As String) As Boolean
                For Each item In Current
                    If item.Text = Token Then Return True
                Next
                Return False
            End Function

            Private Function CalcInverseDocumentFrequency(ByRef Word As String, ByRef Docs As List(Of String)) As Double
                ' Calculate Inverse Document Frequency for the given term in the corpus
                Dim docsWithTerm As Integer = 0
                For Each doc In Docs
                    If doc.Contains(Word) Then
                        docsWithTerm += 1
                    End If
                Next
                Dim idf As Double = Math.Log(Docs.Count / (docsWithTerm + 1)) ' Adding 1 to avoid division by zero
                Return idf
            End Function

            Private Function CalcSequenceEncoding(ByRef Word As String) As Double
                ' Calculate sequence encoding based on the order of appearance in the corpus
                Dim encoding As Double = 0.0
                For Each entry In Current
                    If entry.Text = Word Then
                        encoding += 1
                    End If
                Next
                Return encoding
            End Function

            Private Function CalcTermFrequency(ByRef Word As String) As Double
                ' Calculate Term Frequency for the given term in the corpus
                Dim count As Integer = 0
                For Each entry In Current
                    If entry.Text = Word Then
                        count += 1
                    End If
                Next
                Return count
            End Function

            Private Function CalcTF_IDF(ByRef Word As String) As Double
                ' Calculate TF-IDF (Term Frequency-Inverse Document Frequency) for the given term in the corpus
                Dim tf As Double = CalcTermFrequency(Word)
                Dim idf As Double = CalcInverseDocumentFrequency(Word, Docs)
                Return tf * idf
            End Function

            ''' <summary>
            ''' Feature context is a way to add information with regards to the document,
            ''' Addind context elements such as features.
            ''' Given a Sentiment (positive) , by marking the words in this document
            ''' present against the corpus vocabulary, it could be suggested that these would
            ''' represent that topic in this document
            ''' </summary>
            Public Structure FeatureContext

                ''' <summary>
                ''' List of items Representing the context,
                ''' All entrys contained in the vocabulary are marked with a tag (present)(true)
                ''' if the are in the context else marked false
                ''' giving a one-shot encoding for the context this collection represents,
                ''' Ie:Sentiment/Topic etc
                ''' </summary>
                Public Present As List(Of VocabularyEntry)

                Public Type As String

                ''' <summary>
                ''' Encodes a Feature into the model,
                ''' Provide the label and the document words in the document
                ''' will be marked present in the context
                ''' Later these Oneshot encoding feature maybe used to increase the scoring vectors
                ''' Adding context to the document for a specific feature such as sentiment / Emotion / Topic.
                ''' Each topic should be encoded as a feature in the document
                '''
                ''' </summary>
                ''' <param name="CorpusVocab">Current Vocabulary </param>
                ''' <param name="iDocument"></param>
                ''' <param name="Label"></param>
                ''' <returns></returns>
                Public Shared Function GetDocumentContext(ByRef CorpusVocab As Vocabulary, ByRef iDocument As Document, ByRef Label As String) As Vocabulary.FeatureContext
                    Dim iContext As New Vocabulary.FeatureContext
                    Dim NewVocab As List(Of Vocabulary.VocabularyEntry) = CorpusVocab.Current

                    For Each item In NewVocab
                        For Each _item In iDocument.UniqueWords
                            If item.Text = _item Then
                                'Encode Presence in text
                                item.Present = True
                            End If
                        Next
                    Next
                    iContext.Present = NewVocab
                    iContext.Type = Label
                    Return iContext
                End Function

            End Structure

            Public Structure InputTextRecord
                Public Text As String
                Public Encoding As List(Of Integer)
                Public Inputblocks As List(Of List(Of Integer))
                Public Targetblocks As List(Of List(Of Integer))
                Public blocksize As Integer

                Public Shared Function GetBlocks(ByRef Embedding As List(Of Integer), ByRef Size As Integer, Optional Ofset As Integer = 0) As List(Of List(Of Integer))
                    Dim pos As Integer = 0
                    Dim newPos As Integer = Size
                    Dim blocks As New List(Of List(Of Integer))
                    Dim block As New List(Of Integer)
                    Do While pos < Embedding.Count - 1
                        For i = pos To newPos - 1
                            If Ofset > 0 Then
                                If i + Ofset < Embedding.Count - 1 Then

                                    block.Add(Embedding(i + Ofset))
                                    'block.Add(Embedding(i))
                                Else
                                    block.Add(Embedding(i))
                                End If
                            Else
                                block.Add(Embedding(i))
                            End If

                        Next
                        blocks.Add(block)
                        block = New List(Of Integer)
                        pos = newPos

                        If newPos + Size < Embedding.Count - 1 Then
                            newPos += Size
                        Else
                            newPos = Embedding.Count
                        End If

                    Loop

                    Return blocks
                End Function

                Public Shared Function GetTargetBlocks(ByRef Embedding As List(Of Integer), ByRef Size As Integer) As List(Of List(Of Integer))
                    Dim pos As Integer = 0
                    Dim newPos As Integer = Size
                    Dim blocks As New List(Of List(Of Integer))
                    Dim block As New List(Of Integer)
                    Do While pos < Embedding.Count - 1
                        For i = pos To newPos - 1
                            block.Add(Embedding(i))

                        Next
                        blocks.Add(block)
                        block = New List(Of Integer)
                        pos = newPos
                        If newPos + Size < Embedding.Count - 1 Then
                            newPos += Size
                        Else
                            newPos = Embedding.Count
                        End If

                    Loop

                    Return blocks
                End Function

            End Structure

            Public Class Encode

                Public Shared Function Encode_Text(ByRef Text As String, ByRef Vocab As List(Of VocabularyEntry), ByRef Type As VocabularyType) As List(Of Integer)
                    Dim iOutput As New List(Of Integer)
                    Select Case Type
                        Case VocabularyType.Character
                            Dim Chars = Tokenizer.TokenizeByCharacter(Text)

                            For Each item In Chars
                                If CheckVocabulary(item.Value.ToLower, Vocab) = True Then
                                    iOutput.Add(Decode.DecodeText(item.Value.ToLower, Vocab))
                                End If
                            Next
                        Case VocabularyType.Word
                            Dim Words = Tokenizer.TokenizeByWord(Text)

                            For Each item In Words
                                If CheckVocabulary(item.Value.ToLower, Vocab) = True Then
                                    iOutput.Add(Decode.DecodeText(item.Value.ToLower, Vocab))
                                End If
                            Next
                        Case VocabularyType.Sentence
                            Dim Sents = Tokenizer.TokenizeBySentence(Text)

                            For Each item In Sents
                                If CheckVocabulary(item.Value, Vocab) = True Then
                                    iOutput.Add(Decode.DecodeText(item.Value.ToLower, Vocab))
                                End If
                            Next
                    End Select
                    Return iOutput
                End Function

                Public Shared Function EncodeChars(VocabList As List(Of String)) As List(Of VocabularyEntry)
                    Dim vocabulary As New List(Of VocabularyEntry)
                    Dim EncodingValue As Integer = 1
                    For Each item In VocabList
                        Dim newVocabRecord As New VocabularyEntry
                        newVocabRecord.Encoding = EncodingValue
                        newVocabRecord.Text = item
                        EncodingValue += 1
                        vocabulary.Add(newVocabRecord)
                    Next
                    Return vocabulary
                End Function

                Public Shared Function EncodeWords(VocabList As List(Of String)) As List(Of VocabularyEntry)
                    Dim vocabulary As New List(Of VocabularyEntry)
                    Dim EncodingValue As Integer = 1
                    For Each item In VocabList
                        Dim newVocabRecord As New VocabularyEntry
                        newVocabRecord.Encoding = EncodingValue
                        newVocabRecord.Text = item
                        EncodingValue += 1
                        vocabulary.Add(newVocabRecord)
                    Next
                    Return vocabulary
                End Function

                Public Shared Function AddNewEncoding(ByRef Word As String, ByRef Vocab As List(Of VocabularyEntry)) As List(Of VocabularyEntry)
                    Dim NewVocab As New List(Of VocabularyEntry)
                    If CheckVocabulary(Word, Vocab) = False Then
                        NewVocab = Vocab
                        Dim NewItem As New VocabularyEntry
                        NewItem.Text = Word
                        NewItem.Encoding = Vocab.Count
                        NewVocab.Add(NewItem)
                        Return NewVocab
                    Else
                        Return Vocab
                    End If
                End Function

                Public Shared Function CheckVocabulary(ByRef Word As String, ByRef Vocab As List(Of VocabularyEntry)) As Boolean

                    For Each item In Vocab
                        If item.Text = Word Then
                            Return True
                        End If
                    Next
                    Return False
                End Function

            End Class

            Public Class Decode

                Public Shared Function DecodeInteger(ByRef Lookup As Integer, ByRef Vocabulary As List(Of VocabularyEntry))
                    For Each item In Vocabulary
                        If item.Encoding = Lookup Then
                            Return item.Text
                        End If
                    Next
                    Return "Not found in vocabulary"
                End Function

                Public Shared Function DecodeText(ByRef Lookup As String, ByRef Vocabulary As List(Of VocabularyEntry))
                    For Each item In Vocabulary
                        If item.Text = Lookup Then
                            Return item.Encoding
                        End If
                    Next
                    Return "Not found in vocabulary"
                End Function

            End Class

            Public Class VocabularyEntry
                Public Text As String
                Public Encoding As Integer
                Public Frequency As Integer
                Public Present As Boolean
                Public SequenceEncoding As Integer
                Public TF_IDF As Double

                Public Sub New()

                End Sub

                Public Sub New(text As String, sequenceEncoding As Integer, frequency As Integer, tF_IDF As Double)
                    If text Is Nothing Then
                        Throw New ArgumentNullException(NameOf(text))
                    End If

                    Me.Text = text
                    Me.SequenceEncoding = sequenceEncoding
                    Me.Frequency = frequency
                    Me.TF_IDF = tF_IDF
                End Sub

            End Class

            Public Enum VocabularyType
                Character
                Word
                Sentence
            End Enum

            Private Shared Function CreateCharVocabulary(ByRef text As String) As List(Of VocabularyEntry)

                Dim RecordList = CreateUniqueChars(text)

                Dim vocabulary As List(Of VocabularyEntry) = Encode.EncodeChars(RecordList)
                Return vocabulary
            End Function

            Private Shared Function CreateWordVocabulary(ByRef text As String) As List(Of VocabularyEntry)

                Dim RecordList = CreateUniqueWords(text)

                Dim vocabulary As List(Of VocabularyEntry) = Encode.EncodeWords(RecordList)
                Return vocabulary
            End Function

            Private Shared Function CreateSentenceVocabulary(ByRef text As String) As List(Of VocabularyEntry)

                Dim RecordList = CreateUniqueSentences(text)

                Dim vocabulary As List(Of VocabularyEntry) = Encode.EncodeWords(RecordList)
                Return vocabulary
            End Function

            Public Shared Function UpdateVocabulary(ByRef Text As String, ByRef vocab As List(Of VocabularyEntry))
                Return Encode.AddNewEncoding(Text, vocab)
            End Function

            Public Shared Function CreateUniqueSentences(ByRef Text As String) As List(Of String)
                Dim Words = Tokenizer.TokenizeBySentence(Text)
                Dim WordList As New List(Of String)
                For Each item In Words
                    If WordList.Contains(item.Value.ToLower) = False Then
                        WordList.Add(item.Value.ToLower)
                    End If

                Next

                Return WordList
            End Function

            Public Shared Function CreateUniqueWords(ByRef Text As String) As List(Of String)
                Dim Words = Tokenizer.TokenizeByWord(Text)
                Dim WordList As New List(Of String)
                For Each item In Words
                    If WordList.Contains(item.Value.ToLower) = False Then
                        WordList.Add(item.Value.ToLower)
                    End If

                Next

                Return WordList
            End Function

            Public Shared Function CreateUniqueChars(ByRef Text As String) As List(Of String)
                Dim Chars = Tokenizer.TokenizeByCharacter(Text)
                Dim CharList As New List(Of String)
                For Each item In Chars
                    If CharList.Contains(item.Value.ToLower) = False Then
                        CharList.Add(item.Value.ToLower)
                    End If

                Next

                Return CharList
            End Function

            Public Shared Function CreateVocabulary(ByRef Text As String, vType As VocabularyType) As List(Of VocabularyEntry)
                Select Case vType
                    Case VocabularyType.Character
                        Return CreateCharVocabulary(Text)
                    Case VocabularyType.Word
                        Return CreateWordVocabulary(Text)
                    Case VocabularyType.Sentence
                        Return CreateSentenceVocabulary(Text)
                End Select
                Return New List(Of VocabularyEntry)
            End Function

        End Class

        '    Positional Encoding :
        '    To provide positional information to the model, positional encodings.
        '    These encodings are added to the input embeddings to capture the order of the tokens in the sequence.
        '    Positional Encoding :
        '    To provide positional information to the model, positional encodings.
        '    These encodings are added to the input embeddings to capture the order of the tokens in the sequence.
        Public Class PositionalEncoding
            Private ReadOnly encodingMatrix As List(Of List(Of Double))
            Private InternalVocab As Corpus.Vocabulary

            Public Sub New(maxLength As Integer, embeddingSize As Integer, ByRef Vocab As Corpus.Vocabulary)
                InternalVocab = Vocab
                encodingMatrix = New List(Of List(Of Double))()
                ' Create the encoding matrix
                For pos As Integer = 0 To maxLength - 1
                    Dim encodingRow As List(Of Double) = New List(Of Double)()
                    For i As Integer = 0 To embeddingSize - 1
                        Dim angle As Double = pos / Math.Pow(10000, (2 * i) / embeddingSize)
                        encodingRow.Add(Math.Sin(angle))
                        encodingRow.Add(Math.Cos(angle))
                    Next
                    encodingMatrix.Add(encodingRow)
                Next
            End Sub

            Public Function Encode(inputTokens As List(Of String)) As List(Of List(Of Double))
                Dim encodedInputs As List(Of List(Of Double)) = New List(Of List(Of Double))()

                For Each token As String In inputTokens
                    Dim tokenEncoding As List(Of Double) = New List(Of Double)()

                    ' Find the index of the token in the vocabulary
                    ' For simplicity, let's assume a fixed vocabulary
                    Dim tokenIndex As Integer = GetTokenIndex(token)

                    ' Retrieve the positional encoding for the token
                    If tokenIndex >= 0 Then
                        tokenEncoding = encodingMatrix(tokenIndex)
                    Else
                        ' Handle unknown tokens if necessary
                    End If

                    encodedInputs.Add(tokenEncoding)
                Next

                Return encodedInputs
            End Function

            Private Function GetTokenIndex(token As String) As Integer
                ' Retrieve the index of the token in the vocabulary
                ' For simplicity, let's assume a fixed vocabulary
                Dim vocabulary As List(Of String) = GetVocabulary(InternalVocab)
                Return vocabulary.IndexOf(token)
            End Function

            Private Function GetVocabulary(ByRef Vocab As Corpus.Vocabulary) As List(Of String)
                ' Return the vocabulary list
                ' Modify this function as per your specific vocabulary
                Dim Lst As New List(Of String)
                For Each item In Vocab.Current
                    Lst.Add(item.Text)
                Next
                Return Lst
            End Function

        End Class

        Public Class PositionalDecoder
            Private ReadOnly decodingMatrix As List(Of List(Of Double))
            Private InternalVocab As Corpus.Vocabulary

            Public Sub New(maxLength As Integer, embeddingSize As Integer, ByRef Vocab As Corpus.Vocabulary)
                decodingMatrix = New List(Of List(Of Double))()
                InternalVocab = Vocab
                ' Create the decoding matrix
                For pos As Integer = 0 To maxLength - 1
                    Dim decodingRow As List(Of Double) = New List(Of Double)()

                    For i As Integer = 0 To embeddingSize - 1
                        Dim angle As Double = pos / Math.Pow(10000, (2 * i) / embeddingSize)
                        decodingRow.Add(Math.Sin(angle))
                        decodingRow.Add(Math.Cos(angle))
                    Next

                    decodingMatrix.Add(decodingRow)
                Next
            End Sub

            Public Function Decode(encodedInputs As List(Of List(Of Double))) As List(Of String)
                Dim decodedTokens As List(Of String) = New List(Of String)()

                For Each encoding As List(Of Double) In encodedInputs
                    ' Retrieve the token index based on the encoding
                    Dim tokenIndex As Integer = GetTokenIndex(encoding)

                    ' Retrieve the token based on the index
                    If tokenIndex >= 0 Then
                        Dim token As String = GetToken(tokenIndex)
                        decodedTokens.Add(token)
                    Else
                        ' Handle unknown encodings if necessary
                    End If
                Next

                Return decodedTokens
            End Function

            Private Function GetTokenIndex(encoding As List(Of Double)) As Integer
                ' Retrieve the index of the token based on the encoding
                ' For simplicity, let's assume a fixed vocabulary
                Dim vocabulary As List(Of String) = GetVocabulary(InternalVocab)

                For i As Integer = 0 To decodingMatrix.Count - 1
                    If encoding.SequenceEqual(decodingMatrix(i)) Then
                        Return i
                    End If
                Next

                Return -1 ' Token not found
            End Function

            Private Function GetToken(tokenIndex As Integer) As String
                ' Retrieve the token based on the index
                ' For simplicity, let's assume a fixed vocabulary
                Dim vocabulary As List(Of String) = GetVocabulary(InternalVocab)

                If tokenIndex >= 0 AndAlso tokenIndex < vocabulary.Count Then
                    Return vocabulary(tokenIndex)
                Else
                    Return "Unknown" ' Unknown token
                End If
            End Function

            Private Function GetVocabulary(ByRef Vocab As Corpus.Vocabulary) As List(Of String)
                ' Return the vocabulary list
                ' Modify this function as per your specific vocabulary
                Dim Lst As New List(Of String)
                For Each item In Vocab.Current
                    Lst.Add(item.Text)
                Next
                Return Lst
            End Function

        End Class

    End Class
    Public Structure WordVector
        Dim Freq As Integer
        Public NormalizedEncoding As Integer
        Public OneHotEncoding As Integer
        Public PositionalEncoding As Double()
        Dim PositionalEncodingVector As List(Of Double)
        Dim SequenceEncoding As Integer
        Dim Token As String

        ''' <summary>
        ''' adds positional encoding to list of word_vectors (ie encoded document)
        ''' Presumes a dimensional model of 512
        ''' </summary>
        ''' <param name="DccumentStr">Current Document</param>
        ''' <returns></returns>
        Public Shared Function AddPositionalEncoding(ByRef DccumentStr As List(Of WordVector)) As List(Of WordVector)
            ' Define the dimension of the model
            Dim d_model As Integer = 512
            ' Loop through each word in the sentence and apply positional encoding
            Dim i As Integer = 0
            For Each wrd In DccumentStr

                wrd.PositionalEncoding = CalcPositionalEncoding(i, d_model)
                i += 1
            Next
            Return DccumentStr
        End Function

        ''' <summary>
        ''' creates a list of word vectors sorted by frequency, from the text given
        ''' </summary>
        ''' <param name="Sentence"></param> document
        ''' <returns>vocabulary sorted in order of frequency</returns>
        Public Shared Function CreateSortedVocabulary(ByRef Sentence As String) As List(Of WordVector)
            Dim Vocabulary = WordVector.CreateVocabulary(Sentence)
            Dim NewDict As New List(Of WordVector)
            Dim Words() = Sentence.Split(" ")
            ' Count the frequency of each word
            Dim wordCounts As Dictionary(Of String, Integer) = Words.GroupBy(Function(w) w).ToDictionary(Function(g) g.Key, Function(g) g.Count())
            'Get the top ten words
            Dim TopTen As List(Of KeyValuePair(Of String, Integer)) = wordCounts.OrderByDescending(Function(w) w.Value).Take(10).ToList()

            Dim SortedDict As New List(Of WordVector)
            'Create Sorted List
            Dim Sorted As List(Of KeyValuePair(Of String, Integer)) = wordCounts.OrderByDescending(Function(w) w.Value).ToList()
            'Create Sorted Dictionary
            For Each item In Sorted

                Dim NewToken As New WordVector
                NewToken.Token = item.Key
                NewToken.SequenceEncoding = LookUpSeqEncoding(Vocabulary, item.Key)
                NewToken.Freq = item.Value
                SortedDict.Add(NewToken)

            Next

            Return SortedDict
        End Function

        ''' <summary>
        ''' Creates a unique list of words
        ''' Encodes words by their order of appearance in the text
        ''' </summary>
        ''' <param name="Sentence">document text</param>
        ''' <returns>EncodedWordlist (current vocabulary)</returns>
        Public Shared Function CreateVocabulary(ByRef Sentence As String) As List(Of WordVector)
            Dim inputString As String = "This is a sample sentence."
            If Sentence IsNot Nothing Then
                inputString = Sentence
            End If
            Dim uniqueCharacters As New List(Of String)

            Dim Dictionary As New List(Of WordVector)
            Dim Words() = Sentence.Split(" ")
            'Create unique tokens
            For Each c In Words
                If Not uniqueCharacters.Contains(c) Then
                    uniqueCharacters.Add(c)
                End If
            Next
            'Iterate through unique tokens assigning integers
            For i As Integer = 0 To uniqueCharacters.Count - 1
                'create token entry
                Dim newToken As New WordVector
                newToken.Token = uniqueCharacters(i)
                newToken.SequenceEncoding = i + 1
                'Add to vocab
                Dictionary.Add(newToken)

            Next
            Return UpdateVocabularyFrequencys(Sentence, Dictionary)
        End Function

        ''' <summary>
        ''' Creates embeddings for the sentence provided using the generated vocabulary
        ''' </summary>
        ''' <param name="Sentence"></param>
        ''' <param name="Vocabulary"></param>
        ''' <returns></returns>
        Public Shared Function EncodeWordsToVectors(ByRef Sentence As String, ByRef Vocabulary As List(Of WordVector)) As List(Of WordVector)

            Sentence = Sentence.ToLower
            If Vocabulary Is Nothing Then
                Vocabulary = CreateVocabulary(Sentence)
            End If
            Dim words() As String = Sentence.Split(" ")
            Dim Dict As New List(Of WordVector)
            For Each item In words
                Dim RetSent As New WordVector
                RetSent = GetToken(Vocabulary, item)
                Dict.Add(RetSent)
            Next
            Return NormalizeWords(Sentence, AddPositionalEncoding(Dict))
        End Function

        ''' <summary>
        ''' Decoder
        ''' </summary>
        ''' <param name="Vocabulary">Encoded Wordlist</param>
        ''' <param name="Token">desired token</param>
        ''' <returns></returns>
        Public Shared Function GetToken(ByRef Vocabulary As List(Of WordVector), ByRef Token As String) As WordVector
            For Each item In Vocabulary
                If item.Token = Token Then Return item
            Next
            Return New WordVector
        End Function

        ''' <summary>
        ''' finds the frequency of this token in the sentence
        ''' </summary>
        ''' <param name="Token">token to be defined</param>
        ''' <param name="InputStr">string containing token</param>
        ''' <returns></returns>
        Public Shared Function GetTokenFrequency(ByRef Token As String, ByRef InputStr As String) As Integer
            GetTokenFrequency = 0

            If InputStr.Contains(Token) = True Then
                For Each item In WordVector.GetWordFrequencys(InputStr, " ")
                    If item.Token = Token Then
                        GetTokenFrequency = item.Freq
                    End If
                Next
            End If
        End Function

        ''' <summary>
        ''' Returns frequencys for words
        ''' </summary>
        ''' <param name="_Text"></param>
        ''' <param name="Delimiter"></param>
        ''' <returns></returns>
        Public Shared Function GetTokenFrequencys(ByVal _Text As String, ByVal Delimiter As String) As List(Of WordVector)
            Dim Words As New WordVector
            Dim ListOfWordFrequecys As New List(Of WordVector)
            Dim WordList As List(Of String) = _Text.Split(Delimiter).ToList
            Dim groups = WordList.GroupBy(Function(value) value)
            For Each grp In groups
                Words.Token = grp(0)
                Words.Freq = grp.Count
                ListOfWordFrequecys.Add(Words)
            Next
            Return ListOfWordFrequecys
        End Function

        ''' <summary>
        ''' For Legacy Functionality
        ''' </summary>
        ''' <param name="_Text"></param>
        ''' <param name="Delimiter"></param>
        ''' <returns></returns>
        Public Shared Function GetWordFrequencys(ByVal _Text As String, ByVal Delimiter As String) As List(Of WordVector)
            GetTokenFrequencys(_Text, Delimiter)
        End Function

        ''' <summary>
        ''' Decoder - used to look up a token identity using its sequence encoding
        ''' </summary>
        ''' <param name="EncodedWordlist">Encoded VectorList(vocabulary)</param>
        ''' <param name="EncodingValue">Sequence Encoding Value</param>
        ''' <returns></returns>
        Public Shared Function LookUpBySeqEncoding(ByRef EncodedWordlist As List(Of WordVector), ByRef EncodingValue As Integer) As String
            For Each item In EncodedWordlist
                If item.SequenceEncoding = EncodingValue Then Return item.Token
            Next
            Return EncodingValue
        End Function

        ''' <summary>
        ''' Encoder - used to look up a tokens sequence encoding, in a vocabulary
        ''' </summary>
        ''' <param name="EncodedWordlist">Encoded VectorList(vocabulary) </param>
        ''' <param name="Token">Desired Token</param>
        ''' <returns></returns>
        Public Shared Function LookUpSeqEncoding(ByRef EncodedWordlist As List(Of WordVector), ByRef Token As String) As Integer
            For Each item In EncodedWordlist
                If item.Token = Token Then Return item.SequenceEncoding
            Next
            Return 0
        End Function

        ''' <summary>
        ''' Adds Normalization to Vocabulary(Word-based)
        ''' </summary>
        ''' <param name="Sentence">Doc</param>
        ''' <param name="dict">Vocabulary</param>
        ''' <returns></returns>
        Public Shared Function NormalizeWords(ByRef Sentence As String, ByRef dict As List(Of WordVector)) As List(Of WordVector)
            Dim Count = CountWords(Sentence)
            For Each item In dict
                item.NormalizedEncoding = Count / item.Freq
            Next
            Return dict
        End Function

        ''' <summary>
        ''' Encodes a list of word-vector by a list of strings
        ''' If a token is found in the list it is encoded with a binary 1 if false then 0
        ''' This is useful for categorizing and adding context to the word vector
        ''' </summary>
        ''' <param name="WordVectorList">list of tokens to be encoded (categorized)</param>
        ''' <param name="Vocabulary">Categorical List, Such as a list of positive sentiment</param>
        ''' <returns></returns>
        Public Shared Function OneShotEncoding(ByVal WordVectorList As List(Of WordVector),
                                ByRef Vocabulary As List(Of String)) As List(Of WordVector)
            Dim EncodedList As New List(Of WordVector)
            For Each item In WordVectorList
                Dim Found As Boolean = False
                For Each RefItem In Vocabulary
                    If item.Token = RefItem Then
                        Found = True
                    Else

                    End If
                Next
                If Found = True Then
                    Dim newWordvector As WordVector = item
                    newWordvector.OneHotEncoding = True
                End If
                EncodedList.Add(item)
            Next
            Return EncodedList
        End Function

        ''' <summary>
        ''' Creates a List of Bigram WordVectors Based on the text
        ''' to create the vocabulary file use @ProduceBigramVocabulary
        ''' </summary>
        ''' <param name="Sentence"></param>
        ''' <returns>Encoded list of bigrams and vectors (vocabulary)with frequencies </returns>
        Public Shared Function ProduceBigramDocument(ByRef sentence As String) As List(Of WordVector)

            ' Convert sentence to lowercase and split into words
            Dim words As String() = sentence.ToLower().Split()
            Dim GeneratedBigramsList As New List(Of String)
            Dim bigrams As New Dictionary(Of String, Integer)
            'We start at the first word And go up to the second-to-last word
            For i As Integer = 0 To words.Length - 2
                Dim bigram As String = words(i) & " " & words(i + 1)
                'We check If the bigrams dictionary already contains the bigram.
                'If it does, we increment its frequency by 1.
                'If it doesn't, we add it to the dictionary with a frequency of 1.
                GeneratedBigramsList.Add(bigram)
                If bigrams.ContainsKey(bigram) Then
                    bigrams(bigram) += 1
                Else
                    bigrams.Add(bigram, 1)
                End If
            Next

            'we Loop through the bigrams dictionary(of frequncies) And encode a integer to the bi-gram

            Dim bigramVocab As New List(Of WordVector)
            Dim a As Integer = 1
            For Each kvp As KeyValuePair(Of String, Integer) In bigrams
                Dim newvect As New WordVector
                newvect.Token = kvp.Key
                newvect.Freq = kvp.Value

                bigramVocab.Add(newvect)
            Next
            'create a list from the generated bigrams and
            ''add frequecies from vocabulary of frequecies
            Dim nVocab As New List(Of WordVector)
            Dim z As Integer = 0
            For Each item In GeneratedBigramsList
                'create final token
                Dim NewToken As New WordVector
                NewToken.Token = item
                'add current position in document
                NewToken.SequenceEncoding = GeneratedBigramsList(z)
                'add frequency
                For Each Lookupitem In bigramVocab
                    If item = Lookupitem.Token Then
                        NewToken.Freq = Lookupitem.Freq
                    Else
                    End If
                Next
                'add token
                nVocab.Add(NewToken)
                'update current index
                z += 1
            Next

            'Return bigram document with sequence and frequencys
            Return nVocab
        End Function

        ''' <summary>
        ''' Creates a Vocabulary of unique bigrams from sentence with frequencies adds a sequence vector based on
        ''' its appearence in the text, if item is repeated at multiple locations it is not reflected here
        ''' </summary>
        ''' <param name="Sentence"></param>
        ''' <returns>Encoded list of unique bigrams and vectors (vocabulary)with frequencies </returns>
        Public Shared Function ProduceBigramVocabulary(ByRef sentence As String) As List(Of WordVector)

            ' Convert sentence to lowercase and split into words
            Dim words As String() = sentence.ToLower().Split()
            Dim GeneratedBigramsList As New List(Of String)
            Dim bigrams As New Dictionary(Of String, Integer)
            'We start at the first word And go up to the second-to-last word
            For i As Integer = 0 To words.Length - 2
                Dim bigram As String = words(i) & " " & words(i + 1)
                'We check If the bigrams dictionary already contains the bigram.
                'If it does, we increment its frequency by 1.
                'If it doesn't, we add it to the dictionary with a frequency of 1.
                GeneratedBigramsList.Add(bigram)
                If bigrams.ContainsKey(bigram) Then
                    bigrams(bigram) += 1
                Else
                    bigrams.Add(bigram, 1)
                End If
            Next

            'we Loop through the bigrams dictionary(of frequncies) And encode a integer to the bi-gram

            Dim bigramVocab As New List(Of WordVector)
            Dim a As Integer = 0
            For Each kvp As KeyValuePair(Of String, Integer) In bigrams
                Dim newvect As New WordVector
                newvect.Token = kvp.Key
                newvect.Freq = kvp.Value
                newvect.SequenceEncoding = a + 1
                bigramVocab.Add(newvect)
            Next

            'Return bigram document with sequence and frequencys
            Return bigramVocab
        End Function

        ''' <summary>
        ''' Adds Frequencies to a sequentially encoded word-vector list
        ''' </summary>
        ''' <param name="Sentence">current document</param>
        ''' <param name="EncodedWordlist">Current Vocabulary</param>
        ''' <returns>an encoded word-Vector list with Frequencys attached</returns>
        Public Shared Function UpdateVocabularyFrequencys(ByRef Sentence As String, ByVal EncodedWordlist As List(Of WordVector)) As List(Of WordVector)

            Dim NewDict As New List(Of WordVector)
            Dim Words() = Sentence.Split(" ")
            ' Count the frequency of each word
            Dim wordCounts As Dictionary(Of String, Integer) = Words.GroupBy(Function(w) w).ToDictionary(Function(g) g.Key, Function(g) g.Count())
            'Get the top ten words
            Dim TopTen As List(Of KeyValuePair(Of String, Integer)) = wordCounts.OrderByDescending(Function(w) w.Value).Take(10).ToList()

            'Create Standard Dictionary
            For Each EncodedItem In EncodedWordlist
                For Each item In wordCounts
                    If EncodedItem.Token = item.Key Then
                        Dim NewToken As New WordVector
                        NewToken = EncodedItem
                        NewToken.Freq = item.Value
                        NewDict.Add(NewToken)
                    End If
                Next
            Next

            Return NewDict
        End Function

        ''' <summary>
        ''' Outputs Structure to Jason(JavaScriptSerializer)
        ''' </summary>
        ''' <returns></returns>
        Public Function ToJson() As String
            Dim Converter As New JavaScriptSerializer
            Return Converter.Serialize(Me)
        End Function



#Region "Positional Encoding"

        Private Shared Function CalcPositionalEncoding(ByVal position As Integer, ByVal d_model As Integer) As Double()
            ' Create an empty array to store the encoding
            Dim encoding(d_model - 1) As Double

            ' Loop through each dimension of the model and calculate the encoding value
            For i As Integer = 0 To d_model - 1
                If i Mod 2 = 0 Then
                    encoding(i) = Math.Sin(position / (10000 ^ (i / d_model)))
                Else
                    encoding(i) = Math.Cos(position / (10000 ^ ((i - 1) / d_model)))
                End If
            Next

            ' Return the encoding array
            Return encoding
        End Function

#End Region

#Region "Normalization"

        ''' <summary>
        ''' returns number of Chars in text
        ''' </summary>
        ''' <param name="Sentence">Document</param>
        ''' <returns>number of Chars</returns>
        Private Shared Function CountChars(ByRef Sentence As String) As Integer
            Dim uniqueCharacters As New List(Of String)
            For Each c As Char In Sentence
                uniqueCharacters.Add(c.ToString)
            Next
            Return uniqueCharacters.Count
        End Function

        ''' <summary>
        ''' Returns number of words in text
        ''' </summary>
        ''' <param name="Sentence">Document</param>
        ''' <returns>number of words</returns>
        Private Shared Function CountWords(ByRef Sentence As String) As Integer
            Dim Words() = Sentence.Split(" ")
            Return Words.Count
        End Function

#End Region

    End Structure




End Namespace
Namespace Utilitys
    Namespace NLP_MATH
        Public Class Softmax
            Public Shared Function Softmax(matrix2 As Integer(,)) As Double(,)
                Dim numRows As Integer = matrix2.GetLength(0)
                Dim numColumns As Integer = matrix2.GetLength(1)

                Dim softmaxValues(numRows - 1, numColumns - 1) As Double

                ' Compute softmax values for each row
                For i As Integer = 0 To numRows - 1
                    Dim rowSum As Double = 0

                    ' Compute exponential values and sum of row elements
                    For j As Integer = 0 To numColumns - 1
                        softmaxValues(i, j) = Math.Sqrt(Math.Exp(matrix2(i, j)))
                        rowSum += softmaxValues(i, j)
                    Next

                    ' Normalize softmax values for the row
                    For j As Integer = 0 To numColumns - 1
                        softmaxValues(i, j) /= rowSum
                    Next
                Next

                ' Display the softmax values
                Console.WriteLine("Calculated:" & vbNewLine)
                For i As Integer = 0 To numRows - 1
                    For j As Integer = 0 To numColumns - 1

                        Console.Write(softmaxValues(i, j).ToString("0.0000") & " ")
                    Next
                    Console.WriteLine(vbNewLine & "---------------------")
                Next
                Return softmaxValues
            End Function
            Public Shared Sub Main()
                Dim input() As Double = {1.0, 2.0, 3.0}

                Dim output() As Double = Softmax(input)

                Console.WriteLine("Input: {0}", String.Join(", ", input))
                Console.WriteLine("Softmax Output: {0}", String.Join(", ", output))
                Console.ReadLine()
            End Sub

            Public Shared Function Softmax(ByVal input() As Double) As Double()
                Dim maxVal As Double = input.Max()

                Dim exponentiated() As Double = input.Select(Function(x) Math.Exp(x - maxVal)).ToArray()

                Dim sum As Double = exponentiated.Sum()

                Dim softmaxOutput() As Double = exponentiated.Select(Function(x) x / sum).ToArray()

                Return softmaxOutput
            End Function
        End Class
        Public Class SimilarityCalculator

            Public Shared Function CalculateCosineSimilarity(sentences1 As List(Of String), sentences2 As List(Of String)) As Double
                Dim vectorizer As New SentenceVectorizer()
                Dim vector1 = vectorizer.Vectorize(sentences1)
                Dim vector2 = vectorizer.Vectorize(sentences2)

                Return SimilarityCalculator.CalculateCosineSimilarity(vector1, vector2)
            End Function

            Public Shared Function CalculateCosineSimilarity(vector1 As List(Of Double), vector2 As List(Of Double)) As Double
                If vector1.Count <> vector2.Count Then
                    Throw New ArgumentException("Vector dimensions do not match.")
                End If

                Dim dotProduct As Double = 0
                Dim magnitude1 As Double = 0
                Dim magnitude2 As Double = 0

                For i As Integer = 0 To vector1.Count - 1
                    dotProduct += vector1(i) * vector2(i)
                    magnitude1 += Math.Pow(vector1(i), 2)
                    magnitude2 += Math.Pow(vector2(i), 2)
                Next

                magnitude1 = Math.Sqrt(magnitude1)
                magnitude2 = Math.Sqrt(magnitude2)

                Return dotProduct / (magnitude1 * magnitude2)
            End Function

            Public Shared Function CalculateJaccardSimilarity(sentences1 As List(Of String), sentences2 As List(Of String)) As Double
                Dim set1 As New HashSet(Of String)(sentences1)
                Dim set2 As New HashSet(Of String)(sentences2)

                Return SimilarityCalculator.CalculateJaccardSimilarity(set1, set2)
            End Function

            Public Shared Function CalculateJaccardSimilarity(set1 As HashSet(Of String), set2 As HashSet(Of String)) As Double
                Dim intersectionCount As Integer = set1.Intersect(set2).Count()
                Dim unionCount As Integer = set1.Union(set2).Count()

                Return CDbl(intersectionCount) / CDbl(unionCount)
            End Function

        End Class
        Public Class Tril
            Public Sub Main()
                Dim matrix(,) As Integer = {{1, 2, 3, 9}, {4, 5, 6, 8}, {7, 8, 9, 9}}

                Dim result(,) As Integer = Tril(matrix)

                Console.WriteLine("Matrix:")
                PrintMatrix(matrix)

                Console.WriteLine("Tril Result:")
                PrintMatrix(result)
                Console.ReadLine()
            End Sub


            Public Shared Function Tril(ByVal matrix(,) As Integer) As Integer(,)
                Dim rows As Integer = matrix.GetLength(0)
                Dim cols As Integer = matrix.GetLength(1)

                Dim result(rows - 1, cols - 1) As Integer

                For i As Integer = 0 To rows - 1
                    For j As Integer = 0 To cols - 1
                        If j <= i Then
                            result(i, j) = matrix(i, j)
                        End If
                    Next
                Next

                Return result
            End Function
            Public Shared Function Tril(ByVal matrix(,) As Double) As Double(,)
                Dim rows As Integer = matrix.GetLength(0)
                Dim cols As Integer = matrix.GetLength(1)

                Dim result(rows - 1, cols - 1) As Double

                For i As Integer = 0 To rows - 1
                    For j As Integer = 0 To cols - 1
                        If j <= i Then
                            result(i, j) = matrix(i, j)
                        End If
                    Next
                Next

                Return result
            End Function
            Public Shared Function Tril(ByVal matrix As List(Of List(Of Double))) As List(Of List(Of Double))
                Dim rows As Integer = matrix.Count
                Dim cols As Integer = matrix(0).Count

                Dim result As New List(Of List(Of Double))

                For i As Integer = 0 To rows - 1
                    For j As Integer = 0 To cols - 1
                        If j <= i Then
                            result(i)(j) = matrix(i)(j)
                        End If
                    Next
                Next

                Return result
            End Function
            Public Shared Sub PrintMatrix(ByVal matrix(,) As Double)
                Dim rows As Integer = matrix.GetLength(0)
                Dim cols As Integer = matrix.GetLength(1)

                For i As Integer = 0 To rows - 1
                    For j As Integer = 0 To cols - 1
                        Console.Write(matrix(i, j) & " ")
                    Next
                    Console.WriteLine()
                Next
            End Sub
            Public Shared Sub PrintMatrix(ByVal matrix(,) As Integer)
                Dim rows As Integer = matrix.GetLength(0)
                Dim cols As Integer = matrix.GetLength(1)

                For i As Integer = 0 To rows - 1
                    For j As Integer = 0 To cols - 1
                        Console.Write(matrix(i, j) & " ")
                    Next
                    Console.WriteLine()
                Next
            End Sub
        End Class
    End Namespace
    Namespace TEXT
        Public Module TextProcessingTasks

            <Runtime.CompilerServices.Extension()>
            Public Function PerformTasks(ByRef Txt As String, ByRef Tasks As List(Of TextPreProcessingTasks)) As String

                For Each tsk In Tasks
                    Select Case tsk

                        Case TextPreProcessingTasks.Space_Punctuation

                            Txt = SpacePunctuation(Txt).Replace("  ", " ")
                        Case TextPreProcessingTasks.To_Upper
                            Txt = Txt.ToUpper.Replace("  ", " ")
                        Case TextPreProcessingTasks.To_Lower
                            Txt = Txt.ToLower.Replace("  ", " ")
                        Case TextPreProcessingTasks.Lemmatize_Text
                        Case TextPreProcessingTasks.Tokenize_Characters
                            Txt = TokenizeChars(Txt)
                            Dim Words() As String = Txt.Split(",")
                            Txt &= vbNewLine & "Total Tokens in doc  -" & Words.Count - 1 & ":" & vbNewLine
                        Case TextPreProcessingTasks.Remove_Stop_Words
                            TextExtensions.RemoveStopWords(Txt)
                        Case TextPreProcessingTasks.Tokenize_Words
                            Txt = TokenizeWords(Txt)
                            Dim Words() As String = Txt.Split(",")
                            Txt &= vbNewLine & "Total Tokens in doc  -" & Words.Count - 1 & ":" & vbNewLine
                        Case TextPreProcessingTasks.Tokenize_Sentences
                            Txt = TokenizeSentences(Txt)
                            Dim Words() As String = Txt.Split(",")
                            Txt &= vbNewLine & "Total Tokens in doc  -" & Words.Count - 2 & ":" & vbNewLine
                        Case TextPreProcessingTasks.Remove_Symbols
                            Txt = RemoveSymbols(Txt).Replace("  ", " ")
                        Case TextPreProcessingTasks.Remove_Brackets
                            Txt = RemoveBrackets(Txt).Replace("  ", " ")
                        Case TextPreProcessingTasks.Remove_Maths_Symbols
                            Txt = RemoveMathsSymbols(Txt).Replace("  ", " ")
                        Case TextPreProcessingTasks.Remove_Punctuation
                            Txt = RemovePunctuation(Txt).Replace("  ", " ")
                        Case TextPreProcessingTasks.AlphaNumeric_Only
                            Txt = AlphaNumericOnly(Txt).Replace("  ", " ")
                    End Select
                Next

                Return Txt
            End Function

            Public Enum TextPreProcessingTasks
                Space_Punctuation
                To_Upper
                To_Lower
                Lemmatize_Text
                Tokenize_Characters
                Remove_Stop_Words
                Tokenize_Words
                Tokenize_Sentences
                Remove_Symbols
                Remove_Brackets
                Remove_Maths_Symbols
                Remove_Punctuation
                AlphaNumeric_Only
            End Enum

        End Module
        Public Class TextCorpusChunker
            Implements ICorpusChunker

            Private chunkProcessor As ChunkProcessor
            Private regexFilter As New RegexFilter()

            Public Sub New(chunkType As ChunkType, maxPaddingSize As Integer)
                chunkProcessor = New ChunkProcessor(chunkType, maxPaddingSize)
            End Sub

            Public Function CreateDictionaryVocabulary(data As List(Of String))
                Return VocabularyGenerator.CreateDictionaryVocabulary(data)
            End Function

            Public Function CreatePunctuationVocabulary(data As List(Of String))
                Return VocabularyGenerator.CreatePunctuationVocabulary(data)
            End Function

            Public Function FilterUsingPunctuationVocabulary(data As List(Of String)) As List(Of String) Implements ICorpusChunker.FilterUsingPunctuationVocabulary
                Return regexFilter.FilterUsingRegexPatterns(data, VocabularyGenerator.CreatePunctuationVocabulary(data).ToList())
            End Function

            Public Function GenerateClassificationDataset(data As List(Of String), classes As List(Of String)) As List(Of Tuple(Of String, String)) Implements ICorpusChunker.GenerateClassificationDataset
                Return CorpusCreator.GenerateClassificationDataset(data, classes)
            End Function

            Public Function GeneratePredictiveDataset(data As List(Of String), windowSize As Integer) As List(Of String()) Implements ICorpusChunker.GeneratePredictiveDataset
                Return CorpusCreator.GeneratePredictiveDataset(data, windowSize)
            End Function

            Public Sub LoadEntityListFromFile(filePath As String)
                EntityLoader.LoadEntityListFromFile(filePath)
            End Sub

            Public Function ProcessTextData(rawData As String, useFiltering As Boolean) As List(Of String) Implements ICorpusChunker.ProcessTextData
                Dim chunks As List(Of String) = chunkProcessor.ProcessFile(rawData, useFiltering)

                If VocabularyGenerator.CreatePunctuationVocabulary(chunks) IsNot Nothing Then
                    chunks = regexFilter.FilterUsingRegexPatterns(chunks, VocabularyGenerator.CreatePunctuationVocabulary(chunks).ToList())
                End If

                Return chunks
            End Function

        End Class
        ''' <summary>
        ''' The removal of commonly used words which are only used to create a sentence such as,
        ''' the, on, in of, but
        ''' </summary>
        Public Class RemoveStopWords

            Public StopWords As New List(Of String)

            Public StopWordsArab() As String = {"،", "آض", "آمينَ", "آه",
                    "آهاً", "آي", "أ", "أب", "أجل", "أجمع", "أخ", "أخذ", "أصبح", "أضحى", "أقبل",
                    "أقل", "أكثر", "ألا", "أم", "أما", "أمامك", "أمامكَ", "أمسى", "أمّا", "أن", "أنا", "أنت",
                    "أنتم", "أنتما", "أنتن", "أنتِ", "أنشأ", "أنّى", "أو", "أوشك", "أولئك", "أولئكم", "أولاء",
                    "أولالك", "أوّهْ", "أي", "أيا", "أين", "أينما", "أيّ", "أَنَّ", "أََيُّ", "أُفٍّ", "إذ", "إذا", "إذاً",
                    "إذما", "إذن", "إلى", "إليكم", "إليكما", "إليكنّ", "إليكَ", "إلَيْكَ", "إلّا", "إمّا", "إن", "إنّما",
                    "إي", "إياك", "إياكم", "إياكما", "إياكن", "إيانا", "إياه", "إياها", "إياهم", "إياهما", "إياهن",
                    "إياي", "إيهٍ", "إِنَّ", "ا", "ابتدأ", "اثر", "اجل", "احد", "اخرى", "اخلولق", "اذا", "اربعة", "ارتدّ",
                    "استحال", "اطار", "اعادة", "اعلنت", "اف", "اكثر", "اكد", "الألاء", "الألى", "الا", "الاخيرة", "الان", "الاول",
                    "الاولى", "التى", "التي", "الثاني", "الثانية", "الذاتي", "الذى", "الذي", "الذين", "السابق", "الف", "اللائي",
                    "اللاتي", "اللتان", "اللتيا", "اللتين", "اللذان", "اللذين", "اللواتي", "الماضي", "المقبل", "الوقت", "الى",
                    "اليوم", "اما", "امام", "امس", "ان", "انبرى", "انقلب", "انه", "انها", "او", "اول", "اي", "ايار", "ايام",
                    "ايضا", "ب", "بات", "باسم", "بان", "بخٍ", "برس", "بسبب", "بسّ", "بشكل", "بضع", "بطآن", "بعد", "بعض", "بك",
                    "بكم", "بكما", "بكن", "بل", "بلى", "بما", "بماذا", "بمن", "بن", "بنا", "به", "بها", "بي", "بيد", "بين",
                    "بَسْ", "بَلْهَ", "بِئْسَ", "تانِ", "تانِك", "تبدّل", "تجاه", "تحوّل", "تلقاء", "تلك", "تلكم", "تلكما", "تم", "تينك",
                    "تَيْنِ", "تِه", "تِي", "ثلاثة", "ثم", "ثمّ", "ثمّة", "ثُمَّ", "جعل", "جلل", "جميع", "جير", "حار", "حاشا", "حاليا",
                    "حاي", "حتى", "حرى", "حسب", "حم", "حوالى", "حول", "حيث", "حيثما", "حين", "حيَّ", "حَبَّذَا", "حَتَّى", "حَذارِ", "خلا",
                    "خلال", "دون", "دونك", "ذا", "ذات", "ذاك", "ذانك", "ذانِ", "ذلك", "ذلكم", "ذلكما", "ذلكن", "ذو", "ذوا", "ذواتا", "ذواتي", "ذيت", "ذينك", "ذَيْنِ", "ذِه", "ذِي", "راح", "رجع", "رويدك", "ريث", "رُبَّ", "زيارة", "سبحان", "سرعان", "سنة", "سنوات", "سوف", "سوى", "سَاءَ", "سَاءَمَا", "شبه", "شخصا", "شرع", "شَتَّانَ", "صار", "صباح", "صفر", "صهٍ", "صهْ", "ضد", "ضمن", "طاق", "طالما", "طفق", "طَق", "ظلّ", "عاد", "عام", "عاما", "عامة", "عدا", "عدة", "عدد", "عدم", "عسى", "عشر", "عشرة", "علق", "على", "عليك", "عليه", "عليها", "علًّ", "عن", "عند", "عندما", "عوض", "عين", "عَدَسْ", "عَمَّا", "غدا", "غير", "ـ", "ف", "فان", "فلان", "فو", "فى", "في", "فيم", "فيما", "فيه", "فيها", "قال", "قام", "قبل", "قد", "قطّ", "قلما", "قوة", "كأنّما", "كأين", "كأيّ", "كأيّن", "كاد", "كان", "كانت", "كذا", "كذلك", "كرب", "كل", "كلا", "كلاهما", "كلتا", "كلم", "كليكما", "كليهما", "كلّما", "كلَّا", "كم", "كما", "كي", "كيت", "كيف", "كيفما", "كَأَنَّ", "كِخ", "لئن", "لا", "لات", "لاسيما", "لدن", "لدى", "لعمر", "لقاء", "لك", "لكم", "لكما", "لكن", "لكنَّما", "لكي", "لكيلا", "للامم", "لم", "لما", "لمّا", "لن", "لنا", "له", "لها", "لو", "لوكالة", "لولا", "لوما", "لي", "لَسْتَ", "لَسْتُ", "لَسْتُم", "لَسْتُمَا", "لَسْتُنَّ", "لَسْتِ", "لَسْنَ", "لَعَلَّ", "لَكِنَّ", "لَيْتَ", "لَيْسَ", "لَيْسَا", "لَيْسَتَا", "لَيْسَتْ", "لَيْسُوا", "لَِسْنَا", "ما", "ماانفك", "مابرح", "مادام", "ماذا", "مازال", "مافتئ", "مايو", "متى", "مثل", "مذ", "مساء", "مع", "معاذ", "مقابل", "مكانكم", "مكانكما", "مكانكنّ", "مكانَك", "مليار", "مليون", "مما", "ممن", "من", "منذ", "منها", "مه", "مهما", "مَنْ", "مِن", "نحن", "نحو", "نعم", "نفس", "نفسه", "نهاية", "نَخْ", "نِعِمّا", "نِعْمَ", "ها", "هاؤم", "هاكَ", "هاهنا", "هبّ", "هذا", "هذه", "هكذا", "هل", "هلمَّ", "هلّا", "هم", "هما", "هن", "هنا", "هناك", "هنالك", "هو", "هي", "هيا", "هيت", "هيّا", "هَؤلاء", "هَاتانِ", "هَاتَيْنِ", "هَاتِه", "هَاتِي", "هَجْ", "هَذا", "هَذانِ", "هَذَيْنِ", "هَذِه", "هَذِي", "هَيْهَاتَ", "و", "و6", "وا", "واحد", "واضاف", "واضافت", "واكد", "وان", "واهاً", "واوضح", "وراءَك", "وفي", "وقال", "وقالت", "وقد", "وقف", "وكان", "وكانت", "ولا", "ولم",
                    "ومن", "وهو", "وهي", "ويكأنّ", "وَيْ", "وُشْكَانََ", "يكون", "يمكن", "يوم", "ّأيّان"}

            Public StopWordsDutch() As String = {"aan", "achte", "achter", "af", "al", "alle", "alleen", "alles", "als", "ander", "anders", "beetje",
            "behalve", "beide", "beiden", "ben", "beneden", "bent", "bij", "bijna", "bijv", "blijkbaar", "blijken", "boven", "bv",
            "daar", "daardoor", "daarin", "daarna", "daarom", "daaruit", "dan", "dat", "de", "deden", "deed", "derde", "derhalve", "dertig",
            "deze", "dhr", "die", "dit", "doe", "doen", "doet", "door", "drie", "duizend", "echter", "een", "eens", "eerst", "eerste", "eigen",
            "eigenlijk", "elk", "elke", "en", "enige", "er", "erg", "ergens", "etc", "etcetera", "even", "geen", "genoeg", "geweest", "haar",
            "haarzelf", "had", "hadden", "heb", "hebben", "hebt", "hedden", "heeft", "heel", "hem", "hemzelf", "hen", "het", "hetzelfde",
            "hier", "hierin", "hierna", "hierom", "hij", "hijzelf", "hoe", "honderd", "hun", "ieder", "iedere", "iedereen", "iemand", "iets",
            "ik", "in", "inderdaad", "intussen", "is", "ja", "je", "jij", "jijzelf", "jou", "jouw", "jullie", "kan", "kon", "konden", "kun",
            "kunnen", "kunt", "laatst", "later", "lijken", "lijkt", "maak", "maakt", "maakte", "maakten", "maar", "mag", "maken", "me", "meer",
            "meest", "meestal", "men", "met", "mevr", "mij", "mijn", "minder", "miss", "misschien", "missen", "mits", "mocht", "mochten",
            "moest", "moesten", "moet", "moeten", "mogen", "mr", "mrs", "mw", "na", "naar", "nam", "namelijk", "nee", "neem", "negen",
            "nemen", "nergens", "niemand", "niet", "niets", "niks", "noch", "nochtans", "nog", "nooit", "nu", "nv", "of", "om", "omdat",
            "ondanks", "onder", "ondertussen", "ons", "onze", "onzeker", "ooit", "ook", "op", "over", "overal", "overige", "paar", "per",
            "recent", "redelijk", "samen", "sinds", "steeds", "te", "tegen", "tegenover", "thans", "tien", "tiende", "tijdens", "tja", "toch",
            "toe", "tot", "totdat", "tussen", "twee", "tweede", "u", "uit", "uw", "vaak", "van", "vanaf", "veel", "veertig", "verder",
            "verscheidene", "verschillende", "via", "vier", "vierde", "vijf", "vijfde", "vijftig", "volgend", "volgens", "voor", "voordat",
            "voorts", "waar", "waarom", "waarschijnlijk", "wanneer", "waren", "was", "wat", "we", "wederom", "weer", "weinig", "wel", "welk",
            "welke", "werd", "werden", "werder", "whatever", "wie", "wij", "wijzelf", "wil", "wilden", "willen", "word", "worden", "wordt", "zal",
            "ze", "zei", "zeker", "zelf", "zelfde", "zes", "zeven", "zich", "zij", "zijn", "zijzelf", "zo", "zoals", "zodat", "zou", "zouden",
            "zulk", "zullen"}

            Public StopWordsENG() As String = {"a", "as", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "aint",
                            "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any",
            "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "arent", "around",
            "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming",
            "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief",
            "but", "by", "c", "cmon", "cs", "came", "can", "cant", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com",
            "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldnt",
            "course", "currently", "d", "definitely", "described", "despite", "did", "didnt", "different", "do", "does", "doesnt", "doing", "dont", "done", "down",
            "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever",
            "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed",
            "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives",
            "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadnt", "happens", "hardly", "has", "hasnt", "have", "havent", "having", "he",
            "hes", "hello", "help", "hence", "her", "here", "heres", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his",
            "hither", "hopefully", "how", "howbeit", "however", "i", "id", "ill", "im", "ive", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed",
            "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isnt", "it", "itd", "itll", "its", "its", "itself", "j",
            "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets",
            "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might",
            "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither",
            "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o",
            "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our",
            "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible",
            "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards",
            "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming",
            "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldnt", "since", "six", "so",
            "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying",
            "still", "sub", "such", "sup", "sure", "t", "ts", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "thats", "the",
            "their", "theirs", "them", "themselves", "then", "thence", "there", "theres", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon",
            "these", "they", "theyd", "theyll", "theyre", "theyve", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through",
            "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un",
            "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v",
            "value", "various", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "well", "were", "weve", "welcome", "well",
            "went", "were", "werent", "what", "whats", "whatever", "when", "whence", "whenever", "where", "wheres", "whereafter", "whereas", "whereby", "wherein",
            "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whos", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish",
            "with", "within", "without", "wont", "wonder", "would", "wouldnt", "x", "y", "yes", "yet", "you", "youd", "youll", "youre", "youve", "your", "yours",
            "yourself", "yourselves", "youll", "z", "zero"}

            Public StopWordsFrench() As String = {"a", "abord", "absolument", "afin", "ah", "ai", "aie", "ailleurs", "ainsi", "ait", "allaient", "allo", "allons",
            "allô", "alors", "anterieur", "anterieure", "anterieures", "apres", "après", "as", "assez", "attendu", "au", "aucun", "aucune",
            "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "auraient", "aurait", "auront", "aussi", "autre", "autrefois", "autrement",
            "autres", "autrui", "aux", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avoir", "avons", "ayant", "b",
            "bah", "bas", "basee", "bat", "beau", "beaucoup", "bien", "bigre", "boum", "bravo", "brrr", "c", "car", "ce", "ceci", "cela", "celle",
            "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "cent", "cependant", "certain",
            "certaine", "certaines", "certains", "certes", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque",
            "cher", "chers", "chez", "chiche", "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième",
            "clac", "clic", "combien", "comme", "comment", "comparable", "comparables", "compris", "concernant", "contre", "couic", "crac", "d",
            "da", "dans", "de", "debout", "dedans", "dehors", "deja", "delà", "depuis", "dernier", "derniere", "derriere", "derrière", "des",
            "desormais", "desquelles", "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement", "devant", "devers", "devra",
            "different", "differentes", "differents", "différent", "différente", "différentes", "différents", "dire", "directe", "directement",
            "dit", "dite", "dits", "divers", "diverse", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit", "doivent", "donc",
            "dont", "douze", "douzième", "dring", "du", "duquel", "durant", "dès", "désormais", "e", "effet", "egale", "egalement", "egales", "eh",
            "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin", "entre", "envers", "environ", "es", "est", "et", "etant", "etc",
            "etre", "eu", "euh", "eux", "eux-mêmes", "exactement", "excepté", "extenso", "exterieur", "f", "fais", "faisaient", "faisant", "fait",
            "façon", "feront", "fi", "flac", "floc", "font", "g", "gens", "h", "ha", "hein", "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors",
            "hou", "houp", "hue", "hui", "huit", "huitième", "hum", "hurrah", "hé", "hélas", "i", "il", "ils", "importe", "j", "je", "jusqu", "jusque",
            "juste", "k", "l", "la", "laisser", "laquelle", "las", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "longtemps",
            "lors", "lorsque", "lui", "lui-meme", "lui-même", "là", "lès", "m", "ma", "maint", "maintenant", "mais", "malgre", "malgré", "maximale",
            "me", "meme", "memes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille", "mince", "minimale", "moi", "moi-meme", "moi-même",
            "moindres", "moins", "mon", "moyennant", "multiple", "multiples", "même", "mêmes", "n", "na", "naturel", "naturelle", "naturelles", "ne",
            "neanmoins", "necessaire", "necessairement", "neuf", "neuvième", "ni", "nombreuses", "nombreux", "non", "nos", "notamment", "notre",
            "nous", "nous-mêmes", "nouveau", "nul", "néanmoins", "nôtre", "nôtres", "o", "oh", "ohé", "ollé", "olé", "on", "ont", "onze", "onzième",
            "ore", "ou", "ouf", "ouias", "oust", "ouste", "outre", "ouvert", "ouverte", "ouverts", "o|", "où", "p", "paf", "pan", "par", "parce",
            "parfois", "parle", "parlent", "parler", "parmi", "parseme", "partant", "particulier", "particulière", "particulièrement", "pas",
            "passé", "pendant", "pense", "permet", "personne", "peu", "peut", "peuvent", "peux", "pff", "pfft", "pfut", "pif", "pire", "plein",
            "plouf", "plus", "plusieurs", "plutôt", "possessif", "possessifs", "possible", "possibles", "pouah", "pour", "pourquoi", "pourrais",
            "pourrait", "pouvait", "prealable", "precisement", "premier", "première", "premièrement", "pres", "probable", "probante",
            "procedant", "proche", "près", "psitt", "pu", "puis", "puisque", "pur", "pure", "q", "qu", "quand", "quant", "quant-à-soi", "quanta",
            "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième", "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles",
            "quelqu'un", "quelque", "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare", "rarement", "rares",
            "relative", "relativement", "remarquable", "rend", "rendre", "restant", "reste", "restent", "restrictif", "retour", "revoici",
            "revoilà", "rien", "s", "sa", "sacrebleu", "sait", "sans", "sapristi", "sauf", "se", "sein", "seize", "selon", "semblable", "semblaient",
            "semble", "semblent", "sent", "sept", "septième", "sera", "seraient", "serait", "seront", "ses", "seul", "seule", "seulement", "si",
            "sien", "sienne", "siennes", "siens", "sinon", "six", "sixième", "soi", "soi-même", "soit", "soixante", "son", "sont", "sous", "souvent",
            "specifique", "specifiques", "speculatif", "stop", "strictement", "subtiles", "suffisant", "suffisante", "suffit", "suis", "suit",
            "suivant", "suivante", "suivantes", "suivants", "suivre", "superpose", "sur", "surtout", "t", "ta", "tac", "tant", "tardive", "te",
            "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente", "tes", "tic", "tien", "tienne", "tiennes", "tiens",
            "toc", "toi", "toi-même", "ton", "touchant", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "tres",
            "trois", "troisième", "troisièmement", "trop", "très", "tsoin", "tsouin", "tu", "té", "u", "un", "une", "unes", "uniformement", "unique",
            "uniques", "uns", "v", "va", "vais", "vas", "vers", "via", "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voilà",
            "vont", "vos", "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres", "w", "x", "y", "z", "zut", "à", "â", "ça", "ès", "étaient",
            "étais", "était", "étant", "été", "être", "ô"}

            Public StopWordsItalian() As String = {"IE", "a", "abbastanza", "abbia", "abbiamo", "abbiano", "abbiate", "accidenti", "ad", "adesso", "affinche", "agl", "agli",
                    "ahime", "ahimè", "ai", "al", "alcuna", "alcuni", "alcuno", "all", "alla", "alle", "allo", "allora", "altri", "altrimenti", "altro",
            "altrove", "altrui", "anche", "ancora", "anni", "anno", "ansa", "anticipo", "assai", "attesa", "attraverso", "avanti", "avemmo",
            "avendo", "avente", "aver", "avere", "averlo", "avesse", "avessero", "avessi", "avessimo", "aveste", "avesti", "avete", "aveva",
            "avevamo", "avevano", "avevate", "avevi", "avevo", "avrai", "avranno", "avrebbe", "avrebbero", "avrei", "avremmo", "avremo",
            "avreste", "avresti", "avrete", "avrà", "avrò", "avuta", "avute", "avuti", "avuto", "basta", "bene", "benissimo", "berlusconi",
            "brava", "bravo", "c", "casa", "caso", "cento", "certa", "certe", "certi", "certo", "che", "chi", "chicchessia", "chiunque", "ci",
            "ciascuna", "ciascuno", "cima", "cio", "cioe", "cioè", "circa", "citta", "città", "ciò", "co", "codesta", "codesti", "codesto",
            "cogli", "coi", "col", "colei", "coll", "coloro", "colui", "come", "cominci", "comunque", "con", "concernente", "conciliarsi",
            "conclusione", "consiglio", "contro", "cortesia", "cos", "cosa", "cosi", "così", "cui", "d", "da", "dagl", "dagli", "dai", "dal",
            "dall", "dalla", "dalle", "dallo", "dappertutto", "davanti", "degl", "degli", "dei", "del", "dell", "della", "delle", "dello",
            "dentro", "detto", "deve", "di", "dice", "dietro", "dire", "dirimpetto", "diventa", "diventare", "diventato", "dopo", "dov", "dove",
            "dovra", "dovrà", "dovunque", "due", "dunque", "durante", "e", "ebbe", "ebbero", "ebbi", "ecc", "ecco", "ed", "effettivamente", "egli",
            "ella", "entrambi", "eppure", "era", "erano", "eravamo", "eravate", "eri", "ero", "esempio", "esse", "essendo", "esser", "essere",
            "essi", "ex", "fa", "faccia", "facciamo", "facciano", "facciate", "faccio", "facemmo", "facendo", "facesse", "facessero", "facessi",
            "facessimo", "faceste", "facesti", "faceva", "facevamo", "facevano", "facevate", "facevi", "facevo", "fai", "fanno", "farai",
            "faranno", "fare", "farebbe", "farebbero", "farei", "faremmo", "faremo", "fareste", "faresti", "farete", "farà", "farò", "fatto",
            "favore", "fece", "fecero", "feci", "fin", "finalmente", "finche", "fine", "fino", "forse", "forza", "fosse", "fossero", "fossi",
            "fossimo", "foste", "fosti", "fra", "frattempo", "fu", "fui", "fummo", "fuori", "furono", "futuro", "generale", "gia", "giacche",
            "giorni", "giorno", "già", "gli", "gliela", "gliele", "glieli", "glielo", "gliene", "governo", "grande", "grazie", "gruppo", "ha",
            "haha", "hai", "hanno", "ho", "i", "ieri", "il", "improvviso", "in", "inc", "infatti", "inoltre", "insieme", "intanto", "intorno",
            "invece", "io", "l", "la", "lasciato", "lato", "lavoro", "le", "lei", "li", "lo", "lontano", "loro", "lui", "lungo", "luogo", "là",
            "ma", "macche", "magari", "maggior", "mai", "male", "malgrado", "malissimo", "mancanza", "marche", "me", "medesimo", "mediante",
            "meglio", "meno", "mentre", "mesi", "mezzo", "mi", "mia", "mie", "miei", "mila", "miliardi", "milioni", "minimi", "ministro",
            "mio", "modo", "molti", "moltissimo", "molto", "momento", "mondo", "mosto", "nazionale", "ne", "negl", "negli", "nei", "nel",
            "nell", "nella", "nelle", "nello", "nemmeno", "neppure", "nessun", "nessuna", "nessuno", "niente", "no", "noi", "non", "nondimeno",
            "nonostante", "nonsia", "nostra", "nostre", "nostri", "nostro", "novanta", "nove", "nulla", "nuovo", "o", "od", "oggi", "ogni",
            "ognuna", "ognuno", "oltre", "oppure", "ora", "ore", "osi", "ossia", "ottanta", "otto", "paese", "parecchi", "parecchie",
            "parecchio", "parte", "partendo", "peccato", "peggio", "per", "perche", "perchè", "perché", "percio", "perciò", "perfino", "pero",
            "persino", "persone", "però", "piedi", "pieno", "piglia", "piu", "piuttosto", "più", "po", "pochissimo", "poco", "poi", "poiche",
            "possa", "possedere", "posteriore", "posto", "potrebbe", "preferibilmente", "presa", "press", "prima", "primo", "principalmente",
            "probabilmente", "proprio", "puo", "pure", "purtroppo", "può", "qualche", "qualcosa", "qualcuna", "qualcuno", "quale", "quali",
            "qualunque", "quando", "quanta", "quante", "quanti", "quanto", "quantunque", "quasi", "quattro", "quel", "quella", "quelle",
            "quelli", "quello", "quest", "questa", "queste", "questi", "questo", "qui", "quindi", "realmente", "recente", "recentemente",
            "registrazione", "relativo", "riecco", "salvo", "sara", "sarai", "saranno", "sarebbe", "sarebbero", "sarei", "saremmo", "saremo",
            "sareste", "saresti", "sarete", "sarà", "sarò", "scola", "scopo", "scorso", "se", "secondo", "seguente", "seguito", "sei", "sembra",
            "sembrare", "sembrato", "sembri", "sempre", "senza", "sette", "si", "sia", "siamo", "siano", "siate", "siete", "sig", "solito",
            "solo", "soltanto", "sono", "sopra", "sotto", "spesso", "srl", "sta", "stai", "stando", "stanno", "starai", "staranno", "starebbe",
            "starebbero", "starei", "staremmo", "staremo", "stareste", "staresti", "starete", "starà", "starò", "stata", "state", "stati",
            "stato", "stava", "stavamo", "stavano", "stavate", "stavi", "stavo", "stemmo", "stessa", "stesse", "stessero", "stessi", "stessimo",
            "stesso", "steste", "stesti", "stette", "stettero", "stetti", "stia", "stiamo", "stiano", "stiate", "sto", "su", "sua", "subito",
            "successivamente", "successivo", "sue", "sugl", "sugli", "sui", "sul", "sull", "sulla", "sulle", "sullo", "suo", "suoi", "tale",
            "tali", "talvolta", "tanto", "te", "tempo", "ti", "titolo", "torino", "tra", "tranne", "tre", "trenta", "troppo", "trovato", "tu",
            "tua", "tue", "tuo", "tuoi", "tutta", "tuttavia", "tutte", "tutti", "tutto", "uguali", "ulteriore", "ultimo", "un", "una", "uno",
            "uomo", "va", "vale", "vari", "varia", "varie", "vario", "verso", "vi", "via", "vicino", "visto", "vita", "voi", "volta", "volte",
            "vostra", "vostre", "vostri", "vostro", "è"}

            Public StopWordsSpanish() As String = {"a", "actualmente", "acuerdo", "adelante", "ademas", "además", "adrede", "afirmó", "agregó", "ahi", "ahora",
            "ahí", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí", "alrededor", "ambos", "ampleamos",
            "antano", "antaño", "ante", "anterior", "antes", "apenas", "aproximadamente", "aquel", "aquella", "aquellas", "aquello",
            "aquellos", "aqui", "aquél", "aquélla", "aquéllas", "aquéllos", "aquí", "arriba", "arribaabajo", "aseguró", "asi", "así",
            "atras", "aun", "aunque", "ayer", "añadió", "aún", "b", "bajo", "bastante", "bien", "breve", "buen", "buena", "buenas", "bueno",
            "buenos", "c", "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco", "claro", "comentó", "como", "con",
            "conmigo", "conocer", "conseguimos", "conseguir", "considera", "consideró", "consigo", "consigue", "consiguen", "consigues",
            "contigo", "contra", "cosas", "creo", "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas", "cuanto", "cuantos", "cuatro",
            "cuenta", "cuál", "cuáles", "cuándo", "cuánta", "cuántas", "cuánto", "cuántos", "cómo", "d", "da", "dado", "dan", "dar", "de",
            "debajo", "debe", "deben", "debido", "decir", "dejó", "del", "delante", "demasiado", "demás", "dentro", "deprisa", "desde",
            "despacio", "despues", "después", "detras", "detrás", "dia", "dias", "dice", "dicen", "dicho", "dieron", "diferente", "diferentes",
            "dijeron", "dijo", "dio", "donde", "dos", "durante", "día", "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos",
            "embargo", "empleais", "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra", "enfrente", "enseguida", "entonces",
            "entre", "era", "eramos", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado",
            "estados", "estais", "estamos", "estan", "estar", "estará", "estas", "este", "esto", "estos", "estoy", "estuvo", "está", "están", "ex",
            "excepto", "existe", "existen", "explicó", "expresó", "f", "fin", "final", "fue", "fuera", "fueron", "fui", "fuimos", "g", "general",
            "gran", "grandes", "gueno", "h", "ha", "haber", "habia", "habla", "hablan", "habrá", "había", "habían", "hace", "haceis", "hacemos",
            "hacen", "hacer", "hacerlo", "haces", "hacia", "haciendo", "hago", "han", "hasta", "hay", "haya", "he", "hecho", "hemos", "hicieron",
            "hizo", "horas", "hoy", "hubo", "i", "igual", "incluso", "indicó", "informo", "informó", "intenta", "intentais", "intentamos", "intentan",
            "intentar", "intentas", "intento", "ir", "j", "junto", "k", "l", "la", "lado", "largo", "las", "le", "lejos", "les", "llegó", "lleva",
            "llevar", "lo", "los", "luego", "lugar", "m", "mal", "manera", "manifestó", "mas", "mayor", "me", "mediante", "medio", "mejor", "mencionó",
            "menos", "menudo", "mi", "mia", "mias", "mientras", "mio", "mios", "mis", "misma", "mismas", "mismo", "mismos", "modo", "momento", "mucha",
            "muchas", "mucho", "muchos", "muy", "más", "mí", "mía", "mías", "mío", "míos", "n", "nada", "nadie", "ni", "ninguna", "ningunas", "ninguno",
            "ningunos", "ningún", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "nueva", "nuevas", "nuevo",
            "nuevos", "nunca", "o", "ocho", "os", "otra", "otras", "otro", "otros", "p", "pais", "para", "parece", "parte", "partir", "pasada",
            "pasado", "paìs", "peor", "pero", "pesar", "poca", "pocas", "poco", "pocos", "podeis", "podemos", "poder", "podria", "podriais",
            "podriamos", "podrian", "podrias", "podrá", "podrán", "podría", "podrían", "poner", "por", "porque", "posible", "primer", "primera",
            "primero", "primeros", "principalmente", "pronto", "propia", "propias", "propio", "propios", "proximo", "próximo", "próximos", "pudo",
            "pueda", "puede", "pueden", "puedo", "pues", "q", "qeu", "que", "quedó", "queremos", "quien", "quienes", "quiere", "quiza", "quizas",
            "quizá", "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado", "realizar", "realizó", "repente", "respecto", "s", "sabe",
            "sabeis", "sabemos", "saben", "saber", "sabes", "salvo", "se", "sea", "sean", "segun", "segunda", "segundo", "según", "seis", "ser",
            "sera", "será", "serán", "sería", "señaló", "si", "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sin", "sino", "sobre",
            "sois", "sola", "solamente", "solas", "solo", "solos", "somos", "son", "soy", "soyos", "su", "supuesto", "sus", "suya", "suyas", "suyo",
            "sé", "sí", "sólo", "t", "tal", "tambien", "también", "tampoco", "tan", "tanto", "tarde", "te", "temprano", "tendrá", "tendrán", "teneis",
            "tenemos", "tener", "tenga", "tengo", "tenido", "tenía", "tercera", "ti", "tiempo", "tiene", "tienen", "toda", "todas", "todavia",
            "todavía", "todo", "todos", "total", "trabaja", "trabajais", "trabajamos", "trabajan", "trabajar", "trabajas", "trabajo", "tras",
            "trata", "través", "tres", "tu", "tus", "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "tú", "u", "ultimo", "un", "una", "unas", "uno", "unos",
            "usa", "usais", "usamos", "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va", "vais", "valor", "vamos", "van", "varias", "varios",
            "vaya", "veces", "ver", "verdad", "verdadera", "verdadero", "vez", "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro",
            "vuestros", "w", "x", "y", "ya", "yo", "z", "él", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos", "última", "últimas",
            "último", "últimos"}

            ''' <summary>
            ''' Removes StopWords from sentence
            ''' ARAB/ENG/DUTCH/FRENCH/SPANISH/ITALIAN
            ''' Hopefully leaving just relevant words in the user sentence
            ''' Currently Under Revision (takes too many words)
            ''' </summary>
            ''' <param name="Userinput"></param>
            ''' <returns></returns>
            Public Function RemoveStopWords(ByRef Userinput As String) As String
                ' Userinput = LCase(Userinput).Replace("the", "r")
                For Each item In StopWordsENG
                    Userinput = LCase(Userinput).Replace(item, "")
                Next
                For Each item In StopWordsArab
                    Userinput = Userinput.Replace(item, "")
                Next
                For Each item In StopWordsDutch
                    Userinput = Userinput.Replace(item, "")
                Next
                For Each item In StopWordsFrench
                    Userinput = Userinput.Replace(item, "")
                Next
                For Each item In StopWordsItalian
                    Userinput = Userinput.Replace(item, "")
                Next
                For Each item In StopWordsSpanish
                    Userinput = Userinput.Replace(item, "")
                Next
                Return Userinput
            End Function

            ''' <summary>
            ''' Removes Stop words given a list of stop words
            ''' </summary>
            ''' <param name="Userinput">user input</param>
            ''' <param name="Lst">stop word list</param>
            ''' <returns></returns>
            Public Function RemoveStopWords(ByRef Userinput As String, ByRef Lst As List(Of String)) As String
                For Each item In Lst
                    Userinput = LCase(Userinput).Replace(item, "")
                Next
            End Function

        End Class
        Public Module TextExtensions

            ''' <summary>
            ''' Add full stop to end of String
            ''' </summary>
            ''' <param name="MESSAGE"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function AddFullStop(ByRef MESSAGE As String) As String
                AddFullStop = MESSAGE
                If MESSAGE = "" Then Exit Function
                MESSAGE = Trim(MESSAGE)
                If MESSAGE Like "*." Then Exit Function
                AddFullStop = MESSAGE + "."
            End Function

            ''' <summary>
            ''' Adds string to end of string (no spaces)
            ''' </summary>
            ''' <param name="Str">base string</param>
            ''' <param name="Prefix">Add before (no spaces)</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function AddPrefix(ByRef Str As String, ByVal Prefix As String) As String
                Return Prefix & Str
            End Function

            ''' <summary>
            ''' Adds Suffix to String (No Spaces)
            ''' </summary>
            ''' <param name="Str">Base string</param>
            ''' <param name="Suffix">To be added After</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function AddSuffix(ByRef Str As String, ByVal Suffix As String) As String
                Return Str & Suffix
            End Function

            ''' <summary>
            ''' GO THROUGH EACH CHARACTER AND ' IF PUNCTUATION IE .!?,:'"; REPLACE WITH A SPACE ' IF ,
            ''' OR . THEN CHECK IF BETWEEN TWO NUMBERS, IF IT IS ' THEN LEAVE IT, ELSE REPLACE IT WITH A
            ''' SPACE '
            ''' </summary>
            ''' <param name="STRINPUT">String to be formatted</param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function AlphaNumericalOnly(ByRef STRINPUT As String) As String

                Dim A As Short
                For A = 1 To Len(STRINPUT)
                    If Mid(STRINPUT, A, 1) = "." Or
                    Mid(STRINPUT, A, 1) = "!" Or
                    Mid(STRINPUT, A, 1) = "?" Or
                    Mid(STRINPUT, A, 1) = "," Or
                    Mid(STRINPUT, A, 1) = ":" Or
                    Mid(STRINPUT, A, 1) = "'" Or
                    Mid(STRINPUT, A, 1) = "[" Or
                    Mid(STRINPUT, A, 1) = """" Or
                    Mid(STRINPUT, A, 1) = ";" Then

                        ' BEGIN CHECKING PERIODS AND COMMAS THAT ARE IN BETWEEN NUMBERS '
                        If Mid(STRINPUT, A, 1) = "." Or Mid(STRINPUT, A, 1) = "," Then
                            If Not (A - 1 = 0 Or A = Len(STRINPUT)) Then
                                If Not (IsNumeric(Mid(STRINPUT, A - 1, 1)) Or IsNumeric(Mid(STRINPUT, A + 1, 1))) Then
                                    STRINPUT = Mid(STRINPUT, 1, A - 1) & " " & Mid(STRINPUT, A + 1, Len(STRINPUT) - A)
                                End If
                            Else
                                STRINPUT = Mid(STRINPUT, 1, A - 1) & " " & Mid(STRINPUT, A + 1, Len(STRINPUT) - A)
                            End If
                        Else
                            STRINPUT = Mid(STRINPUT, 1, A - 1) & " " & Mid(STRINPUT, A + 1, Len(STRINPUT) - A)
                        End If

                        ' END CHECKING PERIODS AND COMMAS IN BETWEEN NUMBERS '
                    End If
                Next A
                ' RETURN PUNCTUATION STRIPPED STRING '
                AlphaNumericalOnly = STRINPUT.Replace("  ", " ")
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function AlphaNumericOnly(ByRef txt As String) As String
                Dim NewText As String = ""
                Dim IsLetter As Boolean = False
                Dim IsNumerical As Boolean = False
                For Each chr As Char In txt
                    IsNumerical = False
                    IsLetter = False
                    For Each item In AlphaBet
                        If IsLetter = False Then
                            If chr.ToString = item Then
                                IsLetter = True
                            Else
                            End If
                        End If
                    Next
                    'Check Numerical
                    If IsLetter = False Then
                        For Each item In Numerical
                            If IsNumerical = False Then
                                If chr.ToString = item Then
                                    IsNumerical = True
                                Else
                                End If
                            End If
                        Next
                    Else
                    End If
                    If IsLetter = True Or IsNumerical = True Then
                        NewText &= chr.ToString
                    Else
                        NewText &= " "
                    End If
                Next
                NewText = NewText.Replace("  ", " ")
                Return NewText
            End Function

            'Text
            <Runtime.CompilerServices.Extension()>
            Public Function Capitalise(ByRef MESSAGE As String) As String
                Dim FirstLetter As String
                Capitalise = ""
                If MESSAGE = "" Then Exit Function
                FirstLetter = Left(MESSAGE, 1)
                FirstLetter = UCase(FirstLetter)
                MESSAGE = Right(MESSAGE, Len(MESSAGE) - 1)
                Capitalise = (FirstLetter + MESSAGE)
            End Function

            ''' <summary>
            ''' Capitalizes the text
            ''' </summary>
            ''' <param name="MESSAGE"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function CapitaliseTEXT(ByVal MESSAGE As String) As String
                Dim FirstLetter As String = ""
                CapitaliseTEXT = ""
                If MESSAGE = "" Then Exit Function
                FirstLetter = Left(MESSAGE, 1)
                FirstLetter = UCase(FirstLetter)
                MESSAGE = Right(MESSAGE, Len(MESSAGE) - 1)
                CapitaliseTEXT = (FirstLetter + MESSAGE)
            End Function

            ''' <summary>
            ''' Capitalise the first letter of each word / Tilte Case
            ''' </summary>
            ''' <param name="words">A string - paragraph or sentence</param>
            ''' <returns>String</returns>
            <Runtime.CompilerServices.Extension()>
            Public Function CapitalizeWords(ByVal words As String)
                Dim output As System.Text.StringBuilder = New System.Text.StringBuilder()
                Dim exploded = words.Split(" ")
                If (exploded IsNot Nothing) Then
                    For Each word As String In exploded
                        If word IsNot Nothing Then
                            output.Append(word.Substring(0, 1).ToUpper).Append(word.Substring(1, word.Length - 1)).Append(" ")
                        End If

                    Next
                End If

                Return output.ToString()

            End Function

            ''' <summary>
            '''     A string extension method that query if this object contains the given value.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <param name="value">The value.</param>
            ''' <returns>true if the value is in the string, false if not.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function Contains(this As String, value As String) As Boolean
                Return this.IndexOf(value) <> -1
            End Function

            ''' <summary>
            '''     A string extension method that query if this object contains the given value.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <param name="value">The value.</param>
            ''' <param name="comparisonType">Type of the comparison.</param>
            ''' <returns>true if the value is in the string, false if not.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function Contains(this As String, value As String, comparisonType As StringComparison) As Boolean
                Return this.IndexOf(value, comparisonType) <> -1
            End Function

            ''' <summary>
            ''' Checks if String Contains Letters
            ''' </summary>
            ''' <param name="str"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function ContainsLetters(ByVal str As String) As Boolean

                For i = 0 To str.Length - 1
                    If Char.IsLetter(str.Chars(i)) Then
                        Return True
                    End If
                Next

                Return False

            End Function

            ''' <summary>
            ''' Counts the number of elements in the text, useful for declaring arrays when the element
            ''' length is unknown could be used to split sentence on full stop Find Sentences then again
            ''' on comma(conjunctions) "Find Clauses" NumberOfElements = CountElements(Userinput, delimiter)
            ''' </summary>
            ''' <param name="PHRASE"></param>
            ''' <param name="Delimiter"></param>
            ''' <returns>Integer : number of elements found</returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function CountElements(ByVal PHRASE As String, ByVal Delimiter As String) As Integer
                Dim elementcounter As Integer = 0
                Dim PhraseArray As String()
                PhraseArray = PHRASE.Split(Delimiter)
                elementcounter = UBound(PhraseArray)
                Return elementcounter
            End Function

            ''' <summary>
            ''' counts occurrences of a specific phoneme
            ''' </summary>
            ''' <param name="strIn"></param>
            ''' <param name="strFind"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            <Runtime.CompilerServices.Extension()>
            Public Function CountOccurrences(ByRef strIn As String, ByRef strFind As String) As Integer
                '**
                ' Returns: the number of times a string appears in a string
                '
                '@rem           Example code for CountOccurrences()
                '
                '  ' Counts the occurrences of "ow" in the supplied string.
                '
                '    strTmp = "How now, brown cow"
                '    Returns a value of 4
                '
                '
                'Debug.Print "CountOccurrences(): there are " &  CountOccurrences(strTmp, "ow") &
                '" occurrences of 'ow'" &    " in the string '" & strTmp & "'"
                '
                '@param        strIn Required. String.
                '@param        strFind Required. String.
                '@return       Long.

                Dim lngPos As Integer
                Dim lngWordCount As Integer

                On Error GoTo PROC_ERR

                lngWordCount = 1

                ' Find the first occurrence
                lngPos = InStr(strIn, strFind)

                Do While lngPos > 0
                    ' Find remaining occurrences
                    lngPos = InStr(lngPos + 1, strIn, strFind)
                    If lngPos > 0 Then
                        ' Increment the hit counter
                        lngWordCount = lngWordCount + 1
                    End If
                Loop

                ' Return the value
                CountOccurrences = lngWordCount

PROC_EXIT:
                Exit Function

PROC_ERR:
                MsgBox("Error: " & Err.Number & ". " & Err.Description, , NameOf(CountOccurrences))
                Resume PROC_EXIT

            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function CountVowels(ByVal InputString As String) As Integer
                Dim v(9) As String 'Declare an array  of 10 elements 0 to 9
                Dim vcount As Short 'This variable will contain number of vowels
                Dim flag As Integer
                Dim strLen As Integer
                Dim i As Integer
                v(0) = "a" 'First element of array is assigned small a
                v(1) = "i"
                v(2) = "o"
                v(3) = "u"
                v(4) = "e"
                v(5) = "A" 'Sixth element is assigned Capital A
                v(6) = "I"
                v(7) = "O"
                v(8) = "U"
                v(9) = "E"
                strLen = Len(InputString)

                For flag = 1 To strLen 'It will get every letter of entered string and loop
                    'will terminate when all letters have been examined

                    For i = 0 To 9 'Takes every element of v(9) one by one
                        'Check if current letter is a vowel
                        If Mid(InputString, flag, 1) = v(i) Then
                            vcount = vcount + 1 ' If letter is equal to vowel
                            'then increment vcount by 1
                        End If
                    Next i 'Consider next value of v(i)
                Next flag 'Consider next letter of the entered string

                CountVowels = vcount

            End Function

            ''' <summary>
            ''' Counts tokens in string
            ''' </summary>
            ''' <param name="Str">string to be searched</param>
            ''' <param name="Delimiter">delimiter such as space comma etc</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function CountTokensInString(ByRef Str As String, ByRef Delimiter As String) As Integer
                Dim Words() As String = Split(Str, Delimiter)
                Return Words.Count
            End Function

            ''' <summary>
            ''' Counts the words in a given text
            ''' </summary>
            ''' <param name="NewText"></param>
            ''' <returns>integer: number of words</returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function CountWords(NewText As String) As Integer
                Dim TempArray() As String = NewText.Split(" ")
                CountWords = UBound(TempArray)
                Return CountWords
            End Function

            ''' <summary>
            ''' checks Str contains keyword regardless of case
            ''' </summary>
            ''' <param name="Userinput"></param>
            ''' <param name="Keyword"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function DetectKeyWord(ByRef Userinput As String, ByRef Keyword As String) As Boolean
                Dim mfound As Boolean = False
                If UCase(Userinput).Contains(UCase(Keyword)) = True Or
                        InStr(Userinput, Keyword) > 1 Then
                    mfound = True
                End If

                Return mfound
            End Function

            ''' <summary>
            ''' DETECT IF STATMENT IS AN IF/THEN DETECT IF STATMENT IS AN IF/THEN -- -RETURNS PARTS DETIFTHEN
            ''' = DETECTLOGIC(USERINPUT, "IF", "THEN", IFPART, THENPART)
            ''' </summary>
            ''' <param name="userinput"></param>
            ''' <param name="LOGICA">"IF", can also be replace by "IT CAN BE SAID THAT</param>
            ''' <param name="LOGICB">"THEN" can also be replaced by "it must follow that"</param>
            ''' <param name="IFPART">supply empty string to be used to hold part</param>
            ''' <param name="THENPART">supply empty string to be used to hold part</param>
            ''' <returns>true/false</returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function DetectLOGIC(ByRef userinput As String, ByRef LOGICA As String, ByRef LOGICB As String, ByRef IFPART As String, ByRef THENPART As String) As Boolean
                If InStr(1, userinput, LOGICA, 1) > 0 And InStr(1, userinput, " " & LOGICB & " ", 1) > 0 Then
                    'SPLIT USER INPUT
                    Call SplitPhrase(userinput, " " & LOGICB & " ", IFPART, THENPART)

                    IFPART = Replace(IFPART, LOGICA, "", 1, -1, CompareMethod.Text)
                    THENPART = Replace(THENPART, " " & LOGICB & " ", "", 1, -1, CompareMethod.Text)
                    DetectLOGIC = True
                Else
                    DetectLOGIC = False
                End If
            End Function

            ''' <summary>
            ''' Expand a string such as a field name by inserting a space ahead of each capitalized
            ''' letter (where none exists).
            ''' </summary>
            ''' <param name="inputString"></param>
            ''' <returns>Expanded string</returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function ExpandToWords(ByVal inputString As String) As String
                If inputString Is Nothing Then Return Nothing
                Dim charArray = inputString.ToCharArray
                Dim outStringBuilder As New System.Text.StringBuilder(inputString.Length + 10)
                For index = 0 To charArray.GetUpperBound(0)
                    If Char.IsUpper(charArray(index)) Then
                        'If previous character is also uppercase, don't expand as this may be an acronym.
                        If (index > 0) AndAlso Char.IsUpper(charArray(index - 1)) Then
                            outStringBuilder.Append(charArray(index))
                        Else
                            outStringBuilder.Append(String.Concat(" ", charArray(index)))
                        End If
                    Else
                        outStringBuilder.Append(charArray(index))
                    End If
                Next

                Return outStringBuilder.ToString.Replace("_", " ").Trim

            End Function

            ''' <summary>
            '''     A string extension method that extracts this object.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <param name="predicate">The predicate.</param>
            ''' <returns>A string.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function Extract(this As String, predicate As Func(Of Char, Boolean)) As String
                Return New String(this.ToCharArray().Where(predicate).ToArray())
            End Function

            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractFirstChar(ByRef InputStr As String) As String

                ExtractFirstChar = Left(InputStr, 1)
            End Function

            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractFirstWord(ByRef Statement As String) As String
                Dim StrArr() As String = Split(Statement, " ")
                Return StrArr(0)
            End Function

            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractLastChar(ByRef InputStr As String) As String

                ExtractLastChar = Right(InputStr, 1)
            End Function

            ''' <summary>
            ''' Returns The last word in String
            ''' NOTE: String ois converted to Array then the last element is extracted Count-1
            ''' </summary>
            ''' <param name="InputStr"></param>
            ''' <returns>String</returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractLastWord(ByRef InputStr As String) As String
                Dim TempArr() As String = Split(InputStr, " ")
                Dim Count As Integer = TempArr.Count - 1
                Return TempArr(Count)
            End Function

            ''' <summary>
            '''     A string extension method that extracts the letter described by @this.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <returns>The extracted letter.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function ExtractLetter(this As String) As String
                Return New String(this.ToCharArray().Where(Function(x) [Char].IsLetter(x)).ToArray())
            End Function

            ''' <summary>
            '''     A string extension method that extracts the number described by @this.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <returns>The extracted number.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function ExtractNumber(this As String) As String
                Return New String(this.ToCharArray().Where(Function(x) [Char].IsNumber(x)).ToArray())
            End Function

            ''' <summary>
            ''' extracts string between defined strings
            ''' </summary>
            ''' <param name="value">base sgtring</param>
            ''' <param name="strStart">Start string</param>
            ''' <param name="strEnd">End string</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractStringBetween(ByVal value As String, ByVal strStart As String, ByVal strEnd As String) As String
                If Not String.IsNullOrEmpty(value) Then
                    Dim i As Integer = value.IndexOf(strStart)
                    Dim j As Integer = value.IndexOf(strEnd)
                    Return value.Substring(i, j - i)
                Else
                    Return value
                End If
            End Function

            ''' <summary>
            ''' Extract words Either side of Divider
            ''' </summary>
            ''' <param name="TextStr"></param>
            ''' <param name="Divider"></param>
            ''' <param name="Mode">Front = F Back =B</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function ExtractWordsEitherSide(ByRef TextStr As String, ByRef Divider As String, ByRef Mode As String) As String
                ExtractWordsEitherSide = ""
                Select Case Mode
                    Case "F"
                        Return ExtractWordsEitherSide(TextStr, Divider, "F")
                    Case "B"
                        Return ExtractWordsEitherSide(TextStr, Divider, "B")
                End Select

            End Function

            ' Generate a random number based on the upper and lower bounds of the array,
            'then use that to return the item.
            <Runtime.CompilerServices.Extension()>
            Public Function FetchRandomItem(Of t)(ByRef theArray() As t) As t

                Dim randNumberGenerator As New Random
                Randomize()
                Dim index As Integer = randNumberGenerator.Next(theArray.GetLowerBound(0),
                                                            theArray.GetUpperBound(0) + 1)

                Return theArray(index)

            End Function

            ''' <summary>
            ''' Define the search terms. This list could also be dynamically populated at runtime Find
            ''' sentences that contain all the terms in the wordsToMatch array Note that the number of
            ''' terms to match is not specified at compile time
            ''' </summary>
            ''' <param name="TextStr1">String to be searched</param>
            ''' <param name="Words">List of Words to be detected</param>
            ''' <returns>Sentences containing words</returns>
            <Runtime.CompilerServices.Extension()>
            Public Function FindSentencesContaining(ByRef TextStr1 As String, ByRef Words As List(Of String)) As List(Of String)
                ' Split the text block into an array of sentences.
                Dim sentences As String() = TextStr1.Split(New Char() {".", "?", "!"})

                Dim wordsToMatch(Words.Count) As String
                Dim I As Integer = 0
                For Each item In Words
                    wordsToMatch(I) = item
                    I += 1
                Next

                Dim sentenceQuery = From sentence In sentences
                                    Let w = sentence.Split(New Char() {" ", ",", ".", ";", ":"},
                                                       StringSplitOptions.RemoveEmptyEntries)
                                    Where w.Distinct().Intersect(wordsToMatch).Count = wordsToMatch.Count()
                                    Select sentence

                ' Execute the query

                Dim StrList As New List(Of String)
                For Each str As String In sentenceQuery
                    StrList.Add(str)
                Next
                Return StrList
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function FormatJsonOutput(ByVal jsonString As String) As String
                Dim stringBuilder = New StringBuilder()
                Dim escaping As Boolean = False
                Dim inQuotes As Boolean = False
                Dim indentation As Integer = 0

                For Each character As Char In jsonString

                    If escaping Then
                        escaping = False
                        stringBuilder.Append(character)
                    Else

                        If character = "\"c Then
                            escaping = True
                            stringBuilder.Append(character)
                        ElseIf character = """"c Then
                            inQuotes = Not inQuotes
                            stringBuilder.Append(character)
                        ElseIf Not inQuotes Then

                            If character = ","c Then
                                stringBuilder.Append(character)
                                stringBuilder.Append(vbCrLf)
                                stringBuilder.Append(vbTab, indentation)
                            ElseIf character = "["c OrElse character = "{"c Then
                                stringBuilder.Append(character)
                                stringBuilder.Append(vbCrLf)
                                stringBuilder.Append(vbTab, System.Threading.Interlocked.Increment(indentation))
                            ElseIf character = "]"c OrElse character = "}"c Then
                                stringBuilder.Append(vbCrLf)
                                stringBuilder.Append(vbTab, System.Threading.Interlocked.Decrement(indentation))
                                stringBuilder.Append(character)
                            ElseIf character = ":"c Then
                                stringBuilder.Append(character)
                                stringBuilder.Append(vbTab)
                            ElseIf Not Char.IsWhiteSpace(character) Then
                                stringBuilder.Append(character)
                            End If
                        Else
                            stringBuilder.Append(character)
                        End If
                    End If
                Next

                Return stringBuilder.ToString()
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function FormatText(ByRef Text As String) As String
                Dim FormatTextResponse As String = ""
                'FORMAT USERINPUT
                'turn to uppercase for searching the db
                Text = LTrim(Text)
                Text = RTrim(Text)
                Text = UCase(Text)

                FormatTextResponse = Text
                Return FormatTextResponse
            End Function

            ''' <summary>
            ''' Gets the string after the given string parameter.
            ''' </summary>
            ''' <param name="value">The default value.</param>
            ''' <param name="x">The given string parameter.</param>
            ''' <returns></returns>
            ''' <remarks>Unlike GetBefore, this method trims the result</remarks>
            <System.Runtime.CompilerServices.Extension>
            Public Function GetAfter(value As String, x As String) As String
                Dim xPos = value.LastIndexOf(x, StringComparison.Ordinal)
                If xPos = -1 Then
                    Return [String].Empty
                End If
                Dim startIndex = xPos + x.Length
                Return If(startIndex >= value.Length, [String].Empty, value.Substring(startIndex).Trim())
            End Function

            ''' <summary>
            ''' Gets the string before the given string parameter.
            ''' </summary>
            ''' <param name="value">The default value.</param>
            ''' <param name="x">The given string parameter.</param>
            ''' <returns></returns>
            ''' <remarks>Unlike GetBetween and GetAfter, this does not Trim the result.</remarks>
            <System.Runtime.CompilerServices.Extension>
            Public Function GetBefore(value As String, x As String) As String
                Dim xPos = value.IndexOf(x, StringComparison.Ordinal)
                Return If(xPos = -1, [String].Empty, value.Substring(0, xPos))
            End Function

            ''' <summary>
            ''' Gets the string between the given string parameters.
            ''' </summary>
            ''' <param name="value">The source value.</param>
            ''' <param name="x">The left string sentinel.</param>
            ''' <param name="y">The right string sentinel</param>
            ''' <returns></returns>
            ''' <remarks>Unlike GetBefore, this method trims the result</remarks>
            <System.Runtime.CompilerServices.Extension>
            Public Function GetBetween(value As String, x As String, y As String) As String
                Dim xPos = value.IndexOf(x, StringComparison.Ordinal)
                Dim yPos = value.LastIndexOf(y, StringComparison.Ordinal)
                If xPos = -1 OrElse xPos = -1 Then
                    Return [String].Empty
                End If
                Dim startIndex = xPos + x.Length
                Return If(startIndex >= yPos, [String].Empty, value.Substring(startIndex, yPos - startIndex).Trim())
            End Function

            ''' <summary>
            ''' Returns the first Word
            ''' </summary>
            ''' <param name="Statement"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function GetPrefix(ByRef Statement As String) As String
                Dim StrArr() As String = Split(Statement, " ")
                Return StrArr(0)
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function GetRandItemfromList(ByRef li As List(Of String)) As String
                Randomize()
                Return li.Item(Int(Rnd() * (li.Count - 1)))
            End Function

            ''' <summary>
            ''' Returns random character from string given length of the string to choose from
            ''' </summary>
            ''' <param name="Source"></param>
            ''' <param name="Length"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function GetRndChar(ByVal Source As String, ByVal Length As Integer) As String
                Dim rnd As New Random
                If Source Is Nothing Then Throw New ArgumentNullException(NameOf(Source), "Must contain a string,")
                If Length <= 0 Then Throw New ArgumentException("Length must be a least one.", NameOf(Length))
                Dim s As String = ""
                Dim builder As New System.Text.StringBuilder()
                builder.Append(s)
                For i = 1 To Length
                    builder.Append(Source(rnd.Next(0, Source.Length)))
                Next
                s = builder.ToString()
                Return s
            End Function

            ''' <summary>
            ''' Returns from index to end of file
            ''' </summary>
            ''' <param name="Str">String</param>
            ''' <param name="indx">Index</param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function GetSlice(ByRef Str As String, ByRef indx As Integer) As String
                If indx <= Str.Length Then
                    Str.Substring(indx, Str.Length)
                    Return Str(indx)
                Else
                End If
                Return Nothing
            End Function

            ''' <summary>
            ''' gets the last word
            ''' </summary>
            ''' <param name="InputStr"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function GetSuffix(ByRef InputStr As String) As String
                Dim TempArr() As String = Split(InputStr, " ")
                Dim Count As Integer = TempArr.Count - 1
                Return TempArr(Count)
            End Function

            <System.Runtime.CompilerServices.Extension>
            Public Function GetWordsBetween(ByRef InputStr As String, ByRef StartStr As String, ByRef StopStr As String)
                Return InputStr.ExtractStringBetween(StartStr, StopStr)
            End Function

            ''' <summary>
            '''     A string extension method that query if '@this' satisfy the specified pattern.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <param name="pattern">The pattern to use. Use '*' as wildcard string.</param>
            ''' <returns>true if '@this' satisfy the specified pattern, false if not.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function IsLike(this As String, pattern As String) As Boolean
                ' Turn the pattern into regex pattern, and match the whole string with ^$
                Dim regexPattern As String = "^" + Regex.Escape(pattern) + "$"

                ' Escape special character ?, #, *, [], and [!]
                regexPattern = regexPattern.Replace("\[!", "[^").Replace("\[", "[").Replace("\]", "]").Replace("\?", ".").Replace("\*", ".*").Replace("\#", "\d")

                Return Regex.IsMatch(this, regexPattern)
            End Function

            ''' <summary>
            ''' Checks if string is a reserved VBscipt Keyword
            ''' </summary>
            ''' <param name="keyword"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Function IsReservedWord(ByVal keyword As String) As Boolean
                Dim IsReserved = False
                Select Case LCase(keyword)
                    Case "and" : IsReserved = True
                    Case "as" : IsReserved = True
                    Case "boolean" : IsReserved = True
                    Case "byref" : IsReserved = True
                    Case "byte" : IsReserved = True
                    Case "byval" : IsReserved = True
                    Case "call" : IsReserved = True
                    Case "case" : IsReserved = True
                    Case "class" : IsReserved = True
                    Case "const" : IsReserved = True
                    Case "currency" : IsReserved = True
                    Case "debug" : IsReserved = True
                    Case "dim" : IsReserved = True
                    Case "do" : IsReserved = True
                    Case "double" : IsReserved = True
                    Case "each" : IsReserved = True
                    Case "else" : IsReserved = True
                    Case "elseif" : IsReserved = True
                    Case "empty" : IsReserved = True
                    Case "end" : IsReserved = True
                    Case "endif" : IsReserved = True
                    Case "enum" : IsReserved = True
                    Case "eqv" : IsReserved = True
                    Case "event" : IsReserved = True
                    Case "exit" : IsReserved = True
                    Case "false" : IsReserved = True
                    Case "for" : IsReserved = True
                    Case "function" : IsReserved = True
                    Case "get" : IsReserved = True
                    Case "goto" : IsReserved = True
                    Case "if" : IsReserved = True
                    Case "imp" : IsReserved = True
                    Case "implements" : IsReserved = True
                    Case "in" : IsReserved = True
                    Case "integer" : IsReserved = True
                    Case "is" : IsReserved = True
                    Case "let" : IsReserved = True
                    Case "like" : IsReserved = True
                    Case "long" : IsReserved = True
                    Case "loop" : IsReserved = True
                    Case "lset" : IsReserved = True
                    Case "me" : IsReserved = True
                    Case "mod" : IsReserved = True
                    Case "new" : IsReserved = True
                    Case "next" : IsReserved = True
                    Case "not" : IsReserved = True
                    Case "nothing" : IsReserved = True
                    Case "null" : IsReserved = True
                    Case "on" : IsReserved = True
                    Case "option" : IsReserved = True
                    Case "optional" : IsReserved = True
                    Case "or" : IsReserved = True
                    Case "paramarray" : IsReserved = True
                    Case "preserve" : IsReserved = True
                    Case "private" : IsReserved = True
                    Case "public" : IsReserved = True
                    Case "raiseevent" : IsReserved = True
                    Case "redim" : IsReserved = True
                    Case "rem" : IsReserved = True
                    Case "resume" : IsReserved = True
                    Case "rset" : IsReserved = True
                    Case "select" : IsReserved = True
                    Case "set" : IsReserved = True
                    Case "shared" : IsReserved = True
                    Case "single" : IsReserved = True
                    Case "static" : IsReserved = True
                    Case "stop" : IsReserved = True
                    Case "sub" : IsReserved = True
                    Case "then" : IsReserved = True
                    Case "to" : IsReserved = True
                    Case "true" : IsReserved = True
                    Case "type" : IsReserved = True
                    Case "typeof" : IsReserved = True
                    Case "until" : IsReserved = True
                    Case "variant" : IsReserved = True
                    Case "wend" : IsReserved = True
                    Case "while" : IsReserved = True
                    Case "with" : IsReserved = True
                    Case "xor" : IsReserved = True
                End Select
                Return IsReserved
            End Function

            ''' <summary>
            ''' Returns Propercase Sentence
            ''' </summary>
            ''' <param name="TheString">String to be formatted</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function ProperCase(ByRef TheString As String) As String
                ProperCase = UCase(Left(TheString, 1))

                For i = 2 To Len(TheString)

                    ProperCase = If(Mid(TheString, i - 1, 1) = " ", ProperCase & UCase(Mid(TheString, i, 1)), ProperCase & LCase(Mid(TheString, i, 1)))
                Next i
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function RemoveBrackets(ByRef Txt As String) As String
                'Brackets
                Txt = Txt.Replace("(", "")
                Txt = Txt.Replace("{", "")
                Txt = Txt.Replace("}", "")
                Txt = Txt.Replace("[", "")
                Txt = Txt.Replace("]", "")
                Return Txt
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function RemoveFullStop(ByRef MESSAGE As String) As String
Loop1:
                If Right(MESSAGE, 1) = "." Then MESSAGE = Left(MESSAGE, Len(MESSAGE) - 1) : GoTo Loop1
                Return MESSAGE
            End Function

            ''' <summary>
            '''     A string extension method that removes the letter described by @this.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <returns>A string.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function RemoveLetter(this As String) As String
                Return New String(this.ToCharArray().Where(Function(x) Not [Char].IsLetter(x)).ToArray())
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function RemoveMathsSymbols(ByRef Txt As String) As String
                'Maths Symbols
                Txt = Txt.Replace("+", "")
                Txt = Txt.Replace("=", "")
                Txt = Txt.Replace("-", "")
                Txt = Txt.Replace("/", "")
                Txt = Txt.Replace("*", "")
                Txt = Txt.Replace("<", "")
                Txt = Txt.Replace(">", "")
                Txt = Txt.Replace("%", "")
                Return Txt
            End Function

            ''' <summary>
            '''     A string extension method that removes the number described by @this.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <returns>A string.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function RemoveNumber(this As String) As String
                Return New String(this.ToCharArray().Where(Function(x) Not [Char].IsNumber(x)).ToArray())
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function RemovePunctuation(ByRef Txt As String) As String
                'Punctuation
                Txt = Txt.Replace(",", "")
                Txt = Txt.Replace(".", "")
                Txt = Txt.Replace(";", "")
                Txt = Txt.Replace("'", "")
                Txt = Txt.Replace("_", "")
                Txt = Txt.Replace("?", "")
                Txt = Txt.Replace("!", "")
                Txt = Txt.Replace("&", "")
                Txt = Txt.Replace(":", "")

                Return Txt
            End Function

            ''' <summary>
            ''' Removes StopWords from sentence
            ''' ARAB/ENG/DUTCH/FRENCH/SPANISH/ITALIAN
            ''' Hopefully leaving just relevant words in the user sentence
            ''' Currently Under Revision (takes too many words)
            ''' </summary>
            ''' <param name="Userinput"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function RemoveStopWords(ByRef Userinput As String) As String
                ' Userinput = LCase(Userinput).Replace("the", "r")
                For Each item In StopWordsENG
                    Userinput = LCase(Userinput).Replace(item, "")
                Next
                For Each item In StopWordsArab
                    Userinput = Userinput.Replace(item, "")
                Next
                For Each item In StopWordsDutch
                    Userinput = Userinput.Replace(item, "")
                Next
                For Each item In StopWordsFrench
                    Userinput = Userinput.Replace(item, "")
                Next
                For Each item In StopWordsItalian
                    Userinput = Userinput.Replace(item, "")
                Next
                For Each item In StopWordsSpanish
                    Userinput = Userinput.Replace(item, "")
                Next
                Return Userinput
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function RemoveStopWords(ByRef txt As String, ByRef StopWrds As List(Of String)) As String
                For Each item In StopWrds
                    txt = txt.Replace(item, "")
                Next
                Return txt
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function RemoveSymbols(ByRef Txt As String) As String
                'Basic Symbols
                Txt = Txt.Replace("£", "")
                Txt = Txt.Replace("$", "")
                Txt = Txt.Replace("^", "")
                Txt = Txt.Replace("@", "")
                Txt = Txt.Replace("#", "")
                Txt = Txt.Replace("~", "")
                Txt = Txt.Replace("\", "")
                Return Txt
            End Function

            ''' <summary>
            '''     A string extension method that removes the letter.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <param name="predicate">The predicate.</param>
            ''' <returns>A string.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function RemoveWhere(this As String, predicate As Func(Of Char, Boolean)) As String
                Return New String(this.ToCharArray().Where(Function(x) Not predicate(x)).ToArray())
            End Function

            ''' <summary>
            ''' Advanced search String pattern Wildcard denotes which position 1st =1 or 2nd =2 Send
            ''' Original input &gt; Search pattern to be used &gt; Wildcard requred SPattern = "WHAT
            ''' COLOUR DO YOU LIKE * OR *" Textstr = "WHAT COLOUR DO YOU LIKE red OR black" ITEM_FOUND =
            ''' = SearchPattern(USERINPUT, SPattern, 1) ---- RETURNS RED ITEM_FOUND = =
            ''' SearchPattern(USERINPUT, SPattern, 1) ---- RETURNS black
            ''' </summary>
            ''' <param name="TextSTR">
            ''' TextStr Required. String.EG: "WHAT COLOUR DO YOU LIKE red OR black"
            ''' </param>
            ''' <param name="SPattern">
            ''' SPattern Required. String.EG: "WHAT COLOUR DO YOU LIKE * OR *"
            ''' </param>
            ''' <param name="Wildcard">Wildcard Required. Integer.EG: 1st =1 or 2nd =2</param>
            ''' <returns></returns>
            ''' <remarks>* in search pattern</remarks>
            <Runtime.CompilerServices.Extension()>
            Public Function SearchPattern(ByRef TextSTR As String, ByRef SPattern As String, ByRef Wildcard As Short) As String
                Dim SearchP2 As String
                Dim SearchP1 As String
                Dim TextStrp3 As String
                Dim TextStrp4 As String
                SearchPattern = ""
                SearchP2 = ""
                SearchP1 = ""
                TextStrp3 = ""
                TextStrp4 = ""
                If TextSTR Like SPattern = True Then
                    Select Case Wildcard
                        Case 1
                            Call SplitPhrase(SPattern, "*", SearchP1, SearchP2)
                            TextSTR = Replace(TextSTR, SearchP1, "", 1, -1, CompareMethod.Text)

                            SearchP2 = Replace(SearchP2, "*", "", 1, -1, CompareMethod.Text)
                            Call SplitPhrase(TextSTR, SearchP2, TextStrp3, TextStrp4)

                            TextSTR = TextStrp3

                        Case 2
                            Call SplitPhrase(SPattern, "*", SearchP1, SearchP2)
                            SPattern = Replace(SPattern, SearchP1, " ", 1, -1, CompareMethod.Text)
                            TextSTR = Replace(TextSTR, SearchP1, " ", 1, -1, CompareMethod.Text)

                            Call SplitPhrase(SearchP2, "*", SearchP1, SearchP2)
                            Call SplitPhrase(TextSTR, SearchP1, TextStrp3, TextStrp4)

                            TextSTR = TextStrp4

                    End Select

                    SearchPattern = TextSTR
                    LTrim(SearchPattern)
                    RTrim(SearchPattern)
                Else
                End If

            End Function

            ''' <summary>
            ''' Advanced search String pattern Wildcard denotes which position 1st =1 or 2nd =2 Send
            ''' Original input &gt; Search pattern to be used &gt; Wildcard requred SPattern = "WHAT
            ''' COLOUR DO YOU LIKE * OR *" Textstr = "WHAT COLOUR DO YOU LIKE red OR black" ITEM_FOUND =
            ''' = SearchPattern(USERINPUT, SPattern, 1) ---- RETURNS RED ITEM_FOUND = =
            ''' SearchPattern(USERINPUT, SPattern, 2) ---- RETURNS black
            ''' </summary>
            ''' <param name="TextSTR">TextStr = "Pick Red OR Blue" . String.</param>
            ''' <param name="SPattern">Search String = ("Pick * OR *") String.</param>
            ''' <param name="Wildcard">Wildcard Required. Integer. = 1= Red / 2= Blue</param>
            ''' <returns></returns>
            ''' <remarks>finds the * in search pattern</remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function SearchStringbyPattern(ByRef TextSTR As String, ByRef SPattern As String, ByRef Wildcard As Short) As String
                Dim SearchP2 As String
                Dim SearchP1 As String
                Dim TextStrp3 As String
                Dim TextStrp4 As String
                SearchStringbyPattern = ""
                SearchP2 = ""
                SearchP1 = ""
                TextStrp3 = ""
                TextStrp4 = ""
                If TextSTR Like SPattern = True Then
                    Select Case Wildcard
                        Case 1
                            Call SplitString(SPattern, "*", SearchP1, SearchP2)
                            TextSTR = Replace(TextSTR, SearchP1, "", 1, -1, CompareMethod.Text)

                            SearchP2 = Replace(SearchP2, "*", "", 1, -1, CompareMethod.Text)
                            Call SplitString(TextSTR, SearchP2, TextStrp3, TextStrp4)

                            TextSTR = TextStrp3

                        Case 2
                            Call SplitString(SPattern, "*", SearchP1, SearchP2)
                            SPattern = Replace(SPattern, SearchP1, " ", 1, -1, CompareMethod.Text)
                            TextSTR = Replace(TextSTR, SearchP1, " ", 1, -1, CompareMethod.Text)

                            Call SplitString(SearchP2, "*", SearchP1, SearchP2)
                            Call SplitString(TextSTR, SearchP1, TextStrp3, TextStrp4)

                            TextSTR = TextStrp4

                    End Select

                    SearchStringbyPattern = TextSTR
                    LTrim(SearchStringbyPattern)
                    RTrim(SearchStringbyPattern)
                Else
                End If

            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function SpaceItems(ByRef txt As String, Item As String) As String
                Return txt.Replace(Item, " " & Item & " ")
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function SpacePunctuation(ByRef Txt As String) As String
                For Each item In Symbols
                    Txt = SpaceItems(Txt, item)
                Next
                For Each item In EncapuslationPunctuationEnd
                    Txt = SpaceItems(Txt, item)
                Next
                For Each item In EncapuslationPunctuationStart
                    Txt = SpaceItems(Txt, item)
                Next
                For Each item In GramaticalPunctuation
                    Txt = SpaceItems(Txt, item)
                Next
                For Each item In MathPunctuation
                    Txt = SpaceItems(Txt, item)
                Next
                For Each item In MoneyPunctuation
                    Txt = SpaceItems(Txt, item)
                Next
                Return Txt
            End Function

            ''' <summary>
            ''' SPLITS THE GIVEN PHRASE UP INTO TWO PARTS by dividing word SplitPhrase(Userinput, "and",
            ''' Firstp, SecondP)
            ''' </summary>
            ''' <param name="PHRASE">Sentence to be divided</param>
            ''' <param name="DIVIDINGWORD">String: Word to divide sentence by</param>
            ''' <param name="FIRSTPART">String: firstpart of sentence to be populated</param>
            ''' <param name="SECONDPART">String: Secondpart of sentence to be populated</param>
            ''' <remarks></remarks>
            <Runtime.CompilerServices.Extension()>
            Public Sub SplitPhrase(ByVal PHRASE As String, ByRef DIVIDINGWORD As String, ByRef FIRSTPART As String, ByRef SECONDPART As String)
                Dim POS As Short
                POS = InStr(PHRASE, DIVIDINGWORD)
                If (POS > 0) Then
                    FIRSTPART = Trim(Left(PHRASE, POS - 1))
                    SECONDPART = Trim(Right(PHRASE, Len(PHRASE) - POS - Len(DIVIDINGWORD) + 1))
                Else
                    FIRSTPART = ""
                    SECONDPART = PHRASE
                End If
            End Sub

            ''' <summary>
            ''' SPLITS THE GIVEN PHRASE UP INTO TWO PARTS by dividing word SplitPhrase(Userinput, "and",
            ''' Firstp, SecondP)
            ''' </summary>
            ''' <param name="PHRASE">String: Sentence to be divided</param>
            ''' <param name="DIVIDINGWORD">String: Word to divide sentence by</param>
            ''' <param name="FIRSTPART">String-Returned : firstpart of sentence to be populated</param>
            ''' <param name="SECONDPART">String-Returned : Secondpart of sentence to be populated</param>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Sub SplitString(ByVal PHRASE As String, ByRef DIVIDINGWORD As String, ByRef FIRSTPART As String, ByRef SECONDPART As String)
                Dim POS As Short
                'Check Error
                If DIVIDINGWORD IsNot Nothing And PHRASE IsNot Nothing Then

                    POS = InStr(PHRASE, DIVIDINGWORD)
                    If (POS > 0) Then
                        FIRSTPART = Trim(Left(PHRASE, POS - 1))
                        SECONDPART = Trim(Right(PHRASE, Len(PHRASE) - POS - Len(DIVIDINGWORD) + 1))
                    Else
                        FIRSTPART = ""
                        SECONDPART = PHRASE
                    End If
                Else

                End If
            End Sub

            ''' <summary>
            ''' Split string to List of strings
            ''' </summary>
            ''' <param name="Str">base string</param>
            ''' <param name="Seperator">to be seperated by</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function SplitToList(ByRef Str As String, ByVal Seperator As String) As List(Of String)
                Dim lst As New List(Of String)
                If Str <> "" = True And Seperator <> "" Then

                    Dim Found() As String = Str.Split(Seperator)
                    For Each item In Found
                        lst.Add(item)
                    Next
                Else

                End If
                Return lst
            End Function

            ''' <summary>
            ''' Returns a delimited string from the list.
            ''' </summary>
            ''' <param name="ls"></param>
            ''' <param name="delimiter"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function ToDelimitedString(ls As List(Of String), delimiter As String) As String
                Dim sb As New StringBuilder
                For Each buf As String In ls
                    sb.Append(buf)
                    sb.Append(delimiter)
                Next
                Return sb.ToString.Trim(CChar(delimiter))
            End Function

            ''' <summary>
            ''' Convert object to Json String
            ''' </summary>
            ''' <param name="Item"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function ToJson(ByRef Item As Object) As String
                Dim Converter As New JavaScriptSerializer
                Return Converter.Serialize(Item)

            End Function

            ''' <summary>
            ''' Counts the vowels used (AEIOU)
            ''' </summary>
            ''' <param name="InputString"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            <Runtime.CompilerServices.Extension()>
            Public Function VowelCount(ByVal InputString As String) As Integer
                Dim v(9) As String 'Declare an array  of 10 elements 0 to 9
                Dim vcount As Short 'This variable will contain number of vowels
                Dim flag As Integer
                Dim strLen As Integer
                Dim i As Integer
                v(0) = "a" 'First element of array is assigned small a
                v(1) = "i"
                v(2) = "o"
                v(3) = "u"
                v(4) = "e"
                v(5) = "A" 'Sixth element is assigned Capital A
                v(6) = "I"
                v(7) = "O"
                v(8) = "U"
                v(9) = "E"
                strLen = Len(InputString)

                For flag = 1 To strLen 'It will get every letter of entered string and loop
                    'will terminate when all letters have been examined

                    For i = 0 To 9 'Takes every element of v(9) one by one
                        'Check if current letter is a vowel
                        If Mid(InputString, flag, 1) = v(i) Then
                            vcount = vcount + 1 ' If letter is equal to vowel
                            'then increment vcount by 1
                        End If
                    Next i 'Consider next value of v(i)
                Next flag 'Consider next letter of the enterd string

                VowelCount = vcount

            End Function

        End Module
    End Namespace


    Public Class RegexFilter

        Public Function FilterUsingRegexPatterns(data As List(Of String), patterns As List(Of String)) As List(Of String)
            Dim filteredData As New List(Of String)

            For Each chunk As String In data
                Dim shouldIncludeChunk As Boolean = True

                For Each pattern As String In patterns
                    Dim regex As New Regex(pattern, RegexOptions.IgnoreCase)
                    If regex.IsMatch(chunk) Then
                        shouldIncludeChunk = False
                        Exit For
                    End If
                Next

                If shouldIncludeChunk Then
                    filteredData.Add(chunk)
                End If
            Next

            Return filteredData
        End Function

    End Class
    Public Class EntityLoader
        Public EntityList As List(Of Entity)
        Public EntityTypes As List(Of String)
        Private Random As New Random()

        Public Shared Function DetectEntities(chunks As List(Of String), EntityList As List(Of KeyValuePair(Of String, String))) As List(Of KeyValuePair(Of String, String))
            ' Entity detection logic based on chunks
            Dim entityChunks As New List(Of KeyValuePair(Of String, String))

            ' Example entity detection
            For Each chunk As String In chunks
                For Each entity In EntityList
                    If IsTermPresent(entity.Value, chunk, EntityList) Then
                        entityChunks.Add(entity)
                    End If
                Next
            Next

            Return entityChunks
        End Function

        ''' <summary>
        ''' Checks if a specific term (entity or keyword) is present in the processed text data.
        ''' </summary>
        ''' <param name="term">The term to check.</param>
        ''' <param name="data">The processed text data.</param>
        ''' <returns>True if the term is present; otherwise, false.</returns>
        Public Shared Function IsTermPresent(term As String, data As String, EntityList As List(Of KeyValuePair(Of String, String))) As Boolean
            Return data.ToLower().Contains(term.ToLower())
        End Function

        ''' <summary>
        ''' Loads entity information from a file for filtering and labeling.
        ''' </summary>
        ''' <param name="filePath">The path to the entity list file (text or JSON).</param>
        Public Shared Function LoadEntityListFromFile(filePath As String) As List(Of KeyValuePair(Of String, String))
            ' Load entity list from file (text or JSON)
            Dim fileContent As String = File.ReadAllText(filePath)
            Return JsonConvert.DeserializeObject(Of List(Of KeyValuePair(Of String, String)))(fileContent)
        End Function

        Public Function GenerateNamedEntity() As String
            Dim entityType As String = GetRandomEntity()
            Dim entityName As String = String.Empty
            Dim Words As New List(Of String)
            For Each item In EntityList
                If item.Type = entityType Then
                    Words.Add(item.Value)
                End If
            Next

            entityName = GetRandomItem(Words.ToArray)

            Return entityName
        End Function

        Public Function GetRandomContext() As String
            Dim entity As String = GenerateNamedEntity()
            Dim contextType As String = GetRandomItem(New String() {"before", "after"})

            Select Case contextType
                Case "before"
                    Return $"In the context of {entity},"
                Case "after"
                    Return $"Considering {entity},"
                Case Else
                    Return String.Empty
            End Select
        End Function

        Public Function GetRandomEntity() As String

            Dim index As Integer = Random.Next(0, EntityTypes.Count)
            Return EntityTypes(index)
        End Function

        Public Function GetRandomItem(items As String()) As String
            Dim index As Integer = Random.Next(0, items.Length)
            Return items(index)
        End Function

    End Class
    Public Interface ICorpusChunker

        Function FilterUsingPunctuationVocabulary(data As List(Of String)) As List(Of String)

        Function GenerateClassificationDataset(data As List(Of String), classes As List(Of String)) As List(Of Tuple(Of String, String))

        Function GeneratePredictiveDataset(data As List(Of String), windowSize As Integer) As List(Of String())

        Function ProcessTextData(rawData As String, useFiltering As Boolean) As List(Of String)

    End Interface
    Public Class ChunkProcessor
        Private chunkType As ChunkType
        Private maxSize As Integer

        Public Sub New(chunkType As ChunkType, Optional maxSize As Integer = 0)
            Me.chunkType = chunkType
            Me.maxSize = maxSize
        End Sub

        Public Shared Function ApplyPadding(chunks As List(Of String), ByRef maxsize As Integer) As List(Of String)
            ' Padding logic for text data chunks
            Dim paddedChunks As New List(Of String)

            For Each chunk As String In chunks
                If chunk.Length > maxsize Then
                    ' Apply padding if chunk size exceeds maxSize
                    paddedChunks.Add(chunk.Substring(0, maxsize))
                Else
                    paddedChunks.Add(chunk)
                End If
            Next

            Return paddedChunks
        End Function

        Public Shared Function Chunk(data As String, chunkType As ChunkType, ByRef maxsize As Integer) As List(Of String)
            ' Chunking logic for text data based on chunkType
            Dim chunks As New List(Of String)

            Select Case chunkType
                Case ChunkType.Sentence
                    ' Split into sentences
                    chunks.AddRange(data.Split("."c))
                Case ChunkType.Paragraph
                    ' Split into paragraphs
                    chunks.AddRange(data.Split(Environment.NewLine))
                Case ChunkType.Document
                    ' Treat the whole data as a document
                    chunks.Add(data)
            End Select
            If maxsize > 0 Then
                ' Apply padding based on maxSize
                chunks = ApplyPadding(chunks, maxsize)
            End If

            Return chunks
        End Function

        Public Shared Sub OutputToCSV(data As List(Of String), outputPath As String)
            Using writer As New StreamWriter(outputPath)
                For Each chunk As String In data
                    writer.WriteLine(chunk)
                Next
            End Using
        End Sub

        Public Shared Sub OutputToJSON(data As List(Of String), outputPath As String)
            Dim jsonData As New List(Of Object)
            For Each chunk As String In data
                jsonData.Add(New With {.content = chunk})
            Next
            Dim jsonText As String = JsonConvert.SerializeObject(jsonData, Formatting.Indented)
            File.WriteAllText(outputPath, jsonText)
        End Sub

        Public Shared Sub OutputToListOfLists(data As List(Of String), outputPath As String)
            File.WriteAllLines(outputPath, data)
        End Sub

        Public Shared Sub OutputToStructured(entityChunks As List(Of KeyValuePair(Of String, String)), outputPath As String)
            Dim structuredData As New List(Of Object)
            For Each entityChunk As KeyValuePair(Of String, String) In entityChunks
                structuredData.Add(New With {
                .entityType = entityChunk.Key,
                .content = entityChunk.Value
            })
            Next
            Dim jsonText As String = JsonConvert.SerializeObject(structuredData, Formatting.Indented)
            File.WriteAllText(outputPath, jsonText)
        End Sub

        Public Shared Function ProcessFile(inputPath As String, outputDirectory As String, entityListfilePath As String, maxSize As Integer, useFiltering As Boolean, chunkType As ChunkType) As List(Of String)
            Dim rawData As String = File.ReadAllText(inputPath)
            Dim chunks As List(Of String) = Chunk(rawData, chunkType, maxSize)

            ' Load entity list if filtering is selected
            If useFiltering Then
                Dim filterList = EntityLoader.LoadEntityListFromFile(entityListfilePath)

                ' Detect and output structured entities
                Dim entityChunks As List(Of KeyValuePair(Of String, String)) = EntityLoader.DetectEntities(chunks, filterList)
                OutputToStructured(entityChunks, Path.Combine(outputDirectory, "entity_output.txt"))
            End If
            If maxSize > 0 Then
                ' Apply padding based on maxSize
                chunks = ApplyPadding(chunks, maxSize)
            Else
            End If

            ' Output to different formats
            OutputToListOfLists(chunks, Path.Combine(outputDirectory, "output.txt"))
            OutputToCSV(chunks, Path.Combine(outputDirectory, "output.csv"))
            OutputToJSON(chunks, Path.Combine(outputDirectory, "output.json"))

            ' Create punctuation vocabulary
            Return chunks
        End Function

        Public Function ApplyFiltering(chunks As List(Of String), filterList As List(Of KeyValuePair(Of String, String))) As List(Of String)
            Dim filteredChunks As New List(Of String)

            For Each chunk As String In chunks
                For Each filterItem As KeyValuePair(Of String, String) In filterList
                    If chunk.Contains(filterItem.Value) Then
                        filteredChunks.Add(chunk)
                        Exit For
                    End If
                Next
            Next

            Return filteredChunks
        End Function

        Public Function ApplyPadding(chunks As List(Of String)) As List(Of String)
            ' Padding logic for text data chunks
            Dim paddedChunks As New List(Of String)

            For Each chunk As String In chunks
                If chunk.Length > maxSize Then
                    ' Apply padding if chunk size exceeds maxSize
                    paddedChunks.Add(chunk.Substring(0, maxSize))
                Else
                    paddedChunks.Add(chunk)
                End If
            Next

            Return paddedChunks
        End Function

        Public Function Chunk(data As String, chunkType As ChunkType) As List(Of String)
            ' Chunking logic for text data based on chunkType
            Dim chunks As New List(Of String)

            Select Case chunkType
                Case ChunkType.Sentence
                    ' Split into sentences
                    chunks.AddRange(data.Split("."c))
                Case ChunkType.Paragraph
                    ' Split into paragraphs
                    chunks.AddRange(data.Split(Environment.NewLine))
                Case ChunkType.Document
                    ' Treat the whole data as a document
                    chunks.Add(data)
            End Select
            If maxSize > 0 Then
                ' Apply padding based on maxSize
                chunks = ApplyPadding(chunks)
            End If

            Return chunks
        End Function

        Public Function CustomizeChunkingAndPadding(data As String) As List(Of String)
            Dim chunks As List(Of String) = Chunk(data, chunkType)

            If maxSize > 0 Then
                chunks = ApplyPadding(chunks)
            End If

            Return chunks
        End Function

        ''' <summary>
        ''' Filters out chunks containing specific punctuation marks or symbols.
        ''' </summary>
        ''' <param name="data">The list of processed text data chunks.</param>
        ''' <returns>A list of filtered text data chunks.</returns>
        Public Function FilterUsingPunctuationVocabulary(data As List(Of String), ByRef punctuationVocabulary As HashSet(Of String)) As List(Of String)
            Dim filteredData As New List(Of String)

            For Each chunk As String In data
                Dim symbols As String() = chunk.Split().Where(Function(token) Not Char.IsLetterOrDigit(token(0))).ToArray()

                Dim containsPunctuation As Boolean = False
                For Each symbol As String In symbols
                    If punctuationVocabulary.Contains(symbol) Then
                        containsPunctuation = True
                        Exit For
                    End If
                Next

                If Not containsPunctuation Then
                    filteredData.Add(chunk)
                End If
            Next

            Return filteredData
        End Function

        Public Sub ProcessAndFilterChunks(inputPath As String, outputPath As String, filterListPath As String, chunkType As ChunkType, maxSize As Integer)
            Dim rawData As String = File.ReadAllText(inputPath)
            Dim chunks As List(Of String) = Chunk(rawData, chunkType, maxSize)

            If Not String.IsNullOrEmpty(filterListPath) Then
                Dim filterList As List(Of KeyValuePair(Of String, String)) = EntityLoader.LoadEntityListFromFile(filterListPath)
                chunks = ApplyFiltering(chunks, filterList)
            End If

            ' Apply padding if maxSize is specified
            If maxSize > 0 Then
                chunks = ApplyPadding(chunks, maxSize)
            End If

            ' Output to different formats
            OutputToListOfLists(chunks, Path.Combine(outputPath, "output.txt"))
            OutputToCSV(chunks, Path.Combine(outputPath, "output.csv"))
            OutputToJSON(chunks, Path.Combine(outputPath, "output.json"))
        End Sub

        Public Function ProcessFile(inputPath As String, outputDirectory As String)
            Dim rawData As String = File.ReadAllText(inputPath)
            Dim chunks As List(Of String) = Chunk(rawData, chunkType)

            ' Output to different formats
            OutputToListOfLists(chunks, Path.Combine(outputDirectory, "output.txt"))
            OutputToCSV(chunks, Path.Combine(outputDirectory, "output.csv"))
            OutputToJSON(chunks, Path.Combine(outputDirectory, "output.json"))
            Return chunks
        End Function

    End Class




End Namespace
Public Module Extensions

    ''' <summary>
    ''' Outputs Structure to Jason(JavaScriptSerializer)
    ''' </summary>
    ''' <returns></returns>
    <Runtime.CompilerServices.Extension()>
    Public Function ToJson(ByRef iObject As Object) As String
        Dim Converter As New JavaScriptSerializer
        Return Converter.Serialize(iObject)
    End Function

    Function CalculateWordOverlap(tokens1 As String(), tokens2 As String()) As Integer
        Dim overlap As Integer = 0

        ' Compare each token in sentence 1 with tokens in sentence 2
        For Each token1 As String In tokens1
            For Each token2 As String In tokens2
                ' If the tokens match, increment the overlap count
                If token1.ToLower() = token2.ToLower() Then
                    overlap += 1
                    Exit For ' No need to check further tokens in sentence 2
                End If
            Next
        Next

        Return overlap
    End Function

    Function DetermineEntailment(overlap As Integer) As Boolean
        ' Set a threshold for entailment
        Dim threshold As Integer = 2

        ' Determine entailment based on overlap
        Return overlap >= threshold
    End Function

    <Runtime.CompilerServices.Extension()>
    Public Function CalculateCosineSimilarity(sentence1 As String, sentence2 As String) As Double
        ' Calculate the cosine similarity between two sentences
        Dim words1 As String() = sentence1.Split(" "c)
        Dim words2 As String() = sentence2.Split(" "c)

        Dim intersection As Integer = words1.Intersect(words2).Count()
        Dim similarity As Double = intersection / Math.Sqrt(words1.Length * words2.Length)
        Return similarity
    End Function

    Public Structure Entity
        Public Property EndIndex As Integer
        Public Property StartIndex As Integer
        Public Property Type As String
        Public Property Value As String

        Public Shared Function DetectEntitys(ByRef text As String, EntityList As List(Of Entity)) As List(Of Entity)
            Dim detectedEntitys As New List(Of Entity)()

            ' Perform entity detection logic here
            For Each item In EntityList
                If text.Contains(item.Value) Then
                    detectedEntitys.Add(item)
                End If
            Next

            Return detectedEntitys
        End Function

    End Structure

    Public Enum ChunkType
        Sentence
        Paragraph
        Document
    End Enum


    Public Class PunctuationMarkers
        Public Shared ReadOnly SeperatorPunctuation() As String = {" ", ",", "|"}
        Public Shared ReadOnly Symbols() As String = {"@", "#", "$", "%", "&", "*", "+", "=", "^", "_", "~", "§", "°", "¿", "¡"}
        Public Shared ReadOnly EncapuslationPunctuationEnd() As String = {"}", "]", ">", ")"}
        Public Shared ReadOnly EncapuslationPunctuationStart() As String = {"{", "[", "<", "("}
        Public Shared ReadOnly GramaticalPunctuation() As String = {".", "?", "!", ":", ";", ","}
        Public Shared ReadOnly MathPunctuation = New String() {"+", "-", "*", "/", "=", "<", ">", "≤", "≥", "±", "≈", "≠", "%", "‰", "‱", "^", "_", "√", "∛", "∜", "∫", "∬", "∭", "∮", "∯", "∰", "∇", "∂", "∆", "∏", "∑", "∐", "⨀", "⨁", "⨂", "⨃", "⨄", "∫", "∬", "∭", "∮", "∯", "∰", "∇", "∂", "∆", "∏", "∑", "∐", "⨀", "⨁", "⨂", "⨃", "⨄"}
        Public Shared ReadOnly MoneyPunctuation() As String = {"$", "€", "£", "¥", "₹", "₽", "₿"}
        Public Shared ReadOnly CodePunctuation() As String = {"\", "#", "@", "^"}

        Public Shared ReadOnly Delimiters() As Char = {CType(" ", Char), CType(".", Char),
                    CType(",", Char), CType("?", Char),
                    CType("!", Char), CType(";", Char),
                    CType(":", Char), Chr(10), Chr(13), vbTab}

        Public ReadOnly Property SentenceEndPunctuation As List(Of String)
            Get
                Dim markers() As String = {".", ";", ":", "!", "?"}
                Return markers.ToList
            End Get
        End Property

        Public Shared ReadOnly Property Punctuation As List(Of String)
            Get
                Dim x As New List(Of String)
                x.AddRange(SeperatorPunctuation)
                x.AddRange(Symbols)
                x.AddRange(EncapuslationPunctuationStart)
                x.AddRange(EncapuslationPunctuationEnd)
                x.AddRange(MoneyPunctuation)
                x.AddRange(MathPunctuation)
                x.AddRange(GramaticalPunctuation)
                x.AddRange(CodePunctuation)
                Return x.Distinct.ToList
            End Get
        End Property

    End Class
    Public Enum TokenType
        GramaticalPunctuation
        EncapuslationPunctuationStart
        EncapuslationPunctuationEnd
        MoneyPunctuation
        MathPunctuation
        CodePunctuation
        AlphaBet
        Number
        Symbol
        SeperatorPunctuation
        Ignore
        Word
        Sentence
        Character
        Ngram
        WordGram
        SentenceGram
        BitWord
        Punctuation
        whitespace
    End Enum
    Public Enum TokenizerType
        _Char
        _Word
        _Sentence
        _Paragraph
        _BPE
        _Wordpiece
        _Token
        _TokenID
    End Enum
    Public Class RemoveToken



        Private Shared ReadOnly AlphaBet() As String = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}
        Private Shared ReadOnly Number() As String = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
"30", "40", "50", "60", "70", "80", "90", "00", "000", "0000", "00000", "000000", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
"nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "Billion"}
        Private iStopWords As New List(Of String)

        Private Shared Function AddSuffix(ByRef Str As String, ByVal Suffix As String) As String
            Return Str & Suffix
        End Function
        Public Function GetValidTokens(ByRef InputStr As String) As String
            Dim EndStr As Integer = InputStr.Length
            Dim CharStr As String = ""
            For i = 0 To EndStr - 1
                If GetTokenType(InputStr(i)) <> TokenType.Ignore Then
                    CharStr = AddSuffix(CharStr, InputStr(i))
                Else

                End If
            Next
            Return CharStr
        End Function
        Public Function GetTokenType(ByRef CharStr As String) As TokenType
            For Each item In PunctuationMarkers.SeperatorPunctuation
                If CharStr = item Then Return TokenType.SeperatorPunctuation
            Next
            For Each item In PunctuationMarkers.GramaticalPunctuation
                If CharStr = item Then Return TokenType.GramaticalPunctuation
            Next
            For Each item In PunctuationMarkers.EncapuslationPunctuationStart
                If CharStr = item Then Return TokenType.EncapuslationPunctuationStart
            Next
            For Each item In PunctuationMarkers.EncapuslationPunctuationEnd
                If CharStr = item Then Return TokenType.EncapuslationPunctuationEnd
            Next
            For Each item In PunctuationMarkers.MoneyPunctuation
                If CharStr = item Then Return TokenType.MoneyPunctuation
            Next
            For Each item In PunctuationMarkers.MathPunctuation
                If CharStr = item Then Return TokenType.MathPunctuation
            Next
            For Each item In PunctuationMarkers.CodePunctuation
                If CharStr = item Then Return TokenType.CodePunctuation
            Next
            For Each item In AlphaBet
                If CharStr = item Then Return TokenType.AlphaBet
            Next
            For Each item In Number
                If CharStr = item Then Return TokenType.Number
            Next
            Return TokenType.Ignore
        End Function

        Public Function GetEncapsulated(ByRef Userinput As String) As List(Of String)
            GetEncapsulated = New List(Of String)
            Do Until ContainsEncapsulated(Userinput) = False
                GetEncapsulated.Add(ExtractEncapsulated(Userinput))
            Loop
        End Function
        Public Function ExtractEncapsulated(ByRef Userinput As String) As String
            ExtractEncapsulated = Userinput
            If ContainsEncapsulated(ExtractEncapsulated) = True Then
                If ExtractEncapsulated.Contains("(") = True And ExtractEncapsulated.Contains(")") = True Then
                    ExtractEncapsulated = ExtractEncapsulated.ExtractStringBetween("(", ")")
                End If
                If Userinput.Contains("[") = True And Userinput.Contains("]") = True Then
                    ExtractEncapsulated = ExtractEncapsulated.ExtractStringBetween("[", "]")
                End If
                If Userinput.Contains("{") = True And Userinput.Contains("}") = True Then
                    ExtractEncapsulated = ExtractEncapsulated.ExtractStringBetween("{", "}")
                End If
                If Userinput.Contains("<") = True And Userinput.Contains(">") = True Then
                    ExtractEncapsulated = ExtractEncapsulated.ExtractStringBetween("<", ">")
                End If
            End If
        End Function

        Public Function ContainsEncapsulated(ByRef Userinput As String) As Boolean
            Dim Start = False
            Dim Ending = False
            ContainsEncapsulated = False
            For Each item In PunctuationMarkers.EncapuslationPunctuationStart
                If Userinput.Contains(item) = True Then Start = True
            Next
            For Each item In PunctuationMarkers.EncapuslationPunctuationEnd
                If Userinput.Contains(item) = True Then Ending = True
            Next
            If Start And Ending = True Then
                ContainsEncapsulated = True
            End If
        End Function

        Public Shared Function RemoveBrackets(ByRef Txt As String) As String
            'Brackets
            Txt = Txt.Replace("(", "")
            Txt = Txt.Replace("{", "")
            Txt = Txt.Replace("}", "")
            Txt = Txt.Replace("[", "")
            Txt = Txt.Replace("]", "")
            Return Txt
        End Function

        Public Shared Function RemoveDoubleSpace(ByRef txt As String, Item As String) As String
            Return txt.Replace(Item, "  " & Item & " ")
        End Function

        Public Shared Function RemoveMathsSymbols(ByRef Txt As String) As String
            'Maths Symbols
            Txt = Txt.Replace("+", "")
            Txt = Txt.Replace("=", "")
            Txt = Txt.Replace("-", "")
            Txt = Txt.Replace("/", "")
            Txt = Txt.Replace("*", "")
            Txt = Txt.Replace("<", "")
            Txt = Txt.Replace(">", "")
            Txt = Txt.Replace("%", "")
            Return Txt
        End Function

        Public Shared Function RemovePunctuation(ByRef Txt As String) As String
            'Punctuation
            Txt = Txt.Replace(",", "")
            Txt = Txt.Replace(".", "")
            Txt = Txt.Replace(";", "")
            Txt = Txt.Replace("'", "")
            Txt = Txt.Replace("_", "")
            Txt = Txt.Replace("?", "")
            Txt = Txt.Replace("!", "")
            Txt = Txt.Replace("&", "")
            Txt = Txt.Replace(":", "")

            Return Txt
        End Function

        Public Shared Function RemoveStopWords(ByRef txt As String, ByRef StopWrds As List(Of String)) As String
            For Each item In StopWrds
                txt = txt.Replace(item, "")
            Next
            Return txt
        End Function

        Public Shared Function RemoveSymbols(ByRef Txt As String) As String
            'Basic Symbols
            Txt = Txt.Replace("£", "")
            Txt = Txt.Replace("$", "")
            Txt = Txt.Replace("^", "")
            Txt = Txt.Replace("@", "")
            Txt = Txt.Replace("#", "")
            Txt = Txt.Replace("~", "")
            Txt = Txt.Replace("\", "")
            Return Txt
        End Function

        Public Shared Function RemoveTokenType(ByRef UserStr As String, ByRef nType As TokenType) As String
            Dim AlphaBet() As String = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}
            Dim Number() As String = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
"30", "40", "50", "60", "70", "80", "90", "00", "000", "0000", "00000", "000000", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
"nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "Billion"}

            Select Case nType
                Case TokenType.GramaticalPunctuation
                    For Each item In PunctuationMarkers.GramaticalPunctuation
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next
                Case TokenType.AlphaBet
                    For Each item In AlphaBet
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next
                Case TokenType.CodePunctuation
                    For Each item In PunctuationMarkers.CodePunctuation
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next
                Case TokenType.EncapuslationPunctuationEnd
                    For Each item In PunctuationMarkers.EncapuslationPunctuationEnd
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next
                Case TokenType.EncapuslationPunctuationStart
                    For Each item In PunctuationMarkers.EncapuslationPunctuationStart
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next
                Case TokenType.Ignore
                Case TokenType.MathPunctuation
                    For Each item In PunctuationMarkers.MathPunctuation
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next
                Case TokenType.MoneyPunctuation
                    For Each item In PunctuationMarkers.MoneyPunctuation
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next
                Case TokenType.Number
                    For Each item In Number
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next
                Case TokenType.SeperatorPunctuation
                    For Each item In PunctuationMarkers.SeperatorPunctuation
                        If UCase(UserStr).Contains(UCase(item)) = True Then
                            UserStr = UCase(UserStr).Remove(UCase(item))
                        End If
                    Next

            End Select
            Return UserStr
        End Function

    End Class
    Public Class FrequentTerms

        Public Shared Function FindFrequentBigrams(words As List(Of String), Optional Param_Freq As Integer = 1) As List(Of String)
            Dim bigramCounts As New Dictionary(Of String, Integer)

            For i As Integer = 0 To words.Count - 2
                Dim bigram As String = words(i) & " " & words(i + 1)

                If bigramCounts.ContainsKey(bigram) Then
                    bigramCounts(bigram) += 1
                Else
                    bigramCounts.Add(bigram, 1)
                End If
            Next

            Dim frequentBigrams As New List(Of String)

            For Each pair In bigramCounts
                If pair.Value > Param_Freq Then ' Adjust the threshold as needed
                    frequentBigrams.Add(pair.Key)
                End If
            Next

            Return frequentBigrams
        End Function

        Public Shared Function FindFrequentCharacterBigrams(words As List(Of String), Optional Param_Freq As Integer = 1) As List(Of String)
            Dim bigramCounts As New Dictionary(Of String, Integer)

            For Each word In words
                Dim characters As Char() = word.ToCharArray()

                For i As Integer = 0 To characters.Length - 2
                    Dim bigram As String = characters(i) & characters(i + 1)

                    If bigramCounts.ContainsKey(bigram) Then
                        bigramCounts(bigram) += 1
                    Else
                        bigramCounts.Add(bigram, 1)
                    End If
                Next
            Next

            Dim frequentCharacterBigrams As New List(Of String)

            For Each pair In bigramCounts
                If pair.Value > Param_Freq Then ' Adjust the threshold as needed
                    frequentCharacterBigrams.Add(pair.Key)
                End If
            Next

            Return frequentCharacterBigrams
        End Function

        Public Shared Function FindFrequentCharacterTrigrams(words As List(Of String), Optional Param_Freq As Integer = 1) As List(Of String)
            Dim trigramCounts As New Dictionary(Of String, Integer)

            For Each word In words
                Dim characters As Char() = word.ToCharArray()

                For i As Integer = 0 To characters.Length - 3
                    Dim trigram As String = characters(i) & characters(i + 1) & characters(i + 2)

                    If trigramCounts.ContainsKey(trigram) Then
                        trigramCounts(trigram) += 1
                    Else
                        trigramCounts.Add(trigram, 1)
                    End If
                Next
            Next

            Dim frequentCharacterTrigrams As New List(Of String)

            For Each pair In trigramCounts
                If pair.Value > Param_Freq Then ' Adjust the threshold as needed
                    frequentCharacterTrigrams.Add(pair.Key)
                End If
            Next

            Return frequentCharacterTrigrams
        End Function

        Public Shared Function FindFrequentSentenceBigrams(sentences As List(Of String), Optional Param_Freq As Integer = 1) As List(Of String)
            Dim bigramCounts As New Dictionary(Of String, Integer)

            For i As Integer = 0 To sentences.Count - 2
                Dim bigram As String = sentences(i) & " " & sentences(i + 1)

                If bigramCounts.ContainsKey(bigram) Then
                    bigramCounts(bigram) += 1
                Else
                    bigramCounts.Add(bigram, 1)
                End If
            Next

            Dim frequentSentenceBigrams As New List(Of String)

            For Each pair In bigramCounts
                If pair.Value > Param_Freq Then ' Adjust the threshold as needed
                    frequentSentenceBigrams.Add(pair.Key)
                End If
            Next

            Return frequentSentenceBigrams
        End Function

        Public Shared Function FindFrequentSentenceTrigrams(sentences As List(Of String), Optional Param_Freq As Integer = 1) As List(Of String)
            Dim trigramCounts As New Dictionary(Of String, Integer)

            For i As Integer = 0 To sentences.Count - 3
                Dim trigram As String = sentences(i) & " " & sentences(i + 1) & " " & sentences(i + 2)

                If trigramCounts.ContainsKey(trigram) Then
                    trigramCounts(trigram) += 1
                Else
                    trigramCounts.Add(trigram, 1)
                End If
            Next

            Dim frequentSentenceTrigrams As New List(Of String)

            For Each pair In trigramCounts
                If pair.Value > Param_Freq Then ' Adjust the threshold as needed
                    frequentSentenceTrigrams.Add(pair.Key)
                End If
            Next

            Return frequentSentenceTrigrams
        End Function

        Public Shared Function FindFrequentTrigrams(words As List(Of String), Optional Param_Freq As Integer = 1) As List(Of String)
            Dim trigramCounts As New Dictionary(Of String, Integer)

            For i As Integer = 0 To words.Count - 3
                Dim trigram As String = words(i) & " " & words(i + 1) & " " & words(i + 2)

                If trigramCounts.ContainsKey(trigram) Then
                    trigramCounts(trigram) += 1
                Else
                    trigramCounts.Add(trigram, 1)
                End If
            Next

            Dim frequentTrigrams As New List(Of String)

            For Each pair In trigramCounts
                If pair.Value > Param_Freq Then ' Adjust the threshold as needed
                    frequentTrigrams.Add(pair.Key)
                End If
            Next

            Return frequentTrigrams
        End Function

        Public Shared Function FindFrequentWordBigrams(sentences As List(Of String), Optional Param_Freq As Integer = 1) As List(Of String)
            Dim bigramCounts As New Dictionary(Of String, Integer)

            For Each sentence In sentences
                Dim words As String() = sentence.Split(" "c)

                For i As Integer = 0 To words.Length - 2
                    Dim bigram As String = words(i) & " " & words(i + 1)

                    If bigramCounts.ContainsKey(bigram) Then
                        bigramCounts(bigram) += 1
                    Else
                        bigramCounts.Add(bigram, 1)
                    End If
                Next
            Next

            Dim frequentWordBigrams As New List(Of String)

            For Each pair In bigramCounts
                If pair.Value > Param_Freq Then ' Adjust the threshold as needed
                    frequentWordBigrams.Add(pair.Key)
                End If
            Next

            Return frequentWordBigrams
        End Function

        Public Shared Function FindFrequentWordTrigrams(sentences As List(Of String), Optional Param_Freq As Integer = 1) As List(Of String)
            Dim trigramCounts As New Dictionary(Of String, Integer)

            For Each sentence In sentences
                Dim words As String() = sentence.Split(" "c)

                For i As Integer = 0 To words.Length - 3
                    Dim trigram As String = words(i) & " " & words(i + 1) & " " & words(i + 2)

                    If trigramCounts.ContainsKey(trigram) Then
                        trigramCounts(trigram) += 1
                    Else
                        trigramCounts.Add(trigram, 1)
                    End If
                Next
            Next

            Dim frequentWordTrigrams As New List(Of String)

            For Each pair In trigramCounts
                If pair.Value > Param_Freq Then ' Adjust the threshold as needed
                    frequentWordTrigrams.Add(pair.Key)
                End If
            Next

            Return frequentWordTrigrams
        End Function


        Public Shared Function FindFrequentCharNgrams(Tokens As List(Of String), N As Integer, ByRef Freq_threshold As Integer) As List(Of String)
            Dim NgramCounts As New Dictionary(Of String, Integer)

            For Each word In Tokens
                Dim characters As List(Of String) = Tokenizer.TokenizeToCharacter(word)

                For Each ngram In GetTokenGramCounts(characters, N)
                    'Update Dictionary
                    If NgramCounts.ContainsKey(ngram.Key) Then

                        NgramCounts(ngram.Key) += ngram.Value
                    Else
                        NgramCounts.Add(ngram.Key, ngram.Value)
                    End If

                Next
            Next

            Return Tokenizer.GetHighFreq(NgramCounts, Freq_threshold)
        End Function
        Public Shared Function GetTokenGramCounts(Tokens As List(Of String), N As Integer) As Dictionary(Of String, Integer)
            Dim NgramCounts As New Dictionary(Of String, Integer)

            For Each word In Tokens

                For i As Integer = 0 To Tokens.Count - N
                    Dim Ngram As String = Tokens(i) & Tokens(i + 1)

                    If NgramCounts.ContainsKey(Ngram) Then
                        NgramCounts(Ngram) += 1
                    Else
                        NgramCounts.Add(Ngram, 1)
                    End If
                Next
            Next

            Return NgramCounts
        End Function
        Public Shared Function GetFrequentTokenNgrams(Tokens As List(Of String), N As Integer, ByRef Freq_threshold As Integer) As List(Of String)
            Dim NgramCounts As Dictionary(Of String, Integer) = GetTokenGramCounts(Tokens, N)

            Dim frequentWordNgrams As List(Of String) = Tokenizer.GetHighFreq(NgramCounts, Freq_threshold)

            Return frequentWordNgrams
        End Function


    End Class
    Public Function ReadTextFilesFromDirectory(directoryPath As String) As List(Of String)
        Dim fileList As New List(Of String)()

        Try
            Dim txtFiles As String() = Directory.GetFiles(directoryPath, "*.txt")

            For Each filePath As String In txtFiles
                Dim fileContent As String = File.ReadAllText(filePath)
                fileList.Add(fileContent)
            Next
        Catch ex As Exception
            ' Handle any exceptions that may occur while reading the files.
            Console.WriteLine("Error: " & ex.Message)
        End Try

        Return fileList
    End Function
    <Runtime.CompilerServices.Extension()>
    Public Function SpaceItems(ByRef txt As String, Item As String) As String
        Return txt.Replace(Item, " " & Item & " ")
    End Function
    <Runtime.CompilerServices.Extension()>
    Public Function SplitIntoSubwords(token As String, ByRef ngramLength As Integer) As List(Of String)
        Dim subwordUnits As List(Of String) = New List(Of String)

        For i As Integer = 0 To token.Length - ngramLength
            Dim subword As String = token.Substring(i, ngramLength)
            subwordUnits.Add(subword)
        Next

        Return subwordUnits
    End Function
    <Runtime.CompilerServices.Extension()>
    Public Function SpacePunctuation(ByRef Txt As String) As String
        For Each item In PunctuationMarkers.Punctuation
            Txt = SpaceItems(Txt, item)
        Next

        Return Txt
    End Function
    <Runtime.CompilerServices.Extension()>
    Public Function Tokenize(Document As String, ByRef TokenizerType As TokenizerType) As List(Of String)
        Tokenize = New List(Of String)
        Select Case TokenizerType
            Case TokenizerType._Sentence
                Return Tokenizer.TokenizeToSentence(Document)
            Case TokenizerType._Word
                Return Tokenizer.TokenizeToWord(Document)
            Case TokenizerType._Char
                Return Tokenizer.TokenizeToCharacter(Document)

        End Select

    End Function
    <Runtime.CompilerServices.Extension()>
    Public Function ExtractStringBetween(ByVal value As String, ByVal strStart As String, ByVal strEnd As String) As String
        If Not String.IsNullOrEmpty(value) Then
            Dim i As Integer = value.IndexOf(strStart)
            Dim j As Integer = value.IndexOf(strEnd)
            Return value.Substring(i, j - i)
        Else
            Return value
        End If
    End Function
    Public Class GetContext

        Public Shared Function GetContext(ByRef corpus As List(Of List(Of String)), ByRef WindowSize As Integer) As List(Of String)
            Dim contextWords As New List(Of String)
            For Each doc In corpus

                ' Iterate over each sentence in the corpus
                For Each sentence In doc
                    Dim Words() = Split(sentence, " ")
                    ' Iterate over each word in the sentence
                    For targetIndex = 0 To sentence.Length - 1
                        Dim targetWord As String = sentence(targetIndex)

                        ' Get the context words within the window
                        contextWords = GetContextWordsByIndex(Words.ToList, targetIndex, WindowSize)
                    Next
                Next

            Next
            Return contextWords
        End Function

        Private Shared Function GetContextWordsByIndex(ByVal sentence As List(Of String), ByVal targetIndex As Integer, ByRef Windowsize As Integer) As List(Of String)
            Dim contextWords As New List(Of String)

            For i = Math.Max(0, targetIndex - Windowsize) To Math.Min(sentence.Count - 1, targetIndex + Windowsize)
                If i <> targetIndex Then
                    contextWords.Add(sentence(i))
                End If
            Next

            Return contextWords
        End Function

    End Class
    <Runtime.CompilerServices.Extension()>
    Public Function ReplaceMergedPair(tokens As List(Of String), newUnit As String) As List(Of String)
        Dim mergedTokens As List(Of String) = New List(Of String)

        For Each token As String In tokens
            Dim replacedToken As String = token.Replace(newUnit, " " & newUnit & " ")
            mergedTokens.AddRange(replacedToken.Split(" ").ToList())
        Next

        Return mergedTokens
    End Function
    Public Class ReservedWords
        Public Shared Function IdentifyReservedWords(ByRef Input As String) As String
            Dim reservedWords As List(Of String) = GetReservedWords()

            For Each word In reservedWords
                Input = Input.Replace(word, UCase(word))
            Next

            Return Input
        End Function
        Private Shared Function GetReservedWords() As List(Of String)
            Dim reservedWords As New List(Of String)()

            ' Add VB.NET reserved words to the list
            reservedWords.Add("AddHandler")
            reservedWords.Add("AddressOf")
            reservedWords.Add("Alias")
            reservedWords.Add("And")
            reservedWords.Add("AndAlso")
            reservedWords.Add("As")
            reservedWords.Add("Boolean")
            reservedWords.Add("ByRef")
            reservedWords.Add("Byte")
            reservedWords.Add("ByVal")
            reservedWords.Add("Call")
            reservedWords.Add("Case")
            reservedWords.Add("Catch")
            reservedWords.Add("CBool")
            reservedWords.Add("CByte")
            reservedWords.Add("CChar")
            reservedWords.Add("CDate")
            reservedWords.Add("CDbl")
            reservedWords.Add("CDec")
            reservedWords.Add("Char")
            reservedWords.Add("CInt")
            reservedWords.Add("Class")
            reservedWords.Add("CLng")
            reservedWords.Add("CObj")
            reservedWords.Add("Continue")
            reservedWords.Add("CSByte")
            reservedWords.Add("CShort")
            reservedWords.Add("CSng")
            reservedWords.Add("CStr")
            reservedWords.Add("CType")
            reservedWords.Add("CUInt")
            reservedWords.Add("CULng")
            reservedWords.Add("CUShort")
            reservedWords.Add("Date")
            reservedWords.Add("Decimal")
            reservedWords.Add("Declare")
            reservedWords.Add("Default")
            reservedWords.Add("Delegate")
            reservedWords.Add("Dim")
            reservedWords.Add("DirectCast")
            reservedWords.Add("Do")
            reservedWords.Add("Double")
            reservedWords.Add("Each")
            reservedWords.Add("Else")
            reservedWords.Add("ElseIf")
            reservedWords.Add("End")
            reservedWords.Add("EndIf")
            reservedWords.Add("Enum")
            reservedWords.Add("Erase")
            reservedWords.Add("Error")
            reservedWords.Add("Event")
            reservedWords.Add("Exit")
            reservedWords.Add("False")
            reservedWords.Add("Finally")
            reservedWords.Add("For")
            reservedWords.Add("Friend")
            reservedWords.Add("Function")
            reservedWords.Add("Get")
            reservedWords.Add("GetType")
            reservedWords.Add("GetXMLNamespace")
            reservedWords.Add("Global")
            reservedWords.Add("GoSub")
            reservedWords.Add("GoTo")
            reservedWords.Add("Handles")
            reservedWords.Add("If")
            reservedWords.Add("Implements")
            reservedWords.Add("Imports")
            reservedWords.Add("In")
            reservedWords.Add("Inherits")
            reservedWords.Add("Integer")
            reservedWords.Add("Interface")
            reservedWords.Add("Is")
            reservedWords.Add("IsNot")
            reservedWords.Add("Let")
            reservedWords.Add("Lib")
            reservedWords.Add("Like")
            reservedWords.Add("Long")
            reservedWords.Add("Loop")
            reservedWords.Add("Me")
            reservedWords.Add("Mod")
            reservedWords.Add("Module")
            reservedWords.Add("MustInherit")
            reservedWords.Add("MustOverride")
            reservedWords.Add("MyBase")
            reservedWords.Add("MyClass")
            reservedWords.Add("Namespace")
            reservedWords.Add("Narrowing")
            reservedWords.Add("New")
            reservedWords.Add("Next")
            reservedWords.Add("Not")
            reservedWords.Add("Nothing")
            reservedWords.Add("NotInheritable")
            reservedWords.Add("NotOverridable")
            reservedWords.Add("Object")
            reservedWords.Add("Of")
            reservedWords.Add("On")
            reservedWords.Add("Operator")
            reservedWords.Add("Option")
            reservedWords.Add("Optional")
            reservedWords.Add("Or")
            reservedWords.Add("OrElse")
            reservedWords.Add("Overloads")
            reservedWords.Add("Overridable")
            reservedWords.Add("Overrides")
            reservedWords.Add("ParamArray")
            reservedWords.Add("Partial")
            reservedWords.Add("Private")
            reservedWords.Add("Property")
            reservedWords.Add("Protected")
            reservedWords.Add("Public")
            reservedWords.Add("RaiseEvent")
            reservedWords.Add("ReadOnly")
            reservedWords.Add("ReDim")
            reservedWords.Add("RemoveHandler")
            reservedWords.Add("Resume")
            reservedWords.Add("Return")
            reservedWords.Add("SByte")
            reservedWords.Add("Select")
            reservedWords.Add("Set")
            reservedWords.Add("Shadows")
            reservedWords.Add("Shared")
            reservedWords.Add("Short")
            reservedWords.Add("Single")
            reservedWords.Add("Static")
            reservedWords.Add("Step")
            reservedWords.Add("Stop")
            reservedWords.Add("String")
            reservedWords.Add("Structure")
            reservedWords.Add("Sub")
            reservedWords.Add("SyncLock")
            reservedWords.Add("Then")
            reservedWords.Add("Throw")
            reservedWords.Add("To")
            reservedWords.Add("True")
            reservedWords.Add("Try")
            reservedWords.Add("TryCast")
            reservedWords.Add("TypeOf")
            reservedWords.Add("UInteger")
            reservedWords.Add("ULong")
            reservedWords.Add("UShort")
            reservedWords.Add("Using")
            reservedWords.Add("Variant")
            reservedWords.Add("Wend")
            reservedWords.Add("When")
            reservedWords.Add("While")
            reservedWords.Add("Widening")
            reservedWords.Add("With")
            reservedWords.Add("WithEvents")
            reservedWords.Add("WriteOnly")
            reservedWords.Add("Xor")

            Return reservedWords
        End Function

        ''' <summary>
        ''' Checks if string is a reserved VBscipt Keyword
        ''' </summary>
        ''' <param name="keyword"></param>
        ''' <returns></returns>
        Public Shared Function IsReservedWord(ByVal keyword As String) As Boolean
            Dim IsReserved = False
            Select Case LCase(keyword)
                Case "and" : IsReserved = True
                Case "as" : IsReserved = True
                Case "boolean" : IsReserved = True
                Case "byref" : IsReserved = True
                Case "byte" : IsReserved = True
                Case "byval" : IsReserved = True
                Case "call" : IsReserved = True
                Case "case" : IsReserved = True
                Case "class" : IsReserved = True
                Case "const" : IsReserved = True
                Case "currency" : IsReserved = True
                Case "debug" : IsReserved = True
                Case "dim" : IsReserved = True
                Case "do" : IsReserved = True
                Case "double" : IsReserved = True
                Case "each" : IsReserved = True
                Case "else" : IsReserved = True
                Case "elseif" : IsReserved = True
                Case "empty" : IsReserved = True
                Case "end" : IsReserved = True
                Case "endif" : IsReserved = True
                Case "enum" : IsReserved = True
                Case "eqv" : IsReserved = True
                Case "event" : IsReserved = True
                Case "exit" : IsReserved = True
                Case "false" : IsReserved = True
                Case "for" : IsReserved = True
                Case "function" : IsReserved = True
                Case "get" : IsReserved = True
                Case "goto" : IsReserved = True
                Case "if" : IsReserved = True
                Case "imp" : IsReserved = True
                Case "implements" : IsReserved = True
                Case "in" : IsReserved = True
                Case "integer" : IsReserved = True
                Case "is" : IsReserved = True
                Case "let" : IsReserved = True
                Case "like" : IsReserved = True
                Case "long" : IsReserved = True
                Case "loop" : IsReserved = True
                Case "lset" : IsReserved = True
                Case "me" : IsReserved = True
                Case "mod" : IsReserved = True
                Case "new" : IsReserved = True
                Case "next" : IsReserved = True
                Case "not" : IsReserved = True
                Case "nothing" : IsReserved = True
                Case "null" : IsReserved = True
                Case "on" : IsReserved = True
                Case "option" : IsReserved = True
                Case "optional" : IsReserved = True
                Case "or" : IsReserved = True
                Case "paramarray" : IsReserved = True
                Case "preserve" : IsReserved = True
                Case "private" : IsReserved = True
                Case "public" : IsReserved = True
                Case "raiseevent" : IsReserved = True
                Case "redim" : IsReserved = True
                Case "rem" : IsReserved = True
                Case "resume" : IsReserved = True
                Case "rset" : IsReserved = True
                Case "select" : IsReserved = True
                Case "set" : IsReserved = True
                Case "shared" : IsReserved = True
                Case "single" : IsReserved = True
                Case "static" : IsReserved = True
                Case "stop" : IsReserved = True
                Case "sub" : IsReserved = True
                Case "then" : IsReserved = True
                Case "to" : IsReserved = True
                Case "true" : IsReserved = True
                Case "type" : IsReserved = True
                Case "typeof" : IsReserved = True
                Case "until" : IsReserved = True
                Case "variant" : IsReserved = True
                Case "wend" : IsReserved = True
                Case "while" : IsReserved = True
                Case "with" : IsReserved = True
                Case "xor" : IsReserved = True
            End Select
            Return IsReserved
        End Function


    End Class


    ''' <summary>
    ''' Generate a random number based on the upper and lower bounds of the array,
    ''' then use that to return the item.
    ''' </summary>
    ''' <typeparam name="t"></typeparam>
    ''' <param name="theArray"></param>
    ''' <returns></returns>
    <Runtime.CompilerServices.Extension()>
    Public Function FetchRandomItem(Of t)(ByRef theArray() As t) As t

        Dim randNumberGenerator As New Random
        Randomize()
        Dim index As Integer = randNumberGenerator.Next(theArray.GetLowerBound(0),
                                                    theArray.GetUpperBound(0) + 1)

        Return theArray(index)

    End Function

    Public ReadOnly AlphaBet() As String = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
                    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
                    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "}

    Public ReadOnly EncapuslationPunctuationEnd() As String = {"}", "]", ">", ")"}
    Public ReadOnly CodePunctuation() As String = {"\", "#", "@", "^"}
    Public ReadOnly EncapuslationPunctuationStart() As String = {"{", "[", "<", "("}

    Public ReadOnly GramaticalPunctuation() As String = {"?", "!", ":", ";", ",", "_", "&"}

    Public ReadOnly MathPunctuation() As String = {"+", "-", "=", "/", "*", "%", "PLUS", "ADD", "MINUS", "SUBTRACT", "DIVIDE", "DIFFERENCE", "TIMES", "MULTIPLY", "PERCENT", "EQUALS"}

    Public ReadOnly MoneyPunctuation() As String = {"£", "$"}

    Public ReadOnly Number() As String = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "30", "40", "50", "60", "70", "80", "90", "00", "000", "0000", "00000", "000000", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "Billion"}

    Public ReadOnly Numerical() As String = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}

    Public ReadOnly Symbols() As String = {"£", "$", "^", "@", "#", "~", "\"}

    Public StopWords As New List(Of String)

    Public ReadOnly StopWordsArab() As String = {"،", "آض", "آمينَ", "آه",
                    "آهاً", "آي", "أ", "أب", "أجل", "أجمع", "أخ", "أخذ", "أصبح", "أضحى", "أقبل",
                    "أقل", "أكثر", "ألا", "أم", "أما", "أمامك", "أمامكَ", "أمسى", "أمّا", "أن", "أنا", "أنت",
                    "أنتم", "أنتما", "أنتن", "أنتِ", "أنشأ", "أنّى", "أو", "أوشك", "أولئك", "أولئكم", "أولاء",
                    "أولالك", "أوّهْ", "أي", "أيا", "أين", "أينما", "أيّ", "أَنَّ", "أََيُّ", "أُفٍّ", "إذ", "إذا", "إذاً",
                    "إذما", "إذن", "إلى", "إليكم", "إليكما", "إليكنّ", "إليكَ", "إلَيْكَ", "إلّا", "إمّا", "إن", "إنّما",
                    "إي", "إياك", "إياكم", "إياكما", "إياكن", "إيانا", "إياه", "إياها", "إياهم", "إياهما", "إياهن",
                    "إياي", "إيهٍ", "إِنَّ", "ا", "ابتدأ", "اثر", "اجل", "احد", "اخرى", "اخلولق", "اذا", "اربعة", "ارتدّ",
                    "استحال", "اطار", "اعادة", "اعلنت", "اف", "اكثر", "اكد", "الألاء", "الألى", "الا", "الاخيرة", "الان", "الاول",
                    "الاولى", "التى", "التي", "الثاني", "الثانية", "الذاتي", "الذى", "الذي", "الذين", "السابق", "الف", "اللائي",
                    "اللاتي", "اللتان", "اللتيا", "اللتين", "اللذان", "اللذين", "اللواتي", "الماضي", "المقبل", "الوقت", "الى",
                    "اليوم", "اما", "امام", "امس", "ان", "انبرى", "انقلب", "انه", "انها", "او", "اول", "اي", "ايار", "ايام",
                    "ايضا", "ب", "بات", "باسم", "بان", "بخٍ", "برس", "بسبب", "بسّ", "بشكل", "بضع", "بطآن", "بعد", "بعض", "بك",
                    "بكم", "بكما", "بكن", "بل", "بلى", "بما", "بماذا", "بمن", "بن", "بنا", "به", "بها", "بي", "بيد", "بين",
                    "بَسْ", "بَلْهَ", "بِئْسَ", "تانِ", "تانِك", "تبدّل", "تجاه", "تحوّل", "تلقاء", "تلك", "تلكم", "تلكما", "تم", "تينك",
                    "تَيْنِ", "تِه", "تِي", "ثلاثة", "ثم", "ثمّ", "ثمّة", "ثُمَّ", "جعل", "جلل", "جميع", "جير", "حار", "حاشا", "حاليا",
                    "حاي", "حتى", "حرى", "حسب", "حم", "حوالى", "حول", "حيث", "حيثما", "حين", "حيَّ", "حَبَّذَا", "حَتَّى", "حَذارِ", "خلا",
                    "خلال", "دون", "دونك", "ذا", "ذات", "ذاك", "ذانك", "ذانِ", "ذلك", "ذلكم", "ذلكما", "ذلكن", "ذو", "ذوا", "ذواتا", "ذواتي", "ذيت", "ذينك", "ذَيْنِ", "ذِه", "ذِي", "راح", "رجع", "رويدك", "ريث", "رُبَّ", "زيارة", "سبحان", "سرعان", "سنة", "سنوات", "سوف", "سوى", "سَاءَ", "سَاءَمَا", "شبه", "شخصا", "شرع", "شَتَّانَ", "صار", "صباح", "صفر", "صهٍ", "صهْ", "ضد", "ضمن", "طاق", "طالما", "طفق", "طَق", "ظلّ", "عاد", "عام", "عاما", "عامة", "عدا", "عدة", "عدد", "عدم", "عسى", "عشر", "عشرة", "علق", "على", "عليك", "عليه", "عليها", "علًّ", "عن", "عند", "عندما", "عوض", "عين", "عَدَسْ", "عَمَّا", "غدا", "غير", "ـ", "ف", "فان", "فلان", "فو", "فى", "في", "فيم", "فيما", "فيه", "فيها", "قال", "قام", "قبل", "قد", "قطّ", "قلما", "قوة", "كأنّما", "كأين", "كأيّ", "كأيّن", "كاد", "كان", "كانت", "كذا", "كذلك", "كرب", "كل", "كلا", "كلاهما", "كلتا", "كلم", "كليكما", "كليهما", "كلّما", "كلَّا", "كم", "كما", "كي", "كيت", "كيف", "كيفما", "كَأَنَّ", "كِخ", "لئن", "لا", "لات", "لاسيما", "لدن", "لدى", "لعمر", "لقاء", "لك", "لكم", "لكما", "لكن", "لكنَّما", "لكي", "لكيلا", "للامم", "لم", "لما", "لمّا", "لن", "لنا", "له", "لها", "لو", "لوكالة", "لولا", "لوما", "لي", "لَسْتَ", "لَسْتُ", "لَسْتُم", "لَسْتُمَا", "لَسْتُنَّ", "لَسْتِ", "لَسْنَ", "لَعَلَّ", "لَكِنَّ", "لَيْتَ", "لَيْسَ", "لَيْسَا", "لَيْسَتَا", "لَيْسَتْ", "لَيْسُوا", "لَِسْنَا", "ما", "ماانفك", "مابرح", "مادام", "ماذا", "مازال", "مافتئ", "مايو", "متى", "مثل", "مذ", "مساء", "مع", "معاذ", "مقابل", "مكانكم", "مكانكما", "مكانكنّ", "مكانَك", "مليار", "مليون", "مما", "ممن", "من", "منذ", "منها", "مه", "مهما", "مَنْ", "مِن", "نحن", "نحو", "نعم", "نفس", "نفسه", "نهاية", "نَخْ", "نِعِمّا", "نِعْمَ", "ها", "هاؤم", "هاكَ", "هاهنا", "هبّ", "هذا", "هذه", "هكذا", "هل", "هلمَّ", "هلّا", "هم", "هما", "هن", "هنا", "هناك", "هنالك", "هو", "هي", "هيا", "هيت", "هيّا", "هَؤلاء", "هَاتانِ", "هَاتَيْنِ", "هَاتِه", "هَاتِي", "هَجْ", "هَذا", "هَذانِ", "هَذَيْنِ", "هَذِه", "هَذِي", "هَيْهَاتَ", "و", "و6", "وا", "واحد", "واضاف", "واضافت", "واكد", "وان", "واهاً", "واوضح", "وراءَك", "وفي", "وقال", "وقالت", "وقد", "وقف", "وكان", "وكانت", "ولا", "ولم",
                    "ومن", "وهو", "وهي", "ويكأنّ", "وَيْ", "وُشْكَانََ", "يكون", "يمكن", "يوم", "ّأيّان"}

    Public ReadOnly StopWordsDutch() As String = {"aan", "achte", "achter", "af", "al", "alle", "alleen", "alles", "als", "ander", "anders", "beetje",
            "behalve", "beide", "beiden", "ben", "beneden", "bent", "bij", "bijna", "bijv", "blijkbaar", "blijken", "boven", "bv",
            "daar", "daardoor", "daarin", "daarna", "daarom", "daaruit", "dan", "dat", "de", "deden", "deed", "derde", "derhalve", "dertig",
            "deze", "dhr", "die", "dit", "doe", "doen", "doet", "door", "drie", "duizend", "echter", "een", "eens", "eerst", "eerste", "eigen",
            "eigenlijk", "elk", "elke", "en", "enige", "er", "erg", "ergens", "etc", "etcetera", "even", "geen", "genoeg", "geweest", "haar",
            "haarzelf", "had", "hadden", "heb", "hebben", "hebt", "hedden", "heeft", "heel", "hem", "hemzelf", "hen", "het", "hetzelfde",
            "hier", "hierin", "hierna", "hierom", "hij", "hijzelf", "hoe", "honderd", "hun", "ieder", "iedere", "iedereen", "iemand", "iets",
            "ik", "in", "inderdaad", "intussen", "is", "ja", "je", "jij", "jijzelf", "jou", "jouw", "jullie", "kan", "kon", "konden", "kun",
            "kunnen", "kunt", "laatst", "later", "lijken", "lijkt", "maak", "maakt", "maakte", "maakten", "maar", "mag", "maken", "me", "meer",
            "meest", "meestal", "men", "met", "mevr", "mij", "mijn", "minder", "miss", "misschien", "missen", "mits", "mocht", "mochten",
            "moest", "moesten", "moet", "moeten", "mogen", "mr", "mrs", "mw", "na", "naar", "nam", "namelijk", "nee", "neem", "negen",
            "nemen", "nergens", "niemand", "niet", "niets", "niks", "noch", "nochtans", "nog", "nooit", "nu", "nv", "of", "om", "omdat",
            "ondanks", "onder", "ondertussen", "ons", "onze", "onzeker", "ooit", "ook", "op", "over", "overal", "overige", "paar", "per",
            "recent", "redelijk", "samen", "sinds", "steeds", "te", "tegen", "tegenover", "thans", "tien", "tiende", "tijdens", "tja", "toch",
            "toe", "tot", "totdat", "tussen", "twee", "tweede", "u", "uit", "uw", "vaak", "van", "vanaf", "veel", "veertig", "verder",
            "verscheidene", "verschillende", "via", "vier", "vierde", "vijf", "vijfde", "vijftig", "volgend", "volgens", "voor", "voordat",
            "voorts", "waar", "waarom", "waarschijnlijk", "wanneer", "waren", "was", "wat", "we", "wederom", "weer", "weinig", "wel", "welk",
            "welke", "werd", "werden", "werder", "whatever", "wie", "wij", "wijzelf", "wil", "wilden", "willen", "word", "worden", "wordt", "zal",
            "ze", "zei", "zeker", "zelf", "zelfde", "zes", "zeven", "zich", "zij", "zijn", "zijzelf", "zo", "zoals", "zodat", "zou", "zouden",
            "zulk", "zullen"}

    Public ReadOnly StopWordsENG() As String = {"a", "as", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "aint",
            "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any",
            "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "arent", "around",
            "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming",
            "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief",
            "but", "by", "c", "cmon", "cs", "came", "can", "cant", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com",
            "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldnt",
            "course", "currently", "d", "definitely", "described", "despite", "did", "didnt", "different", "do", "does", "doesnt", "doing", "dont", "done", "down",
            "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever",
            "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed",
            "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives",
            "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadnt", "happens", "hardly", "has", "hasnt", "have", "havent", "having", "he",
            "hes", "hello", "help", "hence", "her", "here", "heres", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his",
            "hither", "hopefully", "how", "howbeit", "however", "i", "id", "ill", "im", "ive", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed",
            "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isnt", "it", "itd", "itll", "its", "its", "itself", "j",
            "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets",
            "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might",
            "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither",
            "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o",
            "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our",
            "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible",
            "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards",
            "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming",
            "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldnt", "since", "six", "so",
            "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying",
            "still", "sub", "such", "sup", "sure", "t", "ts", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "thats", "the",
            "their", "theirs", "them", "themselves", "then", "thence", "there", "theres", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon",
            "these", "they", "theyd", "theyll", "theyre", "theyve", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through",
            "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un",
            "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v",
            "value", "various", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "well", "were", "weve", "welcome", "well",
            "went", "were", "werent", "what", "whats", "whatever", "when", "whence", "whenever", "where", "wheres", "whereafter", "whereas", "whereby", "wherein",
            "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whos", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish",
            "with", "within", "without", "wont", "wonder", "would", "wouldnt", "x", "y", "yes", "yet", "you", "youd", "youll", "youre", "youve", "your", "yours",
            "yourself", "yourselves", "youll", "z", "zero"}

    Public ReadOnly StopWordsFrench() As String = {"a", "abord", "absolument", "afin", "ah", "ai", "aie", "ailleurs", "ainsi", "ait", "allaient", "allo", "allons",
            "allô", "alors", "anterieur", "anterieure", "anterieures", "apres", "après", "as", "assez", "attendu", "au", "aucun", "aucune",
            "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "auraient", "aurait", "auront", "aussi", "autre", "autrefois", "autrement",
            "autres", "autrui", "aux", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avoir", "avons", "ayant", "b",
            "bah", "bas", "basee", "bat", "beau", "beaucoup", "bien", "bigre", "boum", "bravo", "brrr", "c", "car", "ce", "ceci", "cela", "celle",
            "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "cent", "cependant", "certain",
            "certaine", "certaines", "certains", "certes", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque",
            "cher", "chers", "chez", "chiche", "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième",
            "clac", "clic", "combien", "comme", "comment", "comparable", "comparables", "compris", "concernant", "contre", "couic", "crac", "d",
            "da", "dans", "de", "debout", "dedans", "dehors", "deja", "delà", "depuis", "dernier", "derniere", "derriere", "derrière", "des",
            "desormais", "desquelles", "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement", "devant", "devers", "devra",
            "different", "differentes", "differents", "différent", "différente", "différentes", "différents", "dire", "directe", "directement",
            "dit", "dite", "dits", "divers", "diverse", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit", "doivent", "donc",
            "dont", "douze", "douzième", "dring", "du", "duquel", "durant", "dès", "désormais", "e", "effet", "egale", "egalement", "egales", "eh",
            "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin", "entre", "envers", "environ", "es", "est", "et", "etant", "etc",
            "etre", "eu", "euh", "eux", "eux-mêmes", "exactement", "excepté", "extenso", "exterieur", "f", "fais", "faisaient", "faisant", "fait",
            "façon", "feront", "fi", "flac", "floc", "font", "g", "gens", "h", "ha", "hein", "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors",
            "hou", "houp", "hue", "hui", "huit", "huitième", "hum", "hurrah", "hé", "hélas", "i", "il", "ils", "importe", "j", "je", "jusqu", "jusque",
            "juste", "k", "l", "la", "laisser", "laquelle", "las", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "longtemps",
            "lors", "lorsque", "lui", "lui-meme", "lui-même", "là", "lès", "m", "ma", "maint", "maintenant", "mais", "malgre", "malgré", "maximale",
            "me", "meme", "memes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille", "mince", "minimale", "moi", "moi-meme", "moi-même",
            "moindres", "moins", "mon", "moyennant", "multiple", "multiples", "même", "mêmes", "n", "na", "naturel", "naturelle", "naturelles", "ne",
            "neanmoins", "necessaire", "necessairement", "neuf", "neuvième", "ni", "nombreuses", "nombreux", "non", "nos", "notamment", "notre",
            "nous", "nous-mêmes", "nouveau", "nul", "néanmoins", "nôtre", "nôtres", "o", "oh", "ohé", "ollé", "olé", "on", "ont", "onze", "onzième",
            "ore", "ou", "ouf", "ouias", "oust", "ouste", "outre", "ouvert", "ouverte", "ouverts", "o|", "où", "p", "paf", "pan", "par", "parce",
            "parfois", "parle", "parlent", "parler", "parmi", "parseme", "partant", "particulier", "particulière", "particulièrement", "pas",
            "passé", "pendant", "pense", "permet", "personne", "peu", "peut", "peuvent", "peux", "pff", "pfft", "pfut", "pif", "pire", "plein",
            "plouf", "plus", "plusieurs", "plutôt", "possessif", "possessifs", "possible", "possibles", "pouah", "pour", "pourquoi", "pourrais",
            "pourrait", "pouvait", "prealable", "precisement", "premier", "première", "premièrement", "pres", "probable", "probante",
            "procedant", "proche", "près", "psitt", "pu", "puis", "puisque", "pur", "pure", "q", "qu", "quand", "quant", "quant-à-soi", "quanta",
            "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième", "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles",
            "quelqu'un", "quelque", "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare", "rarement", "rares",
            "relative", "relativement", "remarquable", "rend", "rendre", "restant", "reste", "restent", "restrictif", "retour", "revoici",
            "revoilà", "rien", "s", "sa", "sacrebleu", "sait", "sans", "sapristi", "sauf", "se", "sein", "seize", "selon", "semblable", "semblaient",
            "semble", "semblent", "sent", "sept", "septième", "sera", "seraient", "serait", "seront", "ses", "seul", "seule", "seulement", "si",
            "sien", "sienne", "siennes", "siens", "sinon", "six", "sixième", "soi", "soi-même", "soit", "soixante", "son", "sont", "sous", "souvent",
            "specifique", "specifiques", "speculatif", "stop", "strictement", "subtiles", "suffisant", "suffisante", "suffit", "suis", "suit",
            "suivant", "suivante", "suivantes", "suivants", "suivre", "superpose", "sur", "surtout", "t", "ta", "tac", "tant", "tardive", "te",
            "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente", "tes", "tic", "tien", "tienne", "tiennes", "tiens",
            "toc", "toi", "toi-même", "ton", "touchant", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "tres",
            "trois", "troisième", "troisièmement", "trop", "très", "tsoin", "tsouin", "tu", "té", "u", "un", "une", "unes", "uniformement", "unique",
            "uniques", "uns", "v", "va", "vais", "vas", "vers", "via", "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voilà",
            "vont", "vos", "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres", "w", "x", "y", "z", "zut", "à", "â", "ça", "ès", "étaient",
            "étais", "était", "étant", "été", "être", "ô"}

    Public ReadOnly StopWordsItalian() As String = {"IE", "a", "abbastanza", "abbia", "abbiamo", "abbiano", "abbiate", "accidenti", "ad", "adesso", "affinche", "agl", "agli",
            "ahime", "ahimè", "ai", "al", "alcuna", "alcuni", "alcuno", "all", "alla", "alle", "allo", "allora", "altri", "altrimenti", "altro",
            "altrove", "altrui", "anche", "ancora", "anni", "anno", "ansa", "anticipo", "assai", "attesa", "attraverso", "avanti", "avemmo",
            "avendo", "avente", "aver", "avere", "averlo", "avesse", "avessero", "avessi", "avessimo", "aveste", "avesti", "avete", "aveva",
            "avevamo", "avevano", "avevate", "avevi", "avevo", "avrai", "avranno", "avrebbe", "avrebbero", "avrei", "avremmo", "avremo",
            "avreste", "avresti", "avrete", "avrà", "avrò", "avuta", "avute", "avuti", "avuto", "basta", "bene", "benissimo", "berlusconi",
            "brava", "bravo", "c", "casa", "caso", "cento", "certa", "certe", "certi", "certo", "che", "chi", "chicchessia", "chiunque", "ci",
            "ciascuna", "ciascuno", "cima", "cio", "cioe", "cioè", "circa", "citta", "città", "ciò", "co", "codesta", "codesti", "codesto",
            "cogli", "coi", "col", "colei", "coll", "coloro", "colui", "come", "cominci", "comunque", "con", "concernente", "conciliarsi",
            "conclusione", "consiglio", "contro", "cortesia", "cos", "cosa", "cosi", "così", "cui", "d", "da", "dagl", "dagli", "dai", "dal",
            "dall", "dalla", "dalle", "dallo", "dappertutto", "davanti", "degl", "degli", "dei", "del", "dell", "della", "delle", "dello",
            "dentro", "detto", "deve", "di", "dice", "dietro", "dire", "dirimpetto", "diventa", "diventare", "diventato", "dopo", "dov", "dove",
            "dovra", "dovrà", "dovunque", "due", "dunque", "durante", "e", "ebbe", "ebbero", "ebbi", "ecc", "ecco", "ed", "effettivamente", "egli",
            "ella", "entrambi", "eppure", "era", "erano", "eravamo", "eravate", "eri", "ero", "esempio", "esse", "essendo", "esser", "essere",
            "essi", "ex", "fa", "faccia", "facciamo", "facciano", "facciate", "faccio", "facemmo", "facendo", "facesse", "facessero", "facessi",
            "facessimo", "faceste", "facesti", "faceva", "facevamo", "facevano", "facevate", "facevi", "facevo", "fai", "fanno", "farai",
            "faranno", "fare", "farebbe", "farebbero", "farei", "faremmo", "faremo", "fareste", "faresti", "farete", "farà", "farò", "fatto",
            "favore", "fece", "fecero", "feci", "fin", "finalmente", "finche", "fine", "fino", "forse", "forza", "fosse", "fossero", "fossi",
            "fossimo", "foste", "fosti", "fra", "frattempo", "fu", "fui", "fummo", "fuori", "furono", "futuro", "generale", "gia", "giacche",
            "giorni", "giorno", "già", "gli", "gliela", "gliele", "glieli", "glielo", "gliene", "governo", "grande", "grazie", "gruppo", "ha",
            "haha", "hai", "hanno", "ho", "i", "ieri", "il", "improvviso", "in", "inc", "infatti", "inoltre", "insieme", "intanto", "intorno",
            "invece", "io", "l", "la", "lasciato", "lato", "lavoro", "le", "lei", "li", "lo", "lontano", "loro", "lui", "lungo", "luogo", "là",
            "ma", "macche", "magari", "maggior", "mai", "male", "malgrado", "malissimo", "mancanza", "marche", "me", "medesimo", "mediante",
            "meglio", "meno", "mentre", "mesi", "mezzo", "mi", "mia", "mie", "miei", "mila", "miliardi", "milioni", "minimi", "ministro",
            "mio", "modo", "molti", "moltissimo", "molto", "momento", "mondo", "mosto", "nazionale", "ne", "negl", "negli", "nei", "nel",
            "nell", "nella", "nelle", "nello", "nemmeno", "neppure", "nessun", "nessuna", "nessuno", "niente", "no", "noi", "non", "nondimeno",
            "nonostante", "nonsia", "nostra", "nostre", "nostri", "nostro", "novanta", "nove", "nulla", "nuovo", "o", "od", "oggi", "ogni",
            "ognuna", "ognuno", "oltre", "oppure", "ora", "ore", "osi", "ossia", "ottanta", "otto", "paese", "parecchi", "parecchie",
            "parecchio", "parte", "partendo", "peccato", "peggio", "per", "perche", "perchè", "perché", "percio", "perciò", "perfino", "pero",
            "persino", "persone", "però", "piedi", "pieno", "piglia", "piu", "piuttosto", "più", "po", "pochissimo", "poco", "poi", "poiche",
            "possa", "possedere", "posteriore", "posto", "potrebbe", "preferibilmente", "presa", "press", "prima", "primo", "principalmente",
            "probabilmente", "proprio", "puo", "pure", "purtroppo", "può", "qualche", "qualcosa", "qualcuna", "qualcuno", "quale", "quali",
            "qualunque", "quando", "quanta", "quante", "quanti", "quanto", "quantunque", "quasi", "quattro", "quel", "quella", "quelle",
            "quelli", "quello", "quest", "questa", "queste", "questi", "questo", "qui", "quindi", "realmente", "recente", "recentemente",
            "registrazione", "relativo", "riecco", "salvo", "sara", "sarai", "saranno", "sarebbe", "sarebbero", "sarei", "saremmo", "saremo",
            "sareste", "saresti", "sarete", "sarà", "sarò", "scola", "scopo", "scorso", "se", "secondo", "seguente", "seguito", "sei", "sembra",
            "sembrare", "sembrato", "sembri", "sempre", "senza", "sette", "si", "sia", "siamo", "siano", "siate", "siete", "sig", "solito",
            "solo", "soltanto", "sono", "sopra", "sotto", "spesso", "srl", "sta", "stai", "stando", "stanno", "starai", "staranno", "starebbe",
            "starebbero", "starei", "staremmo", "staremo", "stareste", "staresti", "starete", "starà", "starò", "stata", "state", "stati",
            "stato", "stava", "stavamo", "stavano", "stavate", "stavi", "stavo", "stemmo", "stessa", "stesse", "stessero", "stessi", "stessimo",
            "stesso", "steste", "stesti", "stette", "stettero", "stetti", "stia", "stiamo", "stiano", "stiate", "sto", "su", "sua", "subito",
            "successivamente", "successivo", "sue", "sugl", "sugli", "sui", "sul", "sull", "sulla", "sulle", "sullo", "suo", "suoi", "tale",
            "tali", "talvolta", "tanto", "te", "tempo", "ti", "titolo", "torino", "tra", "tranne", "tre", "trenta", "troppo", "trovato", "tu",
            "tua", "tue", "tuo", "tuoi", "tutta", "tuttavia", "tutte", "tutti", "tutto", "uguali", "ulteriore", "ultimo", "un", "una", "uno",
            "uomo", "va", "vale", "vari", "varia", "varie", "vario", "verso", "vi", "via", "vicino", "visto", "vita", "voi", "volta", "volte",
            "vostra", "vostre", "vostri", "vostro", "è"}

    Public ReadOnly StopWordsSpanish() As String = {"a", "actualmente", "acuerdo", "adelante", "ademas", "además", "adrede", "afirmó", "agregó", "ahi", "ahora",
            "ahí", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí", "alrededor", "ambos", "ampleamos",
            "antano", "antaño", "ante", "anterior", "antes", "apenas", "aproximadamente", "aquel", "aquella", "aquellas", "aquello",
            "aquellos", "aqui", "aquél", "aquélla", "aquéllas", "aquéllos", "aquí", "arriba", "arribaabajo", "aseguró", "asi", "así",
            "atras", "aun", "aunque", "ayer", "añadió", "aún", "b", "bajo", "bastante", "bien", "breve", "buen", "buena", "buenas", "bueno",
            "buenos", "c", "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco", "claro", "comentó", "como", "con",
            "conmigo", "conocer", "conseguimos", "conseguir", "considera", "consideró", "consigo", "consigue", "consiguen", "consigues",
            "contigo", "contra", "cosas", "creo", "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas", "cuanto", "cuantos", "cuatro",
            "cuenta", "cuál", "cuáles", "cuándo", "cuánta", "cuántas", "cuánto", "cuántos", "cómo", "d", "da", "dado", "dan", "dar", "de",
            "debajo", "debe", "deben", "debido", "decir", "dejó", "del", "delante", "demasiado", "demás", "dentro", "deprisa", "desde",
            "despacio", "despues", "después", "detras", "detrás", "dia", "dias", "dice", "dicen", "dicho", "dieron", "diferente", "diferentes",
            "dijeron", "dijo", "dio", "donde", "dos", "durante", "día", "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos",
            "embargo", "empleais", "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra", "enfrente", "enseguida", "entonces",
            "entre", "era", "eramos", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado",
            "estados", "estais", "estamos", "estan", "estar", "estará", "estas", "este", "esto", "estos", "estoy", "estuvo", "está", "están", "ex",
            "excepto", "existe", "existen", "explicó", "expresó", "f", "fin", "final", "fue", "fuera", "fueron", "fui", "fuimos", "g", "general",
            "gran", "grandes", "gueno", "h", "ha", "haber", "habia", "habla", "hablan", "habrá", "había", "habían", "hace", "haceis", "hacemos",
            "hacen", "hacer", "hacerlo", "haces", "hacia", "haciendo", "hago", "han", "hasta", "hay", "haya", "he", "hecho", "hemos", "hicieron",
            "hizo", "horas", "hoy", "hubo", "i", "igual", "incluso", "indicó", "informo", "informó", "intenta", "intentais", "intentamos", "intentan",
            "intentar", "intentas", "intento", "ir", "j", "junto", "k", "l", "la", "lado", "largo", "las", "le", "lejos", "les", "llegó", "lleva",
            "llevar", "lo", "los", "luego", "lugar", "m", "mal", "manera", "manifestó", "mas", "mayor", "me", "mediante", "medio", "mejor", "mencionó",
            "menos", "menudo", "mi", "mia", "mias", "mientras", "mio", "mios", "mis", "misma", "mismas", "mismo", "mismos", "modo", "momento", "mucha",
            "muchas", "mucho", "muchos", "muy", "más", "mí", "mía", "mías", "mío", "míos", "n", "nada", "nadie", "ni", "ninguna", "ningunas", "ninguno",
            "ningunos", "ningún", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "nueva", "nuevas", "nuevo",
            "nuevos", "nunca", "o", "ocho", "os", "otra", "otras", "otro", "otros", "p", "pais", "para", "parece", "parte", "partir", "pasada",
            "pasado", "paìs", "peor", "pero", "pesar", "poca", "pocas", "poco", "pocos", "podeis", "podemos", "poder", "podria", "podriais",
            "podriamos", "podrian", "podrias", "podrá", "podrán", "podría", "podrían", "poner", "por", "porque", "posible", "primer", "primera",
            "primero", "primeros", "principalmente", "pronto", "propia", "propias", "propio", "propios", "proximo", "próximo", "próximos", "pudo",
            "pueda", "puede", "pueden", "puedo", "pues", "q", "qeu", "que", "quedó", "queremos", "quien", "quienes", "quiere", "quiza", "quizas",
            "quizá", "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado", "realizar", "realizó", "repente", "respecto", "s", "sabe",
            "sabeis", "sabemos", "saben", "saber", "sabes", "salvo", "se", "sea", "sean", "segun", "segunda", "segundo", "según", "seis", "ser",
            "sera", "será", "serán", "sería", "señaló", "si", "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sin", "sino", "sobre",
            "sois", "sola", "solamente", "solas", "solo", "solos", "somos", "son", "soy", "soyos", "su", "supuesto", "sus", "suya", "suyas", "suyo",
            "sé", "sí", "sólo", "t", "tal", "tambien", "también", "tampoco", "tan", "tanto", "tarde", "te", "temprano", "tendrá", "tendrán", "teneis",
            "tenemos", "tener", "tenga", "tengo", "tenido", "tenía", "tercera", "ti", "tiempo", "tiene", "tienen", "toda", "todas", "todavia",
            "todavía", "todo", "todos", "total", "trabaja", "trabajais", "trabajamos", "trabajan", "trabajar", "trabajas", "trabajo", "tras",
            "trata", "través", "tres", "tu", "tus", "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "tú", "u", "ultimo", "un", "una", "unas", "uno", "unos",
            "usa", "usais", "usamos", "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va", "vais", "valor", "vamos", "van", "varias", "varios",
            "vaya", "veces", "ver", "verdad", "verdadera", "verdadero", "vez", "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro",
            "vuestros", "w", "x", "y", "ya", "yo", "z", "él", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos", "última", "últimas",
            "último", "últimos"}

End Module
