Imports System.Numerics
Imports MathNet.Numerics

Namespace Models
    Namespace Storage
        Public Class SearchEngine
            Private documents As List(Of Document)

            Public Sub New()
                documents = New List(Of Document)()
            End Sub

            Public Sub AddDocument(id As Integer, content As String)
                documents.Add(New Document() With {.Id = id, .Content = content})
            End Sub

            Public Function Search(query As String) As List(Of Tuple(Of Integer, String))
                Dim searchResults As New List(Of Tuple(Of Integer, String))()

                For Each doc As Document In documents
                    If doc.Content.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0 Then
                        Dim snippet As String = ExtractContextualSnippet(doc.Content, query)
                        searchResults.Add(New Tuple(Of Integer, String)(doc.Id, snippet))
                    End If
                Next

                Return searchResults
            End Function

            Private Function ExtractContextualSnippet(content As String, query As String) As String
                Dim queryIndex As Integer = content.IndexOf(query, StringComparison.OrdinalIgnoreCase)

                Dim snippetStart As Integer = Math.Max(0, queryIndex - 50)
                Dim snippetEnd As Integer = Math.Min(content.Length, queryIndex + query.Length + 50)

                Dim snippet As String = content.Substring(snippetStart, snippetEnd - snippetStart)

                ' Highlight query terms in the snippet
                snippet = snippet.Replace(query, $"<strong>{query}</strong>")

                Return snippet
            End Function
        End Class

        Public Class Document
            Public Property Index As Integer
            Public Property Content As String
            Public Property Elements As List(Of String)

            Public Property Id As Integer
            Public Sub New(ByVal content As String, ByVal index As Integer)
                Me.Content = content
                Me.Index = index
                ' Split the document content into elements (words) for MinHash.
                Me.Elements = content.Split({" "}, StringSplitOptions.RemoveEmptyEntries).ToList()
            End Sub
            Public Sub New(ByVal id As Integer, ByVal content As String)
                Me.Content = content
                Me.Id = id
                ' Split the document content into elements (words) for MinHash.
                Me.Elements = content.Split({" "}, StringSplitOptions.RemoveEmptyEntries).ToList()
            End Sub

            Public Sub New()
            End Sub
        End Class
        Namespace MinHashAndLSH

            Public Class LSHIndex
                Private HashTables As List(Of Dictionary(Of Integer, List(Of Document)))
                Private NumHashTables As Integer
                Private NumHashFunctionsPerTable As Integer

                Public Sub New(ByVal numHashTables As Integer, ByVal numHashFunctionsPerTable As Integer)
                    ' Initialize the LSH index with the specified number of hash tables and hash functions per table
                    Me.NumHashTables = numHashTables
                    Me.NumHashFunctionsPerTable = numHashFunctionsPerTable
                    HashTables = New List(Of Dictionary(Of Integer, List(Of Document)))(numHashTables)

                    ' Create empty hash tables
                    For i As Integer = 0 To numHashTables - 1
                        HashTables.Add(New Dictionary(Of Integer, List(Of Document))())
                    Next
                End Sub

                Private Function ComputeHash(ByVal content As String, ByVal hashFunctionIndex As Integer) As Integer
                    ' Improved hash function that uses prime numbers to reduce collisions.
                    ' This hash function is for demonstration purposes and can be further optimized for production.
                    Dim p As Integer = 31 ' Prime number
                    Dim hash As Integer = 0
                    For Each ch As Char In content
                        hash = (hash * p + Asc(ch)) Mod Integer.MaxValue
                    Next
                    Return hash
                End Function

                Public Sub AddDocument(ByVal document As Document)
                    ' For each hash table, apply multiple hash functions to the document and insert it into the appropriate bucket.
                    For tableIndex As Integer = 0 To NumHashTables - 1
                        For i As Integer = 0 To NumHashFunctionsPerTable - 1
                            Dim hashCode As Integer = ComputeHash(document.Content, i)
                            If Not HashTables(tableIndex).ContainsKey(hashCode) Then
                                ' If the bucket for the hash code doesn't exist, create it and associate it with an empty list of documents.
                                HashTables(tableIndex)(hashCode) = New List(Of Document)()
                            End If
                            ' Add the document to the bucket associated with the hash code.
                            HashTables(tableIndex)(hashCode).Add(document)
                        Next
                    Next
                End Sub

                Private Function GenerateShingles(ByVal content As String, ByVal shingleSize As Integer) As List(Of String)
                    ' Helper function to generate shingles (substrings) of the given size from the content.
                    Dim shingles As New List(Of String)()
                    If content.Length < shingleSize Then
                        shingles.Add(content)
                        Return shingles
                    End If
                    For i As Integer = 0 To content.Length - shingleSize
                        shingles.Add(content.Substring(i, shingleSize))
                    Next
                    Return shingles
                End Function

                Private Function ComputeSimilarity(ByVal content1 As String, ByVal content2 As String) As Double
                    ' Improved Jaccard similarity function that uses sets of shingles for more accurate results.
                    ' For simplicity, let's use 3-character shingles.
                    Dim set1 As New HashSet(Of String)(GenerateShingles(content1, 3))
                    Dim set2 As New HashSet(Of String)(GenerateShingles(content2, 3))

                    Dim intersectionCount As Integer = set1.Intersect(set2).Count()
                    Dim unionCount As Integer = set1.Count + set2.Count - intersectionCount

                    Return CDbl(intersectionCount) / CDbl(unionCount)
                End Function

                Public Function FindSimilarDocuments(ByVal queryDocument As Document) As List(Of Document)
                    Dim similarDocuments As New List(Of Document)()

                    ' For the query document, compute hash codes for each hash table and look for similar documents.
                    For tableIndex As Integer = 0 To NumHashTables - 1
                        For i As Integer = 0 To NumHashFunctionsPerTable - 1
                            Dim hashCode As Integer = ComputeHash(queryDocument.Content, i)
                            Dim similarDocs As List(Of Document) = FindSimilarDocumentsInHashTable(hashCode, tableIndex)
                            similarDocuments.AddRange(similarDocs)
                        Next
                    Next

                    ' Remove duplicates from the list of similar documents.
                    Dim uniqueSimilarDocs As New List(Of Document)(New HashSet(Of Document)(similarDocuments))

                    ' Sort the similar documents based on their similarity to the query document.
                    uniqueSimilarDocs.Sort(Function(doc1, doc2) ComputeSimilarity(queryDocument.Content, doc2.Content).CompareTo(ComputeSimilarity(queryDocument.Content, doc1.Content)))

                    Return uniqueSimilarDocs
                End Function

                Private Function FindSimilarDocumentsInHashTable(ByVal queryDocHashCode As Integer, ByVal tableIndex As Integer) As List(Of Document)
                    ' Given the hash code of the query document and a hash table index, find all documents in the same bucket.
                    Dim similarDocs As New List(Of Document)()
                    If HashTables(tableIndex).ContainsKey(queryDocHashCode) Then
                        ' If the bucket for the hash code exists, retrieve the list of documents in the bucket.
                        similarDocs.AddRange(HashTables(tableIndex)(queryDocHashCode))
                    End If
                    Return similarDocs
                End Function
            End Class
            Public Class MinHashAndLSHInterface
                Private MinHashIndex As MinHashIndex

                Public Sub New(ByVal numHashFunctions As Integer)
                    MinHashIndex = New MinHashIndex(numHashFunctions)
                End Sub

                Public Sub AddDocument(ByVal content As String, ByVal index As Integer)
                    Dim document As New Document(content, index)
                    MinHashIndex.AddDocument(document)
                End Sub

                Public Function FindSimilarDocuments(ByVal queryContent As String) As List(Of Document)
                    Dim queryDocument As New Document(queryContent, -1)
                    Dim similarDocuments As List(Of Document) = MinHashIndex.FindSimilarDocuments(queryDocument)
                    Return similarDocuments
                End Function
            End Class


            Public Class MinHashVectorDatabase
                Private MinHashIndex As MinHashIndex

                Public Sub New(ByVal numHashFunctions As Integer)
                    MinHashIndex = New MinHashIndex(numHashFunctions)
                End Sub

                Public Sub AddDocument(ByVal content As String, ByVal index As Integer)
                    Dim document As New Document(content, index)
                    MinHashIndex.AddDocument(document)
                End Sub

                Public Function FindSimilarDocuments(ByVal queryContent As String, ByVal threshold As Double) As List(Of Document)
                    Dim queryDocument As New Document(queryContent, -1)
                    Dim similarDocuments As List(Of Document) = MinHashIndex.FindSimilarDocuments(queryDocument)
                    Return similarDocuments
                End Function
            End Class
            Public Class MinHashIndex
                Private NumHashFunctions As Integer
                Private SignatureMatrix As List(Of List(Of Integer))
                Private Buckets As Dictionary(Of Integer, List(Of Document))

                Public Sub New(ByVal numHashFunctions As Integer)
                    Me.NumHashFunctions = numHashFunctions
                    SignatureMatrix = New List(Of List(Of Integer))()
                    Buckets = New Dictionary(Of Integer, List(Of Document))()
                End Sub

                Private Function ComputeHash(ByVal element As String, ByVal hashFunctionIndex As Integer) As Integer
                    ' Use a simple hashing algorithm for demonstration purposes.
                    Dim p As Integer = 31 ' Prime number
                    Dim hash As Integer = 0
                    For Each ch As Char In element
                        hash = (hash * p + Asc(ch)) Mod Integer.MaxValue
                    Next
                    Return hash
                End Function

                Public Sub AddDocument(ByVal document As Document)
                    ' Generate the signature matrix for the given document using the hash functions.
                    For i As Integer = 0 To NumHashFunctions - 1
                        Dim minHash As Integer = Integer.MaxValue
                        For Each element As String In document.Elements
                            Dim hashCode As Integer = ComputeHash(element, i)
                            minHash = Math.Min(minHash, hashCode)
                        Next
                        If SignatureMatrix.Count <= i Then
                            SignatureMatrix.Add(New List(Of Integer)())
                        End If
                        SignatureMatrix(i).Add(minHash)
                    Next

                    ' Add the document to the appropriate bucket for LSH.
                    Dim bucketKey As Integer = SignatureMatrix(NumHashFunctions - 1).Last()
                    If Not Buckets.ContainsKey(bucketKey) Then
                        Buckets(bucketKey) = New List(Of Document)()
                    End If
                    Buckets(bucketKey).Add(document)
                End Sub

                Private Function EstimateSimilarity(ByVal signatureMatrix1 As List(Of Integer), ByVal signatureMatrix2 As List(Of Integer)) As Double
                    Dim matchingRows As Integer = 0
                    For i As Integer = 0 To NumHashFunctions - 1
                        If signatureMatrix1(i) = signatureMatrix2(i) Then
                            matchingRows += 1
                        End If
                    Next

                    Return CDbl(matchingRows) / CDbl(NumHashFunctions)
                End Function

                Public Function FindSimilarDocuments(ByVal queryDocument As Document) As List(Of Document)
                    ' Generate the signature matrix for the query document using the hash functions.
                    Dim querySignature As New List(Of Integer)()
                    For i As Integer = 0 To NumHashFunctions - 1
                        Dim minHash As Integer = Integer.MaxValue
                        For Each element As String In queryDocument.Elements
                            Dim hashCode As Integer = ComputeHash(element, i)
                            minHash = Math.Min(minHash, hashCode)
                        Next
                        querySignature.Add(minHash)
                    Next

                    ' Find candidate similar documents using Locality-Sensitive Hashing (LSH).
                    Dim candidateDocuments As New List(Of Document)()
                    Dim bucketKey As Integer = querySignature(NumHashFunctions - 1)
                    If Buckets.ContainsKey(bucketKey) Then
                        candidateDocuments.AddRange(Buckets(bucketKey))
                    End If

                    ' Refine the candidate documents using MinHash similarity estimation.
                    Dim similarDocuments As New List(Of Document)()
                    For Each candidateDoc As Document In candidateDocuments
                        Dim similarity As Double = EstimateSimilarity(querySignature, SignatureMatrix(candidateDoc.Index))
                        If similarity >= 0.5 Then ' Adjust the threshold as needed.
                            similarDocuments.Add(candidateDoc)
                        End If
                    Next

                    Return similarDocuments
                End Function
            End Class
        End Namespace
        Public Class VectorStorageModel
            Private AudioVectors As Dictionary(Of Integer, List(Of Complex)) = New Dictionary(Of Integer, List(Of Complex))()
            Private ImageVectors As Dictionary(Of Integer, Tuple(Of VectorType, List(Of Double))) = New Dictionary(Of Integer, Tuple(Of VectorType, List(Of Double)))()
            Private TextVectors As Dictionary(Of Integer, List(Of Double)) = New Dictionary(Of Integer, List(Of Double))()
            Public Enum VectorType
                Text
                Image
                Audio
            End Enum

            Public Sub AddAudioVector(id As Integer, vector As List(Of Complex))
                AudioVectors.Add(id, vector)
            End Sub
            Public Sub AddImageVector(id As Integer, vector As List(Of Double))
                ImageVectors.Add(id, Tuple.Create(VectorType.Image, vector))
            End Sub
            Public Sub AddTextVector(id As Integer, vector As List(Of Double))
                ImageVectors.Add(id, Tuple.Create(VectorType.Text, vector))
            End Sub
            Public Sub AddVector(id As Integer, vector As List(Of Double), vectorType As VectorType)
                If vectorType = VectorType.Text Then

                    TextVectors.Add(id, vector)
                ElseIf vectorType = VectorType.Image Then
                    ImageVectors.Add(id, Tuple.Create(VectorType.Image, vector))

                End If
            End Sub
            Public Sub AddVector(id As Integer, vector As List(Of Complex), vectorType As VectorType)
                If vectorType = VectorType.Audio Then
                    AudioVectors.Add(id, vector)
                End If
            End Sub
            Public Function FindSimilarAudioVectors(queryVector As List(Of Complex), numNeighbors As Integer) As List(Of Integer)
                Dim similarVectors As List(Of Integer) = New List(Of Integer)()

                For Each vectorId As Integer In AudioVectors.Keys
                    Dim vectorData As List(Of Complex) = AudioVectors(vectorId)
                    Dim distance As Double = CalculateEuclideanDistanceComplex(queryVector, vectorData)

                    ' Maintain a list of the closest neighbors
                    If similarVectors.Count < numNeighbors Then
                        similarVectors.Add(vectorId)
                    Else
                        Dim maxDistance As Double = GetMaxDistanceComplex(similarVectors, queryVector)
                        If distance < maxDistance Then
                            Dim indexToRemove As Integer = GetIndexOfMaxDistanceComplex(similarVectors, queryVector)
                            similarVectors.RemoveAt(indexToRemove)
                            similarVectors.Add(vectorId)
                        End If
                    End If
                Next

                Return similarVectors
            End Function
            Public Function FindSimilarImageVectors(queryVector As List(Of Double), numNeighbors As Integer) As List(Of Integer)
                Dim similarVectors As List(Of Integer) = New List(Of Integer)

                For Each vectorId As Integer In ImageVectors.Keys
                    Dim vectorType As VectorType
                    Dim vectorData As List(Of Double)
                    Dim vectorTuple As Tuple(Of VectorType, List(Of Double)) = ImageVectors(vectorId)
                    vectorType = vectorTuple.Item1
                    vectorData = vectorTuple.Item2

                    If vectorType = VectorType.Image Then
                        ' Calculate similarity using image-specific logic
                        ' You can integrate the image similarity logic here
                    ElseIf vectorType = VectorType.Text Then
                        ' Calculate similarity using text-specific logic
                        Dim distance As Double = CalculateEuclideanDistance(queryVector, vectorData)

                        ' Maintain a list of the closest neighbors
                        If similarVectors.Count < numNeighbors Then
                            similarVectors.Add(vectorId)
                        Else
                            Dim maxDistance As Double = GetMaxDistance(similarVectors, queryVector)
                            If distance < maxDistance Then
                                Dim indexToRemove As Integer = GetIndexOfMaxDistanceImages(similarVectors, queryVector)
                                similarVectors.RemoveAt(indexToRemove)
                                similarVectors.Add(vectorId)
                            End If
                        End If
                    End If
                Next

                Return similarVectors
            End Function
            Public Function FindSimilarTextVectors(queryVector As List(Of Double), numNeighbors As Integer) As List(Of Integer)
                Dim similarVectors As List(Of Integer) = New List(Of Integer)()

                For Each vectorId As Integer In TextVectors.Keys
                    Dim vector As List(Of Double) = TextVectors(vectorId)
                    Dim distance As Double = CalculateEuclideanDistance(queryVector, vector)

                    ' Maintain a list of the closest neighbors
                    If similarVectors.Count < numNeighbors Then
                        similarVectors.Add(vectorId)
                    Else
                        Dim maxDistance As Double = GetTextVectorsMaxDistance(similarVectors, queryVector)
                        If distance < maxDistance Then
                            Dim indexToRemove As Integer = GetIndexOfMaxDistance(similarVectors, queryVector)
                            similarVectors.RemoveAt(indexToRemove)
                            similarVectors.Add(vectorId)
                        End If
                    End If
                Next

                Return similarVectors
            End Function
            Private Function CalculateEuclideanDistance(vector1 As List(Of Double), vector2 As List(Of Double)) As Double
                Dim sum As Double = 0
                For i As Integer = 0 To vector1.Count - 1
                    sum += Math.Pow(vector1(i) - vector2(i), 2)
                Next
                Return Math.Sqrt(sum)
            End Function
            Private Function CalculateEuclideanDistanceComplex(vector1 As List(Of Complex), vector2 As List(Of Complex)) As Double
                Dim sum As Double = 0
                For i As Integer = 0 To vector1.Count - 1
                    Dim difference As Complex = vector1(i) - vector2(i)
                    sum += difference.MagnitudeSquared
                Next
                Return Math.Sqrt(sum)
            End Function
            Private Function GetIndexOfMaxDistance(vectorIds As List(Of Integer), queryVector As List(Of Double)) As Integer
                Dim maxDistance As Double = Double.MinValue
                Dim indexToRemove As Integer = -1
                For i As Integer = 0 To vectorIds.Count - 1
                    Dim vectorId As Integer = vectorIds(i)
                    Dim vector As List(Of Double) = TextVectors(vectorId)
                    Dim distance As Double = CalculateEuclideanDistance(queryVector, vector)
                    If distance > maxDistance Then
                        maxDistance = distance
                        indexToRemove = i
                    End If
                Next
                Return indexToRemove
            End Function
            Private Function GetIndexOfMaxDistanceComplex(vectorIds As List(Of Integer), queryVector As List(Of Complex)) As Integer
                Dim maxDistance As Double = Double.MinValue
                Dim indexToRemove As Integer = -1
                For i As Integer = 0 To vectorIds.Count - 1
                    Dim vectorId As Integer = vectorIds(i)
                    Dim vectorData As List(Of Complex) = AudioVectors(vectorId)
                    Dim distance As Double = CalculateEuclideanDistanceComplex(queryVector, vectorData)
                    If distance > maxDistance Then
                        maxDistance = distance
                        indexToRemove = i
                    End If
                Next
                Return indexToRemove
            End Function
            Private Function GetIndexOfMaxDistanceImages(vectorIds As List(Of Integer), queryVector As List(Of Double)) As Integer
                Dim maxDistance As Double = Double.MinValue
                Dim indexToRemove As Integer = -1
                For i As Integer = 0 To vectorIds.Count - 1
                    Dim vectorId As Integer = vectorIds(i)
                    Dim vector = ImageVectors(vectorId)


                    Dim distance As Double = CalculateEuclideanDistance(queryVector, vector.Item2)
                    If distance > maxDistance Then
                        maxDistance = distance
                        indexToRemove = i
                    End If
                Next
                Return indexToRemove
            End Function
            Private Function GetMaxDistance(vectorIds As List(Of Integer), queryVector As List(Of Double)) As Double
                Dim maxDistance As Double = Double.MinValue
                For Each vectorId As Integer In vectorIds
                    Dim vectorType As VectorType
                    Dim vectorData As List(Of Double)
                    Dim vectorTuple As Tuple(Of VectorType, List(Of Double)) = ImageVectors(vectorId)
                    vectorType = vectorTuple.Item1
                    vectorData = vectorTuple.Item2

                    If vectorType = VectorType.Text Then
                        Dim distance As Double = CalculateEuclideanDistance(queryVector, vectorData)
                        If distance > maxDistance Then
                            maxDistance = distance
                        End If
                    End If
                Next
                Return maxDistance
            End Function
            Private Function GetMaxDistanceComplex(vectorIds As List(Of Integer), queryVector As List(Of Complex)) As Double
                Dim maxDistance As Double = Double.MinValue
                For Each vectorId As Integer In vectorIds
                    Dim vectorData As List(Of Complex) = AudioVectors(vectorId)
                    Dim distance As Double = CalculateEuclideanDistanceComplex(queryVector, vectorData)
                    If distance > maxDistance Then
                        maxDistance = distance
                    End If
                Next
                Return maxDistance
            End Function
            Private Function GetTextVectorsMaxDistance(vectorIds As List(Of Integer), queryVector As List(Of Double)) As Double
                Dim maxDistance As Double = Double.MinValue
                For Each vectorId As Integer In vectorIds
                    Dim vector As List(Of Double) = TextVectors(vectorId)
                    Dim distance As Double = CalculateEuclideanDistance(queryVector, vector)
                    If distance > maxDistance Then
                        maxDistance = distance
                    End If
                Next
                Return maxDistance
            End Function
        End Class
    End Namespace
End Namespace
