

Imports System.Runtime.CompilerServices
Imports System.Runtime.Serialization.Formatters.Binary
Imports System.Windows.Forms

Namespace Models
    Namespace TokenizerModels
        <Serializable>
        Public Class Token

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
        End Class
        Public Module TokenizerExtensions
            <Extension>
            Public Function ModelImporter(ByRef Filename As String) As Object
                Dim FileStream As New System.IO.FileStream(Filename, System.IO.FileMode.Open)
                Dim Formatter As New BinaryFormatter
                Dim Model As Object = Formatter.Deserialize(FileStream)
                FileStream.Close()

                Return Model
            End Function
            <Extension>
            Public Sub ModelExporter(ByRef Model As Object, Filename As String)
                Dim path As String = Application.StartupPath

                Dim FileStream As New System.IO.FileStream(Filename, System.IO.FileMode.CreateNew)
                Dim Formatter As New BinaryFormatter
                Formatter.Serialize(Model, FileStream)
                FileStream.Close()


            End Sub


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

            <Serializable>
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
            <Runtime.CompilerServices.Extension()>
            Public Function SpaceItems(ByRef txt As String, Item As String) As String
                Return txt.Replace(Item, " " & Item & " ")
            End Function
            Public Class VocabularyPruner
                Public Sub New(maxVocab As Integer, vocabulary As Dictionary(Of String, Integer), lowestVocabularyFreq As Integer)
                    If vocabulary Is Nothing Then
                        Throw New ArgumentNullException(NameOf(vocabulary))
                    End If

                    Me.MaxVocab = maxVocab
                    Me.Vocabulary = vocabulary
                    Me.LowestVocabularyFreq = lowestVocabularyFreq
                End Sub

                ''' <summary>
                ''' Defines max entries in vocabulary before Pruning Rare Words
                ''' </summary>
                ''' <returns></returns>
                Public Property MaxVocab As Integer = 100000
                Public Property Vocabulary As New Dictionary(Of String, Integer)
                Public Property LowestVocabularyFreq As Integer = 1
                Public Function Prune() As Dictionary(Of String, Integer)


                    If Vocabulary.Count > MaxVocab Then
                        PruneVocabulary()
                    End If
                    Return Vocabulary
                End Function

                Private Sub PruneVocabulary()
                    ' Create a list to store tokens to be removed.
                    Dim tokensToRemove As New List(Of String)

                    ' Iterate through the vocabulary and identify tokens to prune.
                    For Each token In Vocabulary
                        Dim tokenId As Integer = token.Value
                        Dim tokenFrequency As Integer = Vocabulary(token.Key)

                        ' Prune the token if it has frequency below the threshold (1) and is not recent (has a lower ID).
                        If tokenFrequency <= LowestVocabularyFreq AndAlso tokenId < Vocabulary.Count - 1 Then
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
            End Class
            <Extension>
            Public Function UpdateVocabulary(vocabulary As Dictionary(Of String, Integer), Term As String) As Dictionary(Of String, Integer)
                If vocabulary(Term) > 0 Then
                    Dim Freq As Integer = vocabulary(Term)
                    Freq += 1
                    vocabulary.Remove(Term)
                    vocabulary.Add(Term, Freq)
                Else
                    vocabulary.Add(Term, 1)
                End If
                Return vocabulary
            End Function
            <Extension>
            Public Function GetHighFreqLst(ByRef Vocabulary As Dictionary(Of String, Integer), ByRef Threshold As Integer) As List(Of String)
                Dim HighFreq As New List(Of String)
                For Each item In Vocabulary
                    If item.Value > Threshold Then
                        HighFreq.Add(item.Key)
                    End If
                Next
                Return HighFreq
            End Function
            Public Function FindFrequentCharacterBigrams(Vocab As List(Of String), ByRef Freq_Threshold As Integer) As List(Of String)
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
            <Extension>
            Public Function UpdateCorpusWithMergedToken(ByRef corpus As List(Of String), pair As String) As List(Of String)
                ' Update the text corpus with the merged token for the next iteration.
                Return corpus.ConvertAll(Function(text) text.Replace(pair, pair.Replace(" ", "_")))
            End Function



            <Runtime.CompilerServices.Extension()>
            Public Function SpacePunctuation(ByRef Txt As String) As String
                For Each item In PunctuationMarkers.Punctuation
                    Txt = SpaceItems(Txt, item)
                Next

                Return Txt
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
        End Module
        <Serializable>
        Public MustInherit Class Tokenizer


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
            Public Shared Function CharGram(Document As String, n As Integer) As List(Of String)
                CharGram = New List(Of String)
                Document = Document.ToLower()
                Document = Document.SpacePunctuation

                ' Generate character n-grams
                For i As Integer = 0 To Document.Length - n
                    Dim ngram As String = Document.Substring(i, n)
                    CharGram.Add(ngram)
                Next

            End Function

            Public Shared Function WordGram(ByRef text As String, n As Integer) As List(Of String)
                WordGram = New List(Of String)
                text = text.ToLower()
                text = text.SpacePunctuation

                ' Split the clean text into individual words
                Dim words() As String = text.Split({" ", ".", ",", ";", ":", "!", "?"}, StringSplitOptions.RemoveEmptyEntries)

                ' Generate n-grams from the words
                For i As Integer = 0 To words.Length - n
                    Dim ngram As String = String.Join(" ", words.Skip(i).Take(n))
                    WordGram.Add(ngram)
                Next

            End Function

            Public Shared Function ParagraphGram(text As String, n As Integer) As List(Of String)
                ParagraphGram = New List(Of String)

                ' Split the text into paragraphs
                Dim paragraphs() As String = text.Split({Environment.NewLine & Environment.NewLine}, StringSplitOptions.RemoveEmptyEntries)

                ' Generate paragraph n-grams
                For i As Integer = 0 To paragraphs.Length - n
                    Dim ngram As String = String.Join(Environment.NewLine & Environment.NewLine, paragraphs.Skip(i).Take(n))
                    ParagraphGram.Add(ngram)
                Next

                Return ParagraphGram
            End Function

            Public Shared Function SentenceGram(text As String, n As Integer) As List(Of String)
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


            Public MustOverride Sub Train(ByRef Corpus As List(Of String))

            Public Overridable Function Tokenize(text As String) As List(Of String)
                Return TokenizeToWord(text)
            End Function

        End Class
        <Serializable>
        Public Class WordPiece
            Inherits Tokenizer

            Property vocabulary As Dictionary(Of String, Integer)
            Public Property maxVocabSize As Integer
            Public ReadOnly Property maxSubwordLength As Integer

            Public Sub New()

                Me.vocabulary = New Dictionary(Of String, Integer)
                Me.maxVocabSize = 1000000
                Me.maxSubwordLength = 20
            End Sub

            Public Overrides Sub Train(ByRef corpus As List(Of String))
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
            Public Overrides Function Tokenize(text As String) As List(Of String)
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
        <Serializable>
        Public Class BPE
            Inherits Tokenizer

            Public Vocabulary As New Dictionary(Of String, Integer)
            Public numMerges As Integer = 1
            Public Sub New()
            End Sub

            Public Overrides Sub Train(ByRef Corpus As List(Of String))
                For Each item In Corpus
                    ' Tokenize the corpus at the character level to get the initial vocabulary
                    Dim characterLevelVocabulary As Dictionary(Of String, Integer) = TrainTokenize(item)

                    ' Merge the most frequent pairs of subwords iteratively
                    For i As Integer = 0 To numMerges - 1
                        Dim mostFrequentPair As SubWord_Pair = FindMostFrequentPair(characterLevelVocabulary)
                        If mostFrequentPair Is Nothing Then
                            Exit For
                        End If

                        Dim newSubword As String = mostFrequentPair.Sub_word_1 + mostFrequentPair.Sub_Word_2
                        characterLevelVocabulary = MergeSubwordPair(characterLevelVocabulary, mostFrequentPair, newSubword)
                    Next
                    For Each Entry In characterLevelVocabulary

                        UpdateVocabulary(Vocabulary, Entry.Key)
                    Next


                Next




            End Sub

            Private Function TrainTokenize(Document As String) As Dictionary(Of String, Integer)
                Dim characterLevelVocabulary As New Dictionary(Of String, Integer)


                For Each character As Char In Document
                    Dim subword As String = character.ToString()

                    If characterLevelVocabulary.ContainsKey(subword) Then
                        characterLevelVocabulary(subword) += 1
                    Else
                        characterLevelVocabulary.Add(subword, 1)
                    End If
                Next


                Return characterLevelVocabulary
            End Function

            Private Shared Function getTokenlist(characterLevelVocabulary As Dictionary(Of String, Integer)) As List(Of String)
                Dim Tokens As New List(Of String)
                For Each item In characterLevelVocabulary
                    Tokens.Add(item.Key)
                Next
                Return Tokens
            End Function

            Private Shared Function FindMostFrequentPair(vocabulary As Dictionary(Of String, Integer)) As SubWord_Pair
                Dim mostFrequentPair As SubWord_Pair = Nothing
                Dim maxFrequency As Integer = 0

                For Each subword1 As String In vocabulary.Keys
                    For Each subword2 As String In vocabulary.Keys
                        If subword1 <> subword2 Then
                            Dim pairFrequency As Integer = CalculatePairFrequency(vocabulary, subword1, subword2)
                            If pairFrequency > maxFrequency Then
                                maxFrequency = pairFrequency
                                mostFrequentPair = New SubWord_Pair(subword1, subword2, pairFrequency)
                            End If
                        End If
                    Next
                Next

                Return mostFrequentPair
            End Function

            Private Shared Function CalculatePairFrequency(vocabulary As Dictionary(Of String, Integer), subword1 As String, subword2 As String) As Integer
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

            Private Shared Function MergeSubwordPair(vocabulary As Dictionary(Of String, Integer), pairToMerge As SubWord_Pair, newSubword As String) As Dictionary(Of String, Integer)
                Dim newVocabulary As New Dictionary(Of String, Integer)

                For Each subword As String In vocabulary.Keys
                    Dim mergedSubword As String = subword.Replace(pairToMerge.Sub_word_1 + pairToMerge.Sub_Word_2, newSubword)
                    newVocabulary(mergedSubword) = vocabulary(subword)
                Next

                Return newVocabulary
            End Function

            Public Overrides Function Tokenize(Document As String) As List(Of String)
                Dim characterLevelVocabulary As New Dictionary(Of String, Integer)


                For Each character As Char In Document
                    Dim subword As String = character.ToString()

                    If characterLevelVocabulary.ContainsKey(subword) Then
                        characterLevelVocabulary(subword) += 1
                    Else
                        characterLevelVocabulary.Add(subword, 1)
                    End If
                Next


                Return getTokenlist(characterLevelVocabulary)
            End Function

            Public Class SubWord_Pair
                Public Property Sub_word_1 As String
                Public Property Sub_Word_2 As String
                Public Property Frequency As Integer

                Public Sub New(Sub_word_1 As String, Sub_Word_2 As String, frequency As Integer)
                    Me.Sub_word_1 = Sub_word_1
                    Me.Sub_Word_2 = Sub_Word_2
                    Me.Frequency = frequency
                End Sub
            End Class
        End Class
        <Serializable>
        Public Class BitWord
            Inherits Tokenizer
            Public Property Vocabulary As Dictionary(Of String, Integer)
            Public Property MaxMergeOperations As Integer

            Public Overrides Sub Train(ByRef Corpus As List(Of String))
                ' Initialize the vocabulary with word-level subword units
                TokenizeCorpus(Corpus)
                Dim mergeOperationsCount As Integer = 0

                While mergeOperationsCount < MaxMergeOperations
                    ' Compute the frequency of subword units in the vocabulary
                    Dim subwordFrequencies As New Dictionary(Of String, Integer)

                    For Each subword In Vocabulary.Keys
                        Dim subwordUnits = TokenizeToCharacter(subword)
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

            Public Function TokenizeCorpus(Corpus As List(Of String)) As List(Of String)
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
                        Dim Para As List(Of String) = TokenizeToParagraph(doc)
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
                        Dim Sents As List(Of String) = TokenizeToSentence(sent)


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
                        Dim Words As List(Of String) = TokenizeToWord(Word)
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
                        Dim Chars As List(Of String) = TokenizeToCharacter(iChar)
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

            Public Overrides Function Tokenize(text As String) As List(Of String)
                Throw New NotImplementedException()
            End Function
        End Class
        <Serializable>
        Public Class TokenID
            Inherits TokenizerToTokens
            Private nextId As Integer = 0
            Public Property Vocabulary As New Dictionary(Of String, Integer)
            Public TokenToID As New Dictionary(Of String, Integer)
            Private IDToToken As New Dictionary(Of Integer, String)
            ''' <summary>
            ''' Pure Tokenizer (will tokenize based on the Tokenizer model settings)
            ''' </summary>
            ''' <param name="Doc"></param>
            ''' <returns></returns>
            Public Shadows Function Encode(Doc As String) As List(Of Integer)
                Dim tokens = TokenizeByWord(Doc)
                Dim tokenIds As New List(Of Integer)

                For Each itoken In tokens
                    Dim tokenId As Integer
                    If TokenToID.ContainsKey(itoken.Value) Then
                        tokenId = TokenToID(itoken.Value)
                    Else
                        'Not registered

                        tokenId = TokenToID(itoken.Value)

                    End If
                    tokenIds.Add(tokenId)

                Next

                Return tokenIds
            End Function
            Public Sub UpdateVocabulary(Token As String)

                If Not Vocabulary.ContainsKey(Token) Then
                    Vocabulary(Token) = nextId
                    nextId += 1
                    TokenToID = Vocabulary.ToDictionary(Function(x) x.Key, Function(x) x.Value)
                    IDToToken = TokenToID.ToDictionary(Function(x) x.Value, Function(x) x.Key)
                End If


            End Sub

            ''' <summary>
            ''' Given  a Set of Token ID Decode the Tokens 
            ''' </summary>
            ''' <param name="tokenIds"></param>
            ''' <returns></returns>
            Public Function Decode(tokenIds As List(Of Integer)) As String
                Dim tokens As New List(Of String)

                For Each tokenId As Integer In tokenIds
                    tokens.Add(IDToToken(tokenId))
                Next

                Return String.Join(" ", tokens)
            End Function


            Public Overloads Function Tokenize(text As String) As List(Of String)
                Dim lst As New List(Of String)
                For Each item In Encode(text)
                    lst.Add(item)
                Next
                Return lst
            End Function
        End Class
        <Serializable>
        Public Class Advanced
            Inherits Tokenizer

            Public Property Vocabulary As Dictionary(Of String, Integer)
            Public ReadOnly Property PairFrequencies As Dictionary(Of String, Integer) = ComputePairFrequencies()
            Public ReadOnly Property maxSubwordLen As Integer = Me.Vocabulary.Max(Function(token) token.Key.Length)
            Private ReadOnly unkToken As String = "<Unk>"
            ''' <summary>
            ''' Defines max entries in vocabulary before Pruning Rare Words
            ''' </summary>
            ''' <returns></returns>
            Public Property MaxVocabSize As Integer = 100000
            Public Sub Prune(pruningThreshold As Integer)
                Dim Pruner As New VocabularyPruner(MaxVocabSize, Vocabulary, pruningThreshold)

                If Vocabulary.Count > MaxVocabSize Then
                    Vocabulary = Pruner.Prune()
                End If

            End Sub


            Public Overrides Sub Train(ByRef Corpus As List(Of String))
                For Each item In Corpus
                    Train(item, 10)
                Next
            End Sub
            Public Overloads Sub Train(text As String, isWordPiece As Boolean, Epochs As Integer)
                If isWordPiece Then
                    TrainWordPiece(text, Epochs)
                Else
                    TrainBPE(text, Epochs)
                End If
                Prune(1)
            End Sub
            Public Overloads Sub Train(text As String, Epochs As Integer)
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

            Public Overloads Function Tokenize(singleDocument As String, isWordPiece As Boolean) As List(Of String)
                ' Tokenize the document using the current vocabulary.
                Dim tokens As List(Of String) = If(isWordPiece, Tokenize(singleDocument, True), Tokenize(singleDocument, False))
                If tokens.Contains(unkToken) = True Then
                    tokens = TrainAndTokenize(singleDocument, isWordPiece, 1)
                End If
                Return tokens
            End Function
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

            Public Overrides Function Tokenize(text As String) As List(Of String)
                Dim Words = Tokenizer.TokenizeToWord(text)
                Dim Tokens As New List(Of String)
                For Each item In Words
                    Tokens.AddRange(TokenizeBitWord(item))
                Next
                Return Tokens
            End Function
            ''' <summary>
            ''' Adds a VocabularyList to this vocabulary
            ''' </summary>
            ''' <param name="initialVocabulary"></param>
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
            Public Function GetVocabularyLst() As List(Of String)
                Return Vocabulary.Keys.ToList()
            End Function



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


        End Class
        <Serializable>
        Public Class TokenizerToTokens





            ''' <summary>
            ''' Pure basic Tokenizer to Tokens
            ''' </summary>
            ''' <param name="Corpus"></param>
            ''' <param name="tokenizationOption">Type Of Tokenization</param>
            ''' <returns></returns>
            Public Function Tokenize(ByRef Corpus As List(Of String), tokenizationOption As TokenizerType) As List(Of Token)
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
            Private Shared ReadOnly AlphaBet() As String = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}
            Private Shared ReadOnly Number() As String = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
"30", "40", "50", "60", "70", "80", "90", "00", "000", "0000", "00000", "000000", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
"nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "Billion"}
            Public Shared Function GetValidTokens(ByRef InputStr As String) As String
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
            Private Shared Function AddSuffix(ByRef Str As String, ByVal Suffix As String) As String
                Return Str & Suffix
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

            Public Shared Function GetTokenType(ByRef CharStr As String) As TokenType
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

        End Class
    End Namespace



End Namespace