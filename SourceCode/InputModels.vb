Imports System.Drawing
Imports System.IO
Imports System.Text.RegularExpressions
Imports System.Web.Script.Serialization
Imports System.Windows.Forms
Imports InputModelling.LanguageModels
Imports InputModelling.Models.EntityModel
Imports InputModelling.Models.MatrixModels
Imports InputModelling.Models.Readers
Imports InputModelling.Models.Tokenizers
Imports InputModelling.Models.VocabularyModelling
Imports InputModelling.Utilitys
Imports Newtonsoft.Json


Namespace Models

    Namespace Chunkers
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
        ''' Separates text into groups(chunks)
        ''' </summary>
        Public Class TextChunking
            ''' <summary>
            ''' a Function In VB which takes a text And splits it Into chunks Of X length  
            ''' </summary>
            ''' <param name="text"></param>
            ''' <param name="chunkLength"></param>
            ''' <returns></returns>
            Function SplitTextIntoChunks(text As String, chunkLength As Integer) As List(Of String)
                Dim chunks As New List(Of String)()

                If chunkLength <= 0 OrElse String.IsNullOrEmpty(text) Then
                    Return chunks
                End If

                Dim length As Integer = text.Length
                Dim index As Integer = 0

                While index < length
                    Dim remainingLength As Integer = length - index
                    Dim chunk As String = If(remainingLength <= chunkLength, text.Substring(index), text.Substring(index, chunkLength))
                    chunks.Add(chunk)
                    index += chunkLength
                End While

                Return chunks
            End Function



            Function SelectRandomChunks(document As String, chunkCount As Integer, chunkLength As Integer) As List(Of String)
                Dim chunks As New List(Of String)()

                If chunkCount <= 0 OrElse String.IsNullOrEmpty(document) OrElse chunkLength <= 0 Then
                    Return chunks
                End If

                Dim random As New Random()
                Dim documentLength As Integer = document.Length

                For i As Integer = 1 To chunkCount
                    Dim startIndex As Integer = random.Next(0, documentLength - chunkLength + 1)
                    Dim chunk As String = document.Substring(startIndex, chunkLength)
                    chunks.Add(chunk)
                Next

                Return chunks
            End Function


            ''' <summary>
            ''' Grammatical person refers to the degree of involvement of a participant in an action, event, or circumstance.
            ''' There are three degrees of grammatical person:
            ''' first person (the speaker),
            ''' second person (someone being spoken to),
            ''' and third person (anyone/anything not being directly addressed).
            ''' </summary>
            Public Class GramaticalPerson
                Private Shared FirstPersonProNouns As List(Of String) = New List(Of String)({" I ", " ME ", " MY", " MINE", " MYSELF", "I ", " US", " OUR", " OURS"})
                Private Shared SecondPersonProNouns As List(Of String) = New List(Of String)({" YOU ", " YOUR ", " YOURSELF ", " YOURSELFS", " YOURSELVES"})
                Private Shared ThirdPersonProNouns As List(Of String) = New List(Of String)({"he", "him", " his", " himself", " she", " her", " hers", " herself", " it", " its", " itself", " they", "them", "their", "theirs", "themselves"})

                ''' <summary>
                ''' Grammatical person refers to the degree of involvement of a participant in an action, event, or circumstance.
                ''' There are three degrees of grammatical person:
                ''' first person (the speaker),
                ''' second person (someone being spoken to),
                ''' and third person (anyone/anything not being directly addressed).
                ''' </summary>
                Public Enum PerspectivePerson
                    First_Person_ME = 0
                    Second_Person_YOU = 1
                    Third_Person_THEM = 2
                    NOBODY = 4
                End Enum

                ''' <summary>
                ''' The cases Of pronouns tell you how they are being used In a sentence.
                ''' </summary>
                Public Enum PerspectiveCase
                    PersonalSubject = 0
                    PersonalObject = 1
                    PersonalPosessive = 2
                    NOBODY = 3
                End Enum

#Region "Perspective Person"

                ''' <summary>
                ''' checks list if it contains item
                ''' </summary>
                ''' <param name="UserSentence"></param>
                ''' <param name="Lst"></param>
                ''' <returns></returns>
                Private Shared Function DETECT_PERSPECTIVE(ByRef UserSentence As String, Lst As List(Of String)) As Boolean
                    DETECT_PERSPECTIVE = False
                    For Each item In Lst
                        If UCase(UserSentence).Contains(UCase(item)) Then Return True
                    Next
                End Function

                ''' <summary>
                ''' RETURNS THE SUBJECT PERSPECTIVE
                '''         ''' IE:
                ''' "ME" - 1ST PERSON -
                ''' "YOU" - SECOND PERSON -
                ''' "THEM" - 3RD PERSON -
                ''' "NOBODY" NO PERSPECTIVE
                ''' </summary>
                ''' <param name="UserInputStr"></param>
                ''' <returns></returns>
                Public Shared Function GetGramiticalPersonStr(ByRef UserInputStr As String) As PerspectivePerson
                    If DETECT_1ST_PERSON(UserInputStr) = True Then Return PerspectivePerson.First_Person_ME
                    If DETECT_2ND_PERSON(UserInputStr) = True Then Return PerspectivePerson.Second_Person_YOU
                    If DETECT_3RD_PERSON(UserInputStr) = True Then Return PerspectivePerson.Third_Person_THEM
                    Return PerspectivePerson.NOBODY
                End Function

                ''' <summary>
                ''' First person definition: first person indicates the speaker.
                ''' First person is the I/we perspective.
                ''' We, us, our,and ourselves are all first-person pronouns.
                ''' Specifically, they are plural first-person pronouns.
                ''' Singular first-person pronouns include I, me, my, mine and myself.
                ''' </summary>
                ''' <returns></returns>
                Public Shared Function DETECT_1ST_PERSON(ByRef UserSentence As String) As Boolean
                    DETECT_1ST_PERSON = False
                    If DETECT_PERSPECTIVE(UserSentence, FirstPersonProNouns) = True Then Return True
                End Function

                ''' <summary>
                ''' Second person definition: second person indicates the addressee.
                ''' Second person is the you perspective.
                ''' The second-person point of view belongs to the person (or people) being addressed.
                ''' This is the “you” perspective.
                ''' the biggest indicator of the second person is the use of second-person pronouns:
                ''' you, your, yours, yourself, yourselves.
                ''' </summary>
                ''' <returns></returns>
                Public Shared Function DETECT_2ND_PERSON(ByRef UserSentence As String) As Boolean
                    DETECT_2ND_PERSON = False
                    If DETECT_PERSPECTIVE(UserSentence, SecondPersonProNouns) = True Then Return True
                End Function

                ''' <summary>
                ''' Third person definition: third person indicates a third party individual other than the speaker.
                ''' Third person is the he/she/it/they perspective.
                ''' The third-person point of view belongs to the person (or people) being talked about.
                ''' The third-person pronouns include
                ''' he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, and themselves.
                ''' </summary>
                ''' <returns></returns>
                Public Shared Function DETECT_3RD_PERSON(ByRef UserSentence As String) As Boolean
                    DETECT_3RD_PERSON = False
                    If DETECT_PERSPECTIVE(UserSentence, ThirdPersonProNouns) = True Then Return True
                End Function

                ''' <summary>
                ''' Returns detected Pronoun indicator
                ''' </summary>
                ''' <param name="Userinput"></param>
                ''' <returns></returns>
                Public Shared Function GetDetectedPersonalProNoun(ByRef Userinput As String) As String
                    Dim lst As New List(Of String)
                    lst.AddRange(FirstPersonProNouns)
                    lst.AddRange(SecondPersonProNouns)
                    lst.AddRange(ThirdPersonProNouns)
                    For Each item In lst
                        If Userinput.Contains(UCase(item)) Then Return UCase(item)
                    Next
                    Return ""
                End Function

#End Region

#Region "Case-The cases Of pronouns tell you how they are being used In a sentence."

                ''' <summary>
                ''' The cases Of pronouns tell you how they are being used In a sentence.
                ''' SUBJECT / OBJECT or POSSESSION - or NONE
                ''' </summary>
                ''' <returns></returns>
                Public Function CheckCase(ByRef Userinput As String) As PerspectiveCase
                    CheckCase = PerspectiveCase.NOBODY
                    If CheckPersonalSubject(Userinput) = True Then
                        Return PerspectiveCase.PersonalSubject
                    Else
                        If CheckPersonalObject(Userinput) = True Then
                            Return PerspectiveCase.PersonalObject
                            If CheckPersonalPossession(Userinput) = True Then
                                Return PerspectiveCase.PersonalPosessive
                            End If
                        End If
                    End If

                End Function

                ''' <summary>
                ''' The cases Of pronouns tell you how they are being used In a sentence.
                ''' </summary>
                Public Shared Function CheckPersonalSubject(ByRef Userinput As String) As Boolean
                    Dim mFirstPersonProNouns As List(Of String) = New List(Of String)({" I", " WE"})
                    Dim mSecondPersonProNouns As List(Of String) = New List(Of String)({" YOU", " US"})
                    Dim mThirdPersonProNouns As List(Of String) = New List(Of String)({" HE", " SHE", " IT"})
                    If DETECT_PERSPECTIVE(Userinput, mFirstPersonProNouns) = True Or
                    DETECT_PERSPECTIVE(Userinput, mSecondPersonProNouns) = True Or
                    DETECT_PERSPECTIVE(Userinput, mThirdPersonProNouns) = True Then
                        Return True
                    Else
                        Return False
                    End If

                End Function

                ''' <summary>
                ''' The cases Of pronouns tell you how they are being used In a sentence.
                ''' </summary>
                Public Shared Function CheckPersonalObject(ByRef Userinput As String) As Boolean
                    Dim mFirstPersonProNouns As List(Of String) = New List(Of String)({"ME"})
                    Dim mSecondPersonProNouns As List(Of String) = New List(Of String)({"YOU", "US"})
                    Dim mThirdPersonProNouns As List(Of String) = New List(Of String)({" HIM", " HER", " IT"})
                    If DETECT_PERSPECTIVE(Userinput, mFirstPersonProNouns) = True Or
                    DETECT_PERSPECTIVE(Userinput, mSecondPersonProNouns) = True Or
                    DETECT_PERSPECTIVE(Userinput, mThirdPersonProNouns) = True Then
                        Return True
                    Else
                        Return False
                    End If
                End Function

                ''' <summary>
                ''' The cases Of pronouns tell you how they are being used In a sentence.
                ''' </summary>
                Public Shared Function CheckPersonalPossession(ByRef Userinput As String) As Boolean
                    Dim mFirstPersonProNouns As List(Of String) = New List(Of String)({" MY", " MINE", " OUR", " OURS"})
                    Dim mSecondPersonProNouns As List(Of String) = New List(Of String)({" YOUR", " YOURS"})
                    Dim mThirdPersonProNouns As List(Of String) = New List(Of String)({" HIS", " HER", " HES", " HE IS", " HERS", " THEIR", " THEIRS"})
                    If DETECT_PERSPECTIVE(Userinput, mFirstPersonProNouns) = True Or
                    DETECT_PERSPECTIVE(Userinput, mSecondPersonProNouns) = True Or
                    DETECT_PERSPECTIVE(Userinput, mThirdPersonProNouns) = True Then
                        Return True
                    Else
                        Return False
                    End If
                End Function

                ''' <summary>
                ''' The cases Of pronouns tell you how they are being used In a sentence.
                ''' "ME" - 1ST PERSON -
                ''' "YOU" - SECOND PERSON -
                ''' "THEM" - 3RD PERSON -
                ''' "NOBODY" NO PERSPECTIVE
                ''' </summary>
                Public Shared Function GetPersonalSubject(ByRef Userinput As String) As String
                    Dim mFirstPersonProNouns As List(Of String) = New List(Of String)({" I", " WE"})
                    Dim mSecondPersonProNouns As List(Of String) = New List(Of String)({" YOU", " US"})
                    Dim mThirdPersonProNouns As List(Of String) = New List(Of String)({" HE", " SHE", " IT"})
                    If DETECT_PERSPECTIVE(Userinput, mFirstPersonProNouns) = True Then
                        Return "ME"
                    Else
                        If DETECT_PERSPECTIVE(Userinput, mSecondPersonProNouns) = True Then
                            Return "YOU"
                        Else
                            If DETECT_PERSPECTIVE(Userinput, mThirdPersonProNouns) = True Then
                                Return "THEM"
                            Else
                                Return "NOBODY"
                            End If
                        End If
                    End If

                End Function

                ''' <summary>
                ''' The cases Of pronouns tell you how they are being used In a sentence.
                ''' "ME" - 1ST PERSON -
                ''' "YOU" - SECOND PERSON -
                ''' "THEM" - 3RD PERSON -
                ''' "NOBODY" NO PERSPECTIVE
                ''' </summary>
                Public Shared Function GetPersonalObject(ByRef Userinput As String) As String
                    Dim mFirstPersonProNouns As List(Of String) = New List(Of String)({"ME"})
                    Dim mSecondPersonProNouns As List(Of String) = New List(Of String)({"YOU", "US"})
                    Dim mThirdPersonProNouns As List(Of String) = New List(Of String)({" HIM", " HER", " IT"})
                    If DETECT_PERSPECTIVE(Userinput, mFirstPersonProNouns) = True Then
                        Return "ME"
                    Else
                        If DETECT_PERSPECTIVE(Userinput, mSecondPersonProNouns) = True Then
                            Return "YOU"
                        Else
                            If DETECT_PERSPECTIVE(Userinput, mThirdPersonProNouns) = True Then
                                Return "THEM"
                            Else
                                Return "NOBODY"
                            End If
                        End If
                    End If
                End Function

                ''' <summary>
                ''' The cases Of pronouns tell you how they are being used In a sentence.
                ''' "ME" - 1ST PERSON -
                ''' "YOU" - SECOND PERSON -
                ''' "THEM" - 3RD PERSON -
                ''' "NOBODY" NO PERSPECTIVE
                ''' </summary>
                Public Shared Function GetPersonalPossession(ByRef Userinput As String) As String
                    Dim mFirstPersonProNouns As List(Of String) = New List(Of String)({" MY", " MINE", " OUR", " OURS"})
                    Dim mSecondPersonProNouns As List(Of String) = New List(Of String)({" YOUR", " YOURS"})
                    Dim mThirdPersonProNouns As List(Of String) = New List(Of String)({" HIS", " HER", " HES", " HE IS", " HERS", " THEIR", " THEIRS"})
                    If DETECT_PERSPECTIVE(Userinput, mFirstPersonProNouns) = True Then
                        Return "ME"
                    Else
                        If DETECT_PERSPECTIVE(Userinput, mSecondPersonProNouns) = True Then
                            Return "YOU"
                        Else
                            If DETECT_PERSPECTIVE(Userinput, mThirdPersonProNouns) = True Then
                                Return "THEM"
                            Else
                                Return "NOBODY"
                            End If
                        End If
                    End If
                End Function

#End Region

            End Class

            ''' <summary>
            ''' counts occurrences of a specific phoneme
            ''' </summary>
            ''' <param name="strIn">  </param>
            ''' <param name="strFind"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            Public Shared Function CountOccurrences(ByRef strIn As String, ByRef strFind As String) As Integer
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



            ''' <summary>
            ''' For example, consider a simple sentence: "NLP information extraction is fun''. This could be tokenized into;
            '''    One-word (sometimes called unigram token) NLP, information, extraction, Is, fun
            '''    Two-word phrase (bigram tokens): NLP information, information extraction, extraction Is, Is fun, fun NLP
            '''    Three-word sentence (trigram tokens): NLP information extraction, information extraction Is, extraction Is fun
            ''' </summary>
            ''' <param name="Str">String to be analyzed</param>
            ''' <param name="Ngrams">Size of token component Group (Must be &gt; 0)</param>
            ''' <returns></returns>
            Public Shared Function GetNgrms(ByRef Str As String, ByRef Ngrams As Integer) As List(Of String)
                Dim NgramArr() As String = Split(Str, " ")
                Dim Length As Integer = NgramArr.Count
                Dim lst As New List(Of String)
                Dim Str2 As String = ""
                If Length - Ngrams > 0 Then

                    For i = 0 To Length - Ngrams
                        Str2 = ""
                        Dim builder As New System.Text.StringBuilder()
                        builder.Append(Str2)
                        For j = 0 To Ngrams - 1
                            builder.Append(NgramArr(i + j) & " ")
                        Next
                        Str2 = builder.ToString()
                        lst.Add(Str2)
                    Next
                Else

                End If
                Return lst
            End Function



            Public Shared Function GetRandomObjectFromList(ByRef Patterns As List(Of Object)) As Object
                Dim rnd = New Random()
                If Patterns.Count > 0 Then

                    Return Patterns(rnd.Next(0, Patterns.Count))
                Else
                    Return Nothing
                End If
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

        'Public Class MarkdownProcessor
        '    Private markdown As Markdown

        '    Public Sub New()
        '        markdown = New Markdown()
        '    End Sub

        '    Public Function ConvertToHtml(markdownText As String) As String
        '        Return markdown.Transform(markdownText)
        '    End Function
        'End Class

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
Namespace Utilitys
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

End Namespace
Public Module Extensions

    Public Function ConvertToDataTable(Of T)(ByVal list As IList(Of T)) As DataTable
        Dim table As New DataTable()
        Dim fields() = GetType(T).GetFields()
        For Each field In fields
            table.Columns.Add(field.Name, field.FieldType)
        Next
        For Each item As T In list
            Dim row As DataRow = table.NewRow()
            For Each field In fields
                row(field.Name) = field.GetValue(item)
            Next
            table.Rows.Add(row)
        Next
        Return table
    End Function

    Public Function CreateDataTable(ByRef HeaderTitles As List(Of String)) As DataTable
        Dim DT As New DataTable
        For Each item In HeaderTitles
            DT.Columns.Add(item, GetType(String))
        Next
        Return DT
    End Function


    <Serializable>
    Public Class Entity

        Public MemberOfEntityTypes As New List(Of EntityType)

        Public Sub New(entityName As String)
            Value = entityName
            MemberOfEntityTypes = New List(Of EntityType)
        End Sub
        Public ReadOnly Property EntityEncoding As Double
            Get
                Dim Count As Double = 0.0
                For Each item In MemberOfEntityTypes
                    Count += item.Value
                Next
                Return Count
            End Get
        End Property
        Public Property EndIndex As Integer
        Public Property StartIndex As Integer
        Public Property Value As String

        ' Deserialize the entity from JSON format
        Public Shared Function FromJson(json As String) As Entity
            Return JsonConvert.DeserializeObject(Of Entity)(json)
        End Function
        Public Function HasType(ByRef type As String) As Boolean
            For Each item In MemberOfEntityTypes
                If item.Type = type Then
                    Return True
                End If
            Next
            Return False
        End Function
        Public Sub AddEntityScore(entityType As EntityType)
            Dim Found As Boolean = False
            For Each item In MemberOfEntityTypes
                If item.Type = entityType.Type Then
                    'UpdateValue
                    item.Value = entityType.Value
                    Found = True
                End If
            Next
            If Found = False Then
                MemberOfEntityTypes.Add(entityType)
                Console.WriteLine($"Added value {Value} for {entityType} to {Value}'s value list.")
            Else

            End If
        End Sub
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
        ' Serialize the entity to JSON format
        Public Function ToJson() As String
            Return JsonConvert.SerializeObject(Me)
        End Function

    End Class
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
