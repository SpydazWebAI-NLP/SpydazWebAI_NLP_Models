Imports System.Text
Imports System.Text.RegularExpressions
Imports System.Web.Script.Serialization
Imports InputModelling.DataObjects
Imports InputModelling.Models.Embeddings.Text
Imports InputModelling.Models.EntityModel

Namespace Utilitys

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
    <ComClass(SentenceSplitter.ClassId, SentenceSplitter.InterfaceId, SentenceSplitter.EventsId)>
    Public Class SentenceSplitter
        Public Const ClassId As String = "28993390-7702-401C-BAB3-38FF97BC1AC9"
        Public Const EventsId As String = "CD334307-F53E-401A-AC6D-3CFDD86FD6F1"
        Public Const InterfaceId As String = "8B3345B1-5D13-4059-829B-B531310144B5"

        ''' <summary>
        ''' punctuation markers for end of sentences(individual thoughts) Set in order of Rank
        ''' </summary>
        Public Shared EndPunctuation() As String = {".", ";", "?", "!", ":"}

        ''' <summary>
        ''' Punctuation(known)
        ''' </summary>
        Public Shared Punctuation() As String = {".", ",", ";", "?", "!", ":", "$", "%", "^", "*", "<", ">",
"/", "@", "(", ")", "'""{", "}", "[", "]", "\", "|", "+", "=", "_", "-"}

        Private mSent As List(Of String)

        ''' <summary>
        ''' Provide text for sentence definition,
        ''' </summary>
        ''' <param name="Text"></param>
        Public Sub New(ByVal Text As String)
            mSent = SplitTextToSentences(Text)
        End Sub

        ''' <summary>
        ''' Returns number of sentences found
        ''' </summary>
        ''' <returns></returns>
        Public ReadOnly Property Count As Integer
            Get
                For Each Sent As String In Sentences
                    Count += 1

                Next
                Return Count
            End Get
        End Property

        Public ReadOnly Property Sentences As List(Of String)
            Get
                Return mSent
            End Get
        End Property

        ''' <summary>
        ''' Removes Trailing Spaces as well as double spaces from Text Also the Text is Capitalized
        ''' </summary>
        ''' <param name="Text"></param>
        ''' <returns></returns>
        Public Shared Function FormatText(ByRef Text As String) As String
            Dim FormatTextResponse As String = ""
            'FORMAT USERINPUT
            'turn to uppercase for searching the db
            Text = LTrim(Text)
            Text = RTrim(Text)
            Text = Text.Replace("  ", " ")
            FormatTextResponse = Text
            Return FormatTextResponse
        End Function

        ''' <summary>
        ''' finds sentences in text or phrase. based on EndPunctuation markers
        ''' </summary>
        ''' <param name="InputStr"></param>
        ''' <returns>Returns a list of sentences defined in the text</returns>
        Public Shared Function GetSentences(ByRef InputStr As String) As List(Of String)
            GetSentences = New List(Of String)
            Dim s As New SentenceSplitter(InputStr)
            For Each Sent As String In s.Sentences
                GetSentences.Add(Sent)
            Next
        End Function

        ''' <summary>
        ''' Removes Punctuation from Text
        ''' </summary>
        ''' <param name="Text"></param>
        ''' <returns>Cleaned Text</returns>
        Public Shared Function RemovePunctuation(ByVal Text As String) As String
            Dim mText As String = Text
            For Each item As String In Punctuation
                mText = mText.Replace(item, " ")
            Next
            mText = mText.Replace("  ", " ")
            Return mText
        End Function

        ''' <summary>
        ''' Splits Sentences by the Punctution offered. As it may be prudent to split by "." then
        ''' after by "," for sub components of the sentence etc
        ''' </summary>
        ''' <param name="mText">          text to be examined</param>
        ''' <param name="mEndPunctuation">Punctuation to be used as end marker</param>
        ''' <returns></returns>
        Public Shared Function SplitTextToSentences(ByVal mText As String, ByVal mEndPunctuation As String) As List(Of String)

            Dim Text As String = mText

            Text = Text.Replace(mEndPunctuation, "#")

            Dim TempSentencesArray() As String = Split(Text, "#")
            Dim mSentences As New List(Of String)
            For Each SentStr As String In TempSentencesArray
                If SentStr <> "" Then
                    mSentences.Add(FormatText(SentStr))
                End If

            Next

            Return mSentences
        End Function

        ''' <summary>
        ''' Splits to sentences based on all end markers in EndPunctuation
        ''' </summary>
        ''' <param name="mText"></param>
        ''' <returns></returns>
        Private Function SplitTextToSentences(ByVal mText As String) As List(Of String)

            Dim Text As String = mText
            For Each item As String In EndPunctuation
                Text = Text.Replace(item, "#")

            Next
            Dim TempSentencesArray() As String = Split(Text, "#")
            Dim mSentences As New List(Of String)
            For Each SentStr As String In TempSentencesArray
                If SentStr <> "" Then
                    mSentences.Add(FormatText(SentStr))
                End If

            Next

            Return mSentences
        End Function

    End Class

    Public Class ICollect

        Public Shared FemaleNames As List(Of String)

        Public Shared MaleNames As List(Of String)

        Public Shared ObjectNames As List(Of String)

        Private Shared commonQuestionHeaders As List(Of String)

        Private Shared iPronouns As List(Of String)

        '' Example entity list to search
        'Dim entityList As New List(Of String)() From {"dolphins"}
        Private Shared questionWords As List(Of String)

        Private Shared semanticPatterns As List(Of String)

        Private conclusions As List(Of String)

        Private hypotheses As List(Of String)

        Private patterns As Dictionary(Of String, String)

        Private premises As List(Of String)

        ''' <summary>
        ''' Returns all Pronouns in the model
        ''' </summary>
        ''' <returns></returns>
        Public Shared ReadOnly Property Pronouns As List(Of String)
            Get
                Dim Lst As New List(Of String)
                Lst.AddRange(MaleNames)
                Lst.AddRange(FemaleNames)
                Lst.AddRange(ObjectNames)
                Lst.AddRange(iPronouns)
                Return Lst.Distinct.ToList
            End Get
        End Property

        Public Shared Property bornInPattern As String = "\b([A-Z][a-z]+)\b relation \(born in\) \b([A-Z][a-z]+)\b"

        Public Shared Property datePattern As String = "\b\d{4}\b"

        Public Shared Property organizationPattern As String = "\b([A-Z][a-z]+)\b"

        ' Regular expression patterns for different entity types
        Public Shared Property personPattern As String = "\b([A-Z][a-z]+)\b"

        Public Shared Property programmingLanguagePattern As String = "\b[A-Z][a-z]+\.[a-z]+\b"

        Public Shared Property wroteBookPattern As String = "\b([A-Z][a-z]+)\b \(wrote a book called\) \b([A-Z][a-z]+)\b"

        Public Shared Function CaptureWordsWithContext(text As String, entityList As List(Of String), contextWords As Integer) As List(Of String)
            Dim words As String() = text.Split(" "c)
            Dim capturedWords As New List(Of String)()

            For i As Integer = 0 To words.Length - 1
                Dim word As String = words(i)
                If entityList.Contains(word) Then
                    Dim startIndex As Integer = Math.Max(0, i - contextWords)
                    Dim endIndex As Integer = Math.Min(words.Length - 1, i + contextWords)
                    Dim capturedWord As String = String.Join(" ", words, startIndex, endIndex - startIndex + 1)
                    capturedWords.Add(capturedWord)
                End If
            Next

            Return capturedWords
        End Function

        ''' <summary>
        ''' Detects entities in the given text.
        ''' </summary>
        ''' <param name="text">The text to be analyzed.</param>
        ''' <param name="EntityList">A list of entities to detect.</param>
        ''' <returns>A list of detected entities in the text.</returns>
        Public Shared Function Detect(ByRef text As String, ByRef EntityList As List(Of String)) As List(Of String)
            Dim Lst As New List(Of String)
            If text Is Nothing Then
                Throw New ArgumentNullException("text")
            End If

            If EntityList Is Nothing Then
                Throw New ArgumentNullException("EntityList")
            End If
            If EntityClassifier.Detect.DetectEntity(text, EntityList) = True Then
                For Each item In EntityList
                    If text.Contains(item) Then
                        Lst.Add(item)
                    End If
                Next
                Return Lst
            Else
                Return New List(Of String)
            End If
        End Function

        ''' <summary>
        ''' Attempts to find Unknown Names(pronouns) identified by thier capitalization
        ''' </summary>
        ''' <param name="words"></param>
        ''' <returns></returns>
        Public Shared Function DetectNamedEntities(ByVal words() As String) As List(Of String)
            Dim namedEntities As New List(Of String)

            For i = 0 To words.Length - 1
                Dim word = words(i)
                If Char.IsUpper(word(0)) AndAlso Not Pronouns.Contains(word.ToLower()) Then
                    namedEntities.Add(word)
                End If
            Next

            Return namedEntities
        End Function

        Public Shared Function ExtractAdjectivePhrases(taggedWords As List(Of KeyValuePair(Of String, String))) As List(Of String)
            Dim adjectivePhrases As New List(Of String)()

            Dim currentPhrase As String = ""
            Dim insideAdjectivePhrase As Boolean = False

            For Each taggedWord In taggedWords
                Dim word As String = taggedWord.Key
                Dim tag As String = taggedWord.Value

                If tag.StartsWith("JJ") Then ' Adjective tag
                    If insideAdjectivePhrase Then
                        currentPhrase += " " & word
                    Else
                        currentPhrase = word
                        insideAdjectivePhrase = True
                    End If
                Else
                    If insideAdjectivePhrase Then
                        adjectivePhrases.Add(currentPhrase)
                        insideAdjectivePhrase = False
                    End If
                End If
            Next

            ' Add the last phrase if it is an adjective phrase
            If insideAdjectivePhrase Then
                adjectivePhrases.Add(currentPhrase)
            End If

            Return adjectivePhrases
        End Function

        ''' <summary>
        ''' Extracts context Entitys , As Well As thier context words
        ''' </summary>
        ''' <param name="itext"></param>
        ''' <param name="contextSize"></param>
        ''' <param name="entities">Values to retrieve context for</param>
        ''' <returns></returns>
        Public Shared Function ExtractCapturedContextIntext(ByRef itext As String, ByVal contextSize As Integer, ByRef entities As List(Of String)) As List(Of CapturedContent)
            Dim wordsWithContext As New List(Of CapturedContent)

            ' Create a regular expression pattern for matching the entities
            Dim pattern As String = "(" + String.Join("|", entities.Select(Function(e) Regex.Escape(e))) + ")"

            ' Add context placeholders to the pattern
            Dim contextPattern As String = "(?:\S+\s+){" + contextSize.ToString() + "}"

            ' Combine the entity pattern and the context pattern
            pattern = contextPattern + "(" + pattern + ")" + contextPattern

            ' Find all matches in the text
            Dim matches As MatchCollection = Regex.Matches(itext, pattern)

            ' Iterate over the matches and extract the words with context
            For Each match As Match In matches
                Dim sequence As String = match.Value.Trim()
                Dim word As String = match.Groups(1).Value.Trim()
                Dim precedingContext As String = match.Groups(2).Value.Trim()
                Dim followingContext As String = match.Groups(3).Value.Trim()

                Dim precedingWords As List(Of String) = Split(precedingContext, " "c, StringSplitOptions.RemoveEmptyEntries).ToList
                Dim followingWords As List(Of String) = Split(followingContext, " "c, StringSplitOptions.RemoveEmptyEntries).ToList

                Dim capturedWord As New CapturedContent(word, precedingWords, followingWords)
                wordsWithContext.Add(capturedWord)
            Next

            Return wordsWithContext
        End Function

        Public Shared Function ExtractNounPhrases(taggedWords As List(Of KeyValuePair(Of String, String))) As List(Of String)
            Dim nounPhrases As New List(Of String)()

            Dim currentPhrase As String = ""
            Dim insideNounPhrase As Boolean = False

            For Each taggedWord In taggedWords
                Dim word As String = taggedWord.Key
                Dim tag As String = taggedWord.Value

                If tag.StartsWith("NN") Then ' Noun tag
                    If insideNounPhrase Then
                        currentPhrase += " " & word
                    Else
                        currentPhrase = word
                        insideNounPhrase = True
                    End If
                Else
                    If insideNounPhrase Then
                        nounPhrases.Add(currentPhrase)
                        insideNounPhrase = False
                    End If
                End If
            Next

            ' Add the last phrase if it is a noun phrase
            If insideNounPhrase Then
                nounPhrases.Add(currentPhrase)
            End If

            Return nounPhrases
        End Function

        ''' <summary>
        ''' Extracts patterns from the text and replaces detected entities with asterisks.
        ''' </summary>
        ''' <param name="text">The text to extract patterns from.</param>
        ''' <param name="EntityList">A list of entities to detect and replace.</param>
        ''' <returns>The extracted pattern with detected entities replaced by asterisks.</returns>
        Public Shared Function ExtractPattern(ByRef text As String, ByRef EntityList As List(Of String)) As String
            If text Is Nothing Then
                Throw New ArgumentNullException("text")
            End If

            If EntityList Is Nothing Then
                Throw New ArgumentNullException("EntityList")
            End If

            Dim Entitys As New List(Of String)
            Dim Str As String = text
            If EntityClassifier.Detect.DetectEntity(text, EntityList) = True Then
                Entitys = EntityClassifier.Detect.DetectEntitysInText(text, EntityList)

                Str = EntityClassifier.Discover.DiscoverShape(Str, Entitys)
                Str = EntityClassifier.Transform.TransformText(Str, Entitys)
            End If
            Return Str
        End Function

        Public Shared Function ExtractVerbPhrases(taggedWords As List(Of KeyValuePair(Of String, String))) As List(Of String)
            Dim verbPhrases As New List(Of String)()

            Dim currentPhrase As String = ""
            Dim insideVerbPhrase As Boolean = False

            For Each taggedWord In taggedWords
                Dim word As String = taggedWord.Key
                Dim tag As String = taggedWord.Value

                If tag.StartsWith("VB") Then ' Verb tag
                    If insideVerbPhrase Then
                        currentPhrase += " " & word
                    Else
                        currentPhrase = word
                        insideVerbPhrase = True
                    End If
                Else
                    If insideVerbPhrase Then
                        verbPhrases.Add(currentPhrase)
                        insideVerbPhrase = False
                    End If
                End If
            Next

            ' Add the last phrase if it is a verb phrase
            If insideVerbPhrase Then
                verbPhrases.Add(currentPhrase)
            End If

            Return verbPhrases
        End Function

        ''' <summary>
        ''' Returns a List of WordsWithCOntext with the Focus word at the center surrounded by its context words,
        ''' it can be a useful pattern chunk which can be used for prediction as a context ngram (min-3)
        ''' </summary>
        ''' <param name="text"></param>
        ''' <param name="focusTerm"></param>
        ''' <param name="precedingWordsCount"></param>
        ''' <param name="followingWordsCount"></param>
        ''' <returns></returns>
        Public Shared Function ExtractWordsWithContext(text As String, focusTerm As String, precedingWordsCount As Integer, followingWordsCount As Integer) As List(Of WordWithContext)
            Dim words As List(Of String) = text.Split(" "c).ToList()
            Dim focusIndex As Integer = words.IndexOf(focusTerm)

            Dim capturedWordsWithEntityContext As New List(Of WordWithContext)()

            If focusIndex <> -1 Then
                Dim startIndex As Integer = Math.Max(0, focusIndex - precedingWordsCount)
                Dim endIndex As Integer = Math.Min(words.Count - 1, focusIndex + followingWordsCount)

                For i As Integer = startIndex To endIndex
                    Dim word As String = words(i)

                    Dim wordWithContext As New WordWithContext() With {
                .word = word,
                      .IsFocusTerm = (i = focusIndex),
                .IsPreceding = (i < focusIndex),
                .IsFollowing = (i > focusIndex)
            }

                    capturedWordsWithEntityContext.Add(wordWithContext)
                Next
            End If

            Return capturedWordsWithEntityContext
        End Function

        ''' <summary>
        ''' Returns a new string with the Focus word at the center surrounded by its context words,
        ''' it can be a useful pattern chunk which can be used for prediction as a context ngram (min-3)
        ''' </summary>
        ''' <param name="text"></param>
        ''' <param name="Word"></param>
        ''' <param name="contextWords"></param>
        ''' <returns></returns>
        Public Shared Function ExtractWordWithContext(text As String, Word As String, contextWords As Integer) As String
            Dim words As String() = text.Split(" "c)

            Dim capturedWord As String = ""
            For i As Integer = 0 To words.Length - 1

                Dim Tword As String = words(i)
                If Word = Tword Then
                    Dim startIndex As Integer = Math.Max(0, i - contextWords)
                    Dim endIndex As Integer = Math.Min(words.Length - 1, i + contextWords)
                    capturedWord = String.Join(" ", words, startIndex, endIndex - startIndex + 1)

                End If
            Next

            Return capturedWord
        End Function

        '1. Objects:
        '   - Objects are typically referred to using nouns. Examples include "car," "book," "tree," "chair," "pen," etc.
        '   - Objects may have specific attributes or characteristics associated with them, such as color, size, shape, etc., which can be mentioned when referring to them.
        Public Shared Function FindAntecedent(words As String(), pronounIndex As Integer, entityList As List(Of String)) As String
            For i As Integer = pronounIndex - 1 To 0 Step -1
                Dim word As String = words(i)
                If entityList.Contains(word) Then
                    Return word
                End If
            Next

            Return ""
        End Function

        Public Shared Function FindNounPhrases(sentence As String) As List(Of String)
            Dim nounPhrases As New List(Of String)()

            ' Split the sentence into individual words
            Dim words() As String = sentence.Split({" "}, StringSplitOptions.RemoveEmptyEntries)

            ' Identify noun phrases
            For i As Integer = 0 To words.Length - 1
                If IsNoun(words(i)) Then
                    Dim nounPhrase As String = words(i)
                    Dim j As Integer = i + 1

                    ' Combine consecutive words until a non-noun word is encountered
                    While j < words.Length AndAlso IsNoun(words(j))
                        nounPhrase += " " & words(j)
                        j += 1
                    End While

                    nounPhrases.Add(nounPhrase)
                End If
            Next

            Return nounPhrases
        End Function

        Public Shared Function FindPhrases(sentence As String, phraseType As String) As List(Of String)
            Dim phrases As New List(Of String)()

            ' Split the sentence into individual words
            Dim words() As String = sentence.Split({" "}, StringSplitOptions.RemoveEmptyEntries)

            ' Identify phrases based on the specified type
            For i As Integer = 0 To words.Length - 1
                Dim currentWord As String = words(i)

                If (phraseType = "verb" AndAlso IsVerb(currentWord)) OrElse
           (phraseType = "adjective" AndAlso IsAdjective(currentWord)) Then

                    Dim phrase As String = currentWord
                    Dim j As Integer = i + 1

                    ' Combine consecutive words until a non-phrase word is encountered
                    While j < words.Length AndAlso (IsVerb(words(j)) OrElse IsAdjective(words(j)))
                        phrase += " " & words(j)
                        j += 1
                    End While

                    phrases.Add(phrase)
                End If
            Next

            Return phrases
        End Function

        Public Shared Function FindPhrases(taggedWords As List(Of KeyValuePair(Of String, String)), phraseType As String) As List(Of String)
            Dim phrases As New List(Of String)()

            ' Identify phrases based on the specified type
            For i As Integer = 0 To taggedWords.Count - 1
                Dim currentWord As String = taggedWords(i).Key
                Dim currentTag As String = taggedWords(i).Value

                If (phraseType = "verb" AndAlso IsVerbTag(currentTag)) OrElse
           (phraseType = "adjective" AndAlso IsAdjectiveTag(currentTag)) Then

                    Dim phrase As String = currentWord
                    Dim j As Integer = i + 1

                    ' Combine consecutive words until a non-phrase word is encountered
                    While j < taggedWords.Count AndAlso (IsVerbTag(taggedWords(j).Value) OrElse IsAdjectiveTag(taggedWords(j).Value))
                        phrase += " " & taggedWords(j).Key
                        j += 1
                    End While

                    phrases.Add(phrase)
                End If
            Next

            Return phrases
        End Function

        '2. Locations:
        '   - Locations are places or areas where entities or objects exist or are situated.
        '   - Locations can be referred to using nouns that represent specific places, such as "home," "office," "school," "park," "store," "gym," "library," etc.
        '   - Locations can also be described using adjectives or prepositional phrases, such as "in the backyard," "at the beach," "on the street," "near the river," etc.
        Public Shared Function GetPronoun(word As String) As String
            ' Add mapping of pronouns to words as needed
            Select Case word
                Case "he"
                    Return "him"
                Case "she"
                    Return "her"
                Case "it"
                    Return "its"
                Case "they"
                    Return "them"
                Case "them"
                    Return "them"
                Case "that"
                    Return "that"
                Case Else
                    Return ""
            End Select
        End Function

        '3. Antecedents:
        '   - Antecedents are the entities or objects that are referred to by pronouns or other referencing words in a sentence.
        '   - Antecedents are typically introduced in a sentence before the pronoun or referencing word. For example, "John went to the store. He bought some groceries."
        '   - Antecedents can be humans, objects, animals, or other entities. The choice of pronouns or referencing words depends on the gender and type of the antecedent. For example, "he" for a male, "she" for a female, "it" for an object, and "they" for multiple entities.
        ''' <summary>
        ''' Pronoun_mapping to normailized value
        ''' </summary>
        ''' <param name="word"></param>
        ''' <returns></returns>
        Public Shared Function GetPronounIndicator(word As String) As String
            ' Add mapping of pronouns to words as needed
            Select Case word
                Case "shes"
                    Return "her"
                Case "his"
                    Return "him"
                Case "hers"
                    Return "her"
                Case "her"
                    Return "her"
                Case "him"
                    Return "him"
                Case "he"
                    Return "him"
                Case "she"
                    Return "her"
                Case "its"
                    Return " it"
                Case "it"
                    Return " it"
                Case "they"
                    Return "them"
                Case "thats"
                    Return "that"
                Case "that"
                    Return "that"
                Case "we"
                    Return "we"
                Case "us"
                    Return "us"
                Case "them"
                    Return "them"
                Case Else
                    Return ""
            End Select
        End Function

        Public Shared Function IsAdjective(word As String) As Boolean
            ' Add your own adjective identification logic here
            ' This is a basic example that checks if the word ends with "ly"
            Return word.EndsWith("ly")
        End Function

        Public Shared Function IsAdjectiveTag(tag As String) As Boolean
            ' Add your own adjective tag identification logic here
            ' This is a basic example that checks if the tag starts with "JJ"
            Return tag.StartsWith("JJ")
        End Function

        Public Shared Function IsAntecedentIndicator(ByVal token As String) As Boolean
            ' List of antecedent indicator words
            Dim antecedentIndicators As String() = {" he", "she", "it", "they", "them", "that", "him", "we", "us", "its", "his", "thats"}

            ' Check if the token is an antecedent indicator
            Return antecedentIndicators.Contains(token.ToLower())
        End Function

        Public Shared Function IsConclusion(ByVal sentence As String) As Boolean
            ' List of indicator phrases for conclusions
            Dim conclusionIndicators As String() = {"therefore", "thus", "consequently", "hence", "in conclusion"}

            ' Check if any of the conclusion indicators are present in the sentence
            For Each indicator In conclusionIndicators
                If sentence.Contains(indicator) Then
                    Return True
                End If
            Next

            Return False
        End Function

        Public Shared Function IsEntityOrPronoun(word As String, ByRef Entitys As List(Of String)) As Boolean
            Dim AntecedantIdentifers() As String = {" he ", "she", "him", "her", "it", "them", "they", "that", "we"}

            ' 1.For simplicity, let's assume any word ending with "s" is a noun/pronoun
            ' 2.For simplicity, let's assume any word referring to a person is a pronoun
            Dim lowerCaseWord As String = word.ToLower()
            For Each item In Entitys
                If item.ToLower = lowerCaseWord Then Return True

            Next
            For Each item In AntecedantIdentifers
                If item.ToLower = lowerCaseWord Then Return True
            Next
            Return False
        End Function

        Public Shared Function IsFemaleNounOrPronoun(ByVal word As String) As Boolean
            Dim ifemaleNouns() As String = {"she", "her", "hers", "shes"}
            Return FemaleNames.Contains(word.ToLower()) OrElse FemaleNames.Contains(word.ToLower() & "s") OrElse ifemaleNouns.Contains(word.ToLower)
        End Function

        ''' <summary>
        ''' female names
        ''' </summary>
        ''' <param name="pronoun"></param>
        ''' <returns></returns>
        Public Shared Function IsFemalePronoun(pronoun As String) As Boolean
            Dim lowerCasePronoun As String = pronoun.ToLower()
            Return lowerCasePronoun = "her" OrElse lowerCasePronoun = "she" OrElse lowerCasePronoun = "hers" OrElse FemaleNames.Contains(pronoun)
        End Function

        Public Shared Function IsMaleNounOrPronoun(ByVal word As String) As Boolean
            Dim imaleNouns() As String = {"him", " he", "his", ""}
            Return MaleNames.Contains(word.ToLower()) OrElse imaleNouns.Contains(word.ToLower)
        End Function

        ''' <summary>
        ''' Malenames
        ''' </summary>
        ''' <param name="pronoun"></param>
        ''' <returns></returns>
        Public Shared Function IsMalePronoun(pronoun As String) As Boolean
            Dim lowerCasePronoun As String = pronoun.ToLower()
            Return lowerCasePronoun = " he" OrElse lowerCasePronoun = "him" OrElse lowerCasePronoun = " his" OrElse MaleNames.Contains(pronoun)
        End Function

        Public Shared Function IsNoun(word As String) As Boolean
            ' Add your own noun identification logic here
            ' You can check for patterns, word lists, or use external resources for more accurate noun detection
            ' This is a basic example that only checks for the first letter being uppercase
            Return Char.IsUpper(word(0))
        End Function

        Public Shared Function IsObjectPronoun(ByVal word As String) As Boolean
            Dim iObjectNames() As String = {"its", "it", "that", "thats"}

            Return iObjectNames.Contains(word.ToLower()) OrElse iObjectNames.Contains(word.ToLower() & "s")
        End Function

        'Possible Output: "The person associated with John is..."
        Public Shared Function IsPersonName(word As String) As Boolean
            ' Implement your custom logic to determine if a word is a person name
            ' Return true if the word is a person name, false otherwise

            ' Example: Check if the word starts with an uppercase letter
            Return Char.IsUpper(word(0))
        End Function

        Public Shared Function IsPremise(ByVal sentence As String) As Boolean
            ' List of indicator phrases for premises
            Dim premiseIndicators As String() = {"based on", "according to", "given", "assuming", "since"}

            ' Check if any of the premise indicators are present in the sentence
            For Each indicator In premiseIndicators
                If sentence.Contains(indicator) Then
                    Return True
                End If
            Next

            Return False
        End Function

        Public Shared Function IsProperNoun(word As String) As Boolean
            ' Implement your custom logic to determine if a word is a proper noun
            ' Return true if the word is a proper noun, false otherwise

            ' Example: Check if the word starts with an uppercase letter
            Return Char.IsUpper(word(0))
        End Function

        Public Shared Function IsQuestion(sentence As String) As Boolean
            ' Preprocess the sentence
            sentence = sentence.ToLower().Trim()

            ' Check for question words
            If StartsWithAny(sentence, questionWords) Then
                Return True
            End If

            ' Check for question marks
            If sentence.EndsWith("?") Then
                Return True
            End If

            ' Check for semantic patterns
            Dim patternRegex As New Regex(String.Join("|", semanticPatterns))
            If patternRegex.IsMatch(sentence) Then
                Return True
            End If

            ' Check for common question headers
            If StartsWithAny(sentence, commonQuestionHeaders) Then
                Return True
            End If

            ' No matching question pattern found
            Return False
        End Function

        Public Shared Function IsVerb(word As String) As Boolean
            ' Add your own verb identification logic here
            ' This is a basic example that checks if the word ends with "ing"
            Return word.EndsWith("ing")
        End Function

        Public Shared Function IsVerbTag(tag As String) As Boolean
            ' Add your own verb tag identification logic here
            ' This is a basic example that checks if the tag starts with "V"
            Return tag.StartsWith("V")
        End Function

        Public Shared Function MatchesAnswerShape(sentence As String, answerShapes As List(Of String)) As Boolean

            ' Check if the sentence matches any of the answer shapes using regex pattern matching
            For Each answerShape In answerShapes
                Dim pattern As String = "\b" + Regex.Escape(answerShape) + "\b"
                If Regex.IsMatch(sentence, pattern, RegexOptions.IgnoreCase) Then
                    Return True
                End If
            Next

            Return False
        End Function

        ' Identify antecedent indicators:
        '   - Look for pronouns or referencing words like "he," "she," "it," "they," "them," "that" in the sentence.
        '   - Check the preceding tokens to identify the most recent entity token with a matching type.
        '   - Use the identified entity as the antecedent indicator.
        ''' <summary>
        ''' finds pronoun antecedants in the text a replaces them with thier names
        ''' </summary>
        ''' <param name="sentence"></param>
        ''' <param name="entityList"></param>
        ''' <returns></returns>
        Public Shared Function ResolveCoreference(sentence As String, entityList As List(Of String)) As String
            Dim words As String() = sentence.Split(" ")

            For i As Integer = 0 To words.Length - 1
                Dim word As String = words(i)
                If entityList.Contains(word) Then
                    Dim pronoun As String = GetPronoun(word)
                    Dim antecedent As String = FindAntecedent(words, i, entityList)
                    If Not String.IsNullOrEmpty(antecedent) Then
                        sentence = sentence.Replace(pronoun, antecedent)
                    End If
                End If
            Next

            Return sentence
        End Function

        Public Function DetectGender(name As String) As String
            ' For simplicity, let's assume any name starting with a vowel is female, and the rest are male
            If IsObjectPronoun(name) Then
                Return "Object"
            ElseIf IsMaleNounOrPronoun(name) Then
                Return "Male"
            ElseIf IsFemaleNounOrPronoun(name) Then
                Return "Female"
            Else
                Return "Unknown"
            End If
        End Function

        ''' <summary>
        ''' Given an Answer shape , Detect and
        ''' Extract All Answers and context sentences
        ''' </summary>
        ''' <param name="text"></param>
        ''' <param name="answerShapes"></param>
        ''' <param name="contextSentences"></param>
        ''' <returns></returns>
        Public Function ExtractAnswersWithContextFromText(text As String, answerShapes As List(Of String), contextSentences As Integer) As List(Of String)
            Dim answers As New List(Of String)()

            ' Split the text into sentences
            Dim sentences As String() = Split(text, ".", StringSplitOptions.RemoveEmptyEntries)

            ' Iterate through each sentence and check for potential answer sentences
            For i As Integer = 0 To sentences.Length - 1
                Dim sentence As String = sentences(i).Trim()

                ' Check if the sentence matches any of the answer shapes
                If MatchesAnswerShape(sentence, answerShapes) Then
                    ' Add the current sentence and the context sentences to the list of potential answer sentences
                    Dim startIndex As Integer = Math.Max(0, i - contextSentences)
                    Dim endIndex As Integer = Math.Min(i + contextSentences, sentences.Length - 1)

                    Dim answer As String = String.Join(" ", sentences, startIndex, endIndex - startIndex + 1).Trim()
                    answers.Add(answer)
                End If
            Next

            Return answers
        End Function

        ''' <summary>
        ''' catches words , context etc by entitys and concat context chunk
        ''' </summary>
        ''' <param name="text"></param>
        ''' <param name="entities"></param>
        ''' <param name="contextSize"></param>
        ''' <returns>complex object</returns>
        Public Function ExtractCapturedContextMatchesInTextByContext(ByVal text As String, ByVal entities As List(Of String),
                                                            ByVal contextSize As Integer) As List(Of (Word As String, PrecedingWords As List(Of String), FollowingWords As List(Of String), Position As Integer))

            Dim wordsWithContext As New List(Of (Word As String, PrecedingWords As List(Of String), FollowingWords As List(Of String), Position As Integer))

            ' Create a regular expression pattern for matching the entities
            Dim pattern As String = "(" + String.Join("|", entities.Select(Function(e) Regex.Escape(e))) + ")"

            ' Add context placeholders to the pattern
            Dim contextPattern As String = "(?:\S+\s+){" + contextSize.ToString() + "}"

            ' Combine the entity pattern and the context pattern
            pattern = contextPattern + "(" + pattern + ")" + contextPattern

            ' Find all matches in the text
            Dim matches As MatchCollection = Regex.Matches(text, pattern)

            ' Iterate over the matches and extract the words with context and position
            For Each match As Match In matches
                Dim wordWithContext As String = match.Groups(0).Value.Trim()
                Dim word As String = match.Groups(1).Value.Trim()
                Dim precedingContext As String = match.Groups(2).Value.Trim()
                Dim followingContext As String = match.Groups(3).Value.Trim()
                Dim position As Integer = match.Index

                Dim precedingWords As List(Of String) = Split(precedingContext, " "c, StringSplitOptions.RemoveEmptyEntries).ToList()
                Dim followingWords As List(Of String) = Split(followingContext, " "c, StringSplitOptions.RemoveEmptyEntries).ToList()

                wordsWithContext.Add((word, precedingWords, followingWords, position))
            Next

            Return wordsWithContext
        End Function

        ''' <summary>
        ''' Creates A context item based item the inputs and searches for matches
        ''' returning the item plus its context and entitys etc
        ''' </summary>
        ''' <param name="itext"></param>
        ''' <param name="contextSize"></param>
        ''' <param name="entities"></param>
        ''' <returns></returns>
        Public Function ExtractCapturedWordsinTextByContext(ByRef itext As String, ByVal contextSize As Integer, ByRef entities As List(Of String)) As List(Of CapturedWord)
            Dim wordsWithContext As New List(Of CapturedWord)()

            ' Create a regular expression pattern for matching the entities
            Dim pattern As String = "(" + String.Join("|", entities.Select(Function(e) Regex.Escape(e))) + ")"

            ' Add context placeholders to the pattern
            Dim contextPattern As String = "(?:\S+\s+){" + contextSize.ToString() + "}"

            ' Combine the entity pattern and the context pattern
            pattern = contextPattern + "(" + pattern + ")" + contextPattern

            ' Find all matches in the text
            Dim matches As MatchCollection = Regex.Matches(itext, pattern)

            ' Iterate over the matches and extract the words with context
            For Each match As Match In matches
                Dim sequence As String = match.Value.Trim()
                Dim word As String = match.Groups(1).Value.Trim()
                Dim precedingContext As String = match.Groups(2).Value.Trim()
                Dim followingContext As String = match.Groups(3).Value.Trim()

                Dim precedingWords As List(Of String) = Split(precedingContext, " "c, StringSplitOptions.RemoveEmptyEntries).ToList
                Dim followingWords As List(Of String) = Split(followingContext, " "c, StringSplitOptions.RemoveEmptyEntries).ToList

                Dim capturedWord As New CapturedWord(word, precedingWords, followingWords, "", "")
                wordsWithContext.Add(capturedWord)
            Next

            Return wordsWithContext
        End Function

        Public Function ExtractPotentialAnswers(text As String, resolvedAntecedents As List(Of String), resolvedEntities As List(Of String)) As List(Of String)
            Dim answers As New List(Of String)()

            ' Split the text into sentences
            Dim sentences As String() = Split(text, ".", StringSplitOptions.RemoveEmptyEntries)

            ' Iterate through each sentence and check for potential answer sentences
            For Each sentence In sentences
                ' Check if the sentence contains any of the resolved antecedents or entities
                If ContainsResolvedAntecedentsOrEntities(sentence, resolvedAntecedents, resolvedEntities) Then

                    ' Add the sentence to the list of potential answer sentences
                    answers.Add(sentence.Trim())
                End If
            Next

            Return answers
        End Function

        ''' <summary>
        ''' captures sentences ending in ?
        ''' </summary>
        ''' <param name="input">text</param>
        ''' <returns>list of recognized question sentences</returns>
        Function ExtractSimpleQuestions(input As String) As List(Of String)
            ' Regular expression pattern to match questions
            Dim questionPattern As String = "([\w\s']+)(\?)"

            ' List to store extracted questions
            Dim questions As New List(Of String)()

            ' Match the pattern in the input text
            Dim regex As New Regex(questionPattern)
            Dim matches As MatchCollection = regex.Matches(input)

            ' Iterate over the matches and extract the questions
            For Each match As Match In matches
                Dim question As String = match.Groups(1).Value.Trim()
                questions.Add(question)
            Next

            Return questions
        End Function

        '3. **Regular Expressions**:
        '   - Input text: "John wrote a book called 'The Adventures of Tom'."
        '   - Pattern: "[A-Z][a-z]+ wrote a book called '[A-Z][a-z]+'"
        '   - Expected output: ["John wrote a book called 'The Adventures of Tom'"]
        Public Function ExtractUsingRegexSearchPattern(text As String, pattern As String) As List(Of String)
            Dim relationships As New List(Of String)()

            ' Use regular expression to match the desired pattern
            Dim matches As MatchCollection = Regex.Matches(text, pattern, RegexOptions.IgnoreCase)

            ' Extract the matched relationships and add them to the list
            For Each match As Match In matches
                Dim relationship As String = match.Value
                relationships.Add(relationship)
            Next

            Return relationships
        End Function

        Public Function FindNearestAntecedent(ByVal pronounIndex As Integer, ByVal words() As String, ByVal entities As Dictionary(Of String, String)) As String
            Dim antecedent As String = ""

            ' Search for nearest preceding noun phrase as antecedent
            For i = pronounIndex - 1 To 0 Step -1
                If entities.ContainsKey(words(i)) Then
                    antecedent = entities(words(i))
                    Exit For
                End If
            Next

            Return antecedent
        End Function

        'Dim answer As String = GenerateAnswer(answerType, entity)
        'Console.WriteLine(answer)
        Public Function GenerateAnswer(answerType As String, entity As String) As String
            ' Implement your custom answer generation logic here
            ' Generate an answer based on the answer type and entity

            Dim answer As String = ""

            ' Example answer generation logic
            Select Case answerType
                Case "location"
                    answer = "The location of " & entity & " is [LOCATION]."
                Case "person"
                    answer = "The person associated with " & entity & " is [PERSON]."
                Case "organization"
                    answer = "The organization " & entity & " is [ORGANIZATION]."
                Case "date"
                    answer = "The date related to " & entity & " is [DATE]."
                Case "year"
                    answer = "The year associated with " & entity & " is [YEAR]."
                Case "language"
                    answer = "The programming language " & entity & " is [LANGUAGE]."
                Case "country"
                    answer = "The country associated with " & entity & " is [COUNTRY]."
                Case Else
                    answer = "The information about " & entity & " is [INFORMATION]."
            End Select

            Return answer
        End Function

        'Possible Output: "Who is person(John)?"
        Public Function GenerateQuestionFromEntity(entity As String) As String
            ' Implement your custom question generation logic here
            ' Generate a question based on the given entity

            Dim question As String = ""

            ' Example question generation logic
            If entity.StartsWith("person") Then
                question = "Who is " & entity & "?"
            ElseIf entity.StartsWith("organization") Then
                question = "What is " & entity & "?"
            ElseIf entity.StartsWith("location") Then
                question = "Where is " & entity & "?"
            Else
                question = "What can you tell me about " & entity & "?"
            End If

            Return question
        End Function

        Public Function GenerateRandomAntecedent(entityLists As Dictionary(Of String, List(Of String))) As String
            Dim random As New Random()
            Dim entityTypes As List(Of String) = New List(Of String)(entityLists.Keys)
            Dim entityType As String = entityTypes(random.Next(entityTypes.Count))
            Dim entities As List(Of String) = entityLists(entityType)

            Return entities(random.Next(entities.Count))
        End Function

        ''' <summary>
        ''' Returns All ProNouns detected from this model
        ''' </summary>
        ''' <param name="words"></param>
        ''' <returns></returns>
        Public Function GetPronounsInText(ByVal words() As String) As List(Of String)
            Dim namedEntities As New List(Of String)

            For i = 0 To words.Length - 1
                Dim word = words(i)
                If Char.IsUpper(word(0)) AndAlso Not Pronouns.Contains(word.ToLower()) Then
                    namedEntities.Add(word)
                End If
            Next

            Return namedEntities
        End Function

        Public Function IdentifyAntecedent(ByVal sentence As String, ByVal entityLists As Dictionary(Of String, List(Of String))) As String
            ' Tokenize the sentence
            Dim tokens As String() = sentence.Split(" "c)

            ' Iterate through the tokens
            For i As Integer = tokens.Length - 1 To 0 Step -1
                Dim token As String = tokens(i)

                ' Iterate through the entity lists
                For Each entityType As String In entityLists.Keys
                    ' Check if the token matches an entity in the current entity list
                    If entityLists(entityType).Contains(token) Then
                        ' Check for antecedent indicators
                        If i > 0 AndAlso IsAntecedentIndicator(tokens(i - 1)) Then
                            ' Return the identified antecedent with its type
                            Return $"{token} ({entityType})"
                        End If
                    End If
                Next
            Next

            ' Return empty string if no antecedent indicator is found
            Return ""
        End Function

        Public Function IsEntity(ByRef Word As String, ByRef Entitys As List(Of String)) As Boolean
            For Each item In Entitys
                If Word = item Then
                    Return True
                End If
            Next
            Return False
        End Function

        Public Function ReplaceTagsInSentence(sentence As String, taggedEntities As Dictionary(Of String, String)) As String
            ' Implement your custom rule-based tag replacement logic here
            ' Analyze the sentence and replace tagged entities with their corresponding tags

            ' Example tag replacement logic
            For Each taggedEntity As KeyValuePair(Of String, String) In taggedEntities
                Dim entity As String = taggedEntity.Key
                Dim tag As String = taggedEntity.Value

                ' Replace the tagged entity with its tag in the sentence
                sentence = sentence.Replace(entity, tag)
            Next

            Return sentence
        End Function

        ''' <summary>
        ''' Enabling to find the antecedant for given entitys or pronouns
        ''' </summary>
        ''' <param name="sentence"></param>
        ''' <param name="Entitys"></param>
        ''' <returns></returns>
        Public Function ResolveAntecedant(sentence As String, ByRef Entitys As List(Of String)) As String

            ' Tokenize the sentence into words
            Dim words = Split(sentence, " ")
            ' Find the position of the pronoun in the sentence
            Dim pronounIndex As Integer = 0
            For Each Pronoun In Entitys

                For Each item In words
                    pronounIndex += 1
                    If item = Pronoun Then
                        Exit For
                    End If
                Next
                If pronounIndex = -1 Then
                    Return "Unknown."
                End If

            Next

            ' Start from the pronoun position and search for antecedents before and after the pronoun
            Dim antecedent As String = ""

            ' Search for antecedents before the pronoun
            For i As Integer = pronounIndex - 1 To 0 Step -1
                Dim currentWord As String = words(i)

                ' Check if the current word is a Entity or a pronoun
                If IsEntityOrPronoun(currentWord, Entitys) Then
                    antecedent = currentWord
                    Exit For
                End If
            Next

            ' If no antecedent is found before the pronoun, search for antecedents after the pronoun
            If antecedent = "" Then
                For i As Integer = pronounIndex + 1 To words.Length - 1
                    Dim currentWord As String = words(i)

                    ' Check if the current word is a Entity or a pronoun
                    If IsEntityOrPronoun(currentWord, Entitys) Then
                        antecedent = currentWord
                        Exit For
                    End If
                Next
            End If

            ' If no antecedent is found, return an appropriate message
            If antecedent = "" Then
                Return "No antecedent found for the pronoun."
            End If

            Return antecedent
        End Function

        ''' <summary>
        ''' Given a name / entity it attempts to find the antcedant
        ''' </summary>
        ''' <param name="sentence"></param>
        ''' <param name="pronoun"></param>
        ''' <returns></returns>
        Public Function ResolveAntecedant(sentence As String, pronoun As String) As String
            ' Tokenize the sentence into words
            Dim words = Split(sentence, " ")

            ' Find the position of the pronoun in the sentence
            Dim pronounIndex As Integer = 0
            For Each item In words
                pronounIndex += 1
                If item = pronoun Then
                    Exit For
                End If
            Next
            If pronounIndex = -1 Then
                Return "Unknown."
            End If

            ' Start from the pronoun position and search for antecedents before and after the pronoun
            Dim antecedent As String = ""

            ' Search for antecedents before the pronoun
            If IsObjectPronoun(pronoun) Then
                ' If pronoun is an object pronoun, no need to look for antecedents
                Return pronoun
            ElseIf IsMalePronoun(pronoun) Then
                ' If pronoun is a male pronoun, search for male antecedents
                For i As Integer = pronounIndex - 1 To 0 Step -1
                    Dim currentWord As String = words(i)

                    ' Check if the current word is a noun or a pronoun
                    If IsMaleNounOrPronoun(currentWord) Then
                        antecedent = currentWord
                        Exit For
                    End If
                Next
            Else
                ' If pronoun is a female pronoun, search for female antecedents
                For i As Integer = pronounIndex - 1 To 0 Step -1
                    Dim currentWord As String = words(i)

                    ' Check if the current word is a noun or a pronoun
                    If IsFemaleNounOrPronoun(currentWord) Then
                        antecedent = currentWord
                        Exit For
                    End If
                Next
            End If

            ' If no antecedent is found, return an appropriate message
            If antecedent = "" Then
                Return "No antecedent found for the pronoun."
            End If

            Return antecedent
        End Function

        Public Function ResolveCoReferences(ByRef iText As String, ByRef entities As Dictionary(Of String, String)) As Dictionary(Of String, String)
            Dim coReferences As New Dictionary(Of String, String)()

            Dim sentences() As String = Split(iText, "."c, StringSplitOptions.RemoveEmptyEntries)

            For Each sentence In sentences
                Dim words() As String = Split(sentence.Trim, " "c, StringSplitOptions.RemoveEmptyEntries)

                ' Identify pronouns and assign antecedents
                For i = 0 To words.Length - 1
                    Dim word = words(i)
                    If Pronouns.Contains(word.ToLower()) Then
                        Dim antecedent = FindNearestAntecedent(i, words, entities)
                        coReferences.Add(word, antecedent)
                    End If
                Next

                ' Identify named entities and update entities dictionary
                Dim namedEntities = DetectNamedEntities(words)
                For Each namedEntity In namedEntities
                    If Not entities.ContainsKey(namedEntity) Then
                        entities.Add(namedEntity, namedEntity)
                    End If
                Next
            Next

            Return coReferences
        End Function

        Public Function TagSentence(sentence As String) As List(Of String)
            ' Implement your custom rule-based sentence tagging logic here
            ' Analyze the sentence and assign tags to words or phrases

            Dim taggedSentence As New List(Of String)()

            ' Example sentence tagging logic
            Dim words() As String = sentence.Split(" "c)
            For Each word As String In words
                Dim tag As String = ""

                ' Assign tags based on some criteria (e.g., part-of-speech, named entity)
                If IsProperNoun(word) Then
                    tag = "NNP" ' Proper noun
                ElseIf IsVerb(word) Then
                    tag = "VB" ' Verb
                ElseIf IsAdjective(word) Then
                    tag = "JJ" ' Adjective
                Else
                    tag = "NN" ' Noun
                End If

                ' Create a tagged word (word/tag) and add it to the tagged sentence
                Dim taggedWord As String = $"{word}/{tag}"
                taggedSentence.Add(taggedWord)
            Next

            Return taggedSentence
        End Function

        Private Shared Function ContainsResolvedAntecedentsOrEntities(sentence As String, resolvedAntecedents As List(Of String), resolvedEntities As List(Of String)) As Boolean
            ' Check if the sentence contains any of the resolved antecedents or entities
            For Each antecedent In resolvedAntecedents
                If sentence.Contains(antecedent) Then
                    Return True
                End If
            Next

            For Each entity In resolvedEntities
                If sentence.Contains(entity) Then
                    Return True
                End If
            Next

            Return False
        End Function

        Public Class iSearch

            ' Extract email addresses from text
            Public Shared Function ExtractEmailAddresses(text As String) As List(Of String)
                Dim emailRegex As New Regex("\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
                Dim emailAddresses As New List(Of String)()

                Dim matches As MatchCollection = emailRegex.Matches(text)

                For Each match As Match In matches
                    emailAddresses.Add(match.Value)
                Next

                Return emailAddresses
            End Function

            ' Information Extraction Functions
            ' Extract phone numbers from text
            Public Shared Function ExtractPhoneNumbers(text As String) As List(Of String)
                Dim phoneRegex As New Regex("\b\d{3}-\d{3}-\d{4}\b")
                Dim phoneNumbers As New List(Of String)()

                Dim matches As MatchCollection = phoneRegex.Matches(text)

                For Each match As Match In matches
                    phoneNumbers.Add(match.Value)
                Next

                Return phoneNumbers
            End Function

            '```
        End Class

    End Class



    Namespace NN
        Public Class Perceptron

            Public Property Weights As Double() ' The weights of the perceptron

            Private Function Sigmoid(x As Double) As Double ' The sigmoid activation function

                Return 1 / (1 + Math.Exp(-x))
            End Function

            ''' <summary>
            ''' the step function rarely performs well except in some rare cases with (0,1)-encoded
            ''' binary data.
            ''' </summary>
            ''' <param name="Value"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            Private Shared Function BinaryThreshold(ByRef Value As Double) As Double

                ' Z = Bias+ (Input*Weight)
                'TransferFunction
                'If Z > 0 then Y = 1
                'If Z < 0 then y = 0

                Return If(Value < 0 = True, 0, 1)
            End Function



            Public Sub New(NumberOfInputs As Integer) ' Constructor that initializes the weights and bias of the perceptron
                CreateWeights(NumberOfInputs)

            End Sub

            Public Sub CreateWeights(NumberOfInputs As Integer) ' Constructor that initializes the weights and bias of the perceptron
                Weights = New Double(NumberOfInputs - 1) {}
                For i As Integer = 0 To NumberOfInputs - 1
                    Weights(i) = Rnd(1.0)
                Next

            End Sub

            ' Function to calculate output
            Public Function ForwardLinear(inputs As Double()) As Integer
                Dim sum = 0.0

                ' Loop through inputs and calculate sum of weights times inputs
                For i = 0 To inputs.Length - 1
                    sum += inputs(i)
                Next

                Return sum
            End Function
            Public Function Forward(inputs As Double()) As Integer
                Dim sum = 0.0

                ' Loop through inputs and calculate sum of weights times inputs
                For i = 0 To inputs.Length - 1
                    sum += Weights(i) * inputs(i)
                Next

                Return sum
            End Function
            Public Function ForwardSigmoid(inputs As Double()) As Double ' Compute the output of the perceptron given an input
                CreateWeights(inputs.Count)
                Dim sum As Double = 0
                'Collect the sum of the inputs * Weight
                For i As Integer = 0 To inputs.Length - 1
                    sum += inputs(i) * Weights(i)
                Next

                'Activate
                'We Return the sigmoid of the sum to produce the output
                Return Sigmoid(sum)
            End Function

            Public Function ForwardBinaryThreshold(inputs As Double()) As Double ' Compute the output of the perceptron given an input
                CreateWeights(inputs.Count)
                Dim sum As Double = 0 ' used to hold the output

                'Collect the sum of the inputs * Weight
                For i As Integer = 0 To inputs.Length - 1
                    sum += inputs(i) * Weights(i)
                Next

                'Activate
                'We Return the sigmoid of the sum to produce the output , Applying the Binary threshold funciton to it
                Return BinaryThreshold(Sigmoid(sum))
            End Function

            ' Function to train the perceptron
            Public Sub Train(inputs As Double(), desiredOutput As Integer, threshold As Double, MaxEpochs As Integer, LearningRate As Double)
                Dim guess = Forward(inputs)
                Dim nError As Integer = 0
                Dim CurrentEpoch = 0

                Do Until threshold < nError Or
                        CurrentEpoch = MaxEpochs
                    CurrentEpoch += 1

                    nError = desiredOutput - guess

                    ' Loop through inputs and update weights based on error and learning rate
                    For i = 0 To inputs.Length - 1
                        _Weights(i) += LearningRate * nError * inputs(i)
                    Next

                Loop

            End Sub

        End Class
        ''' <summary>
        ''' These are the options of transfer functions available to the network
        ''' This is used to select which function to be used:
        ''' The derivative function can also be selected using this as a marker
        ''' </summary>
        Public Enum TransferFunctionType
            none
            sigmoid
            HyperbolTangent
            BinaryThreshold
            RectifiedLinear
            Logistic
            StochasticBinary
            Gaussian
            Signum
        End Enum

        ''' <summary>
        ''' Transfer Function used in the calculation of the following layer
        ''' </summary>
        Public Structure TransferFunction

            ''' <summary>
            ''' Returns a result from the transfer function indicated ; Non Derivative
            ''' </summary>
            ''' <param name="TransferFunct">Indicator for Transfer function selection</param>
            ''' <param name="Input">Input value for node/Neuron</param>
            ''' <returns>result</returns>
            Public Shared Function EvaluateTransferFunct(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
                EvaluateTransferFunct = 0
                Select Case TransferFunct
                    Case TransferFunctionType.none
                        Return Input
                    Case TransferFunctionType.sigmoid
                        Return Sigmoid(Input)
                    Case TransferFunctionType.HyperbolTangent
                        Return HyperbolicTangent(Input)
                    Case TransferFunctionType.BinaryThreshold
                        Return BinaryThreshold(Input)
                    Case TransferFunctionType.RectifiedLinear
                        Return RectifiedLinear(Input)
                    Case TransferFunctionType.Logistic
                        Return Logistic(Input)
                    Case TransferFunctionType.Gaussian
                        Return Gaussian(Input)
                    Case TransferFunctionType.Signum
                        Return Signum(Input)
                End Select
            End Function

            ''' <summary>
            ''' Returns a result from the transfer function indicated ; Non Derivative
            ''' </summary>
            ''' <param name="TransferFunct">Indicator for Transfer function selection</param>
            ''' <param name="Input">Input value for node/Neuron</param>
            ''' <returns>result</returns>
            Public Shared Function EvaluateTransferFunctionDerivative(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
                EvaluateTransferFunctionDerivative = 0
                Select Case TransferFunct
                    Case TransferFunctionType.none
                        Return Input
                    Case TransferFunctionType.sigmoid
                        Return SigmoidDerivitive(Input)
                    Case TransferFunctionType.HyperbolTangent
                        Return HyperbolicTangentDerivative(Input)
                    Case TransferFunctionType.Logistic
                        Return LogisticDerivative(Input)
                    Case TransferFunctionType.Gaussian
                        Return GaussianDerivative(Input)
                End Select
            End Function

            ''' <summary>
            ''' the step function rarely performs well except in some rare cases with (0,1)-encoded
            ''' binary data.
            ''' </summary>
            ''' <param name="Value"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            Private Shared Function BinaryThreshold(ByRef Value As Double) As Double

                ' Z = Bias+ (Input*Weight)
                'TransferFunction
                'If Z > 0 then Y = 1
                'If Z < 0 then y = 0

                Return If(Value < 0 = True, 0, 1)
            End Function

            Private Shared Function Gaussian(ByRef x As Double) As Double
                Gaussian = Math.Exp((-x * -x) / 2)
            End Function

            Private Shared Function GaussianDerivative(ByRef x As Double) As Double
                GaussianDerivative = Gaussian(x) * (-x / (-x * -x))
            End Function

            Private Shared Function HyperbolicTangent(ByRef Value As Double) As Double
                ' TanH(x) = (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x))

                Return Math.Tanh(Value)
            End Function

            Private Shared Function HyperbolicTangentDerivative(ByRef Value As Double) As Double
                HyperbolicTangentDerivative = 1 - (HyperbolicTangent(Value) * HyperbolicTangent(Value)) * Value
            End Function

            'Linear Neurons
            ''' <summary>
            ''' in a liner neuron the weight(s) represent unknown values to be determined the
            ''' outputs could represent the known values of a meal and the inputs the items in the
            ''' meal and the weights the prices of the individual items There are no hidden layers
            ''' </summary>
            ''' <remarks>
            ''' answers are determined by determining the weights of the linear neurons the delta
            ''' rule is used as the learning rule: Weight = Learning rate * Input * LocalError of neuron
            ''' </remarks>
            Private Shared Function Linear(ByRef value As Double) As Double
                ' Output = Bias + (Input*Weight)
                Return value
            End Function

            'Non Linear neurons
            Private Shared Function Logistic(ByRef Value As Double) As Double
                'z = bias + (sum of all inputs ) * (input*weight)
                'output = Sigmoid(z)
                'derivative input = z/weight
                'derivative Weight = z/input
                'Derivative output = output*(1-Output)
                'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

                Return 1 / 1 + Math.Exp(-Value)
            End Function

            Private Shared Function LogisticDerivative(ByRef Value As Double) As Double
                'z = bias + (sum of all inputs ) * (input*weight)
                'output = Sigmoid(z)
                'derivative input = z/weight
                'derivative Weight = z/input
                'Derivative output = output*(1-Output)
                'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

                Return Logistic(Value) * (1 - Logistic(Value))
            End Function

            Private Shared Function RectifiedLinear(ByRef Value As Double) As Double
                'z = B + (input*Weight)
                'If Z > 0 then output = z
                'If Z < 0 then output = 0
                If Value < 0 = True Then

                    Return 0
                Else
                    Return Value
                End If
            End Function

            ''' <summary>
            ''' the log-sigmoid function constrains results to the range (0,1), the function is
            ''' sometimes said to be a squashing function in neural network literature. It is the
            ''' non-linear characteristics of the log-sigmoid function (and other similar activation
            ''' functions) that allow neural networks to model complex data.
            ''' </summary>
            ''' <param name="Value"></param>
            ''' <returns></returns>
            ''' <remarks>1 / (1 + Math.Exp(-Value))</remarks>
            Private Shared Function Sigmoid(ByRef Value As Integer) As Double
                'z = Bias + (Input*Weight)
                'Output = 1/1+e**z
                Return 1 / (1 + Math.Exp(-Value))
            End Function

            Private Shared Function SigmoidDerivitive(ByRef Value As Integer) As Double
                Return Sigmoid(Value) * (1 - Sigmoid(Value))
            End Function

            Private Shared Function Signum(ByRef Value As Integer) As Double
                'z = Bias + (Input*Weight)
                'Output = 1/1+e**z
                Return Math.Sign(Value)
            End Function

            Private Shared Function StochasticBinary(ByRef value As Double) As Double
                'Uncreated
                Return value
            End Function

        End Structure
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
                Dim vectorizer As New Sentence2Vector()
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


        Public Class Summarise

            Public Function GenerateSummary(ByRef Text As String, ByRef Entitys As List(Of String)) As String
                ' Step 5: Generate the summary
                Return String.Join(vbNewLine, ExtractImportantSentencesInText(Text, Entitys, True, 2))
            End Function

            Public Function GenerateSummary(ByVal text As String, ByVal entities As List(Of String), ByVal numContextSentencesBefore As Integer, ByVal numContextSentencesAfter As Integer) As String
                ' Extract important sentences with context
                Dim importantSentences As List(Of String) = ExtractImportantSentencesInText(text, entities, numContextSentencesBefore, numContextSentencesAfter)

                ' Generate the summary
                Dim summary As String = String.Join(". ", importantSentences)

                Return summary
            End Function

            ''' <summary>
            ''' Searches for important sentences in text , identified by the presence of an entity from this list
            ''' These lists can be specific to a particular topic or entity or a search query
            ''' </summary>
            ''' <param name="Text"></param>
            ''' <param name="EntityList">Entity list</param>
            ''' <param name="WithContext"></param>
            ''' <param name="NumberOfContextSentences"></param>
            ''' <returns></returns>
            Public Function ExtractImportantSentencesInText(ByRef Text As String,
                                                           EntityList As List(Of String),
                                                           Optional WithContext As Boolean = False,
                                                Optional NumberOfContextSentences As Integer = 0) As List(Of String)
                Dim Sents As New List(Of String)

                Select Case WithContext
                    Case False

                        For Each Sent In Split(Text, ".")
                            For Each Entity In EntityList
                                If Sent.Contains(Entity) Then
                                    Sents.Add(Sent)
                                End If
                            Next

                        Next
                        Return Sents.Distinct.ToList
                    Case True

                        For Each Sent In Split(Text, ".")
                            For Each Entity In EntityList
                                If Sent.ToLower.Contains(Entity.ToLower) Then
                                    Sents.AddRange(ExtractContextSentences(Text, Sent, NumberOfContextSentences))
                                End If
                            Next

                        Next
                        Return Sents.Distinct.ToList
                End Select

                Return Sents.Distinct.ToList
            End Function

            ''' <summary>
            ''' grabs important sentences from text based on the entity list provided .
            ''' (values or terms or noun phrases or verb phrases) as this is a sentence level search
            ''' it also grabs the context sentences surrounding it based on the inputs
            ''' </summary>
            ''' <param name="text"></param>
            ''' <param name="entityList"></param>
            ''' <param name="numContextSentencesBefore"></param>
            ''' <param name="numContextSentencesAfter"></param>
            ''' <returns></returns>
            Public Function ExtractImportantSentencesInText(ByVal text As String, ByVal entityList As List(Of String), ByVal numContextSentencesBefore As Integer, ByVal numContextSentencesAfter As Integer) As List(Of String)
                Dim importantSentences As New List(Of String)

                For Each sentence In text.Split("."c)
                    For Each entity In entityList
                        If sentence.ToLower.Contains(entity.ToLower) Then
                            ' Add the current sentence and the context sentences
                            importantSentences.AddRange(ExtractContextSentences(text, sentence, numContextSentencesBefore, numContextSentencesAfter))
                            Exit For ' Break out of the inner loop if the entity is found in the sentence
                        End If
                    Next
                Next

                Return importantSentences.Distinct().ToList()
            End Function

            ''' <summary>
            ''' Gets important Sentences in text with or without context
            ''' </summary>
            ''' <param name="Text"></param>
            ''' <param name="EntityList"></param>
            ''' <param name="WithContext"></param>
            ''' <param name="NumberOfContextSentencesBefore"></param>
            ''' <param name="NumberOfContextSentencesAfter"></param>
            ''' <returns></returns>
            Public Function ExtractImportantSentencesInText(ByRef Text As String, EntityList As List(Of String), Optional WithContext As Boolean = False,
                                                Optional NumberOfContextSentencesBefore As Integer = 0,
                                                Optional NumberOfContextSentencesAfter As Integer = 0) As List(Of String)
                Dim importantSentences As New List(Of String)

                For Each sentence In Split(Text, ".")
                    For Each entity In EntityList
                        If sentence.ToLower.Contains(entity.ToLower) Then
                            importantSentences.Add(sentence)
                            Exit For ' Break out of the inner loop if the entity is found in the sentence
                        End If
                    Next
                Next

                If WithContext Then
                    Dim sentencesWithContext As New List(Of String)
                    For Each sentence In importantSentences
                        sentencesWithContext.AddRange(ExtractContextSentences(Text, sentence, NumberOfContextSentencesBefore, NumberOfContextSentencesAfter))
                    Next
                    Return sentencesWithContext
                Else
                    Return importantSentences
                End If
            End Function

            ''' <summary>
            ''' Given an important Sentence Extract its surrounding context Sentences
            ''' </summary>
            ''' <param name="Text"></param>
            ''' <param name="ImportantSentence">Important Sentence to match</param>
            ''' <param name="ConTextInt">Number of Sentences Either Side</param>
            ''' <returns></returns>
            Public Function ExtractContextSentences(ByRef Text As String, ByRef ImportantSentence As String, ByRef ConTextInt As Integer) As List(Of String)
                Dim ContextSentences As New List(Of String)
                Dim CurrentSentences As New List(Of String)
                Dim Count As Integer = 0

                For Each Sent In Split(Text, ".")
                    CurrentSentences.Add(Sent)
                    Count += 1
                    If Sent = ImportantSentence Then
                        'Get Previous sentences

                        For i = 0 To ConTextInt
                            Dim Index = Count - 1
                            If Index >= 0 Or Index < CurrentSentences.Count Then

                                ContextSentences.Add(CurrentSentences(Index))

                            End If
                        Next
                        ContextSentences.Add(ImportantSentence)
                        'GetFollowing Sentences
                        For i = 0 To ConTextInt
                            If Count + i < CurrentSentences.Count Then
                                ContextSentences.Add(CurrentSentences(Count + i))
                            End If
                        Next
                    End If
                Next
                Return ContextSentences
            End Function

            ''' <summary>
            ''' Given an important Sentence Extract its surrounding context Sentences -
            ''' In some cases it may be prudent to grab only a single sentence before and multiple sentences after
            ''' important to know which context is important in which instance
            ''' </summary>
            ''' <param name="text">Document</param>
            ''' <param name="importantSentence">Sentence to be matched</param>
            ''' <param name="numContextSentencesBefore">number of</param>
            ''' <param name="numContextSentencesAfter">number of</param>
            ''' <returns></returns>
            Public Function ExtractContextSentences(ByVal text As String, ByVal importantSentence As String, ByVal numContextSentencesBefore As Integer, ByVal numContextSentencesAfter As Integer) As List(Of String)
                Dim contextSentences As New List(Of String)
                Dim allSentences As List(Of String) = text.Split("."c).ToList()
                Dim sentenceIndex As Integer = allSentences.IndexOf(importantSentence)

                ' Get sentences before the important sentence
                Dim startIndex As Integer = Math.Max(0, sentenceIndex - numContextSentencesBefore)
                For i = startIndex To sentenceIndex - 1
                    contextSentences.Add(allSentences(i))
                Next

                ' Add the important sentence
                contextSentences.Add(importantSentence)

                ' Get sentences after the important sentence
                Dim endIndex As Integer = Math.Min(sentenceIndex + numContextSentencesAfter, allSentences.Count - 1)
                For i = sentenceIndex + 1 To endIndex
                    contextSentences.Add(allSentences(i))
                Next

                Return contextSentences
            End Function

            Public Function GenerateTextFromEntities(entities As List(Of String), storedSentences As List(Of String)) As String
                ' Implement your custom text generation logic here
                ' Generate text using the entities and stored sentences

                Dim generatedText As String = ""

                ' Example text generation logic
                For Each entity As String In entities
                    Dim matchingSentences As List(Of String) = FindSentencesWithEntity(entity, storedSentences)

                    ' Randomly select a sentence from the matching sentences
                    Dim random As New Random()
                    Dim selectedSentence As String = matchingSentences(random.Next(0, matchingSentences.Count))

                    ' Replace the entity tag with the actual entity in the selected sentence
                    Dim generatedSentence As String = selectedSentence.Replace(entity, "<<" & entity & ">>")

                    ' Append the generated sentence to the generated text
                    generatedText &= generatedSentence & " "
                Next

                Return generatedText.Trim()
            End Function

            Public Function FindSentencesWithEntity(entity As String, storedSentences As List(Of String)) As List(Of String)
                ' Implement your custom logic to find sentences that contain the given entity
                ' Return a list of sentences that match the entity

                Dim matchingSentences As New List(Of String)

                ' Example logic: Check if the entity appears in each stored sentence
                For Each sentence As String In storedSentences
                    If sentence.Contains(entity) Then
                        matchingSentences.Add(sentence)
                    End If
                Next

                Return matchingSentences
            End Function

        End Class
        ''' <summary>
        ''' TO USE THE PROGRAM CALL THE FUNCTION PORTERALGORITHM. THE WORD
        ''' TO BE STEMMED SHOULD BE PASSED AS THE ARGUEMENT ARGUEMENT. THE STRING
        ''' RETURNED BY THE FUNCTION IS THE STEMMED WORD
        ''' Porter Stemmer. It follows the algorithm definition
        ''' presented in :
        '''   Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14,
        '''   no. 3, pp 130-137,
        ''' </summary>
        Public Class WordStemmer

            '   (http://www.tartarus.org/~martin/PorterStemmer)

            'Author : Navonil Mustafee
            'Brunel University - student
            'Algorithm Implemented as part for assignment on document visualization

            Public Shared Function StemWord(str As String) As String

                'only strings greater than 2 are stemmed
                If Len(Trim(str)) > 0 Then
                    str = porterAlgorithmStep1(str)
                    str = porterAlgorithmStep2(str)
                    str = porterAlgorithmStep3(str)
                    str = porterAlgorithmStep4(str)
                    str = porterAlgorithmStep5(str)
                End If

                'End of Porter's algorithm.........returning the word
                StemWord = str

            End Function

            Private Shared Function porterAlgorithmStep1(str As String) As String

                On Error Resume Next

                'STEP 1A
                '
                '    SSES -> SS                         caresses  ->  caress
                '    IES  -> I                          ponies    ->  poni
                '                                       ties      ->  ti
                '    SS   -> SS                         caress    ->  caress
                '    S    ->                            cats      ->  cat

                'declaring local variables
                Dim i As Byte
                Dim j As Byte
                Dim step1a(3, 1) As String

                'initializing contents of 2D array
                step1a(0, 0) = "sses"
                step1a(0, 1) = "ss"
                step1a(1, 0) = "ies"
                step1a(1, 1) = "i"
                step1a(2, 0) = "ss"
                step1a(2, 1) = "ss"
                step1a(3, 0) = "s"
                step1a(3, 1) = ""

                'checking word
                For i = 0 To 3 Step 1
                    If porterEndsWith(str, step1a(i, 0)) Then
                        str = porterTrimEnd(str, Len(step1a(i, 0)))
                        str = porterAppendEnd(str, step1a(i, 1))
                        Exit For
                    End If
                Next i

                '--------------------------------------------------------------------------------------------------------

                'STEP 1B
                '
                '   If
                '       (m>0) EED -> EE                     feed      ->  feed
                '                                           agreed    ->  agree
                '   Else
                '       (*v*) ED  ->                        plastered ->  plaster
                '                                           bled      ->  bled
                '       (*v*) ING ->                        motoring  ->  motor
                '                                           sing      ->  sing
                '
                'If the second or third of the rules in Step 1b is successful, the following
                'is done:
                '
                '    AT -> ATE                       conflat(ed)  ->  conflate
                '    BL -> BLE                       troubl(ed)   ->  trouble
                '    IZ -> IZE                       siz(ed)      ->  size
                '    (*d and not (*L or *S or *Z))
                '       -> single letter
                '                                    hopp(ing)    ->  hop
                '                                    tann(ed)     ->  tan
                '                                    fall(ing)    ->  fall
                '                                    hiss(ing)    ->  hiss
                '                                    fizz(ed)     ->  fizz
                '    (m=1 and *o) -> E               fail(ing)    ->  fail
                '                                    fil(ing)     ->  file
                '
                'The rule to map to a single letter causes the removal of one of the double
                'letter pair. The -E is put back on -AT, -BL and -IZ, so that the suffixes
                '-ATE, -BLE and -IZE can be recognised later. This E may be removed in step
                '4.

                'declaring local variables
                Dim m As Byte
                Dim temp As String
                Dim second_third_success As Boolean

                'initializing contents of 2D array
                second_third_success = False

                '(m>0) EED -> EE..else..(*v*) ED  ->(*v*) ING  ->
                If porterEndsWith(str, "eed") Then

                    'counting the number of m's
                    temp = porterTrimEnd(str, Len("eed"))
                    m = porterCountm(temp)

                    If m > 0 Then
                        str = porterTrimEnd(str, Len("eed"))
                        str = porterAppendEnd(str, "ee")
                    End If

                ElseIf porterEndsWith(str, "ed") Then

                    'trim and check for vowel
                    temp = porterTrimEnd(str, Len("ed"))

                    If porterContainsVowel(temp) Then
                        str = porterTrimEnd(str, Len("ed"))
                        second_third_success = True
                    End If

                ElseIf porterEndsWith(str, "ing") Then

                    'trim and check for vowel
                    temp = porterTrimEnd(str, Len("ing"))

                    If porterContainsVowel(temp) Then
                        str = porterTrimEnd(str, Len("ing"))
                        second_third_success = True
                    End If

                End If

                'If the second or third of the rules in Step 1b is SUCCESSFUL, the following
                'is done:
                '
                '    AT -> ATE                       conflat(ed)  ->  conflate
                '    BL -> BLE                       troubl(ed)   ->  trouble
                '    IZ -> IZE                       siz(ed)      ->  size
                '    (*d and not (*L or *S or *Z))
                '       -> single letter
                '                                    hopp(ing)    ->  hop
                '                                    tann(ed)     ->  tan
                '                                    fall(ing)    ->  fall
                '                                    hiss(ing)    ->  hiss
                '                                    fizz(ed)     ->  fizz
                '    (m=1 and *o) -> E               fail(ing)    ->  fail
                '                                    fil(ing)     ->  file

                If second_third_success = True Then             'If the second or third of the rules in Step 1b is SUCCESSFUL

                    If porterEndsWith(str, "at") Then           'AT -> ATE
                        str = porterTrimEnd(str, Len("at"))
                        str = porterAppendEnd(str, "ate")
                    ElseIf porterEndsWith(str, "bl") Then       'BL -> BLE
                        str = porterTrimEnd(str, Len("bl"))
                        str = porterAppendEnd(str, "ble")
                    ElseIf porterEndsWith(str, "iz") Then       'IZ -> IZE
                        str = porterTrimEnd(str, Len("iz"))
                        str = porterAppendEnd(str, "ize")
                    ElseIf porterEndsDoubleConsonent(str) Then  '(*d and not (*L or *S or *Z))-> single letter
                        If Not (porterEndsWith(str, "l") Or porterEndsWith(str, "s") Or porterEndsWith(str, "z")) Then
                            str = porterTrimEnd(str, 1)
                        End If
                    ElseIf porterCountm(str) = 1 Then                           '(m=1 and *o) -> E
                        If porterEndsCVC(str) Then
                            str = porterAppendEnd(str, "e")
                        End If
                    End If

                End If

                '--------------------------------------------------------------------------------------------------------
                '
                'STEP 1C
                '
                '    (*v*) Y -> I                    happy        ->  happi
                '                                    sky          ->  sky

                If porterEndsWith(str, "y") Then

                    'trim and check for vowel
                    temp = porterTrimEnd(str, 1)

                    If porterContainsVowel(temp) Then
                        str = porterTrimEnd(str, Len("y"))
                        str = porterAppendEnd(str, "i")
                    End If

                End If

                'retuning the word
                porterAlgorithmStep1 = str

            End Function

            Private Shared Function porterAlgorithmStep2(str As String) As String

                On Error Resume Next

                'STEP 2
                '
                '    (m>0) ATIONAL ->  ATE           relational     ->  relate
                '    (m>0) TIONAL  ->  TION          conditional    ->  condition
                '                                    rational       ->  rational
                '    (m>0) ENCI    ->  ENCE          valenci        ->  valence
                '    (m>0) ANCI    ->  ANCE          hesitanci      ->  hesitance
                '    (m>0) IZER    ->  IZE           digitizer      ->  digitize
                'Also,
                '    (m>0) BLI    ->   BLE           conformabli    ->  conformable
                '
                '    (m>0) ALLI    ->  AL            radicalli      ->  radical
                '    (m>0) ENTLI   ->  ENT           differentli    ->  different
                '    (m>0) ELI     ->  E             vileli        - >  vile
                '    (m>0) OUSLI   ->  OUS           analogousli    ->  analogous
                '    (m>0) IZATION ->  IZE           vietnamization ->  vietnamize
                '    (m>0) ATION   ->  ATE           predication    ->  predicate
                '    (m>0) ATOR    ->  ATE           operator       ->  operate
                '    (m>0) ALISM   ->  AL            feudalism      ->  feudal
                '    (m>0) IVENESS ->  IVE           decisiveness   ->  decisive
                '    (m>0) FULNESS ->  FUL           hopefulness    ->  hopeful
                '    (m>0) OUSNESS ->  OUS           callousness    ->  callous
                '    (m>0) ALITI   ->  AL            formaliti      ->  formal
                '    (m>0) IVITI   ->  IVE           sensitiviti    ->  sensitive
                '    (m>0) BILITI  ->  BLE           sensibiliti    ->  sensible
                'Also,
                '    (m>0) LOGI    ->  LOG           apologi        -> apolog
                '
                'The test for the string S1 can be made fast by doing a program switch on
                'the penultimate letter of the word being tested. This gives a fairly even
                'breakdown of the possible values of the string S1. It will be seen in fact
                'that the S1-strings in step 2 are presented here in the alphabetical order
                'of their penultimate letter. Similar techniques may be applied in the other
                'steps.

                'declaring local variables
                Dim step2(20, 1) As String
                Dim i As Byte
                Dim temp As String

                'initializing contents of 2D array
                step2(0, 0) = "ational"
                step2(0, 1) = "ate"
                step2(1, 0) = "tional"
                step2(1, 1) = "tion"
                step2(2, 0) = "enci"
                step2(2, 1) = "ence"
                step2(3, 0) = "anci"
                step2(3, 1) = "ance"
                step2(4, 0) = "izer"
                step2(4, 1) = "ize"
                step2(5, 0) = "bli"
                step2(5, 1) = "ble"
                step2(6, 0) = "alli"
                step2(6, 1) = "al"
                step2(7, 0) = "entli"
                step2(7, 1) = "ent"
                step2(8, 0) = "eli"
                step2(8, 1) = "e"
                step2(9, 0) = "ousli"
                step2(9, 1) = "ous"
                step2(10, 0) = "ization"
                step2(10, 1) = "ize"
                step2(11, 0) = "ation"
                step2(11, 1) = "ate"
                step2(12, 0) = "ator"
                step2(12, 1) = "ate"
                step2(13, 0) = "alism"
                step2(13, 1) = "al"
                step2(14, 0) = "iveness"
                step2(14, 1) = "ive"
                step2(15, 0) = "fulness"
                step2(15, 1) = "ful"
                step2(16, 0) = "ousness"
                step2(16, 1) = "ous"
                step2(17, 0) = "aliti"
                step2(17, 1) = "al"
                step2(18, 0) = "iviti"
                step2(18, 1) = "ive"
                step2(19, 0) = "biliti"
                step2(19, 1) = "ble"
                step2(20, 0) = "logi"
                step2(20, 1) = "log"

                'checking word
                For i = 0 To 20 Step 1
                    If porterEndsWith(str, step2(i, 0)) Then
                        temp = porterTrimEnd(str, Len(step2(i, 0)))
                        If porterCountm(temp) > 0 Then
                            str = porterTrimEnd(str, Len(step2(i, 0)))
                            str = porterAppendEnd(str, step2(i, 1))
                        End If
                        Exit For
                    End If
                Next i

                'retuning the word
                porterAlgorithmStep2 = str

            End Function

            Private Shared Function porterAlgorithmStep3(str As String) As String

                On Error Resume Next

                'STEP 3
                '
                '    (m>0) ICATE ->  IC              triplicate     ->  triplic
                '    (m>0) ATIVE ->                  formative      ->  form
                '    (m>0) ALIZE ->  AL              formalize      ->  formal
                '    (m>0) ICITI ->  IC              electriciti    ->  electric
                '    (m>0) ICAL  ->  IC              electrical     ->  electric
                '    (m>0) FUL   ->                  hopeful        ->  hope
                '    (m>0) NESS  ->                  goodness       ->  good

                'declaring local variables
                Dim i As Byte
                Dim temp As String
                Dim step3(6, 1) As String

                'initializing contents of 2D array
                step3(0, 0) = "icate"
                step3(0, 1) = "ic"
                step3(1, 0) = "ative"
                step3(1, 1) = ""
                step3(2, 0) = "alize"
                step3(2, 1) = "al"
                step3(3, 0) = "iciti"
                step3(3, 1) = "ic"
                step3(4, 0) = "ical"
                step3(4, 1) = "ic"
                step3(5, 0) = "ful"
                step3(5, 1) = ""
                step3(6, 0) = "ness"
                step3(6, 1) = ""

                'checking word
                For i = 0 To 6 Step 1
                    If porterEndsWith(str, step3(i, 0)) Then
                        temp = porterTrimEnd(str, Len(step3(i, 0)))
                        If porterCountm(temp) > 0 Then
                            str = porterTrimEnd(str, Len(step3(i, 0)))
                            str = porterAppendEnd(str, step3(i, 1))
                        End If
                        Exit For
                    End If
                Next i

                'retuning the word
                porterAlgorithmStep3 = str

            End Function

            Private Shared Function porterAlgorithmStep4(str As String) As String

                On Error Resume Next

                'STEP 4
                '
                '    (m>1) AL    ->                  revival        ->  reviv
                '    (m>1) ANCE  ->                  allowance      ->  allow
                '    (m>1) ENCE  ->                  inference      ->  infer
                '    (m>1) ER    ->                  airliner       ->  airlin
                '    (m>1) IC    ->                  gyroscopic     ->  gyroscop
                '    (m>1) ABLE  ->                  adjustable     ->  adjust
                '    (m>1) IBLE  ->                  defensible     ->  defens
                '    (m>1) ANT   ->                  irritant       ->  irrit
                '    (m>1) EMENT ->                  replacement    ->  replac
                '    (m>1) MENT  ->                  adjustment     ->  adjust
                '    (m>1) ENT   ->                  dependent      ->  depend
                '    (m>1 and (*S or *T)) ION ->     adoption       ->  adopt
                '    (m>1) OU    ->                  homologou      ->  homolog
                '    (m>1) ISM   ->                  communism      ->  commun
                '    (m>1) ATE   ->                  activate       ->  activ
                '    (m>1) ITI   ->                  angulariti     ->  angular
                '    (m>1) OUS   ->                  homologous     ->  homolog
                '    (m>1) IVE   ->                  effective      ->  effect
                '    (m>1) IZE   ->                  bowdlerize     ->  bowdler
                '
                'The suffixes are now removed. All that remains is a little tidying up.

                'declaring local variables
                Dim i As Byte
                Dim temp As String
                Dim step4(18) As String

                'initializing contents of 2D array
                step4(0) = "al"
                step4(1) = "ance"
                step4(2) = "ence"
                step4(3) = "er"
                step4(4) = "ic"
                step4(5) = "able"
                step4(6) = "ible"
                step4(7) = "ant"
                step4(8) = "ement"
                step4(9) = "ment"
                step4(10) = "ent"
                step4(11) = "ion"
                step4(12) = "ou"
                step4(13) = "ism"
                step4(14) = "ate"
                step4(15) = "iti"
                step4(16) = "ous"
                step4(17) = "ive"
                step4(18) = "ize"

                'checking word
                For i = 0 To 18 Step 1

                    If porterEndsWith(str, step4(i)) Then

                        temp = porterTrimEnd(str, Len(step4(i)))

                        If porterCountm(temp) > 1 Then

                            If porterEndsWith(str, "ion") Then
                                If porterEndsWith(temp, "s") Or porterEndsWith(temp, "t") Then
                                    str = porterTrimEnd(str, Len(step4(i)))
                                    str = porterAppendEnd(str, "")
                                End If
                            Else
                                str = porterTrimEnd(str, Len(step4(i)))
                                str = porterAppendEnd(str, "")
                            End If

                        End If

                        Exit For

                    End If

                Next i

                'retuning the word
                porterAlgorithmStep4 = str

            End Function

            Private Shared Function porterAlgorithmStep5(str As String) As String

                On Error Resume Next

                'STEP 5a
                '
                '    (m>1) E     ->                  probate        ->  probat
                '                                    rate           ->  rate
                '    (m=1 and not *o) E ->           cease          ->  ceas
                '
                'STEP 5b
                '
                '    (m>1 and *d and *L) -> single letter
                '                                    controll       ->  control
                '                                    roll           ->  roll

                'declaring local variables
                Dim i As Byte
                Dim temp As String

                'Step5a
                If porterEndsWith(str, "e") Then            'word ends with e
                    temp = porterTrimEnd(str, 1)
                    If porterCountm(temp) > 1 Then          'm>1
                        str = porterTrimEnd(str, 1)
                    ElseIf porterCountm(temp) = 1 Then      'm=1
                        If Not porterEndsCVC(temp) Then     'not *o
                            str = porterTrimEnd(str, 1)
                        End If
                    End If
                End If

                '--------------------------------------------------------------------------------------------------------
                '
                'Step5b
                If porterCountm(str) > 1 Then
                    If porterEndsDoubleConsonent(str) And porterEndsWith(str, "l") Then
                        str = porterTrimEnd(str, 1)
                    End If
                End If

                'retuning the word
                porterAlgorithmStep5 = str

            End Function

            Private Shared Function porterAppendEnd(str As String, ends As String) As String

                On Error Resume Next

                'returning the appended string
                porterAppendEnd = str + ends

            End Function

            Private Shared Function porterContains(str As String, present As String) As Boolean

                On Error Resume Next

                'checking whether strr contains present
                porterContains = If(InStr(str, present) = 0, False, True)

            End Function

            Private Shared Function porterContainsVowel(str As String) As Boolean

                'checking word to see if vowels are present

                Dim pattern As String

                If Len(str) >= 0 Then

                    'find out the CVC pattern
                    pattern = returnCVCpattern(str)

                    'check to see if the return pattern contains a vowel
                    porterContainsVowel = If(InStr(pattern, "v") = 0, False, True)
                Else
                    porterContainsVowel = False
                End If

            End Function

            Private Shared Function porterCountm(str As String) As Byte

                On Error Resume Next

                'A \consonant\ in a word is a letter other than A, E, I, O or U, and other
                'than Y preceded by a consonant. (The fact that the term `consonant' is
                'defined to some extent in terms of itself does not make it ambiguous.) So in
                'TOY the consonants are T and Y, and in SYZYGY they are S, Z and G. If a
                'letter is not a consonant it is a \vowel\.

                'declaring local variables
                Dim chars() As Byte
                Dim const_vowel As String
                Dim i As Byte
                Dim m As Byte
                Dim flag As Boolean
                Dim pattern As String

                'initializing
                const_vowel = ""
                m = 0
                flag = False

                If Not Len(str) = 0 Then

                    'find out the CVC pattern
                    pattern = returnCVCpattern(str)

                    'converting const_vowel to byte array
                    chars = System.Text.Encoding.Unicode.GetBytes(pattern)

                    'counting the number of m's...
                    For i = 0 To UBound(chars) Step 1
                        If Chr(chars(i)) = "v" Or flag = True Then
                            flag = True
                            If Chr(chars(i)) = "c" Then
                                m = m + 1
                                flag = False
                            End If
                        End If
                    Next i

                End If

                porterCountm = m

            End Function

            Private Shared Function porterEndsCVC(str As String) As Boolean

                On Error Resume Next

                '*o  - the stem ends cvc, where the second c is not W, X or Y (e.g. -WIL, -HOP).

                'declaring local variables
                Dim chars() As Byte
                Dim const_vowel As String
                Dim i As Byte
                Dim pattern As String

                'check to see if atleast 3 characters are present
                If Len(str) >= 3 Then

                    'converting string to byte array

                    chars = System.Text.Encoding.Unicode.GetBytes(str)

                    'find out the CVC pattern
                    pattern = returnCVCpattern(str)

                    'we need to check only the last three characters
                    pattern = Right(pattern, 3)

                    'check to see if the letters in str match the sequence cvc
                    porterEndsCVC = If(pattern = "cvc", If(Not (Chr(chars(UBound(chars))) = "w" Or Chr(chars(UBound(chars))) = "x" Or Chr(chars(UBound(chars))) = "y"), True, False), False)
                Else

                    porterEndsCVC = False

                End If

            End Function

            Private Shared Function porterEndsDoubleConsonent(str As String) As Boolean

                On Error Resume Next

                'checking whether word ends with a double consonant (e.g. -TT, -SS).

                'declaring local variables
                Dim holds_ends As String
                Dim hold_third_last As String
                Dim chars() As Byte

                'first check whether the size of the word is >= 2
                If Len(str) >= 2 Then

                    'extract 2 characters from right of str
                    holds_ends = Right(str, 2)

                    'converting string to byte array
                    chars = System.Text.Encoding.Unicode.GetBytes(holds_ends)

                    'checking if both the characters are same
                    If chars(0) = chars(1) Then

                        'check for double consonent
                        If holds_ends = "aa" Or holds_ends = "ee" Or holds_ends = "ii" Or holds_ends = "oo" Or holds_ends = "uu" Then

                            porterEndsDoubleConsonent = False
                        Else

                            'if the second last character is y, and there are atleast three letters in str
                            If holds_ends = "yy" And Len(str) > 2 Then

                                'extracting the third last character
                                hold_third_last = Right(str, 3)
                                hold_third_last = Left(str, 1)

                                porterEndsDoubleConsonent = If(Not (hold_third_last = "a" Or hold_third_last = "e" Or hold_third_last = "i" Or hold_third_last = "o" Or hold_third_last = "u"), False, True)
                            Else

                                porterEndsDoubleConsonent = True

                            End If

                        End If
                    Else

                        porterEndsDoubleConsonent = False

                    End If
                Else

                    porterEndsDoubleConsonent = False

                End If

            End Function

            Private Shared Function porterEndsWith(str As String, ends As String) As Boolean

                On Error Resume Next

                'declaring local variables
                Dim length_str As Byte
                Dim length_ends As Byte
                Dim hold_ends As String

                'finding the length of the string
                length_str = Len(str)
                length_ends = Len(ends)

                'if length of str is greater than the length of length_ends, only then proceed..else return false
                If length_ends >= length_str Then

                    porterEndsWith = False
                Else

                    'extract characters from right of str
                    hold_ends = Right(str, length_ends)

                    'comparing to see whether hold_ends=ends
                    porterEndsWith = If(StrComp(hold_ends, ends) = 0, True, False)

                End If

            End Function

            Private Shared Function porterTrimEnd(str As String, length As Byte) As String

                On Error Resume Next

                'returning the trimmed string
                porterTrimEnd = Left(str, Len(str) - length)

            End Function

            Private Shared Function returnCVCpattern(str As String) As String

                'local variables
                Dim chars() As Byte
                Dim const_vowel As String = ""
                Dim i As Byte

                'converting string to byte array
                chars = System.Text.Encoding.Unicode.GetBytes(str)

                'checking each character to see if it is a consonent or a vowel. also inputs the information in const_vowel
                For i = 0 To UBound(chars) Step 1

                    If Chr(chars(i)) = "a" Or Chr(chars(i)) = "e" Or Chr(chars(i)) = "i" Or Chr(chars(i)) = "o" Or Chr(chars(i)) = "u" Then
                        const_vowel = const_vowel + "v"
                    ElseIf Chr(chars(i)) = "y" Then
                        'if y is not the first character, only then check the previous character
                        'check to see if previous character is a consonent
                        const_vowel = If(i > 0, If(Not (Chr(chars(i - 1)) = "a" Or Chr(chars(i - 1)) = "e" Or Chr(chars(i - 1)) = "i" Or Chr(chars(i - 1)) = "o" Or Chr(chars(i - 1)) = "u"), const_vowel + "v", const_vowel + "c"), const_vowel + "c")
                    Else
                        const_vowel = const_vowel + "c"
                    End If

                Next i

                returnCVCpattern = const_vowel

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

    End Namespace
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





End Namespace