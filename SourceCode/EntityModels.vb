Imports System.IO
Imports System.Text.RegularExpressions
Imports System.Web.Script.Serialization
Imports System.Windows.Forms
Imports InputModelling.LanguageModels.BaseModels.LanguageModelFactory.Corpus.Document.Sentence
Imports Newtonsoft.Json
Namespace Models

    Namespace EntityModel
        <Serializable>
        Public Class EntityEncoder
            Private EntityLists As New Dictionary(Of String, List(Of String))

            Private VocabularyEntities As New List(Of Entity)

            Private Enum ScoreType
                Standard
                RelationToType
                RelationToEntity
            End Enum

            Public ReadOnly Property HeldEntityLists As List(Of EntityList)
                Get
                    Dim Value As New List(Of EntityList)
                    For Each item In EntityLists
                        Dim newEntityList As New EntityList
                        newEntityList.Entitys = item.Value
                        newEntityList.EntityType = item.Key
                        Value.Add(newEntityList)
                    Next
                    Return Value
                End Get
            End Property

            Public ReadOnly Property Vocabulary As List(Of Entity)
                Get
                    Return VocabularyEntities
                End Get
            End Property

            Public Sub Train(ByRef EntityListCorpus As List(Of EntityList))
                For Each EntityListItem In EntityListCorpus
                    Train(EntityListItem)
                Next
            End Sub

            Public Sub Train(ByRef EntityList As EntityList)
                AddEntityList(EntityList.Entitys, EntityList.EntityType)
            End Sub

            Private Sub AddEntityList(ByRef NewEntityList As List(Of String), ByRef EntityType As String)

                If CheckEntityList(EntityType) = True Then
                    Dim OldList As List(Of String) = EntityLists(EntityType)
                    NewEntityList.AddRange(OldList)
                    EntityLists.Remove(EntityType)
                End If

                Dim LabelValue = 1 / NewEntityList.Count
                For Each NewEntity In NewEntityList
                    If CheckEntity(NewEntity) = True Then
                        Dim Newtype As New EntityType
                        Newtype.Type = EntityType
                        Newtype.Value = LabelValue
                        UpdateEntityScore(NewEntity, Newtype)
                    Else
                        'Create Vocab Entity
                        Dim NewEnt As New Entity(NewEntity)
                        Dim NewEntType As New EntityType
                        NewEntType.Type = EntityType
                        NewEntType.Value = LabelValue
                        NewEnt.AddEntityScore(NewEntType)
                        AddToVocabulary(NewEnt)
                    End If
                Next

                EntityLists.Add(EntityType, NewEntityList)
            End Sub

            Private Sub AddToVocabulary(entity As Entity)
                If VocabularyEntities.Contains(entity) = False Then
                    VocabularyEntities.Add(entity)
                    Console.WriteLine($"Added {entity.Value} to vocabulary.")

                End If

            End Sub

            Private Function CalcEntityTypeLabelScore(ByRef EntityType As String, Optional ByRef Method As ScoreType = ScoreType.Standard)

                Select Case Method
                    Case ScoreType.Standard
                        If EntityLists.Keys.Contains(EntityType) Then
                            Dim iEntityList As List(Of String) = EntityLists(EntityType)
                            Dim count As Integer = Vocabulary.Count / iEntityList.Count

                            Return 1 / count
                        End If
                    Case ScoreType.RelationToEntity
                        If EntityLists.Keys.Contains(EntityType) Then
                            Dim EntityList As List(Of String) = EntityLists(EntityType)
                            'Indicate Proportion of vocabulary which is Entity-type
                            Dim Count1 = 1 / EntityList.Count
                            Dim Count2 = 1 / VocabularyEntities.Count
                            Return Count2 / Count1
                        End If
                    Case ScoreType.RelationToType
                        If EntityLists.Keys.Contains(EntityType) Then
                            Dim EntityList As List(Of String) = EntityLists(EntityType)
                            'Indicate Proportion of vocabulary which is Entity-type
                            Dim count = VocabularyEntities.Count / EntityList.Count

                            Return 1 / count
                        End If
                End Select
                Return 0
            End Function

            Private Function CheckEntity(ByRef Entity As String) As Boolean
                For Each item In VocabularyEntities
                    If item.Value = Entity Then
                        Return True
                    End If
                Next
                Return False
            End Function

            Private Function CheckEntityList(ByRef EntityType As String) As Boolean
                If EntityLists.Keys.Contains(EntityType) = True Then
                    Return True

                End If
                Return False
            End Function

            Private Function GetEntityList(ByRef EntityType As String) As EntityList
                If EntityLists.Keys.Contains(EntityType) = True Then

                    Dim NewEntitylist As New EntityList
                    NewEntitylist.Entitys = EntityLists(EntityType)
                    NewEntitylist.EntityType = EntityType
                    Return NewEntitylist
                End If
                Return New EntityList
            End Function

            Private Function UpdateEntityScore(ByRef Entity As String, ByRef NewType As EntityType) As Boolean
                For Each item In VocabularyEntities
                    If item.Value = Entity Then

                        item.AddEntityScore(NewType)

                        Return True
                    End If
                Next
                Return False
            End Function

            Public Class FileHandler
                Private EntityLists As New Dictionary(Of String, List(Of String))

                Private VocabularyEntities As New List(Of Entity)

                Public Sub New(vocabularyEntities As List(Of Entity), entityLists As Dictionary(Of String, List(Of String)))
                    If vocabularyEntities Is Nothing Then
                        Throw New ArgumentNullException(NameOf(vocabularyEntities))
                    End If

                    If entityLists Is Nothing Then
                        Throw New ArgumentNullException(NameOf(entityLists))
                    End If

                    Me.VocabularyEntities = vocabularyEntities
                    Me.EntityLists = entityLists
                End Sub

                Public ReadOnly Property HeldEntityLists As List(Of EntityList)
                    Get
                        Dim Value As New List(Of EntityList)
                        For Each item In EntityLists
                            Dim newEntityList As New EntityList
                            newEntityList.Entitys = item.Value
                            newEntityList.EntityType = item.Key
                            Value.Add(newEntityList)
                        Next
                        Return Value
                    End Get
                End Property

                Public ReadOnly Property Vocabulary As List(Of Entity)
                    Get
                        Return VocabularyEntities
                    End Get
                End Property

                ' Export an entity list to a JSON file
                Public Sub ExportEntityListToJson(entityList As String, filePath As String)

                    Dim json As String = JsonConvert.SerializeObject(GetEntityList(entityList))
                    File.WriteAllText(filePath, json)
                    Console.WriteLine($"Entity list exported to {filePath}.")
                End Sub

                ' Export the complete model to a binary file
                Public Sub ExportModelToBinary(fileName As String)
                    Using fs As New FileStream(fileName, FileMode.Create)
                        Using writer As New BinaryWriter(fs)
                            writer.Write(JsonConvert.SerializeObject(VocabularyEntities))
                            writer.Write(JsonConvert.SerializeObject(EntityLists))
                        End Using
                    End Using
                    Console.WriteLine($"Model exported to {fileName}.")
                End Sub

                ' Export vocabulary to a JSON file
                Public Sub ExportVocabularyToJson(fileName As String)
                    Dim json As String = JsonConvert.SerializeObject(VocabularyEntities)
                    File.WriteAllText(fileName, json)
                    Console.WriteLine($"Vocabulary exported to {fileName}.")
                End Sub

                ' Import the complete model from a binary file
                Public Sub ImportModelFromBinary(fileName As String)
                    Using fs As New FileStream(fileName, FileMode.Open)
                        Using reader As New BinaryReader(fs)
                            VocabularyEntities = JsonConvert.DeserializeObject(Of List(Of Entity))(reader.ReadString())
                            EntityLists = JsonConvert.DeserializeObject(Of Dictionary(Of String, List(Of String)))(reader.ReadString())
                        End Using
                    End Using
                    Console.WriteLine($"Model imported from {fileName}.")
                End Sub

                ' Import vocabulary from a JSON file
                Public Sub ImportVocabularyFromJson(fileName As String)
                    Dim json As String = File.ReadAllText(fileName)
                    VocabularyEntities = JsonConvert.DeserializeObject(Of List(Of Entity))(json)
                    Console.WriteLine($"Vocabulary imported from {fileName}.")
                End Sub

                Private Function GetEntityList(ByRef EntityType As String) As EntityList
                    If EntityLists.Keys.Contains(EntityType) = True Then

                        Dim NewEntitylist As New EntityList
                        NewEntitylist.Entitys = EntityLists(EntityType)
                        NewEntitylist.EntityType = EntityType
                        Return NewEntitylist
                    End If
                    Return New EntityList
                End Function

            End Class

        End Class
        Public Class EntityClassifier


            ''' <summary>
            ''' basic relationships between predicates
            ''' </summary>
            Private Shared ReadOnly BasicRelations() As String = {"capeable of", "related to", "relates to", "does", "did", "defined as",
                "can be defined as", "is described as", "is a", "is desired", "is an effect of", "is effected by", "was affected", "is a",
                "made of", "is part of", "a part of", "has the properties", "a property of", "used for", "used as", "is located in",
                "situated in", "is on", "is above", "is below", "begins with", "starts with",
                "was born in", "wrote book", "works at", "works in", "married to"}

            ''' <summary>
            ''' (can be used for topics or entity-lists or sentiment lists)
            ''' </summary>
            Private EntityList As New List(Of String)

            ''' <summary>
            ''' type of List (Describer) ie = "Negative" or "Cars" or "Organizations"
            ''' </summary>
            Private EntityType As String

            ''' <summary>
            ''' Is a,Part Of, Married to, Works at ...
            ''' </summary>
            Private Relations As New List(Of String)

            Public Sub New(entityList As List(Of String), entityType As String, relations As List(Of String))
                If entityList Is Nothing Then
                    Throw New ArgumentNullException(NameOf(entityList))
                End If

                If entityType Is Nothing Then
                    Throw New ArgumentNullException(NameOf(entityType))
                End If

                If relations Is Nothing Then
                    Throw New ArgumentNullException(NameOf(relations))
                End If
                relations.AddRange(BasicRelations.ToList)
                Me.EntityList = entityList
                Me.EntityType = entityType
                Me.Relations.AddRange(relations)
                relations = relations.Distinct.ToList
            End Sub

            ''' <summary>
            ''' Enables for extracting Lists as Single Item
            ''' </summary>
            ''' <returns></returns>
            Public ReadOnly Property CurrentEntityList As List(Of (String, String))
                Get
                    Return GetEntityList()
                End Get
            End Property
            Public Shared Function CaptureEntitySentencePatterns(ByVal inputText As String, ByVal EntityList As List(Of Entity)) As List(Of String)
                Dim DiscoveredSentencePatterns As New List(Of String)
                Dim found As Boolean = False
                Dim sents = Split(inputText, ".")
                For Each sent In sents
                    Dim words = Split(sent, " ")
                    For Each itemEntry In EntityList
                        If sent.Contains(itemEntry.Value) Then
                            found = True
                            sent = sent.Replace(itemEntry.Value, "[*]")
                        End If
                    Next
                    If found Then
                        DiscoveredSentencePatterns.Add(sent)
                    End If
                    found = False
                Next
                Return DiscoveredSentencePatterns
            End Function

            Public Shared Function CaptureEntitySentences(ByVal inputText As String, ByVal EntityList As List(Of Entity)) As List(Of String)
                Dim DiscoveredSentencePatterns As New List(Of String)
                Dim found As Boolean = False
                Dim sents = Split(inputText, ".")
                For Each sent In sents
                    Dim words = Split(sent, " ")
                    For Each itemEntry In EntityList
                        If sent.Contains(itemEntry.Value) Then
                            found = True
                            sent = sent.Replace(itemEntry.Value, "[" & itemEntry.MemberOfEntityTypes(0).Type & "]")
                        End If
                    Next
                    If found Then
                        DiscoveredSentencePatterns.Add(sent)
                    End If
                    found = False
                Next
                Return DiscoveredSentencePatterns
            End Function

            Public Shared Function DetectEntities(ByVal Text As String, ByVal EntityList As List(Of Entity)) As List(Of Entity)
                Dim words = Split(Text, " ")
                Dim Detected As New List(Of Entity)
                For Each item In words
                    For Each itemEntity In EntityList
                        If String.Equals(item, itemEntity.Value, StringComparison.OrdinalIgnoreCase) Then
                            Detected.Add(itemEntity)
                        End If
                    Next
                Next
                Return Detected
            End Function

            Public Shared Function DetectEntitys(ByRef Text As String, ByRef Entitylist As List(Of Entity)) As List(Of Entity)
                Dim words = Split(Text, " ")
                Dim Detected As New List(Of Entity)
                For Each item In words
                    For Each itemEntity In Entitylist
                        If item = itemEntity.Value Then
                            Detected.Add(itemEntity)
                        End If
                    Next
                Next
                Return Detected
            End Function

            Public Shared Function GetDiscoveredEntities(ByVal Text As String, ByVal EntityList As List(Of Entity)) As DiscoveredEntitys
                Dim NewEntSentence As New DiscoveredEntitys
                NewEntSentence.OriginalSentence = Text
                NewEntSentence.DiscoveredEntitys = DetectEntities(Text, EntityList)
                NewEntSentence.EntitySearchPattern = CaptureEntitySentencePatterns(Text, EntityList)
                NewEntSentence.EntitySentence = CaptureEntitySentences(Text, EntityList)
                Return NewEntSentence
            End Function

            Public Shared Function GetDiscoveredEntitys(ByRef Text As String, ByRef Entitylist As List(Of Entity)) As DiscoveredEntitys
                Dim NewEntSentence As New DiscoveredEntitys
                NewEntSentence.OriginalSentence = Text
                NewEntSentence.DiscoveredEntitys = DetectEntitys(Text, Entitylist)
                NewEntSentence.EntitySearchPattern = CaptureEntitySentencePatterns(Text, Entitylist)
                NewEntSentence.EntitySentence = CaptureEntitySentences(Text, Entitylist)
                Return NewEntSentence
            End Function
            Public Structure DiscoveredEntitys

                ''' <summary>
                ''' list of discovered entitys
                ''' </summary>
                Public DiscoveredEntitys As List(Of Entity)

                ''' <summary>
                ''' List of Extracted Sentences in the text
                ''' </summary>
                Public EntitySearchPattern As List(Of String)

                ''' <summary>
                ''' Transformed input identifing the entity positions in the text
                ''' with the entitys replaced with thier type
                ''' </summary>
                Public EntitySentence As List(Of String)

                ''' <summary>
                ''' original input
                ''' </summary>
                Public OriginalSentence

            End Structure

            Public Shared Function SplitIntoSentences(ByVal text As String) As String()
                ' Split text into sentences based on punctuation marks (.?!)
                Dim sentences As String() = Regex.Split(text, "(?<=['""A-Za-z0-9][.?!])\s+(?=[A-Z])")
                Return sentences
            End Function

            Public Shared Function TransformText(ByVal Sentence As String, ByVal EntityList As List(Of Entity)) As String
                Dim Discovered As List(Of Entity) = DetectEntities(Sentence, EntityList)
                For Each item In Discovered
                    Sentence = Sentence.Replace(item.Value, "[" & item.MemberOfEntityTypes(0).Type & "]")
                Next
                Return Sentence
            End Function

            ''' <summary>
            ''' Replaces known entities in a sentence with placeholders.
            ''' </summary>
            ''' <param name="sentence">The sentence.</param>
            ''' <returns>The sentence with placeholders.</returns>
            Private Shared Function ReplaceKnownEntitiesWithPlaceholders(ByVal sentence As String, ByRef KnownEntitys As List(Of Entity)) As String
                For Each entity As Entity In KnownEntitys
                    sentence = sentence.Replace(entity.Value, $"[{entity.MemberOfEntityTypes(0).Type}]")
                Next

                Return sentence
            End Function

            ''' <summary>
            ''' Replaces placeholders in a sentence with actual entity names.
            ''' </summary>
            ''' <param name="sentence">The sentence.</param>
            ''' <returns>The sentence with actual entity names.</returns>
            Private Shared Function ReplacePlaceholdersWithActualEntities(ByVal sentence As String, ByRef knownEntitys As List(Of Entity)) As String
                For Each entity As Entity In knownEntitys
                    sentence = sentence.Replace($"[{entity.MemberOfEntityTypes(0).Type}]", entity.Value)
                Next

                Return sentence
            End Function

            ' Method to extract entity relationships
            Public Function DiscoverEntityRelationships(ByVal document As String) As List(Of DiscoveredEntity)
                Dim EntitySentences As New List(Of DiscoveredEntity)
                Dim endOfSentenceMarkers As String() = {".", "!", "?"}

                ' Split the document into sentences
                Dim sentences As String() = document.Split(endOfSentenceMarkers, StringSplitOptions.RemoveEmptyEntries)

                For Each Sentence In sentences
                    Sentence = Sentence.Trim().ToLower()

                    ' Discover entities in the sentence
                    Dim detectedEntities As List(Of String) = DetectEntitysInText(Sentence)

                    ' Find relationships between entities based on patterns/rules
                    Dim relationships As List(Of EntityRelationship) = FindEntityRelationships(detectedEntities, Sentence, Relations)

                    ' Create the DiscoveredEntity object with relationships
                    Dim discoveredEntity As New DiscoveredEntity With {
                                .DiscoveredEntitys = detectedEntities,
                                .DiscoveredSentence = Sentence,
                                .EntitysWithContext = DetectEntitysWithContextInText(Sentence, 2),
                                .SentenceShape = DiscoverShape(Sentence),
                                .Relationships = relationships.Distinct.ToList
                            }

                    EntitySentences.Add(discoveredEntity)
                Next

                Return EntitySentences.Distinct.ToList
            End Function

            ' Add this method to the EntityClassifier class
            Public Function Forwards(ByVal documents As List(Of String)) As List(Of List(Of DiscoveredEntity))
                Dim batchResults As New List(Of List(Of DiscoveredEntity))

                For Each document In documents
                    Dim documentEntities As List(Of DiscoveredEntity) = Forwards(document)
                    batchResults.Add(documentEntities)
                Next

                Return batchResults.Distinct.ToList
            End Function

            ''' <summary>
            ''' Classify Entity Sentences
            ''' </summary>
            ''' <param name="document"></param>
            ''' <returns>Entity Sentences by Type</returns>
            Public Function Forwards(ByVal document As String) As List(Of DiscoveredEntity)
                Dim EntitySentences As New List(Of DiscoveredEntity)
                ' Store the classified sentences in a dictionary
                Dim classifiedSentences As New Dictionary(Of String, List(Of String))
                ' Define a list of possible end-of-sentence punctuation markers
                Dim endOfSentenceMarkers As String() = {".", "!", "?"}

                ' Split the document into sentences
                Dim sentences As String() = document.Split(endOfSentenceMarkers, StringSplitOptions.RemoveEmptyEntries)

                ' Rule-based classification
                For Each sentence In sentences
                    ' Remove leading/trailing spaces and convert to lowercase
                    sentence = sentence.ToLower()

                    'Discover
                    For Each EntityItem In EntityList
                        If sentence.ToLower.Contains(EntityItem.ToLower) Then
                            Dim Sent As New DiscoveredEntity

                            Sent.DiscoveredEntitys = DetectEntitysInText(sentence).Distinct.ToList
                            Sent.DiscoveredSentence = sentence
                            Sent.EntitysWithContext = DetectEntitysWithContextInText(sentence, 5).Distinct.ToList

                            Sent.Relationships = FindEntityRelationships(Sent.DiscoveredEntitys, sentence, Relations).Distinct.ToList
                            Sent.EntitySentence = TransformText(sentence)
                            Sent.SentenceShape = DiscoverShape(sentence)
                            EntitySentences.Add(Sent)

                        End If
                    Next

                Next
                Return EntitySentences.Distinct.ToList
            End Function

            ''' <summary>
            ''' Extracts patterns from the text,
            ''' replaces detected entities with asterisks,
            ''' and replaces the entity label with asterisks.
            ''' This is to replace a specific entity in the text
            ''' </summary>
            '''
            ''' <param name="Text">The text to extract patterns from.</param>
            ''' <returns>The extracted pattern with detected entities and the entity label replaced by asterisks.</returns>
            Public Function TransformText(ByRef Text As String) As String
                If Text Is Nothing Then
                    Throw New ArgumentNullException("text")
                End If

                Dim Entitys As New List(Of String)
                Dim Str As String = Text
                If DetectEntity(Text) = True Then
                    Entitys = DetectEntitysInText(Text)
                    ' Str = DiscoverShape(Str)
                    Str = Transform.TransformText(Str.ToLower, Entitys, EntityType)
                End If
                Return Str
            End Function

            ''' <summary>
            ''' Checks if any entities from the EntityList are present in the text.
            ''' </summary>
            ''' <param name="text">The text to be checked.</param>
            ''' <returns>True if any entities are detected, False otherwise.</returns>
            Private Function DetectEntity(ByRef text As String) As Boolean
                If text Is Nothing Then
                    Throw New ArgumentNullException("text")
                End If

                For Each item In EntityList
                    If text.ToLower.Contains(item.ToLower) Then
                        Return True
                    End If
                Next
                Return False
            End Function

            ''' <summary>
            ''' Detects entities in the given text.
            ''' </summary>
            ''' <returns>A list of detected entities in the text.</returns>
            Private Function DetectEntitysInText(ByRef text As String) As List(Of String)
                Dim Lst As New List(Of String)
                If text Is Nothing Then
                    Throw New ArgumentNullException("text")
                End If

                If DetectEntity(text) = True Then
                    For Each item In EntityList
                        If text.ToLower.Contains(item.ToLower) Then
                            Lst.Add(item)
                        End If
                    Next
                    Return Lst.Distinct.ToList
                Else

                End If
                Return New List(Of String)
            End Function

            Private Function DetectEntitysWithContextInText(ByRef text As String, Optional contextLength As Integer = 1) As List(Of String)
                Dim Lst As New List(Of String)
                If text Is Nothing Then
                    Throw New ArgumentNullException("text")
                End If

                If EntityList Is Nothing Then
                    Throw New ArgumentNullException("EntityList")
                End If
                If DetectEntity(text) = True Then
                    For Each item In EntityList
                        If text.ToLower.Contains(item.ToLower) Then
                            'Add Context
                            Lst.Add(ExtractEntityContextFromText(text, item, contextLength))
                        End If
                    Next
                    Return Lst.Distinct.ToList
                Else
                    Return New List(Of String)
                End If
            End Function

            ''' <summary>
            ''' Discovers shapes in the text and replaces the detected entities with entity labels.
            ''' </summary>
            ''' <param name="text">The text to discover shapes in.</param>
            ''' <returns>The text with detected entities replaced by entity labels.</returns>
            Private Function DiscoverShape(ByRef text As String) As String
                If text Is Nothing Then
                    Throw New ArgumentNullException("text")
                End If
                Dim Entitys As New List(Of String)
                Dim Str As String = text
                If DetectEntity(text) = True Then
                    Entitys = DetectEntitysInText(text)

                    Str = Transform.TransformText(Str, Entitys, "*")

                End If
                Return Str
            End Function

            ''' <summary>
            ''' Extracts Entity With Context;
            '''
            ''' </summary>
            ''' <param name="text">doc</param>
            ''' <param name="entity">Entity Value</param>
            ''' <param name="contextLength"></param>
            ''' <returns>a concat string</returns>
            Private Function ExtractEntityContextFromText(ByVal text As String,
                                                ByVal entity As String,
                                                Optional contextLength As Integer = 4) As String
                Dim entityIndex As Integer = text.ToLower.IndexOf(entity.ToLower)
                Dim contextStartIndex As Integer = Math.Max(0, entityIndex - contextLength)
                Dim contextEndIndex As Integer = Math.Min(text.Length - 1, entityIndex + entity.Length + contextLength)
                Dim NewEntity As New Entity("")
                NewEntity.StartIndex = contextStartIndex
                NewEntity.EndIndex = contextEndIndex
                NewEntity.Value = text.Substring(contextStartIndex, contextEndIndex - contextStartIndex + 1)

                Return text.Substring(contextStartIndex, contextEndIndex - contextStartIndex + 1)
            End Function
            Public Function ExtractEntityContextFromTextasEntity(ByVal text As String,
                                                ByVal entity As String,
                                                Optional contextLength As Integer = 4) As Entity
                Dim entityIndex As Integer = text.ToLower.IndexOf(entity.ToLower)
                Dim contextStartIndex As Integer = Math.Max(0, entityIndex - contextLength)
                Dim contextEndIndex As Integer = Math.Min(text.Length - 1, entityIndex + entity.Length + contextLength)
                Dim NewEntity As New Entity("")
                NewEntity.StartIndex = contextStartIndex
                NewEntity.EndIndex = contextEndIndex
                NewEntity.Value = text.Substring(contextStartIndex, contextEndIndex - contextStartIndex + 1)

                Return NewEntity
            End Function

            ' Method to find entity relationships based on patterns/rules
            Public Function FindEntityRelationships(ByVal entities As List(Of String), ByVal sentence As String, ByRef ConceptRelations As List(Of String)) As List(Of EntityRelationship)
                ' Define relationship patterns/rules based on the specific use case
                ' For example, "works at," "is the CEO of," etc.

                Dim relationships As New List(Of EntityRelationship)

                ' Sample rule for demonstration (Assuming "works at" relationship)
                For i = 0 To entities.Count - 1
                    For j = 0 To entities.Count - 1
                        If i <> j Then
                            For Each Relation In ConceptRelations
                                If sentence.ToLower.Contains(entities(i).ToLower & " " & Relation.ToLower & " " & entities(j).ToLower) Then
                                    relationships.Add(New EntityRelationship With {
                                                .SourceEntity = entities(i),
                                                .TargetEntity = entities(j),
                                                .RelationshipType = Relation,
                                                .Sentence = ExtractPredicateRelation(sentence.ToLower, Relation.ToLower)
                                            })
                                End If
                            Next
                        End If
                    Next
                Next

                ' Add more rules and patterns as needed for your specific use case
                ' Example: "is the CEO of," "married to," etc.

                Return relationships.Distinct.ToList
            End Function

            Public Shared Function ExtractPredicateRelation(Sentence As String, ByRef LinkingVerb As String) As String
                ' Implement your custom dependency parsing logic here
                ' Analyze the sentence and extract the relationships
                Dim relationship As String = ""

                ' Example relationship extraction logic
                If Sentence.ToLower.Contains(" " & LinkingVerb.ToLower & " ") Then
                    Dim subject As String = ""
                    Dim iobject As String = ""
                    Discover.SplitPhrase(Sentence.ToLower, LinkingVerb.ToLower, subject, iobject)

                    relationship = $"{subject} -  {LinkingVerb } - {iobject}"

                End If

                Return relationship
            End Function

            Private Function GetEntityList() As List(Of (String, String))
                Dim iEntityList As New List(Of (String, String))
                For Each entity In EntityList
                    iEntityList.Add((entity, EntityType))
                Next

                Return iEntityList.Distinct.ToList
            End Function



            Public Class Detect

                ''' <summary>
                ''' Checks if any entities from the EntityList are present in the text.
                ''' </summary>
                ''' <param name="text">The text to be checked.</param>
                ''' <returns>True if any entities are detected, False otherwise.</returns>
                Public Shared Function DetectEntity(ByRef text As String, ByRef EntityList As List(Of String)) As Boolean
                    If text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If

                    For Each item In EntityList
                        If text.Contains(item) Then
                            Return True
                        End If
                    Next
                    Return False
                End Function

                ' Modify DetectEntity and DetectEntitysInText methods to handle multiple entity types
                Public Shared Function DetectEntity(ByRef text As String, ByRef entityTypes As Dictionary(Of String, List(Of String))) As Boolean
                    If text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If

                    For Each iEntityType In entityTypes
                        For Each item In iEntityType.Value
                            If text.Contains(item) Then
                                Return True
                            End If
                        Next
                    Next

                    Return False
                End Function

                Public Shared Function DetectEntitysInText(ByRef text As String, ByRef Entitylist As List(Of String)) As List(Of String)
                    Dim Lst As New List(Of String)
                    If text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If

                    If Entitylist Is Nothing Then
                        Throw New ArgumentNullException("EntityList")
                    End If
                    If DetectEntity(text, Entitylist) = True Then
                        For Each item In Entitylist
                            If text.Contains(item) Then
                                Lst.Add(item)
                            End If
                        Next
                        Return Lst.Distinct.ToList
                    Else
                        Return New List(Of String)
                    End If
                End Function

                Public Shared Function DetectEntitysInText(ByRef Text As String, EntityList As List(Of String), ByRef EntityLabel As String) As List(Of String)
                    If Text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If

                    If EntityList Is Nothing Then
                        Throw New ArgumentNullException("Entitylist")
                    End If
                    If EntityLabel Is Nothing Then
                        Throw New ArgumentNullException("Entitylabel")
                    End If
                    Dim str As String = Text
                    Dim output As New List(Of String)
                    If EntityClassifier.Detect.DetectEntity(Text, EntityList) = True Then

                        Dim DetectedEntitys As List(Of String) = EntityClassifier.Detect.DetectEntitysInText(Text, EntityList)
                        Dim Shape As String = Discover.DiscoverShape(Text, EntityList, EntityLabel)
                        Dim pattern As String = Discover.ExtractPattern(Text, EntityList, EntityLabel)
                        output = DetectedEntitys
                    Else

                    End If
                    Return output.Distinct.ToList
                End Function

                Public Shared Function DetectEntitysInText(ByRef text As String, ByRef entityTypes As Dictionary(Of String, List(Of String))) As List(Of Entity)
                    Dim detectedEntities As New List(Of Entity)

                    If text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If

                    For Each iEntityType In entityTypes
                        For Each item In iEntityType.Value
                            If text.Contains(item) Then
                                Dim startIndex As Integer = text.IndexOf(item)
                                Dim endIndex As Integer = startIndex + item.Length - 1
                                Dim IEnt As New Entity(item)
                                IEnt.StartIndex = startIndex
                                IEnt.EndIndex = endIndex
                                Dim Itype As New EntityType
                                Itype.Type = iEntityType.Key
                                IEnt.MemberOfEntityTypes.Add(Itype)
                                detectedEntities.Add(IEnt)

                            End If
                        Next
                    Next

                    Return detectedEntities.Distinct.ToList
                End Function

                ''' <summary>
                ''' Predicts the position of an entity relative to the focus term within the context words.
                ''' </summary>
                ''' <param name="contextWords">The context words.</param>
                ''' <param name="focusTerm">The focus term.</param>
                ''' <returns>The predicted entity position.</returns>
                Public Shared Function PredictEntityPosition(contextWords As List(Of String), focusTerm As String) As String
                    Dim termIndex As Integer = contextWords.IndexOf(focusTerm)

                    If termIndex >= 0 Then
                        If termIndex < contextWords.Count - 1 Then
                            Return "After"
                        ElseIf termIndex > 0 Then
                            Return "Before"
                        End If
                    End If

                    Return "None"
                End Function

                Public Shared Function WordIsEntity(ByRef Word As String, ByRef Entitys As List(Of String)) As Boolean
                    For Each item In Entitys
                        If Word = item Then
                            Return True
                        End If
                    Next
                    Return False
                End Function

                ''' <summary>
                ''' 2. **Named Entity Recognition (NER)**:
                '''  - Input sentence: "John Smith and Sarah Johnson went to New York."
                '''   - Expected output: ["John Smith", "Sarah Johnson", "New York"]
                ''' </summary>
                ''' <param name="sentence"></param>
                ''' <param name="Entitys">list of entity values</param>
                ''' <param name="EntityLabel">Entity label</param>
                ''' <returns></returns>
                Public Shared Function DetectNamedEntities(sentence As String, ByRef Entitys As List(Of String), ByRef EntityLabel As String) As List(Of Entity)
                    ' Implement your custom named entity recognition logic here
                    ' Analyze the sentence and extract the named entities

                    Dim entities As New List(Of Entity)

                    ' Example named entity extraction logic
                    Dim words() As String = sentence.Split(" "c)
                    For i As Integer = 0 To words.Length - 1
                        For Each item In Entitys
                            If item.ToLower = words(i) Then



                                Dim ent As New Entity(words(i))
                                Dim Itype As New EntityType
                                Itype.Type = EntityLabel
                                ent.MemberOfEntityTypes.Add(Itype)

                                entities.Add(ent)
                            End If
                        Next

                    Next

                    Return entities.Distinct.ToList
                End Function

            End Class

            Public Class Discover


                Public Shared Function CalculateWordOverlap(tokens1 As String(), tokens2 As String()) As Integer
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

                Public Shared Function CompareAndScoreSentences(sentence1 As String, sentence2 As String) As Double
                    ' Implement your custom sentence comparison and scoring logic here
                    ' Compare the two sentences and assign a similarity score

                    ' Example sentence comparison and scoring logic
                    Dim similarityScore As Double = 0.0

                    ' Calculate the similarity score based on some criteria (e.g., word overlap)
                    Dim words1() As String = sentence1.Split(" "c)
                    Dim words2() As String = sentence2.Split(" "c)

                    Dim commonWordsCount As Integer = words1.Intersect(words2).Count()
                    Dim totalWordsCount As Integer = words1.Length + words2.Length

                    If totalWordsCount > 0 Then
                        similarityScore = CDbl(commonWordsCount) / CDbl(totalWordsCount)
                    End If

                    Return similarityScore
                End Function

                ''' <summary>
                ''' Attempts to find Entitys identified by thier capitalization
                ''' </summary>
                ''' <param name="words"></param>
                ''' <returns></returns>
                Public Shared Function DetectNamedEntities(ByVal words() As String) As List(Of String)
                    Dim namedEntities As New List(Of String)

                    For i = 0 To words.Length - 1
                        Dim word = words(i)
                        If Char.IsUpper(word(0)) Then
                            namedEntities.Add(word)
                        End If
                    Next

                    Return namedEntities.Distinct.ToList
                End Function

                Public Shared Function DetermineEntailment(overlap As Integer) As Boolean
                    ' Set a threshold for entailment
                    Dim threshold As Integer = 2

                    ' Determine entailment based on overlap
                    Return overlap >= threshold
                End Function

                ''' <summary>
                ''' Discovers shapes in the text and replaces the detected entities with entity labels.
                ''' </summary>
                ''' <param name="text">The text to discover shapes in.</param>
                ''' <param name="EntityList">A list of entities to detect and replace.</param>
                ''' <returns>The text with detected entities replaced by entity labels.</returns>
                Public Shared Function DiscoverShape(ByRef text As String, ByRef EntityList As List(Of String)) As String
                    If text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If

                    If EntityList Is Nothing Then
                        Throw New ArgumentNullException("Entitylist")
                    End If

                    Dim Entitys As New List(Of String)
                    Dim Str As String = text
                    If EntityClassifier.Detect.DetectEntity(text, EntityList) = True Then
                        Entitys = EntityClassifier.Detect.DetectEntitysInText(text, EntityList)

                        Str = Transform.TransformText(Str, Entitys)

                    End If
                    Return Str
                End Function

                Public Shared Function DiscoverShape(ByRef Text As String, Entitylist As List(Of String), ByRef EntityLabel As String) As String
                    If Text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If
                    If EntityLabel Is Nothing Then
                        Throw New ArgumentNullException("Entitylabel")
                    End If
                    If Entitylist Is Nothing Then
                        Throw New ArgumentNullException("Entitylist")
                    End If
                    Dim Entitys As New List(Of String)
                    Dim Str As String = Text
                    If EntityClassifier.Detect.DetectEntity(Text, Entitylist) = True Then
                        Entitys = EntityClassifier.Detect.DetectEntitysInText(Text, Entitylist)

                        Str = Transform.TransformText(Str, Entitys, EntityLabel)

                    End If
                    Return Str
                End Function

                ''' <summary>
                ''' Extracts patterns from the text, replaces detected entities with asterisks, and replaces the entity label with asterisks.
                ''' </summary>
                ''' <param name="Text">The text to extract patterns from.</param>
                ''' <param name="Entitylist">A list of entities to detect and replace.</param>
                ''' <param name="EntityLabel">The label to replace detected entities with.</param>
                ''' <returns>The extracted pattern with detected entities and the entity label replaced by asterisks.</returns>
                Public Shared Function ExtractPattern(ByRef Text As String, Entitylist As List(Of String), ByRef EntityLabel As String) As String
                    If Text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If

                    If Entitylist Is Nothing Then
                        Throw New ArgumentNullException("EntityList")
                    End If
                    If EntityLabel Is Nothing Then
                        Throw New ArgumentNullException("EntityLabel")
                    End If
                    Dim Entitys As New List(Of String)
                    Dim Str As String = Text
                    If EntityClassifier.Detect.DetectEntity(Text, Entitylist) = True Then
                        Entitys = EntityClassifier.Detect.DetectEntitysInText(Text, Entitylist)
                        Str = Discover.DiscoverShape(Str, Entitys)
                        Str = Transform.TransformText(Str, Entitys)
                        Str = Str.Replace("[" & EntityLabel & "]", "*")
                    End If
                    Return Str
                End Function

                Public Shared Function ExtractPredicateRelations(text As String, ByRef LinkingVerb As String) As List(Of String)
                    ' Implement your custom dependency parsing logic here
                    ' Analyze the sentence and extract the relationships

                    Dim relationships As New List(Of String)
                    For Each sentence In Split(text, ".)").ToList

                        ' Example relationship extraction logic
                        If sentence.ToLower.Contains(" " & LinkingVerb.ToLower & " ") Then
                            Dim relationParts() As String = sentence.Split(" " & LinkingVerb.ToLower & " ")
                            Dim subject As String = relationParts(0).Trim()
                            Dim iobject As String = relationParts(1).Trim()

                            Dim relationship As String = $"{subject} -  {LinkingVerb } - {iobject}"
                            relationships.Add(relationship)
                        End If
                    Next
                    Return relationships.Distinct.ToList
                End Function

                ''' <summary>
                ''' SPLITS THE GIVEN PHRASE UP INTO TWO PARTS by dividing word SplitPhrase(Userinput, "and",
                ''' Firstp, SecondP)
                ''' </summary>
                ''' <param name="PHRASE">      Sentence to be divided</param>
                ''' <param name="DIVIDINGWORD">String: Word to divide sentence by</param>
                ''' <param name="FIRSTPART">   String: firstpart of sentence to be populated</param>
                ''' <param name="SECONDPART">  String: Secondpart of sentence to be populated</param>
                ''' <remarks></remarks>
                Public Shared Sub SplitPhrase(ByVal PHRASE As String, ByRef DIVIDINGWORD As String, ByRef FIRSTPART As String, ByRef SECONDPART As String)
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

                Public Shared Function ExtractUniqueWordsInText(ByVal text As String) As List(Of String)
                    Dim regex As New Regex("\b\w+\b") ' Matches individual words
                    Dim matches As MatchCollection = regex.Matches(text)
                    Dim uniqueEntities As New List(Of String)

                    For Each match As Match In matches
                        Dim entity As String = match.Value
                        If Not uniqueEntities.Contains(entity) Then
                            uniqueEntities.Add(entity)
                        End If
                    Next

                    Return uniqueEntities
                End Function

                ''' <summary>
                ''' Given a list of Entity-lists and a single entity-list
                ''' Extract a new list (AssociatedEntities)
                ''' </summary>
                ''' <param name="entityList"> a single entity-list ,
                ''' Possibly assert(value) discovered list of entitys, Or a particular intresting list</param>
                ''' <param name="storedEntityLists"></param>
                ''' <returns></returns>
                Public Shared Function GetAssociatedEntities(entityList As List(Of String), storedEntityLists As List(Of List(Of String))) As List(Of String)
                    Dim associatedEntities As New List(Of String)

                    ' Iterate over each stored entity list
                    For Each storedList As List(Of String) In storedEntityLists
                        ' Check if any entity from the input entity list appears in the stored list
                        Dim matchedEntities = storedList.Intersect(entityList)

                        ' Add the matched entities to the associated entities list
                        associatedEntities.AddRange(matchedEntities)
                    Next

                    ' Remove duplicates from the associated entities list
                    associatedEntities = associatedEntities.Distinct().ToList()

                    Return associatedEntities
                End Function

                Public Shared Function GetEntityCountInText(ByVal text As String, ByVal entity As String) As Integer
                    Dim regex As New Regex(Regex.Escape(entity))
                    Return regex.Matches(text).Count
                End Function

                ''' <summary>
                ''' Get stored entity's
                ''' </summary>
                ''' <returns>List of stored entity's</returns>
                Public Shared Function GetEntityLists(ByRef ConnectionStr As String) As List(Of Entity)
                    Dim DbSubjectLst As New List(Of Entity)
                    Dim SQL As String = "SELECT * FROM Entity's "
                    Using conn = New System.Data.OleDb.OleDbConnection(ConnectionStr)
                        Using cmd = New System.Data.OleDb.OleDbCommand(SQL, conn)
                            conn.Open()
                            Try
                                Dim dr = cmd.ExecuteReader()
                                While dr.Read()

                                    Dim ent As New Entity(UCase(dr("Entity's").ToString))
                                    Dim Itype As New EntityType
                                    Itype.Type = UCase(dr("EntityListName").ToString)
                                    ent.MemberOfEntityTypes.Add(Itype)
                                    DbSubjectLst.Add(ent)
                                End While
                            Catch e As Exception
                                ' Do some logging or something.
                                MessageBox.Show("There was an error accessing your data. GetDBConceptNYM: " & e.ToString())
                            End Try
                        End Using
                    End Using
                    Return DbSubjectLst.Distinct.ToList
                End Function

            End Class

            Public Class Transform

                Public Shared Function RemoveEntitiesFromText(ByVal text As String, ByVal entities As List(Of String)) As String
                    For Each entity As String In entities
                        text = text.Replace(entity, String.Empty)
                    Next

                    Return text
                End Function

                Public Shared Function ReplaceEntities(sentence As String, ByRef ENTITYS As List(Of String)) As String
                    ' Replace discovered entities in the sentence with their entity type
                    For Each entity As String In ENTITYS
                        Dim entityType As String = entity.Substring(0, entity.IndexOf("("))
                        Dim entityValue As String = entity.Substring(entity.IndexOf("(") + 1, entity.Length - entity.IndexOf("(") - 2)
                        sentence = sentence.Replace(entityValue, entityType)
                    Next

                    Return sentence
                End Function

                ''' <summary>
                ''' Replaces the encapsulated [Entity Label] with an asterisk (*)
                ''' ie the [Entity] walked
                ''' </summary>
                ''' <param name="text">The text to be modified.</param>
                ''' <param name="Entitylabel">The label to replace the entities with.</param>
                ''' <returns>The text with entities replaced by the label.</returns>
                Public Shared Function ReplaceEntityLabel(ByRef text As String, ByRef EntityLabel As String, Value As String) As String
                    If text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If
                    If EntityLabel Is Nothing Then
                        Throw New ArgumentNullException("EntityLabel")
                    End If
                    Dim str As String = text
                    str = str.Replace("[" & EntityLabel & "]", Value)
                    Return str
                End Function

                Public Shared Function TransformText(ByRef Text As String, ByRef Entitys As List(Of String), Optional iLabel As String = "Entity") As String
                    If Text Is Nothing Then
                        Throw New ArgumentNullException("text")
                    End If

                    If Entitys Is Nothing Then
                        Throw New ArgumentNullException("Entitys")
                    End If

                    Dim Str As String = Text
                    For Each item In Entitys
                        Str = Str.ToLower.Replace(item.ToLower, "[" & iLabel & "]")
                    Next
                    Return Str
                End Function

            End Class

        End Class
        <Serializable>
        Public Class RuleBasedEntityRecognizer
            Private Shared entityPatterns As Dictionary(Of String, String)

            ''' <summary>
            ''' Represents a captured word and its associated information.
            ''' </summary>
            <Serializable>
            Public Structure CapturedWord
                ''' <summary>
                ''' The captured word.
                ''' </summary>
                Public Property Word As String
                ''' <summary>
                ''' The list of preceding words.
                ''' </summary>
                Public Property PrecedingWords As List(Of String)
                ''' <summary>
                ''' The list of following words.
                ''' </summary>
                Public Property FollowingWords As List(Of String)
                ''' <summary>
                ''' The person associated with the word.
                ''' </summary>
                Public Property Person As String
                ''' <summary>
                ''' The location associated with the word.
                ''' </summary>
                Public Property Location As String
                ''' <summary>
                ''' The recognized entity.
                ''' </summary>
                Public Property Entity As String
                ''' <summary>
                ''' Indicates whether the word is recognized as an entity.
                ''' </summary>
                Public Property IsEntity As Boolean
                ''' <summary>
                ''' The entity type of the word.
                ''' </summary>
                Public Property EntityType As String
                ''' <summary>
                ''' The list of entity types associated with the word.
                ''' </summary>
                Public Property EntityTypes As List(Of String)
                ''' <summary>
                ''' Indicates whether the word is the focus term.
                ''' </summary>
                Public Property IsFocusTerm As Boolean
                ''' <summary>
                ''' Indicates whether the word is a preceding word.
                ''' </summary>
                Public Property IsPreceding As Boolean
                ''' <summary>
                ''' Indicates whether the word is a following word.
                ''' </summary>
                Public Property IsFollowing As Boolean
                ''' <summary>
                ''' The context words.
                ''' </summary>
                Public Property ContextWords As List(Of String)

                ''' <summary>
                ''' Initializes a new instance of the <see cref="CapturedWord"/> structure.
                ''' </summary>
                ''' <param name="word">The captured word.</param>
                ''' <param name="precedingWords">The list of preceding words.</param>
                ''' <param name="followingWords">The list of following words.</param>
                ''' <param name="person">The person associated with the word.</param>
                ''' <param name="location">The location associated with the word.</param>
                Public Sub New(ByVal word As String, ByVal precedingWords As List(Of String), ByVal followingWords As List(Of String), ByVal person As String, ByVal location As String)
                    Me.Word = word
                    Me.PrecedingWords = precedingWords
                    Me.FollowingWords = followingWords
                    Me.Person = person
                    Me.Location = location
                End Sub
            End Structure
            Public Enum EntityPositionPrediction
                None
                Before
                After
            End Enum
            ''' <summary>
            ''' Performs a Sub-search within the given context words to recognize entities.
            ''' </summary>
            ''' <param name="contextWords">(applied After) The context words to search within.</param>
            ''' <param name="targetWord">The target word to recognize entities in.</param>
            ''' <returns>A list of captured words with entity information.</returns>
            Public Function PerformAfterSubSearch(ByVal contextWords As List(Of String), ByVal targetWord As String) As List(Of CapturedWord)
                Dim recognizedEntities As New List(Of CapturedWord)()
                Dim NewPat As String = targetWord
                For Each contextWord As String In contextWords

                    NewPat &= " " & contextWord
                    Dim entities As List(Of CapturedWord) = RecognizeEntities(contextWord & " " & targetWord)

                    If entities.Count > 0 Then
                        recognizedEntities.AddRange(entities)
                    End If
                Next

                Return recognizedEntities
            End Function
            ''' <summary>
            ''' Performs a subsearch within the given context words to recognize entities.
            ''' </summary>
            ''' <param name="contextWords">(Applied before) The context words to search within.</param>
            ''' <param name="targetWord">The target word to recognize entities in.</param>
            ''' <returns>A list of captured words with entity information.</returns>
            Public Function PerformBeforeSubSearch(ByVal contextWords As List(Of String), ByVal targetWord As String) As List(Of CapturedWord)
                Dim recognizedEntities As New List(Of CapturedWord)()
                Dim NewPat As String = targetWord
                For Each contextWord As String In contextWords

                    NewPat = contextWord & " " & NewPat
                    Dim entities As List(Of CapturedWord) = RecognizeEntities(contextWord & " " & targetWord)

                    If entities.Count > 0 Then
                        recognizedEntities.AddRange(entities)
                    End If
                Next

                Return recognizedEntities
            End Function

            Public Shared Sub Main()
                Dim recognizer As New RuleBasedEntityRecognizer()

                ' Configure entity patterns
                recognizer.ConfigureEntityPatterns()

                ' Example input text
                Dim inputText As String = "John went to the store and met Mary."

                ' Capture words with entity context
                Dim capturedWords As List(Of RuleBasedEntityRecognizer.CapturedWord) = recognizer.CaptureWordsWithEntityContext(inputText, "store", 2, 2)

                ' Display captured words and their entity information
                For Each capturedWord As RuleBasedEntityRecognizer.CapturedWord In capturedWords
                    Console.WriteLine("Word: " & capturedWord.Word)
                    Console.WriteLine("Is Entity: " & capturedWord.IsEntity)
                    Console.WriteLine("Entity Types: " & String.Join(", ", capturedWord.EntityTypes))
                    Console.WriteLine("Is Focus Term: " & capturedWord.IsFocusTerm)
                    Console.WriteLine("Is Preceding: " & capturedWord.IsPreceding)
                    Console.WriteLine("Is Following: " & capturedWord.IsFollowing)
                    Console.WriteLine("Context Words: " & String.Join(" ", capturedWord.ContextWords))
                    Console.WriteLine()
                Next

                Console.ReadLine()
            End Sub
            ''' <summary>
            ''' Configures the entity patterns by adding them to the recognizer.
            ''' </summary>
            Public Sub ConfigureEntityPatterns()
                ' Define entity patterns
                Me.AddEntityPattern("Person", "John|Mary|David")
                Me.AddEntityPattern("Location", "store|office|park")
                ' Add more entity patterns as needed
            End Sub

            ''' <summary>
            ''' Gets the entity types associated with a given word.
            ''' </summary>
            ''' <param name="word">The word to get entity types for.</param>
            ''' <returns>A list of entity types associated with the word.</returns>
            Public Function GetEntityTypes(ByVal word As String) As List(Of String)
                Dim recognizedEntities As List(Of CapturedWord) = RuleBasedEntityRecognizer.RecognizeEntities(word)
                Return recognizedEntities.Select(Function(entity) entity.EntityType).ToList()
            End Function

            ''' <summary>
            ''' Captures words with their context based on a focus term and the number of preceding and following words to include.
            ''' </summary>
            ''' <param name="text">The input text.</param>
            ''' <param name="focusTerm">The focus term to capture.</param>
            ''' <param name="precedingWordsCount">The number of preceding words to capture.</param>
            ''' <param name="followingWordsCount">The number of following words to capture.</param>
            ''' <returns>A list of WordWithContext objects containing captured words and their context information.</returns>
            Public Function CaptureWordsWithEntityContext(ByVal text As String, ByVal focusTerm As String, ByVal precedingWordsCount As Integer, ByVal followingWordsCount As Integer) As List(Of CapturedWord)
                Dim words As List(Of String) = text.Split(" "c).ToList()
                Dim focusIndex As Integer = words.IndexOf(focusTerm)

                Dim capturedWordsWithEntityContext As New List(Of CapturedWord)()

                If focusIndex <> -1 Then
                    Dim startIndex As Integer = Math.Max(0, focusIndex - precedingWordsCount)
                    Dim endIndex As Integer = Math.Min(words.Count - 1, focusIndex + followingWordsCount)

                    Dim contextWords As List(Of String) = words.GetRange(startIndex, endIndex - startIndex + 1)

                    Dim prediction As EntityPositionPrediction = PredictEntityPosition(contextWords, focusTerm)

                    For i As Integer = startIndex To endIndex
                        Dim word As String = words(i)

                        Dim entityTypes As List(Of String) = GetEntityTypes(word)

                        If entityTypes.Count = 0 AndAlso prediction <> EntityPositionPrediction.None Then
                            Dim isLowConfidenceEntity As Boolean = (prediction = EntityPositionPrediction.After AndAlso i > focusIndex) OrElse
                                                           (prediction = EntityPositionPrediction.Before AndAlso i < focusIndex)

                            If isLowConfidenceEntity Then
                                entityTypes.Add("Low Confidence Entity")
                            End If
                        End If

                        Dim wordWithContext As New CapturedWord() With {
                    .Word = word,
                    .IsEntity = entityTypes.Count > 0,
                    .EntityTypes = entityTypes,
                    .IsFocusTerm = (i = focusIndex),
                    .IsPreceding = (i < focusIndex),
                    .IsFollowing = (i > focusIndex),
                    .ContextWords = contextWords
                }

                        capturedWordsWithEntityContext.Add(wordWithContext)
                    Next
                End If

                Return capturedWordsWithEntityContext
            End Function

            ''' <summary>
            ''' Predicts the position of an entity relative to the focus term within the context words.
            ''' </summary>
            ''' <param name="contextWords">The context words.</param>
            ''' <param name="focusTerm">The focus term.</param>
            ''' <returns>The predicted entity position.</returns>
            Public Function PredictEntityPosition(ByVal contextWords As List(Of String), ByVal focusTerm As String) As EntityPositionPrediction
                Dim termIndex As Integer = contextWords.IndexOf(focusTerm)

                If termIndex >= 0 Then
                    If termIndex < contextWords.Count - 1 Then
                        Return EntityPositionPrediction.After
                    ElseIf termIndex > 0 Then
                        Return EntityPositionPrediction.Before
                    End If
                End If

                Return EntityPositionPrediction.None
            End Function

            ''' <summary>
            ''' Initializes a new instance of the <see cref="RuleBasedEntityRecognizer"/> class.
            ''' </summary>
            Public Sub New()
                entityPatterns = New Dictionary(Of String, String)()
            End Sub

            ''' <summary>
            ''' Adds an entity pattern to the recognizer.
            ''' </summary>
            ''' <param name="entityType">The entity type.</param>
            ''' <param name="pattern">The regular expression pattern.</param>
            Public Sub AddEntityPattern(ByVal entityType As String, ByVal pattern As String)
                entityPatterns.Add(entityType, pattern)
            End Sub

            ''' <summary>
            ''' Recognizes entities in the given text.
            ''' </summary>
            ''' <param name="text">The text to recognize entities in.</param>
            ''' <returns>A list of captured words with entity information.</returns>
            Public Shared Function RecognizeEntities(ByVal text As String) As List(Of CapturedWord)
                Dim capturedEntities As New List(Of CapturedWord)()

                For Each entityType As String In entityPatterns.Keys
                    Dim pattern As String = entityPatterns(entityType)
                    Dim matches As MatchCollection = Regex.Matches(text, pattern)

                    For Each match As Match In matches
                        capturedEntities.Add(New CapturedWord() With {
                    .Entity = match.Value,
                    .EntityType = entityType
                })
                    Next
                Next

                Return capturedEntities
            End Function
        End Class
        <Serializable>
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
                    If item.HasType(entityType) Then
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
        <Serializable>
        Public Structure DiscoveredEntity

            ''' <summary>
            ''' Discovered Entity
            ''' </summary>
            Public DiscoveredEntitys As List(Of String)

            ''' <summary>
            ''' Associated Sentence
            ''' </summary>
            Public DiscoveredSentence As String

            ''' <summary>
            ''' Entity Sentence
            ''' </summary>
            Public EntitySentence As String

            ''' <summary>
            ''' Entity with surrounding Context
            ''' </summary>
            Public EntitysWithContext As List(Of String)

            Public Relationships As List(Of EntityRelationship)

            ''' <summary>
            ''' Associated Sentence Sentence Shape
            ''' </summary>
            Public SentenceShape As String

            ''' <summary>
            ''' Outputs Structure to Jason(JavaScriptSerializer)
            ''' </summary>
            ''' <returns></returns>
            Public Function ToJson() As String
                Dim Converter As New JavaScriptSerializer
                Return Converter.Serialize(Me)
            End Function



        End Structure
        ' New property for relationships

        ' New structure to represent entity relationships
        <Serializable>
        Public Structure EntityRelationship
            Public Property RelationshipType As String
            Public Property Sentence As String
            Public Property SourceEntity As String
            Public Property TargetEntity As String
        End Structure
        <Serializable>
        Public Structure EntityList
            Public Entitys As List(Of String)
            Public EntityType As String

            ' Deserialize the entity from JSON format
            Public Shared Function FromJson(json As String) As EntityList
                Return JsonConvert.DeserializeObject(Of EntityList)(json)
            End Function

            ' Serialize the entity to JSON format
            Public Function ToJson() As String
                Return JsonConvert.SerializeObject(Me)
            End Function

        End Structure
        <Serializable>
        Public Structure EntityType
            Dim Type As String
            Dim Value As Double

            ' Deserialize the entity from JSON format
            Public Shared Function FromJson(json As String) As EntityType
                Return JsonConvert.DeserializeObject(Of EntityType)(json)
            End Function

            ' Serialize the entity to JSON format
            Public Function ToJson() As String
                Return JsonConvert.SerializeObject(Me)
            End Function

        End Structure


    End Namespace


End Namespace
Namespace DataObjects

    Enum ConclusionTypes
        Affirmative_Conclusion
        Conditional_Conclusion
        Negative_Conclusion
        Recommendation_Conclusion
        Prediction_Conclusion
    End Enum
    <Serializable>
    Public Class AnswerType

        Public Sub New(ByVal type As String, ByVal entities As List(Of String))
            Me.Type = type
            Me.Entities = entities
        End Sub

        Public Property Entities As List(Of String)
        Public Property Type As String
    End Class
    <Serializable>
    Public Structure CapturedContent
        Public Sub New(ByVal word As String, ByVal precedingWords As List(Of String), ByVal followingWords As List(Of String))
            Me.Word = word
            Me.PrecedingWords = precedingWords
            Me.FollowingWords = followingWords

        End Sub

        Public Property FollowingWords As List(Of String)
        Public Property PrecedingWords As List(Of String)
        Public Property Word As String
    End Structure
    <Serializable>
    Public Structure CapturedWord
        Public Sub New(ByVal word As String, ByVal precedingWords As List(Of String), ByVal followingWords As List(Of String), ByVal person As String, ByVal location As String)
            Me.Word = word
            Me.PrecedingWords = precedingWords
            Me.FollowingWords = followingWords
            Me.Person = person
            Me.Location = location
        End Sub

        ''' <summary>
        ''' Gets or sets the context words.
        ''' </summary>
        Public Property ContextWords As List(Of String)

        Public Property Entity As String
        Public Property EntityType As String
        ''' <summary>
        ''' Gets or sets the entity types associated with the word.
        ''' </summary>
        Public Property EntityTypes As List(Of String)

        Public Property FollowingWords As List(Of String)
        ''' <summary>
        ''' Gets or sets a value indicating whether the word is recognized as an entity.
        ''' </summary>
        Public Property IsEntity As Boolean

        ''' <summary>
        ''' Gets or sets a value indicating whether the word is the focus term.
        ''' </summary>
        Public Property IsFocusTerm As Boolean

        ''' <summary>
        ''' Gets or sets a value indicating whether the word is a following word.
        ''' </summary>
        Public Property IsFollowing As Boolean

        ''' <summary>
        ''' Gets or sets a value indicating whether the word is a preceding word.
        ''' </summary>
        Public Property IsPreceding As Boolean

        Public Property Location As String
        Public Property Person As String
        Public Property PrecedingWords As List(Of String)
        Public Property Word As String
    End Structure
    <Serializable>
    Public Structure NlpReport

        Public EntityLists As List(Of Entity)
        Public SearchPatterns As List(Of SemanticPattern)
        Public UserText As String

        Public Sub New(ByRef Usertext As String, Entitylists As List(Of Entity), ByRef SearchPatterns As List(Of SemanticPattern))
            Me.UserText = Usertext
            Me.EntityLists = Entitylists
            Me.SearchPatterns = SearchPatterns
        End Sub

    End Structure
    ''' <summary>
    ''' Used to retrieve Learning Patterns
    ''' Learning Pattern / Nym
    ''' </summary>
    <Serializable>
    Public Structure SemanticPattern

        ''' <summary>
        ''' Tablename in db
        ''' </summary>
        Public Shared SemanticPatternTable As String = "SemanticPatterns"

        ''' <summary>
        ''' Used to hold the connection string
        ''' </summary>
        Public ConnectionStr As String

        ''' <summary>
        ''' used to identify patterns
        ''' </summary>
        Public NymStr As String

        ''' <summary>
        ''' Search patterns A# is B#
        ''' </summary>
        Public SearchPatternStr As String

        Public Sub New(ConnectionStr As String)
            Me.New()
            Me.ConnectionStr = ConnectionStr
        End Sub

        ''' <summary>
        ''' filters collection of patterns by nym
        ''' </summary>
        ''' <param name="Patterns">patterns </param>
        ''' <param name="NymStr">nym to be segmented</param>
        ''' <returns></returns>
        Public Shared Function FilterSemanticPatternsbyNym(ByRef Patterns As List(Of SemanticPattern), ByRef NymStr As String) As List(Of SemanticPattern)
            Dim Lst As New List(Of SemanticPattern)
            For Each item In Patterns
                If item.NymStr = NymStr Then
                    Lst.Add(item)
                End If
            Next
            If Lst.Count > 0 Then
                Return Lst
            Else
                Return Nothing
            End If
        End Function

        ''' <summary>
        ''' Gets all Semantic Patterns From Table
        ''' </summary>
        ''' <param name="iConnectionStr"></param>
        ''' <param name="TableName"></param>
        ''' <returns></returns>
        Public Shared Function GetDBSemanticPatterns(ByRef iConnectionStr As String, ByRef TableName As String) As List(Of SemanticPattern)
            Dim DbSubjectLst As New List(Of SemanticPattern)

            Dim SQL As String = "SELECT * FROM " & TableName
            Using conn = New System.Data.OleDb.OleDbConnection(iConnectionStr)
                Using cmd = New System.Data.OleDb.OleDbCommand(SQL, conn)
                    conn.Open()
                    Try
                        Dim dr = cmd.ExecuteReader()
                        While dr.Read()
                            Dim NewKnowledge As New SemanticPattern With {
                                .NymStr = dr("Nym").ToString(),
                                .SearchPatternStr = dr("SemanticPattern").ToString()
                            }
                            DbSubjectLst.Add(NewKnowledge)
                        End While
                    Catch e As Exception
                        ' Do some logging or something.
                        MessageBox.Show("There was an error accessing your data. GetDBSemanticPatterns: " & e.ToString())
                    End Try
                End Using
            End Using
            Return DbSubjectLst
        End Function
        ''' <summary>
        ''' Gets all Semantic Patterns From Table
        ''' </summary>
        ''' <param name="TableName"></param>
        ''' <returns></returns>
        Public Function GetDBSemanticPatterns(ByRef TableName As String) As List(Of SemanticPattern)
            Dim DbSubjectLst As New List(Of SemanticPattern)

            Dim SQL As String = "SELECT * FROM " & TableName
            Using conn = New System.Data.OleDb.OleDbConnection(ConnectionStr)
                Using cmd = New System.Data.OleDb.OleDbCommand(SQL, conn)
                    conn.Open()
                    Try
                        Dim dr = cmd.ExecuteReader()
                        While dr.Read()
                            Dim NewKnowledge As New SemanticPattern With {
                                .NymStr = dr("Nym").ToString(),
                                .SearchPatternStr = dr("SemanticPattern").ToString()
                            }
                            DbSubjectLst.Add(NewKnowledge)
                        End While
                    Catch e As Exception
                        ' Do some logging or something.
                        MessageBox.Show("There was an error accessing your data. GetDBSemanticPatterns: " & e.ToString())
                    End Try
                End Using
            End Using
            Return DbSubjectLst
        End Function

        ''' <summary>
        ''' gets semantic patterns from table based on query SQL
        ''' </summary>
        ''' <param name="iConnectionStr"></param>
        ''' <param name="Query"></param>
        ''' <returns></returns>
        Public Shared Function GetDBSemanticPatternsbyQuery(ByRef iConnectionStr As String, ByRef Query As String) As List(Of SemanticPattern)
            Dim DbSubjectLst As New List(Of SemanticPattern)

            Dim SQL As String = Query
            Using conn = New System.Data.OleDb.OleDbConnection(iConnectionStr)
                Using cmd = New System.Data.OleDb.OleDbCommand(SQL, conn)
                    conn.Open()
                    Try
                        Dim dr = cmd.ExecuteReader()
                        While dr.Read()
                            Dim NewKnowledge As New SemanticPattern With {
                                    .NymStr = dr("Nym").ToString(),
                                    .SearchPatternStr = dr("SemanticPattern").ToString()
                                }
                            DbSubjectLst.Add(NewKnowledge)
                        End While
                    Catch e As Exception
                        ' Do some logging or something.
                        MessageBox.Show("There was an error accessing your data. GetDBSemanticPatterns: " & e.ToString())
                    End Try
                End Using
            End Using
            Return DbSubjectLst
        End Function
        ''' <summary>
        ''' gets semantic patterns from table based on query SQL
        ''' </summary>
        ''' <param name="Query"></param>
        ''' <returns></returns>
        Public Function GetDBSemanticPatternsbyQuery(ByRef Query As String) As List(Of SemanticPattern)
            Dim DbSubjectLst As New List(Of SemanticPattern)

            Dim SQL As String = Query
            Using conn = New System.Data.OleDb.OleDbConnection(ConnectionStr)
                Using cmd = New System.Data.OleDb.OleDbCommand(SQL, conn)
                    conn.Open()
                    Try
                        Dim dr = cmd.ExecuteReader()
                        While dr.Read()
                            Dim NewKnowledge As New SemanticPattern With {
                                    .NymStr = dr("Nym").ToString(),
                                    .SearchPatternStr = dr("SemanticPattern").ToString()
                                }
                            DbSubjectLst.Add(NewKnowledge)
                        End While
                    Catch e As Exception
                        ' Do some logging or something.
                        MessageBox.Show("There was an error accessing your data. GetDBSemanticPatterns: " & e.ToString())
                    End Try
                End Using
            End Using
            Return DbSubjectLst
        End Function

        ''' <summary>
        ''' gets random pattern from list
        ''' </summary>
        ''' <param name="Patterns"></param>
        ''' <returns></returns>
        Public Shared Function GetRandomPattern(ByRef Patterns As List(Of SemanticPattern)) As SemanticPattern
            Dim rnd = New Random()
            If Patterns.Count > 0 Then

                Return Patterns(rnd.Next(0, Patterns.Count))
            Else
                Return Nothing
            End If
        End Function

        ''' <summary>
        ''' used to generalize patterns into general search patterns
        ''' (a# is b#) to (* is a *)
        ''' </summary>
        ''' <param name="Patterns"></param>
        ''' <returns></returns>
        Public Shared Function InsertWildcardsIntoPatterns(ByRef Patterns As List(Of SemanticPattern)) As List(Of SemanticPattern)
            For Each item In Patterns
                item.SearchPatternStr.Replace("A#", "*")
                item.SearchPatternStr.Replace("B#", "*")
            Next
            Return Patterns
        End Function

        ''' <summary>
        ''' Adds a New Semantic pattern
        ''' </summary>
        ''' <param name="NewSemanticPattern"></param>
        Public Shared Function AddSemanticPattern(ByRef iConnectionStr As String, ByRef Tablename As String, ByRef NewSemanticPattern As SemanticPattern) As Boolean
            AddSemanticPattern = False
            If NewSemanticPattern.NymStr IsNot Nothing And NewSemanticPattern.SearchPatternStr IsNot Nothing Then

                Dim sql As String = "INSERT INTO " & Tablename & " (Nym, SemanticPattern) VALUES ('" & NewSemanticPattern.NymStr & "','" & NewSemanticPattern.SearchPatternStr & "')"

                Using conn = New System.Data.OleDb.OleDbConnection(iConnectionStr)

                    Using cmd = New System.Data.OleDb.OleDbCommand(sql, conn)
                        conn.Open()
                        Try
                            cmd.ExecuteNonQuery()
                            AddSemanticPattern = True
                        Catch ex As Exception
                            MessageBox.Show("There was an error accessing your data. AddSemanticPattern: " & ex.ToString())
                        End Try
                    End Using
                End Using
            Else
            End If
        End Function
        ''' <summary>
        ''' Adds a New Semantic pattern
        ''' </summary>
        ''' <param name="NewSemanticPattern"></param>
        Public Function AddSemanticPattern(ByRef Tablename As String, ByRef NewSemanticPattern As SemanticPattern) As Boolean
            AddSemanticPattern = False
            If NewSemanticPattern.NymStr IsNot Nothing And NewSemanticPattern.SearchPatternStr IsNot Nothing Then

                Dim sql As String = "INSERT INTO " & Tablename & " (Nym, SemanticPattern) VALUES ('" & NewSemanticPattern.NymStr & "','" & NewSemanticPattern.SearchPatternStr & "')"

                Using conn = New System.Data.OleDb.OleDbConnection(ConnectionStr)

                    Using cmd = New System.Data.OleDb.OleDbCommand(sql, conn)
                        conn.Open()
                        Try
                            cmd.ExecuteNonQuery()
                            AddSemanticPattern = True
                        Catch ex As Exception
                            MessageBox.Show("There was an error accessing your data. AddSemanticPattern: " & ex.ToString())
                        End Try
                    End Using
                End Using
            Else
            End If
        End Function
        Public Shared Function CheckIfSemanticPatternDetected(ByRef iConnectionStr As String, ByRef TableName As String, ByRef Userinput As String) As Boolean
            CheckIfSemanticPatternDetected = False
            For Each item In InsertWildcardsIntoPatterns(GetDBSemanticPatterns(iConnectionStr, TableName))
                If Userinput Like item.SearchPatternStr Then
                    Return True
                End If
            Next
        End Function
        Public Function CheckIfSemanticPatternDetected(ByRef TableName As String, ByRef Userinput As String) As Boolean
            CheckIfSemanticPatternDetected = False
            For Each item In InsertWildcardsIntoPatterns(GetDBSemanticPatterns(ConnectionStr, TableName))
                If Userinput Like item.SearchPatternStr Then
                    Return True
                End If
            Next
        End Function
        Public Function GetDetectedSemanticPattern(ByRef TableName As String, ByRef Userinput As String) As SemanticPattern
            GetDetectedSemanticPattern = Nothing
            For Each item In InsertWildcardsIntoPatterns(GetDBSemanticPatterns(ConnectionStr, TableName))
                If Userinput Like item.SearchPatternStr Then
                    Return item
                End If
            Next
        End Function
        Public Shared Function GetDetectedSemanticPattern(ByRef iConnectionStr As String, ByRef TableName As String, ByRef Userinput As String) As SemanticPattern
            GetDetectedSemanticPattern = Nothing
            For Each item In InsertWildcardsIntoPatterns(GetDBSemanticPatterns(iConnectionStr, TableName))
                If Userinput Like item.SearchPatternStr Then
                    Return item
                End If
            Next
        End Function

        ''' <summary>
        ''' output in json format
        ''' </summary>
        ''' <returns></returns>
        Public Function ToJson() As String
            Dim Converter As New JavaScriptSerializer
            Return Converter.Serialize(Me)
        End Function

    End Structure

    <Serializable>
    Public Structure WordWithContext
        ''' <summary>
        ''' Gets or sets the context words.
        ''' </summary>
        Public Property ContextWords As List(Of String)

        ''' <summary>
        ''' Gets or sets the entity types associated with the word.
        ''' </summary>
        Public Property EntityTypes As List(Of String)

        ''' <summary>
        ''' Gets or sets a value indicating whether the word is recognized as an entity.
        ''' </summary>
        Public Property IsEntity As Boolean

        ''' <summary>
        ''' Gets or sets a value indicating whether the word is the focus term.
        ''' </summary>
        Public Property IsFocusTerm As Boolean

        ''' <summary>
        ''' Gets or sets a value indicating whether the word is a following word.
        ''' </summary>
        Public Property IsFollowing As Boolean

        ''' <summary>
        ''' Gets or sets a value indicating whether the word is a preceding word.
        ''' </summary>
        Public Property IsPreceding As Boolean

        ''' <summary>
        ''' Gets or sets the captured word.
        ''' </summary>
        Public Property Word As String
    End Structure
End Namespace

Namespace Utilitys

    ' Latent Dirichlet Allocation (LDA) algorithm
    <Serializable>
    Public Class Latent_Dirichlet_Allocation




        'Public Class Document
        '    Public Property Words As List(Of Word)
        'End Class


        <Serializable>
        Public Class WordCount
            Public Property WordCount As Dictionary(Of String, Integer)

            Public Sub New()
                WordCount = New Dictionary(Of String, Integer)()
            End Sub

            Public Sub IncrementCount(word As Clause.Word)
                If Not WordCount.ContainsKey(word.text) Then
                    WordCount(word.text) = 0
                End If

                WordCount(word.text) += 1
            End Sub

            Public Sub DecrementCount(word As Clause.Word)
                If WordCount.ContainsKey(word.text) Then
                    WordCount(word.text) -= 1
                    If WordCount(word.text) = 0 Then
                        WordCount.Remove(word.text)
                    End If
                End If
            End Sub

            Public Function GetCount(word As Clause.Word) As Integer
                If WordCount.ContainsKey(word.text) Then
                    Return WordCount(word.text)
                End If

                Return 0
            End Function
        End Class

        Private documents As List(Of Clause)
        Private numTopics As Integer
        Private topicWordCounts As Dictionary(Of Integer, WordCount)
        Private topicCounts As List(Of Integer)
        Private wordTopicAssignments As List(Of Integer)

        Public Sub New(documents As List(Of Clause), numTopics As Integer)
            Me.documents = documents
            Me.numTopics = numTopics
            topicWordCounts = New Dictionary(Of Integer, WordCount)()
            topicCounts = New List(Of Integer)()
            wordTopicAssignments = New List(Of Integer)()
        End Sub

        Public Sub TrainModel(numIterations As Integer)
            InitializeModel()

            For i As Integer = 0 To numIterations - 1
                For j As Integer = 0 To documents.Count - 1
                    SampleTopicsForDocument(documents(j))
                Next
            Next
        End Sub

        Private Sub InitializeModel()
            Dim wordCount As Integer = 0

            For Each document In documents
                For Each word In document.Words
                    Dim topic = CInt(Math.Floor(Rnd() * numTopics))
                    wordTopicAssignments.Add(topic)
                    wordCount += 1

                    If Not topicWordCounts.ContainsKey(topic) Then
                        topicWordCounts(topic) = New WordCount()
                    End If

                    topicWordCounts(topic).IncrementCount(word)
                    topicCounts.Add(topic)
                Next
            Next

            Console.WriteLine("Number of words: " & wordCount)
        End Sub

        Private Sub SampleTopicsForDocument(document As Clause)
            For Each word In document.Words
                Dim oldTopic = wordTopicAssignments(word.Position)
                topicWordCounts(oldTopic).DecrementCount(word)

                Dim topicDistribution = CalculateTopicDistribution(document, word)
                Dim newTopic = SampleFromDistribution(topicDistribution)

                topicWordCounts(newTopic).IncrementCount(word)
                wordTopicAssignments(word.Position) = newTopic
            Next
        End Sub

        Private Function CalculateTopicDistribution(document As Clause, word As Clause.Word) As Double()
            Dim distribution(numTopics - 1) As Double

            For i As Integer = 0 To numTopics - 1
                distribution(i) = CalculateTopicProbability(i, document, word)
            Next

            Return distribution
        End Function

        Private Function CalculateTopicProbability(topic As Integer, document As Clause, word As Clause.Word) As Double
            Dim wordCountInTopic = topicWordCounts(topic).GetCount(word)

            Dim totalWordCountInTopic As Integer = 0

            For Each assignment In wordTopicAssignments
                If assignment = topic Then
                    totalWordCountInTopic += 1
                End If
            Next

            Return (wordCountInTopic + 1) / (totalWordCountInTopic + topicCounts.Count)
        End Function

        Private Function SampleFromDistribution(distribution As Double()) As Integer
            Dim rnd = New Random()

            For i As Integer = 1 To distribution.Length - 1
                distribution(i) += distribution(i - 1)
            Next

            Dim randomValue = rnd.NextDouble() * distribution(distribution.Length - 1)
            Dim sample As Integer

            For i As Integer = 0 To distribution.Length - 1
                If randomValue < distribution(i) Then
                    sample = i
                    Exit For
                End If
            Next

            Return sample
        End Function

        Public Sub PrintTopics()
            For Each topic In topicWordCounts.Keys
                Console.WriteLine("Topic " & topic)
                Dim topicWordCount = topicWordCounts(topic).WordCount
                Dim totalWordCount As Integer = 0

                For Each assignment In wordTopicAssignments
                    If assignment = topic Then
                        totalWordCount += 1
                    End If
                Next
                For Each word In topicWordCount.Keys
                    Dim count = topicWordCount(word)
                    Dim probability = count / totalWordCount
                    Console.WriteLine("   " & word & ": " & probability)
                Next

                Console.WriteLine()
            Next
        End Sub
    End Class



End Namespace