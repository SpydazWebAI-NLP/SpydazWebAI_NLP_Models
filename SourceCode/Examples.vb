Imports System.IO
Imports System.Numerics
Imports System.Windows.Forms
Imports InputModelling.Models
Imports InputModelling.Models.Chunkers
Imports InputModelling.Models.Embeddings
Imports InputModelling.Models.Embeddings.Audio
Imports InputModelling.Models.Embeddings.Images
Imports InputModelling.Models.Embeddings.Storage
Imports InputModelling.Models.Embeddings.Storage.MinHashAndLSH
Imports InputModelling.Models.Embeddings.Text
Imports InputModelling.Models.Embeddings.Text.Word2Vector
Imports InputModelling.Models.Entailment
Imports InputModelling.Models.Entailment.ContextAnalyzer
Imports InputModelling.Models.Entailment.SentenceClassifier
Imports InputModelling.Models.EntityModel
Imports InputModelling.Models.LanguageModels
Imports InputModelling.Models.LanguageModels.Corpus.Vocabulary
Imports InputModelling.Models.Nodes
Imports InputModelling.Models.Readers
Imports InputModelling.Models.TokenizerModels
Imports InputModelling.Models.Trees.BeliefTree
Imports InputModelling.Models.VocabularyModelling
Imports InputModelling.Utilitys

Namespace Examples

    Public Module Example
        Public Function iLangModelTrainTest() As iLangModel.FeedForwardNetwork
            ' Create the input and target training data
            Dim inputs As New List(Of List(Of Double))()
            Dim targets As New List(Of List(Of Double))()

            ' AND logic gate training data
            inputs.Add(New List(Of Double)() From {0, 0})
            inputs.Add(New List(Of Double)() From {0, 1})
            inputs.Add(New List(Of Double)() From {1, 0})
            inputs.Add(New List(Of Double)() From {1, 1})

            targets.Add(New List(Of Double)() From {0})
            targets.Add(New List(Of Double)() From {0})
            targets.Add(New List(Of Double)() From {0})
            targets.Add(New List(Of Double)() From {1})

            ' Create a feed-forward neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
            Dim network As New iLangModel.FeedForwardNetwork(inputSize:=2, hiddenSize:=2, outputSize:=1)

            ' Train the network using the training data for 100 epochs with a learning rate of 0.1
            network.Train(inputs, targets, epochs:=100, learningRate:=0.1)

            ' Test the trained network
            Console.WriteLine("Testing the trained network:")

            For i As Integer = 0 To inputs.Count - 1
                Dim inputVector As List(Of Double) = inputs(i)
                Dim targetVector As List(Of Double) = targets(i)

                Dim outputVector = network.Forward(inputs)

                Console.WriteLine("Input: {0}, Target: {1}, Output: {2}", String.Join(", ", inputVector), String.Join(", ", targetVector), String.Join(", ", outputVector))
            Next

            Return network
        End Function
        Public Sub IlangModelExample()
            ' Create an instance of the FeedForwardNetwork
            Dim feedForwardNN As iLangModel.FeedForwardNetwork = iLangModelTrainTest()

            ' Define the input sequence for the logical AND operation
            Dim inputSequence As List(Of List(Of Double)) = New List(Of List(Of Double))() From
            {
                New List(Of Double)() From {0, 0},
                New List(Of Double)() From {0, 1},
                New List(Of Double)() From {1, 0},
                New List(Of Double)() From {1, 1}
            }

            ' Apply the forward pass to get the predicted outputs
            Dim output As List(Of List(Of Double)) = feedForwardNN.Forward(inputSequence)

            ' Display the input sequence and predicted outputs
            Console.WriteLine("Input Sequence:")
            For Each inputVector As List(Of Double) In inputSequence
                Console.WriteLine(String.Join(", ", inputVector))
            Next

            Console.WriteLine("Predicted Outputs:")
            For Each outputVector As List(Of Double) In output
                Console.WriteLine(Math.Round(outputVector(0))) ' Round the output to the nearest integer (0 or 1)
            Next

            Console.ReadLine()
        End Sub

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
        Public Sub RiskAssessmentExample()
            ' Creating instances of RiskNode for Option A and Option B
            Dim optionA As New RiskNode()
            Dim optionB As New RiskNode()

            ' Setting properties for Option A
            optionA.Cost = 1000
            optionA.Gain = 2000
            optionA.Probability = 60

            ' Calculating expected monetary value for Option A
            optionA.ExpectedMonetaryValue = optionA.Gain * (optionA.Probability / 100)

            ' Calculating regret for Option A
            optionA.Regret = optionA.Gain - optionA.ExpectedMonetaryValue

            ' Setting properties for Option B
            optionB.Cost = 1500
            optionB.Gain = 3000
            optionB.Probability = 40

            ' Calculating expected monetary value for Option B
            optionB.ExpectedMonetaryValue = optionB.Gain * (optionB.Probability / 100)

            ' Calculating regret for Option B
            optionB.Regret = optionB.Gain - optionB.ExpectedMonetaryValue

            ' Comparing options and making a decision
            If optionA.ExpectedMonetaryValue > optionB.ExpectedMonetaryValue Then
                Console.WriteLine("Option A is more favorable.")
            ElseIf optionB.ExpectedMonetaryValue > optionA.ExpectedMonetaryValue Then
                Console.WriteLine("Option B is more favorable.")
            Else
                Console.WriteLine("Both options have equal expected monetary values.")
            End If

            ' Displaying calculated values
            Console.WriteLine("Option A:")
            Console.WriteLine($"  Expected Monetary Value: {optionA.ExpectedMonetaryValue}")
            Console.WriteLine($"  Regret: {optionA.Regret}")

            Console.WriteLine("Option B:")
            Console.WriteLine($"  Expected Monetary Value: {optionB.ExpectedMonetaryValue}")
            Console.WriteLine($"  Regret: {optionB.Regret}")
        End Sub
        Public Sub iCorpusExample()
            'Create Vocabulary
            Dim iCorpus As String = "the quick brown fox, jumped over the lazy dog."
            Dim NewVocabulary = Corpus.Vocabulary.CreateVocabulary(iCorpus, Corpus.Vocabulary.VocabularyType.Word)
            Console.WriteLine("vocabulary List: ")
            Dim str As String = ""
            For Each item In NewVocabulary
                str &= "entry :" & item.Text & vbTab & "Value :" & item.Encoding & vbNewLine

            Next
            Console.WriteLine(str)
            'Encode InputText
            Dim InputText As String = iCorpus

            Dim InputLayer As New InputTextRecord
            InputLayer.Text = iCorpus
            Console.WriteLine("Input layer: ")
            InputLayer.Encoding = Encode.Encode_Text(InputText, NewVocabulary, VocabularyType.Word)
            Console.WriteLine("Input Text: " & "[" & InputLayer.Text & "]" & vbNewLine)
            Console.WriteLine("Input Embedding: ")
            str = "["
            For Each item In InputLayer.Encoding
                str &= item & " "
            Next
            str &= "] "
            Console.WriteLine(str)
            Console.WriteLine(vbNewLine)
            'get inputs
            InputLayer.blocksize = 4
            InputLayer.Inputblocks = InputTextRecord.GetBlocks(InputLayer.Encoding, InputLayer.blocksize)
            Console.WriteLine("Input BlockSize: " & InputLayer.blocksize)
            Console.WriteLine("Input Blocks ")
            For Each lst In InputLayer.Inputblocks

                Dim block As String = ""
                For Each item In lst
                    block &= item & " "
                Next
                Console.WriteLine("[" & block & "]")
            Next
            Console.WriteLine(vbNewLine)
            Dim ofset = 1
            'get targets(add ofset to get targets further in the future   ofset < blocksize)

            InputLayer.Targetblocks = InputTextRecord.GetTargetBlocks(InputLayer.Encoding, InputLayer.blocksize)

            Console.WriteLine("Target BlockSize: " & InputLayer.blocksize)
            Console.WriteLine("Target ofset    : " & ofset)
            Console.WriteLine("Target Blocks  ")
            For Each lst In InputLayer.Targetblocks

                Dim block As String = ""
                For Each item In lst
                    block &= item & " "
                Next
                Console.WriteLine("[" & block & "]")
            Next
            Console.ReadLine()

        End Sub

        Sub MedicalDiagnosisExample()
            ' Define nodes and states
            Dim feverNode As New BeliefNode("Fever", New List(Of String) From {"High", "Moderate", "Low"})
            Dim coughNode As New BeliefNode("Cough", New List(Of String) From {"High", "Moderate", "Low"})
            Dim soreThroatNode As New BeliefNode("SoreThroat", New List(Of String) From {"Yes", "No"})
            Dim influenzaNode As New BeliefNode("Influenza", New List(Of String) From {"Yes", "No"})

            ' Create the belief network
            Dim diagnosisNetwork As New BeliefNetwork()
            diagnosisNetwork.AddNode(feverNode)
            diagnosisNetwork.AddNode(coughNode)
            diagnosisNetwork.AddNode(soreThroatNode)
            diagnosisNetwork.AddNode(influenzaNode)

            ' Add parent-child relationships
            diagnosisNetwork.AddEdge(feverNode, influenzaNode)
            diagnosisNetwork.AddEdge(coughNode, influenzaNode)
            diagnosisNetwork.AddEdge(soreThroatNode, influenzaNode)

            ' Load training data
            ' Provide training data for each node
            Dim trainingData As New Dictionary(Of String, Dictionary(Of List(Of String), Double))()

            ' Training data for Fever node
            Dim feverTrainingData As New Dictionary(Of List(Of String), Double)()
            feverTrainingData.Add(New List(Of String) From {"High"}, 0.3)
            feverTrainingData.Add(New List(Of String) From {"Moderate"}, 0.5)
            feverTrainingData.Add(New List(Of String) From {"Low"}, 0.2)
            trainingData.Add("Fever", feverTrainingData)

            ' Training data for Cough node
            Dim coughTrainingData As New Dictionary(Of List(Of String), Double)()
            coughTrainingData.Add(New List(Of String) From {"High"}, 0.4)
            coughTrainingData.Add(New List(Of String) From {"Moderate"}, 0.3)
            coughTrainingData.Add(New List(Of String) From {"Low"}, 0.3)
            trainingData.Add("Cough", coughTrainingData)

            ' Training data for SoreThroat node
            Dim soreThroatTrainingData As New Dictionary(Of List(Of String), Double)()
            soreThroatTrainingData.Add(New List(Of String) From {"Yes"}, 0.1)
            soreThroatTrainingData.Add(New List(Of String) From {"No"}, 0.9)
            trainingData.Add("SoreThroat", soreThroatTrainingData)

            ' Training data for Influenza node
            Dim influenzaTrainingData As New Dictionary(Of List(Of String), Double)()
            influenzaTrainingData.Add(New List(Of String) From {"High", "High", "Yes"}, 0.8)
            influenzaTrainingData.Add(New List(Of String) From {"High", "High", "No"}, 0.2)
            ' Add more conditional probabilities...
            trainingData.Add("Influenza", influenzaTrainingData)

            ' Provide training data...

            ' Define conditional probabilities
            diagnosisNetwork.DefineCPT("Fever", trainingData("Fever"))
            diagnosisNetwork.DefineCPT("Cough", trainingData("Cough"))
            diagnosisNetwork.DefineCPT("SoreThroat", trainingData("SoreThroat"))
            diagnosisNetwork.DefineCPT("Influenza", trainingData("Influenza"))

            ' Create evidence
            Dim evidence As New Dictionary(Of BeliefNode, String)()
            evidence.Add(feverNode, "High")
            evidence.Add(coughNode, "High")
            evidence.Add(soreThroatNode, "Yes")

            ' Predict Influenza
            Dim predictedOutcome As String = diagnosisNetwork.Predict(influenzaNode, evidence)
            Console.WriteLine("Predicted Influenza: " & predictedOutcome)

            ' Display network structure
            diagnosisNetwork.DisplayNetworkStructure()

            ' Visualize the network as a tree
            diagnosisNetwork.DisplayAsTree()

            ' Export network data to a file
            diagnosisNetwork.ExportToFile("diagnosis_network.txt")
        End Sub

        Public Sub TrillExample()
            Dim matrix(,) As Integer = {{1, 2, 3, 9}, {4, 5, 6, 8}, {7, 8, 9, 9}}

            Dim result(,) As Integer = NN.Tril.Tril(matrix)

            Console.WriteLine("Matrix:")
            NN.Tril.PrintMatrix(matrix)

            Console.WriteLine("Tril Result:")
            NN.Tril.PrintMatrix(result)
            Console.ReadLine()
        End Sub

        Public Sub ExampleConclusionDetector()
            Dim documents As List(Of String) = TrainingData.TrainingData.GenerateRandomTrainingData(ConclusionDetector.GetInternalConclusionIndicators)
            Dim count As Integer = 0
            For Each hypothesisStorage As CapturedType In ConclusionDetector.GetSentences(TrainingData.TrainingData.GenerateRandomTrainingData(ConclusionDetector.GetInternalConclusionIndicators))
                count += 1
                Console.WriteLine(count & ":")
                Console.WriteLine($"Sentence: {hypothesisStorage.Sentence}")
                Console.WriteLine("Hypothesis Classification: " & hypothesisStorage.SubType)
                Console.WriteLine("Logical Relationship: " & hypothesisStorage.LogicalRelation_)
                Console.WriteLine()
            Next
            Console.ReadLine()

        End Sub

        Public Sub ExamplePronounResolver()
            Dim sentence As String = "John gave Mary a book, and she thanked him for it."

            Dim iresolve As New PronounResolver
            Dim Pronouns As List(Of String) = iresolve.FemalePersonalNouns.ToList
            Pronouns.AddRange(iresolve.MalePersonalNouns)
            Pronouns.AddRange(iresolve.MalePronouns)
            Pronouns.AddRange(iresolve.FemalePronouns)
            Dim resolvedAntecedent As String = ""
            For Each item In Pronouns
                If sentence.ToLower.Contains(item.ToLower) Then


                    resolvedAntecedent = iresolve.ResolvePronoun(sentence.ToLower, item.ToLower)
                    Console.WriteLine($"Resolved antecedent for '{item}': {resolvedAntecedent}")
                    resolvedAntecedent = iresolve.ResolveGender(item)
                    Console.WriteLine($"Resolved Gender for '{item}': {resolvedAntecedent}")
                End If
            Next
            Console.WriteLine("finished")
            Console.ReadLine()


        End Sub
        Public Sub ExampleEntityClassifier()
            ' Example usage of EntityClassifier

            ' Sample input text
            Dim inputText As String = "John Smith and Sarah Johnson went to New York. They both work at XYZ Company. Jane works in the office, She is a NLP Specialist"

            ' Sample list of named entities
            Dim namedEntities As New List(Of String) From {"John Smith", "Sarah Johnson", "Jane", "New York", "XYZ Company", "the Office", "Specialist", "NLP"}

            ' Sample list of linking verbs for relationship extraction
            Dim linkingVerbs As New List(Of String) From {"went to", "work at", "works in"}
            Dim iClassifier As New EntityClassifier(namedEntities, "Business", linkingVerbs)

            Dim Result As List(Of DiscoveredEntity) = iClassifier.Forwards(inputText)
            ' Display the detected entities

            For Each item In Result.Distinct.ToList
                Console.WriteLine()
                Console.WriteLine("RESULT :")
                Console.WriteLine("-------------------------------------------")
                Console.WriteLine("Sentence :")
                Console.WriteLine(item.DiscoveredSentence)
                Console.WriteLine()
                Console.WriteLine("Detected Entities:")
                For Each Ent In item.DiscoveredEntitys
                    Console.WriteLine(Ent)
                Next
                Console.WriteLine()
                Console.WriteLine("Entitys With Context:")
                For Each Ent In item.EntitysWithContext
                    Console.WriteLine(Ent)
                Next
                Console.WriteLine()
                Console.WriteLine("Sentence Shape:")
                Console.WriteLine(item.SentenceShape)
                Console.WriteLine()
                Console.WriteLine("Entity Sentence :")
                Console.WriteLine(item.EntitySentence)
                Console.WriteLine()
                Console.WriteLine("Relationships:")
                For Each Ent In item.Relationships
                    Console.WriteLine("Source :" & Ent.SourceEntity)
                    Console.WriteLine("RelationType :" & Ent.RelationshipType)
                    Console.WriteLine("Target :" & Ent.TargetEntity)
                    Console.WriteLine("Relations: " & Ent.Sentence)
                    Console.WriteLine()
                Next
                Console.WriteLine("......................................")

            Next
            Console.WriteLine(".......Relation Extraction........")
            Console.WriteLine()
            For Each Ent In iClassifier.DiscoverEntityRelationships(inputText)
                Console.WriteLine("Related Entities:")
                For Each item In Ent.DiscoveredEntitys
                    Console.WriteLine(item)

                Next
                Console.WriteLine()
                Console.WriteLine("Relationships:")
                Console.WriteLine()
                For Each item In Ent.Relationships

                    Console.WriteLine("Source :" & item.SourceEntity)
                    Console.WriteLine("RelationType :" & item.RelationshipType)
                    Console.WriteLine("Target :" & item.TargetEntity)
                    Console.WriteLine("Relations: " & item.Sentence)
                    Console.WriteLine()

                Next
                Console.WriteLine()
                Console.WriteLine("__________")
                Console.WriteLine()
            Next
            Console.ReadLine()

        End Sub

        Public Sub ExampleContextAnalyzer()




            Dim sentences As New List(Of String)()
            sentences.Add("This is a premise sentence.")
            sentences.Add("Therefore, the conclusion follows.")
            sentences.Add("Based on the hypothesis, we can conclude that...")
            sentences.Add("In conclusion, the experiment results support the theory.")
            sentences.Add("This is a premise sentence.")
            sentences.Add("Therefore, the conclusion follows.")

            sentences.Add("Based on the hypothesis, we can conclude that...")
            sentences.Add("In conclusion, the experiment results support the theory.")
            sentences.Add("The question is whether...")
            sentences.Add("The answer to this question is...")
            sentences.Add("Please follow the instructions carefully.")
            sentences.Add("The task requires you to...")

            sentences.AddRange(TrainingData.TrainingData.iGetTrainingSentences)
            Dim analyzer As New ContextAnalyzer()
            Dim statementGroups As List(Of StatementGroup) = analyzer.GroupStatements(sentences)
            ' statementGroups.AddRange(analyzer.GetContextStatements(sentences))
            For Each group As StatementGroup In statementGroups
                Console.WriteLine("Statement Type: " & group.Type)
                Console.WriteLine("Sentences:")
                For Each sentence As String In group.Sentences
                    Console.WriteLine("- " & sentence)
                Next
                Console.WriteLine()
            Next
            Console.ReadLine()
        End Sub
        Public Sub ExampleSentenceClassifier()
            Dim documents As List(Of String) = TrainingData.TrainingData.iGetTrainingSentences()
            Console.WriteLine("Docs :" & documents.Count)
            Dim lst As List(Of ClassifiedSentence) = ClassifySentences(documents)
            Console.WriteLine("Premises")
            Console.WriteLine()
            Dim Count As Integer = 0
            For Each item As ClassifiedSentence In lst
                If item.Type = "Premise" Then
                    Count += 1
                    Console.WriteLine(Count & ":")

                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next
            Console.WriteLine()
            Console.WriteLine()
            Console.WriteLine("Hypotheses")
            Console.WriteLine()
            Count = 0
            For Each item As ClassifiedSentence In lst
                If item.Type = "Hypotheses" Then
                    Count += 1
                    Console.WriteLine(Count & ":")
                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next

            Console.WriteLine()
            Console.WriteLine()
            Console.WriteLine("Conclusions")
            Console.WriteLine()
            Count = 0
            For Each item As ClassifiedSentence In lst
                If item.Type = "Conclusion" Then
                    Count += 1
                    Console.WriteLine(Count & ":")
                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next

            Console.WriteLine()
            Console.WriteLine()
            Console.WriteLine("Questions")
            Console.WriteLine()
            Count = 0
            For Each item As ClassifiedSentence In lst
                If item.Type = "Question" Then
                    Count += 1
                    Console.WriteLine(Count & ":")
                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next





            Count = 0
            Console.WriteLine()
            Console.WriteLine()
            Console.WriteLine("Unclassified")
            Console.WriteLine()
            For Each item As ClassifiedSentence In lst
                If item.Type = "Unknown Classification" Or item.Entity.SubType = "Unknown" Or item.Entity.LogicalRelation_ = "Unknown" Then
                    Count += 1
                    Console.WriteLine(Count & ":")
                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next
            Console.ReadLine()

        End Sub



        ' Usage Example:
        Public Sub CorpusWordListReaderExample()
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
        Public Sub ContextAnalyzerExample()




            Dim sentences As New List(Of String)()
            sentences.Add("This is a premise sentence.")
            sentences.Add("Therefore, the conclusion follows.")
            sentences.Add("Based on the hypothesis, we can conclude that...")
            sentences.Add("In conclusion, the experiment results support the theory.")
            sentences.Add("This is a premise sentence.")
            sentences.Add("Therefore, the conclusion follows.")

            sentences.Add("Based on the hypothesis, we can conclude that...")
            sentences.Add("In conclusion, the experiment results support the theory.")
            sentences.Add("The question is whether...")
            sentences.Add("The answer to this question is...")
            sentences.Add("Please follow the instructions carefully.")
            sentences.Add("The task requires you to...")

            sentences.AddRange(TrainingData.TrainingData.iGetTrainingSentences)
            Dim analyzer As New ContextAnalyzer()
            Dim statementGroups As List(Of ContextAnalyzer.StatementGroup) = analyzer.GroupStatements(sentences)
            ' statementGroups.AddRange(analyzer.GetContextStatements(sentences))
            For Each group As ContextAnalyzer.StatementGroup In statementGroups
                Console.WriteLine("Statement Type: " & group.Type)
                Console.WriteLine("Sentences:")
                For Each sentence As String In group.Sentences
                    Console.WriteLine("- " & sentence)
                Next
                Console.WriteLine()
            Next
            Console.ReadLine()
        End Sub
        Public Sub SentenceClassifierExample()
            Dim documents As List(Of String) = TrainingData.TrainingData.iGetTrainingSentences()
            Console.WriteLine("Docs :" & documents.Count)
            Dim lst As List(Of ClassifiedSentence) = ClassifySentences(documents)
            Console.WriteLine("Premises")
            Console.WriteLine()
            Dim Count As Integer = 0
            For Each item As ClassifiedSentence In lst
                If item.Type = "Premise" Then
                    Count += 1
                    Console.WriteLine(Count & ":")

                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next
            Console.WriteLine()
            Console.WriteLine()
            Console.WriteLine("Hypotheses")
            Console.WriteLine()
            Count = 0
            For Each item As ClassifiedSentence In lst
                If item.Type = "Hypotheses" Then
                    Count += 1
                    Console.WriteLine(Count & ":")
                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next

            Console.WriteLine()
            Console.WriteLine()
            Console.WriteLine("Conclusions")
            Console.WriteLine()
            Count = 0
            For Each item As ClassifiedSentence In lst
                If item.Type = "Conclusion" Then
                    Count += 1
                    Console.WriteLine(Count & ":")
                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next

            Console.WriteLine()
            Console.WriteLine()
            Console.WriteLine("Questions")
            Console.WriteLine()
            Count = 0
            For Each item As ClassifiedSentence In lst
                If item.Type = "Question" Then
                    Count += 1
                    Console.WriteLine(Count & ":")
                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next





            Count = 0
            Console.WriteLine()
            Console.WriteLine()
            Console.WriteLine("Unclassified")
            Console.WriteLine()
            For Each item As ClassifiedSentence In lst
                If item.Type = "Unknown Classification" Or item.Entity.SubType = "Unknown" Or item.Entity.LogicalRelation_ = "Unknown" Then
                    Count += 1
                    Console.WriteLine(Count & ":")
                    Console.WriteLine($"Sentence: {item.Entity.Sentence}")
                    Console.WriteLine("Classification: " & item.Type)
                    Console.WriteLine("Sub Type: " & item.Entity.SubType)
                    Console.WriteLine("Logical Relation Type: " & item.Entity.LogicalRelation_)
                    Console.WriteLine()
                Else
                End If

            Next
            Console.ReadLine()

        End Sub
        Public Sub PremiseDetectorExample()
            Dim detect As New EntityModels
            Dim count As Integer = 0

            For Each item In TrainingData.TrainingData.GenerateRandomTrainingData(HypothesisDetector.GetInternalHypothesisIndicators)
                If SentenceClassifier.EntityModels.DetectPremise(item) <> "" Then


                    Dim x = EntityModels.GetSentence(item)

                    count += 1
                    Console.WriteLine(count & ":")
                    Console.WriteLine($"Sentence {x.Sentence}")
                    Console.WriteLine($"Premise Subtype {x.SubType}")
                    Console.WriteLine("Classification " & x.LogicalRelation_)
                    Console.WriteLine()
                End If

            Next



            Console.ReadLine()
        End Sub

        Public Sub ConclusionDetectorExample()
            Dim documents As List(Of String) = TrainingData.TrainingData.GenerateRandomTrainingData(ConclusionDetector.GetInternalConclusionIndicators)
            Dim count As Integer = 0
            For Each hypothesisStorage As CapturedType In ConclusionDetector.GetSentences(TrainingData.TrainingData.GenerateRandomTrainingData(ConclusionDetector.GetInternalConclusionIndicators))
                count += 1
                Console.WriteLine(count & ":")
                Console.WriteLine($"Sentence: {hypothesisStorage.Sentence}")
                Console.WriteLine("Hypothesis Classification: " & hypothesisStorage.SubType)
                Console.WriteLine("Logical Relationship: " & hypothesisStorage.LogicalRelation_)
                Console.WriteLine()
            Next
            Console.ReadLine()

        End Sub
        Public Sub HypothesisDetectorExample()
            Dim documents As List(Of String) = TrainingData.TrainingData.GenerateRandomTrainingData(HypothesisDetector.GetInternalHypothesisIndicators)
            Dim count As Integer = 0
            For Each hypothesisStorage As CapturedType In HypothesisDetector.GetSentences(TrainingData.TrainingData.GenerateRandomTrainingData(HypothesisDetector.GetInternalHypothesisIndicators))
                count += 1
                Console.WriteLine(count & ":")
                Console.WriteLine($"Sentence: {hypothesisStorage.Sentence}")
                Console.WriteLine("Hypothesis Classification: " & hypothesisStorage.SubType)
                Console.WriteLine("Logical Relationship: " & hypothesisStorage.LogicalRelation_)
                Console.WriteLine()
            Next
            Console.ReadLine()

        End Sub


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


        Public Sub DecisionNodeExample()
            ' Create decision nodes
            Dim ageDecision As New DecisionNode("age", "adult")
            Dim weatherDecision As New DecisionNode("weather", "sunny")
            Dim ageLeftDecision As New DecisionNode("age", "child")
            Dim weatherLeftDecision As New DecisionNode("weather", "rainy")
            Dim activityDecision As New DecisionNode("activity", "yes")
            Dim noActivityDecision As New DecisionNode("activity", "no")

            ' Connect nodes to build the decision tree
            ageDecision.Left = ageLeftDecision
            ageDecision.Right = weatherDecision
            ageLeftDecision.Left = noActivityDecision
            ageLeftDecision.Right = weatherLeftDecision
            weatherLeftDecision.Left = noActivityDecision
            weatherLeftDecision.Right = activityDecision
            weatherDecision.Left = activityDecision

            ' Define a sample feature set
            Dim features As New Dictionary(Of String, String) From {
            {"age", "child"},
            {"weather", "rainy"}
        }

            ' Make a decision using the decision tree
            Dim decisionTreeRoot As DecisionNode = ageDecision
            Dim result As String = decisionTreeRoot.MakeDecision(features)
            Console.WriteLine("Decision: " & result) ' Expected output: "no"
        End Sub

        Public Sub BinaryTreeExample()
            Dim rootNode As New BinaryNode(10)
            rootNode.insert(5)
            rootNode.insert(15)
            rootNode.insert(3)
            rootNode.insert(7)
            rootNode.insert(12)
            rootNode.insert(18)

            ' Create a TreeView control to visualize the binary tree
            Dim treeView As New TreeView()

            ' Set the TreeView property for the root node
            rootNode.Tree = treeView

            ' Print the binary tree in different orders
            Console.WriteLine("In-Order Traversal:")
            rootNode.PrintInOrder()

            Console.WriteLine(vbNewLine & "Post-Order Traversal:")
            rootNode.PrintPostOrder()

            Console.WriteLine(vbNewLine & "Pre-Order Traversal:")
            rootNode.PrintPreOrder()

            ' Display the TreeView control to visualize the binary tree
            Dim form As New Form()
            form.Controls.Add(treeView)
            form.ShowDialog()
        End Sub


        Public Sub ExampleTrieTreePrediction()
            Dim iTree As New Trees.TrieTree
            iTree = iTree.MakeTrieTree()
            Dim trainingPaths As List(Of List(Of String)) = iTree.root.GetPathsToLeafNodes()

            Dim wordgramModel As New Predict()
            wordgramModel.n = 2 ' Set the desired n-gram order
            wordgramModel.Train(trainingPaths)

            Dim generatedSentence As String = wordgramModel.ForwardsSentence()
            Console.WriteLine("Generated Sentence: " & generatedSentence)
        End Sub

    End Module
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
    Public Class TokenizerExample
        Public Sub Main()
            Dim Corpus As List(Of String) = GetDocumentCorpus()
            Dim sentences As New List(Of String) From {
            "I love apples.",
            "Bananas are tasty."}
            Dim Tokenizer As New Advanced
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
