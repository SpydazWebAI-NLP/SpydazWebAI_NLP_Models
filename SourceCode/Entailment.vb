Imports System.IO
Imports System.Text.RegularExpressions
Imports System.Windows.Forms
Imports InputModelling.Models.Readers

Namespace Models
    Namespace Entailment
        Public Structure CapturedType

            Public Property Sentence As String
            Public Property LogicalRelation_ As String
            Public Property SubType As String
        End Structure
        Public Structure ClassifiedSentence
            Public Classified As Boolean
            Public Type As String
            Public Entity As CapturedType
        End Structure
        Public Structure ClassificationRule
            Public Property Type As String
            Public Property Subtype As String
            Public Property Relationship As String
            Public Property Patterns As List(Of Regex)
        End Structure
        Public Class EntailmentClassifier
            Public conclusionIndicators As String() = {"therefore", "thus", "consequently", "hence", "in conclusion", "Therefore", "Thus", "As a result"}
            Public hypothesisIndicators As String() = {"if", "when", "suppose that", "let's say", "assuming that", "Suppose", "Assume", "In the case that", "Given that"}
            Public premiseIndicators As String() = {"based on", "according to", "given", "assuming", "since", "According to", "Based on", "In light of", "Considering", "The evidence suggests"}
            Private premises As New List(Of String)
            Private hypotheses As New List(Of String)
            Private conclusions As New List(Of String)

            Public Sub New(conclusionIndicators() As String, hypothesisIndicators() As String, premiseIndicators() As String)
                If conclusionIndicators Is Nothing Then
                    Throw New ArgumentNullException(NameOf(conclusionIndicators))
                End If

                If hypothesisIndicators Is Nothing Then
                    Throw New ArgumentNullException(NameOf(hypothesisIndicators))
                End If

                If premiseIndicators Is Nothing Then
                    Throw New ArgumentNullException(NameOf(premiseIndicators))
                End If

                Me.conclusionIndicators = conclusionIndicators
                Me.hypothesisIndicators = hypothesisIndicators
                Me.premiseIndicators = premiseIndicators
            End Sub

            Public Sub New()
            End Sub

            Public Function IsPremise(ByVal sentence As String) As Boolean

                ' Check if any of the premise indicators are present in the sentence
                For Each indicator In premiseIndicators
                    If sentence.Contains(indicator) Then
                        Return True
                    End If
                Next

                Return False
            End Function

            Public Function IsHypothesis(ByVal sentence As String) As Boolean
                ' List of indicator phrases for hypotheses

                ' Check if any of the hypothesis indicators are present in the sentence
                For Each indicator In hypothesisIndicators
                    If sentence.Contains(indicator) Then
                        Return True
                    End If
                Next

                Return False
            End Function

            Public Function IsConclusion(ByVal sentence As String) As Boolean
                ' List of indicator phrases for conclusions

                ' Check if any of the conclusion indicators are present in the sentence
                For Each indicator In conclusionIndicators
                    If sentence.Contains(indicator) Then
                        Return True
                    End If
                Next

                Return False
            End Function

            Public Function Classify(ByVal document As String) As Dictionary(Of String, List(Of String))

                ' Define a list of possible end-of-sentence punctuation markers
                Dim endOfSentenceMarkers As String() = {".", "!", "?"}

                ' Split the document into sentences
                Dim sentences As String() = document.Split(endOfSentenceMarkers, StringSplitOptions.RemoveEmptyEntries)

                ' Rule-based classification
                For Each sentence In sentences
                    ' Remove leading/trailing spaces and convert to lowercase
                    sentence = sentence.Trim().ToLower()

                    ' Check if the sentence is a premise, hypothesis, or conclusion
                    If IsPremise(sentence) Then
                        premises.Add(sentence)
                    ElseIf IsHypothesis(sentence) Then
                        hypotheses.Add(sentence)
                    ElseIf IsConclusion(sentence) Then
                        conclusions.Add(sentence)
                    End If
                Next

                ' Store the classified sentences in a dictionary
                Dim classifiedSentences As New Dictionary(Of String, List(Of String))
                classifiedSentences.Add("Premise", premises)
                classifiedSentences.Add("Hypothesis", hypotheses)
                classifiedSentences.Add("Conclusion", conclusions)

                Return classifiedSentences
            End Function

            Public Sub DisplayClassifiedSentences()
                Console.WriteLine("Premises:")
                For Each premise In premises
                    Console.WriteLine(premise)
                Next

                Console.WriteLine("Hypotheses:")
                For Each hypothesis In hypotheses
                    Console.WriteLine(hypothesis)
                Next

                Console.WriteLine("Conclusions:")
                For Each conclusion In conclusions
                    Console.WriteLine(conclusion)
                Next

                Console.WriteLine()
            End Sub

            Public Function ClassifySentence(sentence As String, document As String) As String
                ' Rule-based sentence classification

                For Each indicator As String In premiseIndicators
                    If sentence.StartsWith(indicator) OrElse sentence.Contains(indicator) Then
                        Return "Premise"
                    End If
                Next

                For Each indicator As String In hypothesisIndicators
                    If sentence.StartsWith(indicator) OrElse sentence.Contains(indicator) Then
                        Return "Hypothesis"
                    End If
                Next

                For Each indicator As String In conclusionIndicators
                    If sentence.StartsWith(indicator) OrElse sentence.Contains(indicator) Then
                        Return "Conclusion"
                    End If
                Next

                ' Rule 2: Syntactic Patterns

                For Each item In premiseIndicators
                    Dim premisePattern As String = "(?i)(\b" & item & "\b.*?)"
                    If Regex.IsMatch(sentence, premisePattern) Then Return "Premise"
                Next
                For Each item In conclusionIndicators
                    Dim conclusionPattern As String = "(?i)(\b" & item & "\b.*?)"
                    If Regex.IsMatch(sentence, conclusionPattern) Then Return "Conclusion"
                Next
                For Each item In hypothesisIndicators
                    Dim hypothesisPattern As String = "(?i)(\b" & item & "\b.*?)"
                    If Regex.IsMatch(sentence, hypothesisPattern) Then Return "Hypothesis"
                Next

                Return "Unknown"
            End Function

            ''' <summary>
            ''' Attempts to Resolve any relational sentences
            ''' </summary>
            ''' <param name="document"></param>
            ''' <returns></returns>
            Public Function ClassifyAndResolve(ByVal document As String) As Dictionary(Of String, String)
                Dim premises As New Dictionary(Of Integer, String)()
                Dim hypotheses As New Dictionary(Of Integer, String)()
                Dim conclusions As New Dictionary(Of Integer, String)()

                ' Split the document into sentences
                Dim sentences As String() = document.Split(New String() {". "}, StringSplitOptions.RemoveEmptyEntries)

                ' Rule-based resolution
                Dim index As Integer = 1
                For Each sentence In sentences
                    ' Remove leading/trailing spaces and convert to lowercase
                    sentence = sentence.Trim().ToLower()

                    ' Check if the sentence is a premise, hypothesis, or conclusion
                    If IsPremise(sentence) Then
                        premises.Add(index, sentence)
                    ElseIf IsHypothesis(sentence) Then
                        hypotheses.Add(index, sentence)
                    ElseIf IsConclusion(sentence) Then
                        conclusions.Add(index, sentence)
                    End If

                    index += 1
                Next

                ' Resolve the relationships based on the antecedents
                Dim resolvedSentences As New Dictionary(Of String, String)()
                For Each conclusionKvp As KeyValuePair(Of Integer, String) In conclusions
                    Dim conclusionIndex As Integer = conclusionKvp.Key
                    Dim conclusionSentence As String = conclusionKvp.Value

                    ' Find the antecedent hypothesis for the conclusion
                    Dim hypothesisIndex As Integer = conclusionIndex - 1
                    Dim hypothesisSentence As String = ""
                    If hypotheses.ContainsKey(hypothesisIndex) Then
                        hypothesisSentence = hypotheses(hypothesisIndex)
                    End If

                    ' Find the antecedent premises for the hypothesis
                    Dim premiseIndexes As New List(Of Integer)()
                    For i As Integer = hypothesisIndex - 1 To 1 Step -1
                        If premises.ContainsKey(i) Then
                            premiseIndexes.Add(i)
                        Else
                            Exit For
                        End If
                    Next

                    ' Build the resolved sentences
                    Dim resolvedSentence As String = ""
                    If Not String.IsNullOrEmpty(hypothesisSentence) Then
                        resolvedSentence += "Hypothesis: " + hypothesisSentence + " "
                    End If
                    For i As Integer = premiseIndexes.Count - 1 To 0 Step -1
                        resolvedSentence += "Premise " + (premiseIndexes.Count - i).ToString() + ": " + premises(premiseIndexes(i)) + " "
                    Next
                    resolvedSentence += "Conclusion: " + conclusionSentence

                    resolvedSentences.Add("Conclusion " + conclusionIndex.ToString(), resolvedSentence)
                Next

                Return resolvedSentences
            End Function

            Public Function ResolveKnown(ByVal classifiedSentences As Dictionary(Of String, List(Of String))) As Dictionary(Of String, String)
                Dim resolvedSentences As New Dictionary(Of String, String)()

                ' Resolve relationships based on the antecedents
                Dim premiseCount As Integer = classifiedSentences("Premise").Count
                Dim hypothesisCount As Integer = classifiedSentences("Hypothesis").Count
                Dim conclusionCount As Integer = classifiedSentences("Conclusion").Count

                ' Check if the counts are consistent for resolution
                If hypothesisCount = conclusionCount AndAlso hypothesisCount = 1 AndAlso premiseCount >= 1 Then
                    Dim hypothesis As String = classifiedSentences("Hypothesis")(0)
                    Dim conclusion As String = classifiedSentences("Conclusion")(0)

                    Dim resolvedSentence As String = "Hypothesis: " + hypothesis + Environment.NewLine
                    For i As Integer = 1 To premiseCount
                        Dim premise As String = classifiedSentences("Premise")(i - 1)
                        resolvedSentence += "Premise " + i.ToString() + ": " + premise + Environment.NewLine
                    Next
                    resolvedSentence += "Conclusion: " + conclusion
                    resolvedSentences.Add("Resolved", resolvedSentence)
                Else
                    resolvedSentences.Add("Error", "Unable to resolve relationships. Counts are inconsistent.")
                End If

                Return resolvedSentences
            End Function

        End Class
        Public Class LogicalDependencyClassifier
            Private Shared ReadOnly CauseAndEffectPattern As Regex = New Regex("(?i)(cause|effect|result in|lead to|because|due to|consequently)")
            Private Shared ReadOnly ComparisonPattern As Regex = New Regex("(?i)(compared to|greater than|less than|similar to|different from|between)")
            Private Shared ReadOnly ConditionPattern As Regex = New Regex("(?i)(if|unless|only if|when|provided that|in the case of)")
            Private Shared ReadOnly GeneralizationPattern As Regex = New Regex("(?i)(all|every|always|none|never|in general|could|would|maybe|Is a)")
            Private Shared ReadOnly TemporalSequencePattern As Regex = New Regex("(?i)(before|after|during|while|subsequently|previously|simultaneously|when|at the time of|next)")

            Public Shared CauseEntityList As String() = {"cause", "reason", "factor", "based on", "indicates", "lead to", "due to", "consequently", "because", "was provided"}
            Public Shared EffectEntityList As String() = {"effect", "result", "outcome", "was the consequence of", "end process of", "because of", "reason for"}
            Public Shared ComparableObject1EntityList As String() = {"first object", "object A"}
            Public Shared ComparableObject2EntityList As String() = {"second object", "object B"}
            Public Shared ConditionEntityList As String() = {"condition", "requirement", "prerequisite", "if", "when", "then", "but", "And", "Not", "Or", "less than", "greater than"}
            Public Shared GeneralizedObjectEntityList As String() = {"generalized object", "common element", "universal attribute"}
            Public Shared Event1EntityList As String() = {"first event", "event A"}
            Public Shared Event2EntityList As String() = {"second event", "event B"}

            Public Shared ReadOnly DependencyPatterns As Dictionary(Of String, List(Of Regex)) = New Dictionary(Of String, List(Of Regex)) From {
            {"Causal Dependency", New List(Of Regex) From {
                New Regex(".*Is\s+the\s+cause\s+of\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*leads\s+to\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*causes\s+.*", RegexOptions.IgnoreCase),
                CauseAndEffectPattern
            }},
            {"Comparison Dependency", New List(Of Regex) From {
                ComparisonPattern
            }},
            {"Conditional Dependency", New List(Of Regex) From {
                ConditionPattern
            }},
            {"Generalization Dependency", New List(Of Regex) From {
                GeneralizationPattern
            }},
            {"Temporal Sequence Dependency", New List(Of Regex) From {
                TemporalSequencePattern
            }},
            {"Premise", New List(Of Regex) From {
                New Regex(".*infers\s+that\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*Is\s+deduced\s+from\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*drawn\s+from\s+.*", RegexOptions.IgnoreCase),
                New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*would\s+have\s+occurred\s+if\s+.*", RegexOptions.IgnoreCase),
                New Regex("Based\s+on\s+statistics,\s+.*", RegexOptions.IgnoreCase),
                New Regex("According\s+to\s+the\s+survey,\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*Is\s+similar\s+to\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*Is\s+analogous\s+to\s+.*", RegexOptions.IgnoreCase),
                New Regex("For\s+example,\s+.*", RegexOptions.IgnoreCase),
                New Regex("In\s+support\s+of\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*Is\s+backed\s+by\s+.*", RegexOptions.IgnoreCase),
                New Regex("In\s+general,\s+.*", RegexOptions.IgnoreCase),
                New Regex("Typically,\s+.*", RegexOptions.IgnoreCase),
                New Regex("Most\s+of\s+the\s+time,\s+.*", RegexOptions.IgnoreCase),
                New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*relies\s+on\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*Is\s+the\s+cause\s+of\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*leads\s+to\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*causes\s+.*", RegexOptions.IgnoreCase),
                New Regex("In\s+fact,\s+.*", RegexOptions.IgnoreCase),
                New Regex("Indeed,\s+.*", RegexOptions.IgnoreCase),
                New Regex(".*Is\s+a\s+fact\s+that\s+.*", RegexOptions.IgnoreCase),
                CauseAndEffectPattern,
                ComparisonPattern,
                ConditionPattern,
                GeneralizationPattern,
                TemporalSequencePattern
            }}
        }

            Public Function DetectLogicalDependancyType(ByVal premise As String) As String
                For Each premisePattern In DependencyPatterns
                    If premisePattern.Value.Any(Function(indicatorRegex) indicatorRegex.IsMatch(premise)) Then
                        Return premisePattern.Key
                    End If
                Next

                Return "Unknown"
            End Function

            Public Shared Function ClassifyLogicalDependency(statement As String) As String
                If CauseAndEffectPattern.IsMatch(statement) Then
                    Return "Cause And Effect"
                ElseIf ComparisonPattern.IsMatch(statement) Then
                    Return "Comparison"
                ElseIf ConditionPattern.IsMatch(statement) Then
                    Return "Condition"
                ElseIf GeneralizationPattern.IsMatch(statement) Then
                    Return "Generalization"
                ElseIf TemporalSequencePattern.IsMatch(statement) Then
                    Return "Temporal Sequence"
                Else
                    Return "Unknown"
                End If
            End Function

        End Class
        Public Class SentenceClassifier



            Private ReadOnly ClassificationRules As List(Of ClassificationRule)
            Public Shared Function IsPremise(ByVal sentence As String) As Boolean
                ' List of indicator phrases for premises
                Dim indicatorPhrases As New List(Of String) From {"premise"}
                If EntityModels.DetectPremise(sentence) <> "" Then Return True
                For Each phrase As String In indicatorPhrases
                    ' Match the phrase at the beginning of the sentence
                    Dim match As Match = Regex.Match(sentence, "^\s*" + phrase + ":", RegexOptions.IgnoreCase)
                    If match.Success Then
                        Return True
                    End If
                Next
                If EntityModels.ClassifySentence(sentence) = "Unknown" Then
                    Return False
                Else
                    Return True
                End If
                Return False
            End Function
            Public Shared Function IsConclusion(ByVal sentence As String) As Boolean
                ' List of indicator phrases for hypotheses
                Dim indicatorPhrases As New List(Of String) From {"Conclusion"}
                If ConclusionDetector.DetectConclusion(sentence) <> "" Then Return True
                For Each phrase As String In indicatorPhrases
                    ' Match the phrase at the beginning of the sentence
                    Dim match As Match = Regex.Match(sentence, "^\s*" + phrase + ":", RegexOptions.IgnoreCase)
                    If match.Success Then
                        Return True
                    End If
                Next
                If ConclusionDetector.ClassifySentence(sentence) = "Unclassified" Then
                    Return False
                Else
                    Return True
                End If
                Return False
            End Function
            Public Shared Function IsHypothesis(ByVal sentence As String) As Boolean
                ' List of indicator phrases for hypotheses
                Dim indicatorPhrases As New List(Of String) From {"hypothesis"}
                If HypothesisDetector.DetectHypothesis(sentence) <> "" Then Return True
                For Each phrase As String In indicatorPhrases
                    ' Match the phrase at the beginning of the sentence
                    Dim match As Match = Regex.Match(sentence, "^\s*" + phrase + ":", RegexOptions.IgnoreCase)
                    If match.Success Then
                        Return True
                    End If
                Next
                If HypothesisDetector.ClassifySentence(sentence) = "Unclassified" Then
                    Return False
                Else
                    Return True
                End If
                Return False
            End Function
            Public Shared Function IsQuestion(ByVal sentence As String) As Boolean
                ' List of indicator phrases for hypotheses
                Dim indicatorPhrases As New List(Of String) From {"Question"}
                If QuestionDetector.ClassifySentence(sentence) <> "Unknown" Then Return True
                For Each phrase As String In indicatorPhrases
                    ' Match the phrase at the beginning of the sentence
                    Dim match As Match = Regex.Match(sentence, "^\s*" + phrase + ":", RegexOptions.IgnoreCase)
                    If match.Success Then
                        Return True
                    End If
                Next

                Return False
            End Function

            Public Shared Function ClassifySentences(ByRef Sentences As List(Of String)) As List(Of ClassifiedSentence)
                Dim lst As New List(Of ClassifiedSentence)
                For Each item In Sentences
                    Dim classified As New ClassifiedSentence
                    classified.Classified = False
                    Dim Captured As New CapturedType
                    If IsPremise(item) = True Then
                        If EntityModels.DetectPremise(item) <> "" Then

                            classified.Type = "Premise"
                            classified.Entity = EntityModels.GetSentence(item)
                            classified.Classified = True
                            lst.Add(classified)
                        Else
                            'Captured = New CapturedType
                            'Captured.Sentence = item
                            'Captured.SubType = PremiseDetector.ExtractPremiseSubtype(item)
                            'Captured.LogicalRelation_ = PremiseDetector.ClassifySentence(item)
                            'classified.Entity = Captured
                            'classified.Type = "Premise"
                            'classified.Classified = True
                            'lst.Add(classified)
                        End If
                    Else
                    End If



                    If IsHypothesis(item) = True Then
                        If HypothesisDetector.DetectHypothesis(item) <> "" Then

                            classified.Type = "Hypotheses"
                            classified.Entity = HypothesisDetector.GetSentence(item)
                            classified.Classified = True
                            lst.Add(classified)
                        Else
                            'Captured = New CapturedType
                            'Captured.Sentence = item
                            'Captured.SubType = HypothesisDetector.ClassifyHypothesis(item)
                            'Captured.LogicalRelation_ = HypothesisDetector.ClassifySentence(item)
                            'classified.Entity = Captured
                            'classified.Type = "Hypotheses"
                            'classified.Classified = True
                            'lst.Add(classified)
                        End If
                    Else
                    End If


                    If IsConclusion(item) = True Then
                        If ConclusionDetector.DetectConclusion(item) <> "" Then

                            classified.Type = "Conclusion"
                            classified.Entity = ConclusionDetector.GetSentence(item)
                            classified.Classified = True
                            lst.Add(classified)
                        Else
                            'Captured = New CapturedType
                            'Captured.Sentence = item
                            'Captured.SubType = ConclusionDetector.ClassifyConclusion(item)
                            'Captured.LogicalRelation_ = ConclusionDetector.ClassifySentence(item)
                            'classified.Entity = Captured
                            'classified.Type = "Conclusion"
                            'classified.Classified = True
                            'lst.Add(classified)
                        End If
                    Else
                    End If

                    If IsQuestion(item) = True Then
                        If QuestionDetector.ClassifySentence(item) <> "Unclassified" Then

                            classified.Type = "Question"
                            classified.Entity = QuestionDetector.GetSentence(item)
                            classified.Classified = True
                            lst.Add(classified)
                        Else
                            'Captured = New CapturedType
                            'Captured.Sentence = item
                            'Captured.SubType = PremiseDetector.ExtractPremiseSubtype(item)
                            'Captured.LogicalRelation_ = PremiseDetector.ClassifySentence(item)
                            'classified.Entity = Captured
                            'classified.Type = "Question"
                            'classified.Classified = True
                            'lst.Add(classified)
                        End If
                    Else
                    End If

                    'Else
                    If classified.Classified = False Then

                        classified.Type = "Unknown Classification"
                        Captured = New CapturedType
                        Captured.Sentence = item
                        Captured.LogicalRelation_ = LogicalDependencyClassifier.ClassifyLogicalDependency(item)
                        Captured.SubType = LogicalDependencyClassifier.ClassifyLogicalDependency(item)
                        classified.Entity = Captured
                        lst.Add(classified)


                    End If


                Next

                Return lst.Distinct.ToList
            End Function

            Public Sub New()

                ClassificationRules = New List(Of ClassificationRule)

            End Sub
            Public Shared Function InitializeClassificationRules() As List(Of ClassificationRule)



                ' Add your existing classification rules here
                Dim ClassificationRules As New List(Of ClassificationRule)






                ' Inductive Reasoning Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Premise",
        .Subtype = "Inductive Reasoning",
        .Relationship = "Supports",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bBased on\b", RegexOptions.IgnoreCase),
            New Regex("^\bObserving\b", RegexOptions.IgnoreCase),
            New Regex("^\bEmpirical evidence suggests\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Conclusion",
        .Subtype = "Inductive Reasoning",
        .Relationship = "Inferred From",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bTherefore, it can be inferred\b", RegexOptions.IgnoreCase),
            New Regex("^\bIt is likely that\b", RegexOptions.IgnoreCase),
            New Regex("^\bGeneralizing from the evidence\b", RegexOptions.IgnoreCase)
        }
    })

                ' Deductive Reasoning Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Premise",
        .Subtype = "Deductive Reasoning",
        .Relationship = "Supports",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bGiven that\b", RegexOptions.IgnoreCase),
            New Regex("^\bIf\b", RegexOptions.IgnoreCase),
            New Regex("^\bAssuming\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Conclusion",
        .Subtype = "Deductive Reasoning",
        .Relationship = "Follows From",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bTherefore\b", RegexOptions.IgnoreCase),
            New Regex("^\bIt follows that\b", RegexOptions.IgnoreCase),
            New Regex("^\bSo\b", RegexOptions.IgnoreCase)
        }
    })

                ' Abductive Reasoning Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Hypothesis",
        .Subtype = "Abductive Reasoning",
        .Relationship = "Supports",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bIt is possible that\b", RegexOptions.IgnoreCase),
            New Regex("^\bTo explain the observation\b", RegexOptions.IgnoreCase),
            New Regex("^\bSuggesting a likely explanation\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Conclusion",
        .Subtype = "Abductive Reasoning",
        .Relationship = "Explains",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bTherefore, the best explanation is\b", RegexOptions.IgnoreCase),
            New Regex("^\bThe most plausible conclusion is\b", RegexOptions.IgnoreCase),
            New Regex("^\bThe evidence supports the hypothesis that\b", RegexOptions.IgnoreCase)
        }
    })








                ' Straw Man Argument Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "FallacyPremise ",
        .Subtype = "Straw Man Argument",
        .Relationship = "Incorrect truth",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bMisrepresenting\b", RegexOptions.IgnoreCase),
            New Regex("^\bExaggerating\b", RegexOptions.IgnoreCase),
            New Regex("^\bDistorting\b", RegexOptions.IgnoreCase)
        }
    })

                ' Fallacy Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Fallacy Premise",
        .Subtype = "deductive",
        .Relationship = "Circular argument",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bAd Hominem\b", RegexOptions.IgnoreCase),
            New Regex("^\bCircular Reasoning\b", RegexOptions.IgnoreCase),
            New Regex("^\bFalse Cause\b", RegexOptions.IgnoreCase)
        }
    })




                ' Inductive Reasoning Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Premise",
        .Subtype = "Inductive Reasoning",
        .Relationship = "Inductive",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bBased on\b", RegexOptions.IgnoreCase),
            New Regex("^\bObserving\b", RegexOptions.IgnoreCase),
            New Regex("^\bEmpirical evidence suggests\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Conclusion",
        .Subtype = "Inductive Reasoning",
        .Relationship = "Inductive",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bTherefore, it can be inferred\b", RegexOptions.IgnoreCase),
            New Regex("^\bIt is likely that\b", RegexOptions.IgnoreCase),
            New Regex("^\bGeneralizing from the evidence\b", RegexOptions.IgnoreCase)
        }
    })

                ' Deductive Reasoning Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Premise",
        .Subtype = "Deductive Reasoning",
        .Relationship = "Deductive Premise",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bGiven that\b", RegexOptions.IgnoreCase),
            New Regex("^\bIf\b", RegexOptions.IgnoreCase),
            New Regex("^\bAssuming\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Conclusion",
        .Subtype = "Deductive Reasoning",
        .Relationship = "Follow Up Premise",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bTherefore\b", RegexOptions.IgnoreCase),
            New Regex("^\bIt follows that\b", RegexOptions.IgnoreCase),
            New Regex("^\bSo\b", RegexOptions.IgnoreCase)
        }
    })

                ' Abductive Reasoning Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Hypothesis",
        .Subtype = "Abductive Reasoning",
        .Relationship = "Possibility Premise",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bIt is possible that\b", RegexOptions.IgnoreCase),
            New Regex("^\bTo explain the observation\b", RegexOptions.IgnoreCase),
            New Regex("^\bSuggesting a likely explanation\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Conclusion",
        .Subtype = "Abductive Reasoning",
        .Relationship = "Supporting Evidence Premise",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bTherefore, the best explanation is\b", RegexOptions.IgnoreCase),
            New Regex("^\bThe most plausible conclusion is\b", RegexOptions.IgnoreCase),
            New Regex("^\bThe evidence supports the hypothesis that\b", RegexOptions.IgnoreCase)
        }
    })



                ' Question Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Question",
        .Subtype = "General",
        .Relationship = "General Question",
        .Patterns = New List(Of Regex)() From {
            New Regex("^(?:What|Who|Where|When|Why|How)\b", RegexOptions.IgnoreCase),
            New Regex("^Is\b", RegexOptions.IgnoreCase),
            New Regex("^\bCan\b", RegexOptions.IgnoreCase),
            New Regex("^\bAre\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Question",
        .Subtype = "Comparison",
        .Relationship = "Compare Premise",
        .Patterns = New List(Of Regex)() From {
            New Regex("^(?:Which|What|Who)\b.*\b(is|are)\b.*\b(?:better|worse|superior|inferior|more|less)\b", RegexOptions.IgnoreCase),
            New Regex("^(?:How|In what way)\b.*\b(?:different|similar|alike)\b", RegexOptions.IgnoreCase),
            New Regex("^(?:Compare|Contrast)\b", RegexOptions.IgnoreCase)
        }
    })

                ' Answer Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Answer",
        .Subtype = "General",
        .Relationship = "Confirm/Deny",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bYes\b", RegexOptions.IgnoreCase),
            New Regex("^\bNo\b", RegexOptions.IgnoreCase),
            New Regex("^\bMaybe\b", RegexOptions.IgnoreCase),
            New Regex("^\bI don't know\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Answer",
        .Subtype = "Comparison",
        .Relationship = "Compare Premise",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bA is\b.*\b(?:better|worse|superior|inferior|more|less)\b", RegexOptions.IgnoreCase),
            New Regex("^\bA is\b.*\b(?:different|similar|alike)\b", RegexOptions.IgnoreCase),
            New Regex("^\bIt depends\b", RegexOptions.IgnoreCase),
            New Regex("^\bBoth\b", RegexOptions.IgnoreCase)
        }
    })

                ' Hypothesis Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Hypothesis",
        .Subtype = "General",
        .Relationship = "Hypothesize",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bIf\b", RegexOptions.IgnoreCase),
            New Regex("^\bAssuming\b", RegexOptions.IgnoreCase),
            New Regex("^\bSuppose\b", RegexOptions.IgnoreCase),
            New Regex("^\bHypothesize\b", RegexOptions.IgnoreCase)
        }
    })

                ' Conclusion Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Conclusion",
        .Subtype = "General",
        .Relationship = "Follow On Conclusion",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bTherefore\b", RegexOptions.IgnoreCase),
            New Regex("^\bThus\b", RegexOptions.IgnoreCase),
            New Regex("^\bHence\b", RegexOptions.IgnoreCase),
            New Regex("^\bConsequently\b", RegexOptions.IgnoreCase)
        }
    })

                ' Premise Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Premise",
        .Subtype = "General",
        .Relationship = "Reason",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bBecause\b", RegexOptions.IgnoreCase),
            New Regex("^\bSince\b", RegexOptions.IgnoreCase),
            New Regex("^\bGiven that\b", RegexOptions.IgnoreCase),
            New Regex("^\bConsidering\b", RegexOptions.IgnoreCase)
        }
    })
                ' Add new classification rules
                ClassificationRules.Add(New ClassificationRule With {
            .Type = "Dependency",
            .Subtype = "Cause and Effect",
            .Patterns = New List(Of Regex)() From {
                New Regex("(?i)(cause|effect|result in|lead to|because|due to|consequently)")
            }
        })

                ClassificationRules.Add(New ClassificationRule With {
            .Type = "Dependency",
            .Subtype = "Comparison",
            .Patterns = New List(Of Regex)() From {
                New Regex("(?i)(compared to|greater than|less than|similar to|different from)")
            }
        })

                ClassificationRules.Add(New ClassificationRule With {
            .Type = "Dependency",
            .Subtype = "Condition",
            .Patterns = New List(Of Regex)() From {
                New Regex("(?i)(if|unless|only if|when|provided that|in the case of)")
            }
        })

                ClassificationRules.Add(New ClassificationRule With {
            .Type = "Dependency",
            .Subtype = "Generalization",
            .Patterns = New List(Of Regex)() From {
                New Regex("(?i)(all|every|always|none|never|in general)")
            }
        })

                ClassificationRules.Add(New ClassificationRule With {
            .Type = "Dependency",
            .Subtype = "Temporal Sequence",
            .Patterns = New List(Of Regex)() From {
                New Regex("(?i)(before|after|during|while|subsequently|previously|simultaneously)")
            }
        })
                ' Add new classification rules
                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Dependency",
                .Subtype = "Control Flow",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(if the|as long as|when)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Dependency",
                .Subtype = "Function Call",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to calculate|to display|to show|to invoke|to utilize|to get)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Code Task",
                .Subtype = "Variable Assignment",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(store the|assign the|set the)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "System Task",
                .Subtype = "File Manipulation",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to read|to write|to extract|to save|to open|to close)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Code Task",
                .Subtype = "Exception Handling",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(wrap the|enclose the|employ a)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Code Task",
                .Subtype = "Object-Oriented Programming",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to create|to access|to instantiate|to retrieve)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Tokenization",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to split|to tokenize|use a tokenizer)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Lemmatization",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to apply|to convert|use a lemmatizer)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Named Entity Recognition (NER)",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to apply|to detect|use a named entity recognizer)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Sentiment Analysis",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to perform|to analyze|use a sentiment classifier)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Text Classification",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to classify|to perform|use a text classifier)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Language Translation",
                .Patterns = New List(Of Regex)() From {
            New Regex("(?i)(to translate|to perform|use a machine translation model)")
        }
    })

                ClassificationRules.Add(New ClassificationRule With {
        .Type = "Task",
        .Subtype = "Text Summarization",
        .Patterns = New List(Of Regex)() From {
            New Regex("(?i)(to generate|to perform|use a text summarizer)")
        }
    })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Word Embeddings",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to represent|to perform|use word embeddings)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Text Similarity",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to measure|to determine|use a text similarity metric)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Part-of-Speech Tagging",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to assign|to perform|use a part-of-speech tagger)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Dependency Parsing",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to analyze|to perform|perform dependency parsing)")
                }
            })

                ClassificationRules.Add(New ClassificationRule With {
                .Type = "Task",
                .Subtype = "Topic Modeling",
                .Patterns = New List(Of Regex)() From {
                    New Regex("(?i)(to identify|to perform|use topic modeling)")
                }
            })
                Return ClassificationRules
            End Function

            Public Class HypothesisDetector

                Private Shared Function GetEntitys() As String()
                    Return {"Effect", "relationship", "influence", "impact", "difference", "association", "correlation", "effectiveness.", "Significant", "statistically significant", "noticeable", "measurable", "Increase", "decrease", "positive", "negative", "Difference between", "change in", "effect on", "effect of,", "relationship between."}
                End Function

                Private Shared HypothesisPatterns As Dictionary(Of String, String()) = GetInternalHypothesisPatterns()

                Private Shared HypothesisIndicators As String() = {"Hypothesis", "Assumption", "theory", "in theory", "in practice",
            "proposal", "proposes", "it can be proposed", "Supposition", "supposedly", "supposes", "conjecture", "connects",
            "concludes", "follows that", "in light of", "in reflection", "reflects", "statistical", "strong relationship", "correlation", "exactly"}


                Public Shared Function GetSentence(document As String) As CapturedType



                    Dim hypotheses As String = DetectHypothesis(document)


                    Dim classification As String = ClassifyHypothesis(hypotheses)
                    Dim logicalRelationship As String = ClassifySentence(hypotheses)

                    Dim hypothesisStorage As New CapturedType With {
                    .Sentence = hypotheses,
                    .LogicalRelation_ = logicalRelationship,
                    .SubType = classification
                }




                    Return hypothesisStorage
                End Function


                Public Shared Function GetSentences(documents As List(Of String)) As List(Of CapturedType)
                    Dim storage As New List(Of CapturedType)

                    For Each document As String In documents
                        Dim hypotheses As List(Of String) = DetectHypotheses(document)

                        For Each hypothesis As String In hypotheses
                            Dim classification As String = ClassifyHypothesis(hypothesis)
                            Dim logicalRelationship As String = ClassifySentence(hypothesis)

                            Dim hypothesisStorage As New CapturedType With {
                    .Sentence = hypothesis,
                    .SubType = classification,
                    .LogicalRelation_ = logicalRelationship
                }
                            storage.Add(hypothesisStorage)
                        Next
                    Next

                    Return storage.Distinct.ToList
                End Function
                Public Shared Function DetectHypotheses(document As String, HypothesisPatterns As Dictionary(Of String, String())) As List(Of String)
                    Dim hypotheses As New List(Of String)

                    Dim sentences As String() = document.Split("."c)
                    For Each sentence As String In sentences
                        sentence = sentence.Trim().ToLower

                        ' Check if the sentence contains any indicator terms
                        If sentence.ContainsAny(HypothesisIndicators) Then
                            hypotheses.Add(sentence)
                        End If
                        If sentence.ContainsAny(GetEntitys) Then
                            hypotheses.Add(sentence)
                        End If

                        For Each hypothesesPattern In HypothesisPatterns
                            For Each indicatorPhrase In hypothesesPattern.Value
                                Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                                Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                                Dim matches = regex.Matches(sentence)

                                For Each match As Match In matches
                                    hypotheses.Add(match.Value)
                                Next
                            Next
                        Next
                    Next

                    Return hypotheses
                End Function

                Public Shared Function DetectHypothesis(ByRef document As String) As String


                    Dim sentence = document.Trim().ToLower

                    ' Check if the sentence contains any indicator terms
                    If sentence.ContainsAny(GetInternalHypothesisIndicators.ToArray) Then
                        Return sentence
                    End If
                    If sentence.ToLower.ContainsAny(GetEntitys) Then
                        Return sentence
                    End If

                    For Each hypothesesPattern In GetInternalHypothesisPatterns()

                        For Each indicatorPhrase In hypothesesPattern.Value
                            Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                            Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                            Dim matches = regex.Matches(sentence.ToLower)

                            For Each match As Match In matches
                                Return sentence
                            Next
                        Next
                    Next

                    Return ""
                End Function


                Public Shared Function DetectHypotheses(document As String) As List(Of String)
                    Return DetectHypotheses(document, HypothesisPatterns)
                End Function

                Public Shared Function ClassifyHypothesis(hypothesis As String) As String
                    Dim lowercaseHypothesis As String = hypothesis.ToLower()

                    If lowercaseHypothesis.ContainsAny(GetCompositeHypothesisIndicators.ToArray) Then
                        Return "Composite Hypotheses"
                    ElseIf lowercaseHypothesis.ContainsAny(GetNonDirectionalHypothesisIndicators.ToArray) Then
                        Return "Non Directional Hypothesis"
                    ElseIf lowercaseHypothesis.ContainsAny(GetDirectionalHypothesisIndicators.ToArray) Then
                        Return "Directional Hypothesis"
                    ElseIf lowercaseHypothesis.ContainsAny(GetNullHypothesisIndicators.ToArray) Then
                        Return "Null Hypothesis"
                    ElseIf lowercaseHypothesis.ContainsAny(GetAlternativeHypothesisIndicators.ToArray) Then
                        Return "Alternative Hypothesis"
                    ElseIf lowercaseHypothesis.ContainsAny(GetGeneralHypothesisIndicators.ToArray) Then
                        Return "General Hypothesis"
                    ElseIf lowercaseHypothesis.ContainsAny(GetResearchHypothesisIndicators.ToArray) Then
                        Return "Research Hypothesis"
                    End If
                    For Each hypothesesPattern In GetInternalHypothesisPatterns()

                        For Each indicatorPhrase In hypothesesPattern.Value
                            Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                            Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                            Dim matches = regex.Matches(lowercaseHypothesis)

                            For Each match As Match In matches
                                Return hypothesesPattern.Key
                            Next
                        Next
                    Next

                    ' Check classification rules
                    For Each rule In InitializeClassificationRules()
                        For Each pattern In rule.Patterns
                            If pattern.IsMatch(lowercaseHypothesis) Then
                                Return rule.Subtype
                            End If
                        Next
                    Next

                    Return LogicalDependencyClassifier.ClassifyLogicalDependency(lowercaseHypothesis)


                    Return "Unclassified"
                End Function

                Public Shared Function ClassifySentence(ByVal sentence As String) As String
                    Dim lowercaseSentence As String = sentence.ToLower()
                    If ClassifyHypothesis(lowercaseSentence) = "Unclassified" Then
                        For Each hypothesesPattern In GetInternalHypothesisPatterns()

                            For Each indicatorPhrase In hypothesesPattern.Value
                                Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                                Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                                Dim matches = regex.Matches(lowercaseSentence)

                                For Each match As Match In matches
                                    Return hypothesesPattern.Key
                                Next
                            Next
                        Next

                        ' Check classification rules
                        For Each rule In InitializeClassificationRules()
                            For Each pattern In rule.Patterns
                                If pattern.IsMatch(lowercaseSentence) Then
                                    Return rule.Relationship
                                End If
                            Next
                        Next

                    Else
                        'detect logical relation
                        Return LogicalDependencyClassifier.ClassifyLogicalDependency(lowercaseSentence)


                    End If


                    ' If no match found, return unknown
                    Return EntityModels.ClassifySentence(lowercaseSentence)
                End Function
                Public Shared Sub Main()
                    Dim documents As List(Of String) = TrainingData.TrainingData.GenerateRandomTrainingData(GetInternalHypothesisIndicators)
                    Dim count As Integer = 0
                    For Each hypothesisStorage As CapturedType In HypothesisDetector.GetSentences(TrainingData.TrainingData.GenerateRandomTrainingData(GetInternalHypothesisIndicators))
                        count += 1
                        Console.WriteLine(count & ":")
                        Console.WriteLine($"Sentence: {hypothesisStorage.Sentence}")
                        Console.WriteLine("Hypothesis Classification: " & hypothesisStorage.SubType)
                        Console.WriteLine("Logical Relationship: " & hypothesisStorage.LogicalRelation_)
                        Console.WriteLine()
                    Next
                    Console.ReadLine()

                End Sub

                ''' <summary>
                ''' used to detect type of classification of hypothosis
                ''' </summary>
                ''' <returns></returns>
                Public Shared Function GetInternalHypothesisPatterns() As Dictionary(Of String, String())
                    Return New Dictionary(Of String, String()) From {
    {"Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?:hypothesis|assumption|theory|proposal|supposition|conjecture|concludes|assumes|correlates)\"}},
    {"Research Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?:significant|effects|has an effect|induces|strong correlation|statistical)\b"}},
    {"Null Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?:no significant relationship|no relationship between|nothing|null|no effect)\b"}},
    {"Alternative Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?:is a significant relationship|significant relationship between)\b"}},
    {"Directional Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?:increase|decrease|loss|gain|position|correlation|above|below|before|after|precedes|preceding|following|follows|precludes)\b"}},
    {"Non-Directional Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?:significant difference|no change|unchanged|unchangeable)\b"}},
    {"Diagnostic Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?:diagnostic hypothesis|can identify|characteristic of|feature of)\b"}},
    {"Descriptive Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?:describes|it follows that|comprises of|comprises|builds towards)\b"}},
          {"Casual Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(causal hypothesis|causes|leads to|results in)\b"}},
           {"Explanatory Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?i)\b(?:explanatory hypothesis|explains|reason for|cause of)\b"}},
            {"Predictive Hypothesis", {"(?i)\b[A-Z][^.!?]*\b(?i)\b(prediction|forecast|projection|predicts|fore-casted:projects:projection)\b"}
}}
                End Function

                Public Shared Function GetGeneralHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("assuming")
                    lst.Add("assuming")
                    lst.Add("theory")
                    lst.Add("proposed")
                    lst.Add("indicates")
                    lst.Add("conjecture")
                    lst.Add("correlates")


                    Return lst
                End Function
                Public Shared Function GetResearchHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("significant")
                    lst.Add("effects")
                    lst.Add("has an effect")
                    lst.Add("induces")
                    lst.Add("strong correlation")
                    lst.Add("statistically")
                    lst.Add("statistics show")
                    lst.Add("it can be said")
                    lst.Add("it has been shown")
                    lst.Add("been proved")

                    Return lst
                End Function
                Public Shared Function GetDirectionalHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("increase")
                    lst.Add("decrease")
                    lst.Add("loss")
                    lst.Add("gain")
                    lst.Add("position")
                    lst.Add("correlation")
                    lst.Add("above")
                    lst.Add("below")
                    lst.Add("before")
                    lst.Add("after")
                    lst.Add("precedes")
                    lst.Add("follows")
                    lst.Add("following")
                    lst.Add("gaining")
                    lst.Add("precursor")

                    Return lst
                End Function
                Public Shared Function GetInternalHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.AddRange(GetCompositeHypothesisIndicators)
                    lst.AddRange(GetNonDirectionalHypothesisIndicators)
                    lst.AddRange(GetAlternativeHypothesisIndicators)
                    lst.AddRange(GetDirectionalHypothesisIndicators)
                    lst.AddRange(GetNullHypothesisIndicators)
                    lst.AddRange(GetResearchHypothesisIndicators)
                    lst.AddRange(GetGeneralHypothesisIndicators)
                    Return lst
                End Function
                Private Shared Function GetAlternativeHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("significant relationship")
                    lst.Add("relationship between")
                    lst.Add("great significance")
                    lst.Add("signify")

                    Return lst
                End Function
                Private Shared Function GetNonDirectionalHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("significant difference")
                    lst.Add("no change")
                    lst.Add("unchangeable")
                    lst.Add("unchanged")

                    Return lst
                End Function
                Private Shared Function GetCompositeHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)

                    lst.Add("leads to")
                    lst.Add("consequence of")
                    lst.Add("it follows that")
                    lst.Add("comprises of")
                    lst.Add("comprises")
                    lst.Add("builds towards")
                    Return lst
                End Function

                Private Shared Function GetNullHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("no significant relationship")
                    lst.Add("no relationship between")
                    lst.Add("no significance")
                    lst.Add("does not signify")
                    lst.Add("no effect")
                    lst.Add("no changes")
                    Return lst
                End Function
            End Class

            Public Class EntityModels

                Public Shared Function ContainsAny(text As String, indicators As String()) As Boolean
                    For Each indicator As String In indicators
                        If text.Contains(indicator) Then
                            Return True
                        End If
                    Next

                    Return False
                End Function
                Public Shared Function ClassifySentence(ByVal sentence As String) As String
                    Dim lowercaseSentence As String = sentence.ToLower()
                    For Each premisePattern In GetInternalPremisePatterns()

                        For Each indicatorPhrase In premisePattern.Value
                            Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                            Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                            Dim matches = regex.Matches(lowercaseSentence)

                            For Each match As Match In matches
                                Return premisePattern.Key
                            Next
                        Next
                    Next
                    ' Check classification rules
                    For Each rule In InitializeClassificationRules()

                        For Each pattern In rule.Patterns
                            If pattern.IsMatch(lowercaseSentence) Then
                                Return rule.Subtype

                            End If
                        Next
                    Next




                    Return LogicalDependencyClassifier.ClassifyLogicalDependency(lowercaseSentence)


                    ' If no match found, return unknown
                    Return "Unknown"
                End Function

                Public Function ClassifyPremises(ByVal document As String) As List(Of String)
                    ' Add your premise detection logic here
                    ' Return a list of premises found in the document

                    Dim Premise As List(Of String) = DetectPremises(document)
                    ' Placeholder implementation
                    Return Premise
                End Function

                Private Shared PremisePatterns As Dictionary(Of String, String())
                Private Shared DependencyPatterns As Dictionary(Of String, List(Of Regex))
                Private Shared DeductiveDependencyPatterns As List(Of Regex)
                Private Shared InductiveDependencyPatterns As List(Of Regex)
                Private Shared ContrapositiveDependencyPatterns As List(Of Regex)
                Private Shared ConditionalDependencyPatterns As List(Of Regex)
                Private Shared CausalDependencyPatterns As List(Of Regex)
                Private Shared BiconditionalDependencyPatterns As List(Of Regex)
                Private Shared InferenceDependencyPatterns As List(Of Regex)
                Private Shared CounterfactualDependencyPatterns As List(Of Regex)
                Private Shared StatisticalDependencyPatterns As List(Of Regex)
                Private Shared AnalogicalDependencyPatterns As List(Of Regex)
                Public Shared ReadOnly DeductiveDependencyIndicators As String() = {"If", "Then"}
                Public Shared ReadOnly InductiveDependencyIndicators As String() = {"Every time", "Every instance"}
                Public Shared ReadOnly ContrapositiveDependencyIndicators As String() = {"If Not", "Then Not"}
                Public Shared ReadOnly ConditionalDependencyIndicators As String() = {"If", "When"}
                Public Shared ReadOnly CausalDependencyIndicators As String() = {"Because", "Due to"}
                Public Shared ReadOnly BiconditionalDependencyIndicators As String() = {"If And only if", "before", "after", "above", "below"}
                Public Shared ReadOnly InferenceDependencyIndicators As String() = {"Based on", "From"}
                Public Shared ReadOnly CounterfactualDependencyIndicators As String() = {"If Not", "Then Not"}
                Public Shared ReadOnly StatisticalDependencyIndicators As String() = {"Based on statistics", "According to the survey"}
                Public Shared ReadOnly AnalogicalDependencyIndicators As String() = {"Similar to", "Analogous to"}
                Public Shared ReadOnly SupportingPremiseIndicators As String() = {"For", "In support of", "Research has shown that", "Studies have demonstrated that", "Experts agree that", "Evidence suggests that", "Data indicates that", "Statistics show that", "In accordance with", "Based on the findings of", "According to the research"}
                Public Shared ReadOnly GeneralizationPremiseIndicators As String() = {"In general", "Typically", "In general", "Typically", "On average", "As a rule", "Commonly", "In most cases", "Generally speaking", "Universally", "As a general principle", "Across the board"}
                Public Shared ReadOnly ConditionalPremiseIndicators As String() = {"If", "When", "then", "Given that", "On the condition that", "Assuming that", "Provided that", "In the event that", "Whenever", "In case", "Under the circumstances that"}
                Public Shared ReadOnly AnalogicalPremiseIndicators As String() = {"Similar to", "Analogous to", "Similar to", "Just as", "Like", "Comparable to", "In the same way that", "Analogous to", "Corresponding to", "Resembling", "As if", "In a similar fashion"}
                Public Shared ReadOnly CausalPremiseIndicators As String() = {"Because", "Due to", "Because", "Since", "As a result of", "Due to", "Caused by", "Leads to", "Results in", "Owing to", "Contributes to", "Is responsible for"}
                Public Shared ReadOnly FactualPremiseIndicators As String() = {"In fact", "Indeed", "It Is a fact that", "It Is well-established that", "Historically, it has been proven that", "Scientifically speaking", "Empirical evidence confirms that", "Observations reveal that", "Documented sources state that", "In reality", "Undeniably"}

                Public Shared Function GetSentences(documents As List(Of String)) As List(Of CapturedType)
                    Dim storage As New List(Of CapturedType)
                    For Each document As String In documents
                        Dim premises As List(Of String) = EntityModels.DetectPremises(document, EntityModels.GetInternalPremisePatterns)

                        For Each premise As String In premises
                            Dim premiseSubtype As String = EntityModels.ExtractPremiseSubtype(premise)
                            If premiseSubtype = "Unknown" Then

                                premiseSubtype = ClassifySentence(premise)
                            End If

                            Dim classification As String = LogicalDependencyClassifier.ClassifyLogicalDependency(premise)

                            Dim premiseStorage As New CapturedType With {
                    .Sentence = premise,
                    .SubType = premiseSubtype,
                    .LogicalRelation_ = classification}
                            storage.Add(premiseStorage)
                        Next
                    Next
                    Return storage.Distinct.ToList
                End Function
                Public Shared Function GetSentence(document As String) As CapturedType


                    Dim premise As String = EntityModels.DetectPremise(document)


                    Dim premiseSubtype As String = ClassifySentence(premise)
                    If premiseSubtype = "Unknown" Then

                        premiseSubtype = EntityModels.ExtractPremiseSubtype(premise)
                    End If

                    Dim classification As String = LogicalDependencyClassifier.ClassifyLogicalDependency(premise)

                    Dim premiseStorage As New CapturedType With {
                    .Sentence = premise,
                    .SubType = premiseSubtype,
                    .LogicalRelation_ = classification}



                    Return premiseStorage
                End Function

                Public Sub New()
                    PremisePatterns = GetInternalPremisePatterns()
                    InitializeInternalDependancyPatterns()
                    DependencyPatterns = GetInternalDependacyPatterns()
                End Sub


                Public Shared Function GetInternalPremisePatterns() As Dictionary(Of String, String())
                    InitializeInternalDependancyPatterns()
                    Return New Dictionary(Of String, String()) From {
{"Deductive Dependency", DeductiveDependencyIndicators},
{"Inductive Dependency", InductiveDependencyIndicators},
{"Contrapositive Dependency", ContrapositiveDependencyIndicators},
{"Conditional Dependency", ConditionalDependencyIndicators},
{"Causal Dependency", CausalDependencyIndicators},
{"Biconditional Dependency", BiconditionalDependencyIndicators},
{"Inference Dependency", InferenceDependencyIndicators},
{"Counterfactual Dependency", CounterfactualDependencyIndicators},
{"Statistical Dependency", StatisticalDependencyIndicators},
{"Analogical Dependency", AnalogicalDependencyIndicators},
            {"Supporting Premise", SupportingPremiseIndicators},
            {"Generalization Premise", GeneralizationPremiseIndicators},
            {"Conditional Premise", ConditionalPremiseIndicators},
            {"Causal Premise", CausalPremiseIndicators},
            {"Factual Premise", FactualPremiseIndicators}, {"Analogical Premise", AnalogicalPremiseIndicators}}

                End Function

                Private Shared Sub InitializeInternalDependancyPatterns()
                    ' Initialize the patterns for each premise type
                    DeductiveDependencyPatterns = New List(Of Regex) From {
            New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
            New Regex("Given\s+that\s+.*,\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*implies\s+that\s+.*", RegexOptions.IgnoreCase)
        }

                    InductiveDependencyPatterns = New List(Of Regex) From {
            New Regex(".*every\s+time\s+.*,\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*Is\s+often\s+associated\s+with\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*Is\s+usually\s+followed\s+by\s+.*", RegexOptions.IgnoreCase)
        }

                    ContrapositiveDependencyPatterns = New List(Of Regex) From {
            New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*Is\s+Not\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase)
        }

                    ConditionalDependencyPatterns = New List(Of Regex) From {
            New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*depends\s+on\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*Is\s+conditioned\s+by\s+.*", RegexOptions.IgnoreCase)
        }

                    CausalDependencyPatterns = New List(Of Regex) From {
            New Regex(".*Is\s+the\s+cause,\s+which\s+leads\s+to\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*results\s+in\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*causes\s+.*", RegexOptions.IgnoreCase)
        }

                    BiconditionalDependencyPatterns = New List(Of Regex) From {
            New Regex(".*if\s+And\s+only\s+if\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*Is\s+equivalent\s+to\s+.*", RegexOptions.IgnoreCase)
        }

                    InferenceDependencyPatterns = New List(Of Regex) From {
            New Regex("Based\s+on\s+the\s+.*,\s+it\s+can\s+be\s+inferred\s+that\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*implies\s+that\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*leads\s+to\s+the\s+conclusion\s+that\s+.*", RegexOptions.IgnoreCase)
        }

                    CounterfactualDependencyPatterns = New List(Of Regex) From {
            New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*would\s+have\s+been\s+.*,\s+if\s+.*", RegexOptions.IgnoreCase)
        }

                    StatisticalDependencyPatterns = New List(Of Regex) From {
            New Regex("Based\s+on\s+a\s+survey\s+of\s+.*,\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*statistically\s+correlated\s+with\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*Is\s+likely\s+if\s+.*", RegexOptions.IgnoreCase)
        }

                    AnalogicalDependencyPatterns = New List(Of Regex) From {
            New Regex(".*Is\s+similar\s+to\s+.*,\s+which\s+implies\s+that\s+.*", RegexOptions.IgnoreCase),
            New Regex(".*Is\s+analogous\s+to\s+.*,\s+indicating\s+that\s+.*", RegexOptions.IgnoreCase)
        }
                End Sub

                Private Shared Function GetInternalDependacyPatterns() As Dictionary(Of String, List(Of Regex))
                    Return New Dictionary(Of String, List(Of Regex)) From {
                {"Deductive Dependency", New List(Of Regex) From {
                    New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
                    New Regex("Given\s+that\s+.*,\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*implies\s+that\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Inductive Dependency", New List(Of Regex) From {
                    New Regex(".*every\s+time\s+.*,\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+often\s+associated\s+with\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+usually\s+followed\s+by\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Contrapositive Dependency", New List(Of Regex) From {
                    New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+Not\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Conditional Dependency", New List(Of Regex) From {
                    New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*depends\s+on\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+conditioned\s+by\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Causal Dependency", New List(Of Regex) From {
                    New Regex(".*Is\s+the\s+cause,\s+which\s+leads\s+to\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*results\s+in\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*causes\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Biconditional Dependency", New List(Of Regex) From {
                    New Regex(".*if\s+And\s+only\s+if\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+equivalent\s+to\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Inference Dependency", New List(Of Regex) From {
                    New Regex(".*infers\s+that\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+deduced\s+from\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*drawn\s+from\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Counterfactual Dependency", New List(Of Regex) From {
                    New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*would\s+have\s+occurred\s+if\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Statistical Dependency", New List(Of Regex) From {
                    New Regex("Based\s+on\s+statistics,\s+.*", RegexOptions.IgnoreCase),
                    New Regex("According\s+to\s+the\s+survey,\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Analogical Dependency", New List(Of Regex) From {
                    New Regex(".*Is\s+similar\s+to\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+analogous\s+to\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Supporting Premise", New List(Of Regex) From {
                    New Regex("For\s+example,\s+.*", RegexOptions.IgnoreCase),
                    New Regex("In\s+support\s+of\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+backed\s+by\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Generalization Premise", New List(Of Regex) From {
                    New Regex("In\s+general,\s+.*", RegexOptions.IgnoreCase),
                    New Regex("Typically,\s+.*", RegexOptions.IgnoreCase),
                    New Regex("Most\s+of\s+the\s+time,\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Conditional Premise", New List(Of Regex) From {
                    New Regex("If\s+.*,\s+then\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*relies\s+on\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Analogical Premise", New List(Of Regex) From {
                    New Regex(".*Is\s+similar\s+to\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+analogous\s+to\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Causal Premise", New List(Of Regex) From {
                    New Regex(".*Is\s+the\s+cause\s+of\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*leads\s+to\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*causes\s+.*", RegexOptions.IgnoreCase)
                }},
                {"Factual Premise", New List(Of Regex) From {
                    New Regex("In\s+fact,\s+.*", RegexOptions.IgnoreCase),
                    New Regex("Indeed,\s+.*", RegexOptions.IgnoreCase),
                    New Regex(".*Is\s+a\s+fact\s+that\s+.*", RegexOptions.IgnoreCase)
                }}
            }
                End Function

                Public Function DetectPremises(ByVal document As String) As List(Of String)
                    Dim premises As New List(Of String)()

                    For Each premisePattern In PremisePatterns
                        For Each indicatorPhrase In premisePattern.Value
                            Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                            Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                            Dim matches = regex.Matches(document)

                            For Each match As Match In matches
                                premises.Add(match.Value)
                            Next
                        Next
                    Next

                    For Each premiseType As String In PremisePatterns.Keys
                        Dim indicators As String() = PremisePatterns(premiseType)

                        For Each indicator As String In indicators
                            Dim pattern As String = $"(?<=\b{indicator}\b).*?(?=[.]|$)"
                            Dim matches As MatchCollection = Regex.Matches(document, pattern, RegexOptions.IgnoreCase)

                            For Each match As Match In matches
                                premises.Add(match.Value.Trim())
                            Next
                        Next
                    Next

                    Return premises
                End Function
                Public Shared Function DetectPremise(ByVal document As String) As String
                    Dim premises As String = ""

                    For Each premisePattern In GetInternalPremisePatterns()

                        For Each indicatorPhrase In premisePattern.Value
                            Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                            Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                            Dim matches = regex.Matches(document)

                            For Each match As Match In matches
                                Return match.Value
                            Next
                        Next
                    Next


                    Return premises
                End Function

                Public Shared Function DetectPremises(ByVal document As String, ByRef PremisePatterns As Dictionary(Of String, String())) As List(Of String)
                    Dim premises As New List(Of String)()

                    For Each premisePattern In PremisePatterns
                        For Each indicatorPhrase In premisePattern.Value
                            Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                            Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                            Dim matches = regex.Matches(document)

                            For Each match As Match In matches
                                premises.Add(match.Value)
                            Next
                        Next
                    Next

                    For Each premiseType As String In PremisePatterns.Keys
                        Dim indicators As String() = PremisePatterns(premiseType)

                        For Each indicator As String In indicators
                            Dim pattern As String = $"(?<=\b{indicator}\b).*?(?=[.]|$)"
                            Dim matches As MatchCollection = Regex.Matches(document, pattern, RegexOptions.IgnoreCase)

                            For Each match As Match In matches
                                premises.Add(match.Value.Trim())
                            Next
                        Next
                    Next

                    Return premises
                End Function

                Public Shared Function ExtractPremiseSubtype(ByVal premise As String) As String
                    For Each premisePattern In GetInternalPremisePatterns()

                        If Not premisePattern.Value.Any(Function(indicatorPhrase) premise.Contains(indicatorPhrase)) Then
                            Continue For
                        End If
                        Return premisePattern.Key
                    Next

                    Return "Unknown"
                End Function

                Private Shared Function DetectDependencies(ByVal document As String) As List(Of String)
                    Dim dependencies As New List(Of String)()

                    For Each dependencyPattern In DependencyPatterns
                        For Each regex In dependencyPattern.Value
                            Dim matches = regex.Matches(document)

                            For Each match As Match In matches
                                dependencies.Add(match.Value)
                            Next
                        Next
                    Next

                    Return dependencies
                End Function

                Private Shared Function CheckPremiseSubtype(ByVal premise As String) As String
                    ' Extract the premise subtype from the premise string
                    If premise.Contains("Deductive Dependency") Then
                        Return "Deductive Dependency"
                    ElseIf premise.Contains("Inductive Dependency") Then
                        Return "Inductive Dependency"
                    ElseIf premise.Contains("Contrapositive Dependency") Then
                        Return "Contrapositive Dependency"
                    ElseIf premise.Contains("Conditional Dependency") Then
                        Return "Conditional Dependency"
                    ElseIf premise.Contains("Causal Dependency") Then
                        Return "Causal Dependency"
                    ElseIf premise.Contains("Biconditional Dependency") Then
                        Return "Biconditional Dependency"
                    ElseIf premise.Contains("Inference Dependency") Then
                        Return "Inference Dependency"
                    ElseIf premise.Contains("Counterfactual Dependency") Then
                        Return "Counterfactual Dependency"
                    ElseIf premise.Contains("Statistical Dependency") Then
                        Return "Statistical Dependency"
                    ElseIf premise.Contains("Analogical Dependency") Then
                        Return "Analogical Dependency"
                        ' Use regular expressions or string matching to
                        'extract the premise subtype based on indicator phrases
                    ElseIf premise.ContainsAny(DeductiveDependencyIndicators) Then
                        Return "Deductive Dependency"
                    ElseIf premise.ContainsAny(InductiveDependencyIndicators) Then
                        Return "Inductive Dependency"
                    ElseIf premise.ContainsAny(ContrapositiveDependencyIndicators) Then
                        Return "Contrapositive Dependency"
                    ElseIf premise.ContainsAny(ConditionalDependencyIndicators) Then
                        Return "Conditional Dependency"
                    ElseIf premise.ContainsAny(CausalDependencyIndicators) Then
                        Return "Causal Dependency"
                    ElseIf premise.ContainsAny(BiconditionalDependencyIndicators) Then
                        Return "Biconditional Dependency"
                    ElseIf premise.ContainsAny(InferenceDependencyIndicators) Then
                        Return "Inference Dependency"
                    ElseIf premise.ContainsAny(CounterfactualDependencyIndicators) Then
                        Return "Counterfactual Dependency"
                    ElseIf premise.ContainsAny(StatisticalDependencyIndicators) Then
                        Return "Statistical Dependency"
                    ElseIf premise.ContainsAny(AnalogicalDependencyIndicators) Then
                        Return "Analogical Dependency"
                    ElseIf premise.ContainsAny(SupportingPremiseIndicators) Then
                        Return "Supporting Premise"
                    ElseIf premise.ContainsAny(GeneralizationPremiseIndicators) Then
                        Return "Generalization Premise"
                    ElseIf premise.ContainsAny(ConditionalPremiseIndicators) Then
                        Return "Conditional Premise"
                    ElseIf premise.ContainsAny(AnalogicalPremiseIndicators) Then
                        Return "Analogical Premise"
                    ElseIf premise.ContainsAny(CausalPremiseIndicators) Then
                        Return "Causal Premise"
                    ElseIf premise.ContainsAny(FactualPremiseIndicators) Then
                        Return "Factual Premise"
                    Else
                        Return "Unknown"
                    End If
                End Function

                Public Shared Function ExtractDependencyType(ByVal dependency As String) As String
                    For Each dependencyPattern In GetInternalDependacyPatterns()

                        If dependencyPattern.Value.Any(Function(regex) regex.IsMatch(dependency)) Then
                            Return dependencyPattern.Key
                        End If
                    Next

                    Return "Unknown"
                End Function


            End Class

            Public Class ConclusionDetector

                Private Shared Function GetEntitys() As String()
                    Return {"hence", "thus", "therefore", "Consequently", "accordingly", "association with", "correlation with", "conclusion", "Significant result", "statistically results show", "Contrary to popular belief",
            "in light of", "in summary", "in future", "as described", "in lieu"}

                End Function

                Private Shared ConclusionPatterns As Dictionary(Of String, String()) = GetInternalConclusionPatterns()

                Private Shared ConclusionIndicators As String() = {"conclusion", "Assumption", "theory", "in theory", "in practice",
            "proposal", "proposes", "it can be proposed", "Supposition", "supposedly", "supposes", "conjecture", "connects",
            "concludes", "follows that", "in light of", "in reflection", "This disproves", "statistical", "discovered relationship", "correlation", "exactly"}


                Public Shared Function GetSentence(document As String) As CapturedType



                    Dim hypotheses As String = DetectConclusion(document)


                    Dim classification As String = ClassifyConclusion(hypotheses)
                    Dim logicalRelationship As String = ClassifySentence(hypotheses)

                    Dim hypothesisStorage As New CapturedType With {
                    .Sentence = hypotheses,
                    .LogicalRelation_ = logicalRelationship,
                    .SubType = classification
                }




                    Return hypothesisStorage
                End Function


                Public Shared Function GetSentences(documents As List(Of String)) As List(Of CapturedType)
                    Dim storage As New List(Of CapturedType)
                    For Each document As String In documents
                        Dim hypotheses As List(Of String) = Detectconclusions(document)

                        For Each hypothesis As String In hypotheses
                            Dim classification As String = ClassifyConclusion(hypothesis)
                            Dim logicalRelationship As String = ClassifySentence(hypothesis)

                            Dim hypothesisStorage As New CapturedType With {
                    .Sentence = hypothesis,
                    .LogicalRelation_ = logicalRelationship,
                    .SubType = classification
                }
                            storage.Add(hypothesisStorage)
                        Next
                    Next

                    Return storage.Distinct.ToList
                End Function
                Public Shared Function DetectConclusions(document As String, HypothesisPatterns As Dictionary(Of String, String())) As List(Of String)
                    Dim Conclusions As New List(Of String)

                    Dim sentences As String() = document.Split("."c)
                    For Each sentence As String In sentences
                        sentence = sentence.Trim().ToLower

                        ' Check if the sentence contains any indicator terms
                        If sentence.ContainsAny(ConclusionIndicators) Then
                            Conclusions.Add(sentence)
                        End If
                        If sentence.ContainsAny(GetEntitys) Then
                            Conclusions.Add(sentence)
                        End If

                        For Each conclusionPattern In ConclusionPatterns
                            For Each indicatorPhrase In conclusionPattern.Value
                                Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                                Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                                Dim matches = regex.Matches(sentence)

                                For Each match As Match In matches
                                    Conclusions.Add(match.Value)
                                Next
                            Next
                        Next
                    Next

                    Return Conclusions
                End Function

                Public Shared Function DetectConclusion(ByRef document As String) As String


                    Dim sentence = document.Trim().ToLower

                    ' Check if the sentence contains any indicator terms
                    If sentence.ContainsAny(GetInternalConclusionIndicators.ToArray) Then
                        Return sentence
                    End If
                    If sentence.ToLower.ContainsAny(GetEntitys) Then
                        Return sentence
                    End If

                    For Each hypothesesPattern In GetInternalConclusionPatterns()

                        For Each indicatorPhrase In hypothesesPattern.Value
                            Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                            Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                            Dim matches = regex.Matches(sentence.ToLower)

                            For Each match As Match In matches
                                Return sentence
                            Next
                        Next
                    Next

                    Return ""
                End Function


                Public Shared Function Detectconclusions(document As String) As List(Of String)
                    Return DetectConclusions(document, ConclusionPatterns)
                End Function

                Public Shared Function ClassifyConclusion(hypothesis As String) As String
                    Dim lowercaseHypothesis As String = hypothesis.ToLower()

                    For Each conclusionPattern In GetInternalConclusionPatterns()

                        For Each indicatorPhrase In conclusionPattern.Value
                            Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                            Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                            Dim matches = regex.Matches(lowercaseHypothesis)

                            For Each match As Match In matches
                                Return conclusionPattern.Key
                            Next
                        Next
                    Next

                    ' Check classification rules
                    For Each rule In InitializeClassificationRules()
                        For Each pattern In rule.Patterns
                            If pattern.IsMatch(lowercaseHypothesis) Then
                                Return rule.Subtype
                            End If
                        Next
                    Next

                    Return HypothesisDetector.ClassifySentence(lowercaseHypothesis)


                    Return "Unclassified"
                End Function

                Public Shared Function ClassifySentence(ByVal sentence As String) As String
                    Dim lowercaseSentence As String = sentence.ToLower()
                    If ClassifyConclusion(lowercaseSentence) = "Unclassified" Then
                        For Each hypothesesPattern In GetInternalConclusionPatterns()

                            For Each indicatorPhrase In hypothesesPattern.Value
                                Dim regexPattern = $"(\b{indicatorPhrase}\b).*?(\.|$)"
                                Dim regex As New Regex(regexPattern, RegexOptions.IgnoreCase)
                                Dim matches = regex.Matches(lowercaseSentence)

                                For Each match As Match In matches
                                    Return hypothesesPattern.Key
                                Next
                            Next
                        Next

                        ' Check classification rules
                        For Each rule In InitializeClassificationRules()
                            For Each pattern In rule.Patterns
                                If pattern.IsMatch(lowercaseSentence) Then
                                    Return rule.Relationship
                                End If
                            Next
                        Next

                    Else
                        'detect logical relation
                        Return LogicalDependencyClassifier.ClassifyLogicalDependency(lowercaseSentence)


                    End If


                    ' If no match found, return unknown
                    Return HypothesisDetector.ClassifySentence(lowercaseSentence)
                End Function

                ''' <summary>
                ''' used to detect type of classification of hypothesis
                ''' </summary>
                ''' <returns></returns>
                Public Shared Function GetInternalConclusionPatterns() As Dictionary(Of String, String())
                    Return New Dictionary(Of String, String()) From {
    {"Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:hypothesis|assumption|theory|proposal|supposition|conjecture|concludes|assumes|correlates)\"}},
    {"Research Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:significant|effects|has an effect|induces|strong correlation|statistical)\b"}},
    {"Null Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:no significant relationship|no relationship between|nothing|null|no effect)\b"}},
    {"Alternative Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:is a significant relationship|significant relationship between)\b"}},
    {"Directional Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:increase|decrease|loss|gain|position|correlation|above|below|before|after|precedes|preceding|following|follows|precludes)\b"}},
    {"Non-Directional Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:significant difference|no change|unchanged|unchangeable)\b"}},
    {"Diagnostic Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:diagnostic hypothesis|can identify|characteristic of|feature of|factors entail)\b"}},
    {"Descriptive Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:describes|it follows that|comprises of|comprises|builds towards)\b"}},
         {"Recommendation Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?:recommends|it is suggested that|it is urged|it is advisable|Considering these factors|Based on these findings)\b"}},
                   {"Casual Conclusion", {"(?i)\b[A-Z][^.!?]*\b(causal hypothesis|causes|leads to|results in)\b"}},
                {"Conditional Conclusion", {"(?i)\b[A-Z][^.!?]*\b(provided that|As a result|it leads to|results in|conditionally:based on:due to|because of|under the circumstances)\b"}},
           {"Explanatory Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?i)\b(?:explanatory hypothesis|explains|reason for|cause of)\b"}},
            {"Predictive Conclusion", {"(?i)\b[A-Z][^.!?]*\b(?i)\b(prediction|it is estimated that|based on projections|it is foreseen that|fore-casted|predictive models|projections show)\b"}
}}
                End Function

                Public Shared Function GetGeneralConclusionIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("assuming")
                    lst.Add("assuming")
                    lst.Add("theory")
                    lst.Add("proposed")
                    lst.Add("indicates")
                    lst.Add("conjecture")
                    lst.Add("correlates")


                    Return lst
                End Function
                Public Shared Function GetResearchHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("significant")
                    lst.Add("effects")
                    lst.Add("has an effect")
                    lst.Add("induces")
                    lst.Add("strong correlation")
                    lst.Add("statistically")
                    lst.Add("statistics show")
                    lst.Add("it can be said")
                    lst.Add("it has been shown")
                    lst.Add("been proved")

                    Return lst
                End Function
                Public Shared Function GetDirectionalHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("increase")
                    lst.Add("decrease")
                    lst.Add("loss")
                    lst.Add("gain")
                    lst.Add("position")
                    lst.Add("correlation")
                    lst.Add("above")
                    lst.Add("below")
                    lst.Add("before")
                    lst.Add("after")
                    lst.Add("precedes")
                    lst.Add("follows")
                    lst.Add("following")
                    lst.Add("gaining")
                    lst.Add("precursor")

                    Return lst
                End Function
                Public Shared Function GetInternalConclusionIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.AddRange(GetCompositeHypothesisIndicators)
                    lst.AddRange(GetNonDirectionalHypothesisIndicators)
                    lst.AddRange(GetAlternativeHypothesisIndicators)
                    lst.AddRange(GetDirectionalHypothesisIndicators)
                    lst.AddRange(GetNullHypothesisIndicators)
                    lst.AddRange(GetResearchHypothesisIndicators)
                    lst.AddRange(GetGeneralConclusionIndicators)
                    Return lst
                End Function
                Private Shared Function GetAlternativeHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("significant relationship")
                    lst.Add("relationship between")
                    lst.Add("great significance")
                    lst.Add("signify")

                    Return lst
                End Function
                Private Shared Function GetNonDirectionalHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("significant difference")
                    lst.Add("no change")
                    lst.Add("unchangeable")
                    lst.Add("unchanged")

                    Return lst
                End Function
                Private Shared Function GetCompositeHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)

                    lst.Add("leads to")
                    lst.Add("consequence of")
                    lst.Add("it follows that")
                    lst.Add("comprises of")
                    lst.Add("comprises")
                    lst.Add("builds towards")
                    Return lst
                End Function

                Private Shared Function GetNullHypothesisIndicators() As List(Of String)
                    Dim lst As New List(Of String)
                    lst.Add("no significant relationship")
                    lst.Add("This contradicts")
                    lst.Add("no significance")
                    lst.Add("does not signify")
                    lst.Add("this disproves")
                    lst.Add("Contrary to popular belief")
                    lst.Add("this negates")
                    Return lst
                End Function
            End Class
            Public Class QuestionDetector

                Private ReadOnly ClassificationRules As List(Of ClassificationRule)

                Private Shared ReadOnly CauseEffectQuestionIndicators As String() = {"why does", "how does", "what causes"}
                Private Shared ReadOnly ComparativeQuestionIndicators As String() = {"which", "what is the difference", "how does"}
                Private Shared ReadOnly DependentQuestionIndicators As String() = {"if", "unless", "whether", "in case"}
                Private Shared ReadOnly DescriptiveQuestionIndicators As String() = {"describe", "explain", "tell me about"}
                Private Shared ReadOnly HypotheticalQuestionIndicators As String() = {"what if", "imagine", "suppose", "assume"}
                Private Shared ReadOnly IndependentQuestionIndicators As String() = {"what", "who", "where", "when", "why", "how"}
                Private Shared ReadOnly LocationalQuestionIndicators As String() = {"where is", "where was", "where are"}
                Private Shared ReadOnly SocialQuestionIndicators As String() = {"who is", "who was", "who were", "do you", "do they"}
                Private Shared ReadOnly TemporalQuestionIndicators As String() = {"when is", "when was", "when were", "when are", "what day", "what time"}
                Private Shared ReadOnly QuestionPattern As Regex = New Regex("^\s*(?:what|who|where|when|why|how|if|unless|whether|in case|which|what is the difference|how does|why does|describe|explain|tell me about|what if|imagine|suppose|assume)\b", RegexOptions.IgnoreCase)

                Public Shared Function ClassifySentence(sentence As String) As String
                    Dim lowercaseSentence As String = sentence.ToLower()

                    If IsIndependentQuestion(lowercaseSentence) Then
                        Return "Independent Question"
                    ElseIf IsDependentQuestion(lowercaseSentence) Then
                        Return "Dependent Question"
                    ElseIf IsComparativeQuestion(lowercaseSentence) Then
                        Return "Comparative Question"
                    ElseIf IsCauseEffectQuestion(lowercaseSentence) Then
                        Return "Cause-Effect Question"
                    ElseIf IsDescriptiveQuestion(lowercaseSentence) Then
                        Return "Descriptive Question"
                    ElseIf IsHypotheticalQuestion(lowercaseSentence) Then
                        Return "Hypothetical Question"
                    ElseIf IsTemporalQuestion(lowercaseSentence) Then
                        Return "Temporal Question"
                    ElseIf IsSocialQuestion(lowercaseSentence) Then
                        Return "Social Question"
                    ElseIf IsLocationalQuestion(lowercaseSentence) Then
                        Return "Locational Question"
                    Else
                        Return "Unclassified"
                    End If
                End Function
                Private Shared Function IsQuestionType(sentence As String, indicators As String()) As Boolean
                    Return StartsWithAny(sentence, indicators)
                End Function



                Public Shared Function GetSentence(sentence As String) As CapturedType
                    Dim lowercaseSentence As String = sentence.ToLower()
                    Dim newType As New CapturedType With {
        .sentence = lowercaseSentence,
        .LogicalRelation_ = LogicalDependencyClassifier.ClassifyLogicalDependency(lowercaseSentence)
    }

                    If IsQuestionType(lowercaseSentence, IndependentQuestionIndicators) Then
                        newType.SubType = "Independent Question"
                    ElseIf IsQuestionType(lowercaseSentence, DependentQuestionIndicators) Then
                        newType.SubType = "Dependent Question"
                    ElseIf IsQuestionType(lowercaseSentence, ComparativeQuestionIndicators) Then
                        newType.SubType = "Comparative Question"
                    ElseIf IsQuestionType(lowercaseSentence, CauseEffectQuestionIndicators) Then
                        newType.SubType = "Cause-Effect Question"
                    ElseIf IsQuestionType(lowercaseSentence, DescriptiveQuestionIndicators) Then
                        newType.SubType = "Descriptive Question"
                    ElseIf IsQuestionType(lowercaseSentence, HypotheticalQuestionIndicators) Then
                        newType.SubType = "Hypothetical Question"
                    ElseIf IsQuestionType(lowercaseSentence, TemporalQuestionIndicators) Then
                        newType.SubType = "Temporal Question"
                    ElseIf IsQuestionType(lowercaseSentence, SocialQuestionIndicators) Then
                        newType.SubType = "Social Question"
                    ElseIf IsQuestionType(lowercaseSentence, LocationalQuestionIndicators) Then
                        newType.SubType = "Locational Question"
                    Else
                        newType.SubType = "Unclassified"
                    End If

                    Return newType
                End Function

                Private Shared Function IsLocationalQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(LocationalQuestionIndicators)
                End Function
                Private Shared Function IsSocialQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(SocialQuestionIndicators)
                End Function
                Private Shared Function IsTemporalQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(TemporalQuestionIndicators)
                End Function
                Private Shared Function IsCauseEffectQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(CauseEffectQuestionIndicators)
                End Function

                Private Shared Function IsComparativeQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(ComparativeQuestionIndicators)
                End Function

                Private Shared Function IsDependentQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(DependentQuestionIndicators)
                End Function

                Private Shared Function IsDescriptiveQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(DescriptiveQuestionIndicators)
                End Function

                Private Shared Function IsHypotheticalQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(HypotheticalQuestionIndicators)
                End Function

                Private Shared Function IsIndependentQuestion(sentence As String) As Boolean
                    Return sentence.StartsWithAny(IndependentQuestionIndicators)
                End Function

                Private Shared Function StartsWithAny(ByVal input As String, ByVal values As String()) As Boolean
                    For Each value As String In values
                        If input.StartsWith(value, StringComparison.OrdinalIgnoreCase) Then
                            Return True
                        End If
                    Next
                    Return False
                End Function

                Public Shared Sub Main()
                    ' Example usage
                    Dim sentences As String() = {
                "What is the effect of smoking on health?",
                "How does exercise affect weight loss?",
                "If it rains, will the event be canceled?",
                "Describe the process of photosynthesis.",
                "What if I don't submit the assignment?",
                "Who discovered penicillin?",
                "Where is the nearest hospital?",
                "When was the Declaration of Independence signed?",
                "Why is the sky blue?",
                "How does a computer work?"
            }

                    For Each sentence As String In sentences
                        Dim questionType = GetSentence(sentence)
                        Console.WriteLine("Sentence: " & questionType.Sentence)
                        Console.WriteLine("Question Type: " & questionType.SubType)
                        Console.WriteLine("Logical Relation: " & questionType.LogicalRelation_)
                        Console.WriteLine()
                    Next

                    Console.ReadLine()
                End Sub
            End Class

        End Class
        Public Class LogicalArgumentClassifier
            Private Shared ClassificationRules As List(Of ClassificationRule)

            Public Shared Function ClassifyText(ByVal text As String) As List(Of ClassificationRule)
                Dim matchedRules As New List(Of ClassificationRule)()

                For Each rule As ClassificationRule In ClassificationRules
                    For Each pattern As Regex In rule.Patterns
                        If pattern.IsMatch(text) Then
                            matchedRules.Add(rule)
                            Exit For
                        End If
                    Next
                Next

                Return matchedRules
            End Function

            Public Shared Sub Main()
                ' Define the classification rules
                ClassificationRules = New List(Of ClassificationRule)()

                ' Question Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Question",
        .Subtype = "General",
        .Relationship = "General Question",
        .Patterns = New List(Of Regex)() From {
            New Regex("^(?:What|Who|Where|When|Why|How)\b", RegexOptions.IgnoreCase),
            New Regex("^Is\b", RegexOptions.IgnoreCase),
            New Regex("^\bCan\b", RegexOptions.IgnoreCase),
            New Regex("^\bAre\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Question",
        .Subtype = "Comparison",
        .Relationship = "Compare Premise",
        .Patterns = New List(Of Regex)() From {
            New Regex("^(?:Which|What|Who)\b.*\b(is|are)\b.*\b(?:better|worse|superior|inferior|more|less)\b", RegexOptions.IgnoreCase),
            New Regex("^(?:How|In what way)\b.*\b(?:different|similar|alike)\b", RegexOptions.IgnoreCase),
            New Regex("^(?:Compare|Contrast)\b", RegexOptions.IgnoreCase)
        }
    })

                ' Answer Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Answer",
        .Subtype = "General",
        .Relationship = "Confirm/Deny",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bYes\b", RegexOptions.IgnoreCase),
            New Regex("^\bNo\b", RegexOptions.IgnoreCase),
            New Regex("^\bMaybe\b", RegexOptions.IgnoreCase),
            New Regex("^\bI don't know\b", RegexOptions.IgnoreCase)
        }
    })

                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Answer",
        .Subtype = "Comparison",
        .Relationship = "Compare Premise",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bA is\b.*\b(?:better|worse|superior|inferior|more|less)\b", RegexOptions.IgnoreCase),
            New Regex("^\bA is\b.*\b(?:different|similar|alike)\b", RegexOptions.IgnoreCase),
            New Regex("^\bIt depends\b", RegexOptions.IgnoreCase),
            New Regex("^\bBoth\b", RegexOptions.IgnoreCase)
        }
    })

                ' Hypothesis Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Hypothesis",
        .Subtype = "General",
        .Relationship = "Hypothesize",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bIf\b", RegexOptions.IgnoreCase),
            New Regex("^\bAssuming\b", RegexOptions.IgnoreCase),
            New Regex("^\bSuppose\b", RegexOptions.IgnoreCase),
            New Regex("^\bHypothesize\b", RegexOptions.IgnoreCase)
        }
    })

                ' Conclusion Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Conclusion",
        .Subtype = "General",
        .Relationship = "Follow On Conclusion",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bTherefore\b", RegexOptions.IgnoreCase),
            New Regex("^\bThus\b", RegexOptions.IgnoreCase),
            New Regex("^\bHence\b", RegexOptions.IgnoreCase),
            New Regex("^\bConsequently\b", RegexOptions.IgnoreCase)
        }
    })

                ' Premise Rules
                ClassificationRules.Add(New ClassificationRule() With {
        .Type = "Premise",
        .Subtype = "General",
        .Relationship = "Reason",
        .Patterns = New List(Of Regex)() From {
            New Regex("^\bBecause\b", RegexOptions.IgnoreCase),
            New Regex("^\bSince\b", RegexOptions.IgnoreCase),
            New Regex("^\bGiven that\b", RegexOptions.IgnoreCase),
            New Regex("^\bConsidering\b", RegexOptions.IgnoreCase)
        }
    })

                ' Test the function
                Dim text As String = "What is the hypothesis for this experiment?"
                Dim matchedRules As List(Of ClassificationRule) = ClassifyText(text)

                Console.WriteLine("Matched Rules:")
                For Each rule As ClassificationRule In matchedRules
                    Console.WriteLine("Type: " & rule.Type)
                    Console.WriteLine("Subtype: " & rule.Subtype)
                    Console.WriteLine("Relationship: " & rule.Relationship)
                    Console.WriteLine()
                Next

                Console.ReadLine()
            End Sub
        End Class
        Public Class ContextAnalyzer
            Public Class StatementGroup
                ''' <summary>
                ''' Gets or sets the type of the statement group (e.g., Premise, Conclusion, Hypothesis).
                ''' </summary>
                Public Property Type As String

                ''' <summary>
                ''' Gets or sets the list of sentences in the statement group.
                ''' </summary>
                Public Property Sentences As List(Of String)

                ''' <summary>
                ''' Initializes a new instance of the StatementGroup class.
                ''' </summary>
                Public Sub New()
                    Sentences = New List(Of String)()
                End Sub
            End Class
            Public Function GroupStatements(ByVal sentences As List(Of String)) As List(Of StatementGroup)
                Dim statementGroups As New List(Of StatementGroup)()
                Dim previousSentenceType As String = ""

                ' Placeholder for sentence classification logic
                Dim sentenceTypes = SentenceClassifier.ClassifySentences(sentences)

                For Each item In sentenceTypes
                    Dim sentenceType = item.Type

                    ' Perform context analysis based on the previous sentence type and current sentence
                    ' Apply rules or patterns to group the sentences accordingly

                    ' Example rule: If the current sentence is a premise and the previous sentence is a conclusion, group them together
                    If sentenceType = "Premise" AndAlso previousSentenceType = "Conclusion" Then
                        ' Check if the group exists, otherwise create a new group
                        Dim premiseGroup = statementGroups.FirstOrDefault(Function(g) g.Type = "Premise")
                        If premiseGroup Is Nothing Then
                            premiseGroup = New StatementGroup() With {.Type = "Premise"}
                            statementGroups.Add(premiseGroup)
                        End If

                        ' Add the current premise sentence to the existing group
                        premiseGroup.Sentences.Add(item.Entity.Sentence)
                    End If

                    ' Example rule: If the current sentence is a premise and the previous sentence is a Hypothesis, group them together
                    If sentenceType = "Premise" AndAlso previousSentenceType = "Hypothesis" Then
                        ' Check if the group exists, otherwise create a new group
                        Dim premiseGroup = statementGroups.FirstOrDefault(Function(g) g.Type = "Premise")
                        If premiseGroup Is Nothing Then
                            premiseGroup = New StatementGroup() With {.Type = "Premise"}
                            statementGroups.Add(premiseGroup)
                        End If

                        ' Add the current premise sentence to the existing group
                        premiseGroup.Sentences.Add(item.Entity.Sentence)
                    End If

                    ' Add more rules or patterns to group other sentence types

                    ' Update the previous sentence type for the next iteration
                    previousSentenceType = sentenceType
                Next

                ' Return the grouped statements
                Return statementGroups
            End Function

            ''' <summary>
            ''' Attempts to find context Sentences for discovered premise of conclusions or hypotheses etc
            ''' </summary>
            ''' <param name="sentences"></param>
            ''' <returns>Type EG Premise / Partner Sentences  conclusions / hypotheses</returns>
            Public Function GetContextStatements(ByVal sentences As List(Of String)) As Dictionary(Of String, List(Of String))
                Dim statementGroups As New Dictionary(Of String, List(Of String))()
                Dim previousSentenceType As String = ""


                Dim sentenceTypes = SentenceClassifier.ClassifySentences(sentences)
                For Each item In sentenceTypes
                    Dim SentenceType = item.Type
                    ' Perform context analysis based on the previous sentence type and current sentence
                    ' Apply rules or patterns to group the sentences accordingly

                    ' Example rule: If the current sentence is a premise and the previous sentence is a conclusion, group them together
                    If SentenceType = "Premise" AndAlso previousSentenceType = "Conclusion" Then
                        ' Check if the group exists, otherwise create a new group
                        If Not statementGroups.ContainsKey(previousSentenceType) Then
                            statementGroups(previousSentenceType) = New List(Of String)()
                        End If

                        ' Add the current premise sentence to the existing group
                        statementGroups(previousSentenceType).Add(item.Entity.Sentence)
                    End If

                    ' Example rule: If the current sentence is a Conclusion and the previous sentence is a Hypotheses, group them together
                    If SentenceType = "Conclusion" AndAlso previousSentenceType = "Hypotheses" Then
                        ' Check if the group exists, otherwise create a new group
                        If Not statementGroups.ContainsKey(previousSentenceType) Then
                            statementGroups(previousSentenceType) = New List(Of String)()
                        End If

                        ' Add the current premise sentence to the existing group
                        statementGroups(previousSentenceType).Add(item.Entity.Sentence)
                    End If

                    ' Example rule: If the current sentence is a premise and the previous sentence is a Hypotheses, group them together
                    If SentenceType = "Premise" AndAlso previousSentenceType = "Hypotheses" Then
                        ' Check if the group exists, otherwise create a new group
                        If Not statementGroups.ContainsKey(previousSentenceType) Then
                            statementGroups(previousSentenceType) = New List(Of String)()
                        End If

                        ' Add the current premise sentence to the existing group
                        statementGroups(previousSentenceType).Add(item.Entity.Sentence)
                    End If
                    ' Add more rules or patterns to group other sentence types

                    ' Update the previous sentence type for the next iteration
                    previousSentenceType = SentenceType
                Next

                ' Return the grouped statements
                Return statementGroups
            End Function
        End Class
        Public Class PronounResolver
            Public Function ResolvePronoun(sentence As String, pronoun As String) As String
                ' Tokenize the sentence into words
                Dim words As String() = Split(sentence, " ",)

                ' Find the position of the pronoun in the sentence
                Dim pronounIndex As Integer = Array.IndexOf(words, pronoun)

                ' Search for antecedents before the pronoun
                For i As Integer = pronounIndex - 1 To 0 Step -1
                    Dim currentWord As String = words(i)
                    ' Check if the current word is a noun or a pronoun
                    If IsNounOrPronoun(currentWord.ToLower) Then
                        Return currentWord
                    End If
                Next

                ' Search for antecedents after the pronoun
                For i As Integer = pronounIndex + 1 To words.Length - 1
                    Dim currentWord As String = words(i)
                    ' Check if the current word is a noun or a pronoun
                    If IsNounOrPronoun(currentWord.ToLower) Then
                        Return currentWord
                    End If
                Next

                ' If no antecedent is found, return an appropriate message
                Return "No antecedent found for the pronoun."
            End Function
            Public Function ResolveGender(ByRef Pronoun As String) As String
                ' If IsProNoun(Pronoun) = True Then

                If IsFemale(Pronoun) = True Then Return "Female"

                If IsMale(Pronoun) = True Then Return "Male"

                'End If
                Return "Non-Binary"
            End Function


            Private Nounlist As New List(Of String)
            Private MaleNames As New List(Of String)
            Private FemaleNames As New List(Of String)
            Public MalePronouns As New List(Of String)
            Public FemalePronouns As New List(Of String)
            Private PronounList As New List(Of String)
            Public MalePersonalNouns() As String = {"him", " he", "his"}
            Public FemalePersonalNouns() As String = {"she", " her", "hers"}
            Public Sub New()
                LoadLists()

            End Sub

            Private Sub LoadLists()
                Dim corpusRoot As String = Application.StartupPath & "\data\"
                Dim wordlistPath As String = Path.Combine(corpusRoot, "NounList.txt")
                Dim wordlistReader As New WordListReader(wordlistPath)
                Nounlist = wordlistReader.GetWords()
                wordlistPath = Path.Combine(corpusRoot, "ProNounList.txt")
                wordlistReader = New WordListReader(wordlistPath)
                PronounList = wordlistReader.GetWords()
                wordlistPath = Path.Combine(corpusRoot, "MaleNames.txt")
                wordlistReader = New WordListReader(wordlistPath)
                MaleNames = wordlistReader.GetWords()
                wordlistPath = Path.Combine(corpusRoot, "FemaleNames.txt")
                wordlistReader = New WordListReader(wordlistPath)
                FemaleNames = wordlistReader.GetWords()
                wordlistPath = Path.Combine(corpusRoot, "MalePronouns.txt")
                wordlistReader = New WordListReader(wordlistPath)
                MalePronouns = wordlistReader.GetWords()
                wordlistPath = Path.Combine(corpusRoot, "FemalePronouns.txt")
                wordlistReader = New WordListReader(wordlistPath)
                FemalePronouns = wordlistReader.GetWords()
            End Sub
            Private Function IsNounOrPronoun(ByRef Word As String) As Boolean
                If IsNoun(Word) = True Then Return True
                If IsProNoun(Word) = True Then Return True

                Return False
            End Function
            Public Function IsProNoun(ByRef Word As String) As Boolean
                For Each item In PronounList
                    If Word.ToLower = item.ToLower Then Return True
                Next
                For Each item In FemalePronouns
                    If Word.ToLower = item.ToLower Then Return True
                Next
                For Each item In MalePronouns
                    If Word.ToLower = item.ToLower Then Return True
                Next
                For Each item In FemalePersonalNouns
                    If Word.ToLower = item.ToLower Then Return True
                Next
                For Each item In MalePersonalNouns
                    If Word.ToLower = item.ToLower Then Return True
                Next
                Return False
            End Function
            Public Function IsMale(ByRef Word As String) As Boolean
                For Each item In MaleNames
                    If Word.ToLower = item.ToLower Then Return True
                Next
                For Each item In MalePronouns
                    If Word.ToLower = item.ToLower Then Return True
                Next
                For Each item In MalePersonalNouns
                    If Word.ToLower = item.ToLower Then Return True
                Next
                Return False
            End Function
            Public Function IsFemale(ByRef Word As String)
                For Each item In FemaleNames
                    If Word.ToLower = item.ToLower Then Return True
                Next
                For Each item In FemalePronouns
                    If Word.ToLower = item.ToLower Then Return True
                Next
                For Each item In FemalePersonalNouns
                    If Word.ToLower = item.ToLower Then Return True
                Next
                Return False
            End Function
            Public Function IsNoun(ByRef Word As String) As Boolean
                For Each item In Nounlist
                    If Word.ToLower = item.ToLower Then Return True
                Next

                Return False
            End Function
        End Class

    End Namespace
End Namespace