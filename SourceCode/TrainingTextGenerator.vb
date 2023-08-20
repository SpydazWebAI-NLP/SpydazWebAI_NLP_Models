Imports System.Text
Imports InputModelling.Models.EntityModel

Namespace TrainingData
    Public Class TrainingTextGenerator
        Public EntityTypes As List(Of String)
        Public EntityList As List(Of Entity)

        Private Shared Random As New Random()

        Public Subjects As String()
        Public Verbs As String()
        Public Objects As String()
        Function GetCVS() As List(Of String)
            Dim lst As New List(Of String)
            lst.Add(DataScientistCV)
            lst.Add(WebDeveloperCV)
            lst.Add(SoftwareDeveloperCV)

            Return lst
        End Function
        Function DataScientistCV() As String
            Return "
Name: Jane Smith
Profession: Data Scientist
Experience: 8 years
Skills: Python, R, Machine Learning
Qualifications: Ph.D. in Data Science
"

        End Function

        ' CV Examples
        Function SoftwareDeveloperCV()
            Return "
Name: John Doe
Profession: Software Developer
Experience: 5 years
Skills: Python, Java
Qualifications: Bachelor's degree in Computer Science
"

        End Function
        Function WebDeveloperCV()
            Return "
Name: Mark Johnson
Profession: Web Developer
Experience: 10 years
Skills: HTML, CSS, JavaScript, PHP
Qualifications: Bachelor's degree in Web Development
"

        End Function
        Public Sub New(entityTypes As List(Of String),
                   entityList As List(Of Entity),
                   subjects() As String,
                   verbs() As String,
                   objects() As String)
            If entityTypes Is Nothing Then
                Throw New ArgumentNullException(NameOf(entityTypes))
            End If

            If entityList Is Nothing Then
                Throw New ArgumentNullException(NameOf(entityList))
            End If

            If subjects Is Nothing Then
                Throw New ArgumentNullException(NameOf(subjects))
            End If

            If verbs Is Nothing Then
                Throw New ArgumentNullException(NameOf(verbs))
            End If

            If objects Is Nothing Then
                Throw New ArgumentNullException(NameOf(objects))
            End If

            Me.EntityTypes = entityTypes
            Me.EntityList = entityList
            Me.Subjects = subjects
            Me.Verbs = verbs
            Me.Objects = objects
        End Sub
        Public Shared Function GetAnotatedTopicList() As List(Of String)
            Dim annotatedData As New List(Of String)
            annotatedData.Add("Gravity: concept")
            annotatedData.Add("Force: concept")
            annotatedData.Add("Energy: concept")
            annotatedData.Add("Electricity: concept")
            annotatedData.Add("Magnetism: concept")
            annotatedData.Add("Atoms: concept")
            annotatedData.Add("Light: concept")
            annotatedData.Add("Sound: concept")
            annotatedData.Add("Friction: concept")
            annotatedData.Add("Heat: concept")
            annotatedData.Add("Acceleration: concept")
            annotatedData.Add("Velocity: concept")
            annotatedData.Add("Mass: concept")
            annotatedData.Add("Density: concept")
            annotatedData.Add("Volume: concept")
            annotatedData.Add("Pressure: concept")
            annotatedData.Add("Temperature: concept")
            annotatedData.Add("Electric circuit: concept")
            annotatedData.Add("Solar system: concept")
            annotatedData.Add("Photosynthesis: concept")
            annotatedData.Add("Chemical reactions: concept")
            annotatedData.Add("Simple machines: concept")
            annotatedData.Add("Ecosystem: concept")
            annotatedData.Add("States of matter: concept")
            annotatedData.Add("Genetics: concept")
            annotatedData.Add("Plant growth: concept")
            annotatedData.Add("Human body: concept")
            annotatedData.Add("Pollution: concept")
            annotatedData.Add("Water cycle: concept")
            annotatedData.Add("Weather: concept")
            annotatedData.Add("Electric motors: concept")
            annotatedData.Add("Renewable energy: concept")
            annotatedData.Add("Food chain: concept")
            annotatedData.Add("Space exploration: concept")
            annotatedData.Add("Computer programming: concept")
            annotatedData.Add("Scientific method: concept")
            annotatedData.Add("Photosynthesis: concept")
            annotatedData.Add("Periodic table: concept")
            annotatedData.Add("DNA: concept")
            annotatedData.Add("Motion: concept")
            annotatedData.Add("Newton's laws: concept")
            annotatedData.Add("Gravity: concept")
            annotatedData.Add("Sound waves: concept")
            annotatedData.Add("Electricity: concept")
            annotatedData.Add("Atoms: concept")
            annotatedData.Add("Light: concept")
            annotatedData.Add("Matter: concept")
            annotatedData.Add("Solar energy: concept")
            annotatedData.Add("Chemical reactions: concept")
            annotatedData.Add("Genetics: concept")
            annotatedData.Add("Plant growth: concept")
            annotatedData.Add("Human body: concept")
            ' Biology Entities
            annotatedData.Add("Human: organism")
            annotatedData.Add("Dog: organism")
            annotatedData.Add("Tree: organism")
            annotatedData.Add("Bird: organism")
            annotatedData.Add("Insect: organism")
            annotatedData.Add("Fish: organism")
            annotatedData.Add("Amoeba: organism")
            annotatedData.Add("Bacteria: organism")
            annotatedData.Add("Flower: organism")
            annotatedData.Add("Fruit: organism")
            annotatedData.Add("Photosynthesis: process")
            annotatedData.Add("Cell: part")
            annotatedData.Add("Digestive system: system")
            annotatedData.Add("DNA: molecule")
            annotatedData.Add("Respiratory system: system")
            annotatedData.Add("Life cycle: cycle")
            annotatedData.Add("Circulatory system: system")
            annotatedData.Add("Ecosystem: system")
            annotatedData.Add("Classification: classification")
            annotatedData.Add("Nervous system: system")

            ' Computer Science Entities
            annotatedData.Add("Programming: field")
            annotatedData.Add("Computer: device")
            annotatedData.Add("Algorithm: concept")
            annotatedData.Add("Coding: skill")
            annotatedData.Add("Network: concept")
            annotatedData.Add("Cybersecurity: field")
            annotatedData.Add("Graphics: concept")
            annotatedData.Add("Artificial intelligence: concept")
            annotatedData.Add("Software: component")
            annotatedData.Add("Game: application")
            annotatedData.Add("Internet: network")
            annotatedData.Add("Hardware: component")
            annotatedData.Add("Data encryption: technique")
            annotatedData.Add("Memory: component")
            annotatedData.Add("Cloud computing: concept")
            annotatedData.Add("Robot: application")

            ' NLP Entities
            annotatedData.Add("Natural language processing: field")
            annotatedData.Add("Chatbot: application")
            annotatedData.Add("Voice recognition: technology")
            annotatedData.Add("Sentiment analysis: technique")
            annotatedData.Add("Machine translation: technique")
            annotatedData.Add("Spam filter: application")
            annotatedData.Add("Named entity recognition: technique")
            annotatedData.Add("Text summarization: technique")
            annotatedData.Add("Text classification: technique")
            annotatedData.Add("Topic modeling: technique")
            annotatedData.Add("Search engine: application")
            annotatedData.Add("Part-of-speech tagging: technique")
            annotatedData.Add("Text generation: technique")
            annotatedData.Add("Information extraction: technique")
            annotatedData.Add("Language model: model")

            annotatedData.Add("John Doe: person")
            annotatedData.Add("Emma Smith: person")
            annotatedData.Add("Microsoft: organization")
            annotatedData.Add("Apple Inc.: organization")
            annotatedData.Add("New York City: location")
            annotatedData.Add("London: location")
            annotatedData.Add("Tom Johnson: person")
            annotatedData.Add("Google: organization")
            annotatedData.Add("Paris: location")
            annotatedData.Add("Emily Brown: person")
            annotatedData.Add("Amazon: organization")
            annotatedData.Add("Los Angeles: location")
            annotatedData.Add("Mark Thompson: person")
            annotatedData.Add("Facebook: organization")
            annotatedData.Add("San Francisco: location")
            annotatedData.Add("Robert Anderson: person")
            annotatedData.Add("Twitter: organization")
            annotatedData.Add("Chicago: location")
            annotatedData.Add("Sarah Davis: person")
            annotatedData.Add("IBM: organization")
            annotatedData.Add("Seattle: location")
            annotatedData.Add("James Wilson: person")
            annotatedData.Add("Oracle: organization")
            annotatedData.Add("Berlin: location")
            annotatedData.Add("Jennifer Lee: person")
            annotatedData.Add("Intel: organization")
            annotatedData.Add("Tokyo: location")
            ' Biology Entities
            annotatedData.Add("Human: organism")
            annotatedData.Add("Dog: organism")
            annotatedData.Add("Tree: organism")
            annotatedData.Add("Bird: organism")
            annotatedData.Add("Insect: organism")
            annotatedData.Add("Fish: organism")
            annotatedData.Add("Amoeba: organism")
            annotatedData.Add("Bacteria: organism")
            annotatedData.Add("Flower: organism")
            annotatedData.Add("Fruit: organism")
            annotatedData.Add("Photosynthesis: process")
            annotatedData.Add("Cell: part")
            annotatedData.Add("Digestive system: system")
            annotatedData.Add("DNA: molecule")
            annotatedData.Add("Respiratory system: system")
            annotatedData.Add("Life cycle: cycle")
            annotatedData.Add("Circulatory system: system")
            annotatedData.Add("Ecosystem: system")
            annotatedData.Add("Classification: classification")
            annotatedData.Add("Nervous system: system")

            ' Computer Science Entities
            annotatedData.Add("Programming: field")
            annotatedData.Add("Computer: device")
            annotatedData.Add("Algorithm: concept")
            annotatedData.Add("Coding: skill")
            annotatedData.Add("Network: concept")
            annotatedData.Add("Cybersecurity: field")
            annotatedData.Add("Graphics: concept")
            annotatedData.Add("Artificial intelligence: concept")
            annotatedData.Add("Software: component")
            annotatedData.Add("Game: application")
            annotatedData.Add("Internet: network")
            annotatedData.Add("Hardware: component")
            annotatedData.Add("Data encryption: technique")
            annotatedData.Add("Memory: component")
            annotatedData.Add("Cloud computing: concept")
            annotatedData.Add("Robot: application")

            ' NLP Entities
            annotatedData.Add("Natural language processing: field")
            annotatedData.Add("Chatbot: application")
            annotatedData.Add("Voice recognition: technology")
            annotatedData.Add("Sentiment analysis: technique")
            annotatedData.Add("Machine translation: technique")
            annotatedData.Add("Spam filter: application")
            annotatedData.Add("Named entity recognition: technique")
            annotatedData.Add("Text summarization: technique")
            annotatedData.Add("Text classification: technique")
            annotatedData.Add("Topic modeling: technique")
            annotatedData.Add("Search engine: application")
            annotatedData.Add("Part-of-speech tagging: technique")
            annotatedData.Add("Text generation: technique")
            annotatedData.Add("Information extraction: technique")
            annotatedData.Add("Language model: model")

            Return annotatedData
        End Function
        ''' <summary>
        ''' allows for adding quick training data lists
        ''' </summary>
        ''' <param name="annotatedData"></param>
        ''' <returns></returns>
        Public Function AddAnnotatedData(annotatedData As List(Of String)) As List(Of Entity)
            Dim Entitys As New List(Of Entity)
            For Each entry In annotatedData
                Dim parts As String() = entry.Split(":")
                Dim text As String = parts(0).Trim()
                Dim entity As String = parts(1).Trim()
                Dim NewEnt As Entity

                For Each item In entity
                    Dim NewType As New EntityType
                    NewEnt = New Entity(item)
                    NewType.Type = text
                    NewEnt.MemberOfEntityTypes.Add(NewType)

                    Entitys.Add(NewEnt)
                Next

            Next
            Return Entitys
        End Function
        Public Function GenerateSentence() As String
            Dim subject As String = GetRandomItem(Subjects)
            Dim verb As String = GetRandomItem(Verbs)
            Dim iobject As String = GetRandomItem(Objects)

            Dim sentence As String = $"{subject} {verb} {iobject}."

            ' Add contextual variation
            Dim context As String = GetRandomContext()
            sentence = $"{context} {sentence}"

            Return sentence
        End Function
        Private Function GetRandomEntity() As String

            Dim index As Integer = Random.Next(0, EntityTypes.Count)
            Return EntityTypes(index)
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
        Private Function GetRandomContext() As String
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

        Private Shared Function GetRandomItem(items As String()) As String
            Dim index As Integer = Random.Next(0, items.Length)
            Return items(index)
        End Function
        ''' <summary>
        ''' Returns a list of various code snippets
        ''' </summary>
        ''' <returns></returns>
        Public Function LoadSourceCodeTrainingData() As List(Of String)
            ' Training data
            Dim trainingData As List(Of String) = New List(Of String) From {
                "public void method1() {",
                "    Console.WriteLine(""Hello, world!"");",
                "}",
                "",
                "public int method2(int x, int y) {",
                "    int sum = x + y;",
                "    return sum;",
                "}"
            }

            trainingData.Add("Sub HelloWorld()")
            trainingData.Add("    Console.WriteLine(""Hello, World!"")")
            trainingData.Add("End Sub")

            Console.ReadLine()
            trainingData.Add("Public Sub CalculateSum(a As Integer, b As Integer) As Integer")
            trainingData.Add("    Return a + b")
            trainingData.Add("End Sub")

            trainingData.Add("Public Function GetMaxValue(numbers As List(Of Integer)) As Integer")
            trainingData.Add("    Return numbers.Max()")
            trainingData.Add("End Function")

            trainingData.Add("Public Sub PrintMessage(message As String)")
            trainingData.Add("    Console.WriteLine(message)")
            trainingData.Add("End Sub")

            trainingData.Add("Public Function IsPrime(number As Integer) As Boolean")
            trainingData.Add("    If number < 2 Then")
            trainingData.Add("        Return False")
            trainingData.Add("    End If")
            trainingData.Add("    For i As Integer = 2 To Math.Sqrt(number)")
            trainingData.Add("        If number Mod i = 0 Then")
            trainingData.Add("            Return False")
            trainingData.Add("        End If")
            trainingData.Add("    Next")
            trainingData.Add("    Return True")
            trainingData.Add("End Function")

            trainingData.Add("Public Sub SortArray(numbers As Integer())")
            trainingData.Add("    Array.Sort(numbers)")
            trainingData.Add("End Sub")

            trainingData.Add("Public Function ConvertToUpperCase(text As String) As String")
            trainingData.Add("    Return text.ToUpper()")
            trainingData.Add("End Function")

            trainingData.Add("Public Sub WaitForInput()")
            trainingData.Add("    Console.ReadLine()")
            trainingData.Add("End Sub")

            trainingData.Add("Public Function ConcatenateStrings(strings As String()) As String")
            trainingData.Add("    Return String.Concat(strings)")
            trainingData.Add("End Function")

            trainingData.Add("Public Sub PerformTask(task As Action)")
            trainingData.Add("    task.Invoke()")
            trainingData.Add("End Sub")
            'Dim trainingData As List(Of String) = New List(Of String)()
            trainingData.Add("Public Class MyClass")
            trainingData.Add("    Public Sub MyMethod()")
            trainingData.Add("        Console.WriteLine(""Hello, world!"")")
            trainingData.Add("    End Sub")
            trainingData.Add("End Class")

            trainingData.Add("Public Class MyClass")
            trainingData.Add("    Public Sub MyMethod()")
            trainingData.Add("        Console.WriteLine(""Hello, world!"")")
            trainingData.Add("    End Sub")
            trainingData.Add("End Class")

            trainingData.Add("Public Class Rectangle")
            trainingData.Add("    Private width As Integer")
            trainingData.Add("    Private height As Integer")
            trainingData.Add("    Public Sub SetDimensions(w As Integer, h As Integer)")
            trainingData.Add("        width = w")
            trainingData.Add("        height = h")
            trainingData.Add("    End Sub")
            trainingData.Add("    Public Function GetArea() As Integer")
            trainingData.Add("        Return width * height")
            trainingData.Add("    End Function")
            trainingData.Add("End Class")
            trainingData.Add("Public Class Singleton")
            trainingData.Add("    Private Shared instance As Singleton")
            trainingData.Add("    Private Sub New()")
            trainingData.Add("    End Sub")
            trainingData.Add("    Public Shared Function GetInstance() As Singleton")
            trainingData.Add("        If instance Is Nothing Then")
            trainingData.Add("            instance = New Singleton()")
            trainingData.Add("        End If")
            trainingData.Add("        Return instance")
            trainingData.Add("    End Function")
            trainingData.Add("End Class")

            trainingData.Add("Public Interface ISubject")
            trainingData.Add("    Sub Attach(observer As IObserver)")
            trainingData.Add("    Sub Detach(observer As IObserver)")
            trainingData.Add("    Sub Notify()")
            trainingData.Add("End Interface")

            trainingData.Add("Public Class ConcreteSubject Implements ISubject")
            trainingData.Add("    Private observers As List(Of IObserver) = New List(Of IObserver)()")
            trainingData.Add("    Public Sub Attach(observer As IObserver) Implements ISubject.Attach")
            trainingData.Add("        observers.Add(observer)")
            trainingData.Add("    End Sub")
            trainingData.Add("    Public Sub Detach(observer As IObserver) Implements ISubject.Detach")
            trainingData.Add("        observers.Remove(observer)")
            trainingData.Add("    End Sub")
            trainingData.Add("    Public Sub Notify() Implements ISubject.Notify")
            trainingData.Add("        For Each observer As IObserver In observers")
            trainingData.Add("            observer.Update()")
            trainingData.Add("        Next")
            trainingData.Add("    End Sub")
            trainingData.Add("End Class")

            trainingData.Add("Public Interface IObserver")
            trainingData.Add("    Sub Update()")
            trainingData.Add("End Interface")

            trainingData.Add("Public Class ConcreteObserver Implements IObserver")
            trainingData.Add("    Public Sub Update() Implements IObserver.Update")
            trainingData.Add("        ' Perform observer update logic here")
            trainingData.Add("    End Sub")
            trainingData.Add("End Class")

            trainingData.Add("Public Sub Main()")
            trainingData.Add("    Dim rect As New Rectangle()")
            trainingData.Add("    rect.SetDimensions(5, 10)")
            trainingData.Add("    Dim area As Integer = rect.GetArea()")
            trainingData.Add("    Console.WriteLine(""Area: "" & area)")
            trainingData.Add("End Sub")
            Return trainingData
        End Function
        Public entityRegExTypes As New Dictionary(Of String, String)() From {
        {"PERSON", "([A-Z][a-zA-Z]+)"},
        {"ORGANIZATION", "([A-Z][\w&]+(?:\s[A-Z][\w&]+)*)"},
        {"LOCATION", "([A-Z][\w\s&-]+)"},
        {"DATE", "(\d{1,2}\/\d{1,2}\/\d{2,4})"},
        {"TIME", "(\d{1,2}:\d{2}(?::\d{2})?)"},
        {"MONEY", "(\$\d+(?:\.\d{2})?)"},
        {"PERCENT", "(\d+(?:\.\d+)?%)"}
       }



        Public Function GetEntitySentencePatterns() As List(Of String)
            ' Add question patterns to the list
            Dim patterns As New List(Of String)()
            patterns.Add("What is the formula for calculating [attribute]?")
            patterns.Add("How does [entity] work?")
            patterns.Add("What are the basic principles of [entity]?")
            patterns.Add("What is the difference between [entity1] and [entity2]?")
            patterns.Add("How does [entity] affect our daily life?")
            patterns.Add("Why is [entity] important in [subject]?")
            patterns.Add("What are the real-world applications of [entity]?")
            patterns.Add("How do scientists study [entity]?")
            patterns.Add("What are some interesting facts about [entity]?")
            patterns.Add("Can you explain [concept] in simple terms?")
            patterns.Add("What are the different types of [entity]?")
            patterns.Add("How does [entity] help us understand [concept]?")
            patterns.Add("Why is [concept] important in [subject]?")
            patterns.Add("What are the steps involved in [process]?")
            patterns.Add("Can you provide examples of [entity] in everyday life?")
            patterns.Add("What are the main components of [entity]?")
            patterns.Add("How does [entity] relate to [subject]?")
            patterns.Add("What happens when [action] [entity]?")
            patterns.Add("What are the advantages and disadvantages of [entity]?")
            patterns.Add("How can I perform [action] with [entity]?")
            patterns.Add("What are the key features of [entity]?")
            patterns.Add("Can you explain the concept of [entity] using an analogy?")
            patterns.Add("What are the safety precautions when dealing with [entity]?")
            patterns.Add("How does [entity] compare to [similar entity]?")
            patterns.Add("What are the properties of [entity]?")
            patterns.Add("How does [entity] interact with [other entity]?")
            patterns.Add("Why is it important to study [subject]?")
            patterns.Add("What are the limitations of [entity]?")
            patterns.Add("What are the practical uses of [concept]?")
            patterns.Add("How does [entity] impact the environment?")
            patterns.Add("Can you explain the concept of [entity] through an experiment?")
            patterns.Add("What are the future advancements in [subject]?")
            patterns.Add("How does [entity] contribute to sustainable development?")
            patterns.Add("What are the ethical considerations related to [entity]?")
            patterns.Add("How can we solve [problem] using [entity]?")
            patterns.Add("What are the challenges faced in studying [subject]?")
            patterns.Add("How does [entity] support scientific research?")
            patterns.Add("What are the key discoveries or inventions related to [subject]?")
            patterns.Add("How can we use [entity] to improve our daily lives?")
            patterns.Add("What are the different branches or fields within [subject]?")
            patterns.Add("What are the common misconceptions about [entity]?")
            patterns.Add("How does [entity] impact the economy?")
            patterns.Add("Can you provide a step-by-step explanation of [process]?")
            patterns.Add("What are the implications of [entity] on society?")
            patterns.Add("How does [entity] contribute to technological advancements?")
            patterns.Add("What are the current research areas in [subject]?")
            patterns.Add("How does [entity] help us understand the universe?")
            patterns.Add("What are the key principles of [concept]?")
            patterns.Add("Can you explain [entity] using visual aids?")
            patterns.Add("What are the career opportunities in [subject]?")
            patterns.Add("When did [entity] [action]?")
            patterns.Add("What is the [attribute] of [entity]?")
            patterns.Add("How does [entity] [action]?")
            patterns.Add("Who is [entity]'s [relation]?")
            patterns.Add("Where does [entity] [action]?")
            patterns.Add("Why did [entity] [action]?")
            patterns.Add("How many [entity]s are there?")
            patterns.Add("Which [entity] [action]?")
            patterns.Add("Tell me about [entity].")
            patterns.Add("What are the benefits of [entity]?")
            patterns.Add("Can you explain [entity] in simple terms?")
            patterns.Add("What are the different types of [entity]?")
            patterns.Add("Is [entity] a reliable source?")
            patterns.Add("How can I [action] [entity]?")
            patterns.Add("What are the main features of [entity]?")
            patterns.Add("Can you provide examples of [entity]?")
            patterns.Add("What are the steps to [action] [entity]?")
            patterns.Add("What is the purpose of [entity]?")
            patterns.Add("Are there any alternatives to [entity]?")
            patterns.Add("What are the risks associated with [entity]?")
            patterns.Add("How can I contact [entity]?")
            patterns.Add("Is [entity] available in my country?")
            patterns.Add("What are the system requirements for [entity]?")
            patterns.Add("What is the average cost of [entity]?")
            patterns.Add("Can you recommend any resources for learning about [entity]?")
            patterns.Add("What are the legal implications of [action] [entity]?")
            patterns.Add("How does [entity] compare to similar products?")
            patterns.Add("What are the key components of [entity]?")
            patterns.Add("What are the security measures in place for [entity]?")
            patterns.Add("Can you explain the working principle of [entity]?")
            patterns.Add("How long does it take to [action] [entity]?")
            patterns.Add("What are the licensing options for [entity]?")
            patterns.Add("Is [entity] compatible with [software/tool]?")
            patterns.Add("Can you provide a demo of [entity]?")
            patterns.Add("What are the prerequisites for using [entity]?")
            patterns.Add("How does the pricing model for [entity] work?")
            patterns.Add("What are the current trends in [entity]?")
            patterns.Add("What are the limitations of [entity]?")
            patterns.Add("What are the best practices for using [entity]?")
            patterns.Add("What are the common use cases for [entity]?")
            patterns.Add("How does [entity] handle [specific scenario]?")
            patterns.Add("What are the key performance indicators for [entity]?")
            patterns.Add("How is data privacy ensured in [entity]?")
            patterns.Add("What are the prerequisites for implementing [entity]?")
            patterns.Add("What are the data storage options for [entity]?")
            patterns.Add("What are the customization capabilities of [entity]?")
            patterns.Add("What are the integration options for [entity]?")
            patterns.Add("How is [entity] different from [competitor]?")
            patterns.Add("What are the customer reviews for [entity]?")
            patterns.Add("What is the target audience for [entity]?")
            patterns.Add("What are the key metrics to measure [entity] success?")
            patterns.Add("How does [entity] handle scalability?")
            patterns.Add("What are the deployment options for [entity]?")
            patterns.Add("What is the expected ROI of [entity]?")
            patterns.Add("What are the key challenges in implementing [entity]?")
            patterns.Add("How does [entity] ensure data security?")
            patterns.Add("What are the industry standards supported by [entity]?")
            patterns.Add("What is the reputation of [entity] in the market?")
            patterns.Add("How can I provide feedback or suggestions for [entity]?")
            patterns.Add("What are the training resources available for [entity]?")
            patterns.Add("What are the prerequisites for becoming a user of [entity]?")
            patterns.Add("How does [entity] handle data migration?")
            patterns.Add("What are the compliance regulations applicable to [entity]?")
            patterns.Add("What are the maintenance requirements for [entity]?")
            patterns.Add("What is the uptime guarantee for [entity]?")
            patterns.Add("How does [entity] handle error handling?")
            patterns.Add("What are the system integration requirements for [entity]?")
            patterns.Add("What are the support options available for [entity]?")
            patterns.Add("How can I request a feature enhancement for [entity]?")
            patterns.Add("What are the service level agreements for [entity]?")
            patterns.Add("What are the data analytics capabilities of [entity]?")
            patterns.Add("What is the data retention policy of [entity]?")
            patterns.Add("How does [entity] ensure high availability?")
            patterns.Add("What are the backup and disaster recovery mechanisms of [entity]?")
            patterns.Add("How does [entity] handle user access control?")
            patterns.Add("What are the reporting and analytics features of [entity]?")
            patterns.Add("What are the mobile app support options for [entity]?")
            patterns.Add("How does [entity] handle data synchronization?")
            patterns.Add("What are the data export capabilities of [entity]?")
            patterns.Add("What are the supported payment options for [entity]?")
            patterns.Add("How does [entity] handle user authentication?")
            patterns.Add("What are the content management features of [entity]?")
            patterns.Add("What are the collaboration capabilities of [entity]?")
            patterns.Add("How does [entity] handle version control?")
            patterns.Add("What are the document management capabilities of [entity]?")
            patterns.Add("What are the project management features of [entity]?")
            patterns.Add("How does [entity] handle workflow automation?")
            patterns.Add("What are the email integration options for [entity]?")
            patterns.Add("What are the social media integration capabilities of [entity]?")
            patterns.Add("How does [entity] handle customer support?")
            patterns.Add("What are the CRM (Customer Relationship Management) features of [entity]?")
            patterns.Add("What are the marketing automation capabilities of [entity]?")
            patterns.Add("How does [entity] handle data import?")
            patterns.Add("What are the inventory management features of [entity]?")
            patterns.Add("What are the sales forecasting capabilities of [entity]?")
            patterns.Add("How does [entity] handle order management?")
            patterns.Add("What are the HR (Human Resources) management features of [entity]?")
            patterns.Add("What are the accounting and finance capabilities of [entity]?")
            patterns.Add("How does [entity] handle customer data?")
            patterns.Add("What are the business intelligence features of [entity]?")

            ' Biology Questions
            patterns.Add("How does [organism] reproduce?")
            patterns.Add("What is the process of photosynthesis?")
            patterns.Add("What are the different parts of a cell?")
            patterns.Add("How does the digestive system work?")
            patterns.Add("What is DNA and how does it determine traits?")
            patterns.Add("How do plants make food?")
            patterns.Add("What is the role of the respiratory system?")
            patterns.Add("What are the stages of the life cycle?")
            patterns.Add("How does the circulatory system work?")
            patterns.Add("What are some examples of ecosystems?")
            patterns.Add("What is the classification of [animal/plant]?")
            patterns.Add("How does the nervous system function?")
            patterns.Add("What are the different types of animals?")
            patterns.Add("What is the importance of biodiversity?")
            patterns.Add("How do vaccines work?")
            patterns.Add("What is the process of evolution?")
            patterns.Add("How do plants and animals adapt to their environment?")
            patterns.Add("What are some examples of food chains?")
            patterns.Add("What is the role of DNA in heredity?")
            patterns.Add("How do different animals communicate?")

            ' Computer Science Questions
            patterns.Add("What is programming and how does it work?")
            patterns.Add("What are the different programming languages?")
            patterns.Add("How does a computer store and process information?")
            patterns.Add("What is an algorithm and how is it used?")
            patterns.Add("What are the basic concepts of coding?")
            patterns.Add("How do computers communicate with each other?")
            patterns.Add("What are the different types of computer networks?")
            patterns.Add("What is cybersecurity and why is it important?")
            patterns.Add("How do computers generate and display graphics?")
            patterns.Add("What is artificial intelligence and how is it used?")
            patterns.Add("What are the components of a computer system?")
            patterns.Add("How does data encryption work?")
            patterns.Add("What is the role of software in a computer?")
            patterns.Add("How are computer games developed?")
            patterns.Add("What is the Internet and how does it work?")
            patterns.Add("What is the difference between hardware and software?")
            patterns.Add("How do search engines like Google work?")
            patterns.Add("What are the different types of computer memory?")
            patterns.Add("What is cloud computing and how is it used?")
            patterns.Add("How do robots work and what are their applications?")

            ' NLP Questions
            patterns.Add("What is natural language processing?")
            patterns.Add("How do chatbots understand and respond to text?")
            patterns.Add("What are the applications of NLP in everyday life?")
            patterns.Add("How does voice recognition technology work?")
            patterns.Add("What is sentiment analysis and how is it used?")
            patterns.Add("What are the challenges in machine translation?")
            patterns.Add("How do spam filters detect and block unwanted emails?")
            patterns.Add("What is named entity recognition and why is it important?")
            patterns.Add("How does text summarization work?")
            patterns.Add("What are the steps involved in text classification?")
            patterns.Add("What is topic modeling and how is it used?")
            patterns.Add("How do search engines like Google understand user queries?")
            patterns.Add("What is part-of-speech tagging and why is it useful?")
            patterns.Add("How does sentiment analysis analyze emotions in text?")
            patterns.Add("What are the different techniques used in text generation?")
            patterns.Add("What is information extraction and how is it performed?")
            patterns.Add("How do language models like GPT-3 work?")
            patterns.Add("What are the ethical considerations in NLP?")
            patterns.Add("How can NLP be used for language translation?")
            patterns.Add("What are the limitations of current NLP systems?")

            patterns.Add("When did [organism] [action]?")
            patterns.Add("What is the [attribute] of [organism]?")
            patterns.Add("How does [organism] [action]?")
            patterns.Add("Who is [organism]'s [relation]?")
            patterns.Add("Where does [organism] [action]?")
            patterns.Add("Why did [organism] [action]?")
            patterns.Add("How many [organisms] are there?")
            patterns.Add("Which [organism] [action]?")

            patterns.Add("When did [concept] [action]?")
            patterns.Add("What is the [attribute] of [concept]?")
            patterns.Add("How does [concept] [action]?")
            patterns.Add("Who does [concept] [action]?")
            patterns.Add("Where does [concept] [action]?")
            patterns.Add("Why does [concept] [action]?")
            patterns.Add("How many [concepts] are there?")
            patterns.Add("Which [concept] [action]?")
            patterns.Add("When did [organism] [action]?")
            patterns.Add("What is the [attribute] of [organism]?")
            patterns.Add("How does [organism] [action]?")
            patterns.Add("Who is [organism]'s [relation]?")
            patterns.Add("Where does [organism] [action]?")
            patterns.Add("Why did [organism] [action]?")
            patterns.Add("How many [organisms] are there?")
            patterns.Add("Which [organism] [action]?")
            patterns.Add("When did [technology] [action]?")
            patterns.Add("What is the [attribute] of [technology]?")
            patterns.Add("How does [technology] [action]?")
            patterns.Add("Who uses [technology]?")
            patterns.Add("Where is [technology] used?")
            patterns.Add("Why is [technology] important?")
            patterns.Add("How many [technologies] are there?")
            patterns.Add("Which [technology] [action]?")
            ' Biology Questions
            patterns.Add("How does [organism] reproduce?")
            patterns.Add("What is the process of photosynthesis?")
            patterns.Add("What are the different parts of a cell?")
            patterns.Add("How does the digestive system work?")
            patterns.Add("What is DNA and how does it determine traits?")
            patterns.Add("How do plants make food?")
            patterns.Add("What is the role of the respiratory system?")
            patterns.Add("What are the stages of the life cycle?")
            patterns.Add("How does the circulatory system work?")
            patterns.Add("What are some examples of ecosystems?")
            patterns.Add("What is the classification of [animal/plant]?")
            patterns.Add("How does the nervous system function?")
            patterns.Add("What are the different types of animals?")
            patterns.Add("What is the importance of biodiversity?")
            patterns.Add("How do vaccines work?")
            patterns.Add("What is the process of evolution?")
            patterns.Add("How do plants and animals adapt to their environment?")
            patterns.Add("What are some examples of food chains?")
            patterns.Add("What is the role of DNA in heredity?")
            patterns.Add("How do different animals communicate?")

            ' Computer Science Questions
            patterns.Add("What is programming and how does it work?")
            patterns.Add("What are the different programming languages?")
            patterns.Add("How does a computer store and process information?")
            patterns.Add("What is an algorithm and how is it used?")
            patterns.Add("What are the basic concepts of coding?")
            patterns.Add("How do computers communicate with each other?")
            patterns.Add("What are the different types of computer networks?")
            patterns.Add("What is cybersecurity and why is it important?")
            patterns.Add("How do computers generate and display graphics?")
            patterns.Add("What is artificial intelligence and how is it used?")
            patterns.Add("What are the components of a computer system?")
            patterns.Add("How does data encryption work?")
            patterns.Add("What is the role of software in a computer?")
            patterns.Add("How are computer games developed?")
            patterns.Add("What is the Internet and how does it work?")
            patterns.Add("What is the difference between hardware and software?")
            patterns.Add("How do search engines like Google work?")
            patterns.Add("What are the different types of computer memory?")
            patterns.Add("What is cloud computing and how is it used?")
            patterns.Add("How do robots work and what are their applications?")

            ' NLP Questions
            patterns.Add("What is natural language processing?")
            patterns.Add("How do chatbots understand and respond to text?")
            patterns.Add("What are the applications of NLP in everyday life?")
            patterns.Add("How does voice recognition technology work?")
            patterns.Add("What is sentiment analysis and how is it used?")
            patterns.Add("What are the challenges in machine translation?")
            patterns.Add("How do spam filters detect and block unwanted emails?")
            patterns.Add("What is named entity recognition and why is it important?")
            patterns.Add("How does text summarization work?")
            patterns.Add("What are the steps involved in text classification?")
            patterns.Add("What is topic modeling and how is it used?")
            patterns.Add("How do search engines like Google understand user queries?")
            patterns.Add("What is part-of-speech tagging and why is it useful?")
            patterns.Add("How does sentiment analysis analyze emotions in text?")
            patterns.Add("What are the different techniques used in text generation?")
            patterns.Add("What is information extraction and how is it performed?")
            patterns.Add("How do language models like GPT-3 work?")
            patterns.Add("What are the ethical considerations in NLP?")
            patterns.Add("How can NLP be used for language translation?")
            patterns.Add("What are the limitations of current NLP systems?")

            patterns.Add("When did [entity] [action]?")
            patterns.Add("What is the [attribute] of [entity]?")
            patterns.Add("How does [entity] [action]?")
            patterns.Add("Who is [entity]'s [relation]?")
            patterns.Add("Where does [entity] [action]?")
            patterns.Add("Why did [entity] [action]?")
            patterns.Add("How many [entity]s are there?")
            patterns.Add("Which [entity] [action]?")

            patterns.Add("The formula for calculating [attribute] is [formula].")
            patterns.Add("[Entity] works by [explanation].")
            patterns.Add("The basic principles of [entity] are [principles].")
            patterns.Add("The difference between [entity1] and [entity2] is [difference].")
            patterns.Add("[Entity] affects our daily life by [impact].")
            patterns.Add("The importance of [entity] in [subject] is [importance].")
            patterns.Add("Some real-world applications of [entity] are [applications].")
            patterns.Add("Scientists study [entity] by [study method].")
            patterns.Add("Some interesting facts about [entity] are [facts].")
            patterns.Add("[Concept] can be explained in simple terms as [explanation].")
            patterns.Add("The different types of [entity] include [types].")
            patterns.Add("[Entity] helps us understand [concept] by [explanation].")
            patterns.Add("The importance of [concept] in [subject] is [importance].")
            patterns.Add("The steps involved in [process] are [steps].")
            patterns.Add("Some examples of [entity] in everyday life are [examples].")
            patterns.Add("The main components of [entity] are [components].")
            patterns.Add("[Entity] relates to [subject] by [relation].")
            patterns.Add("When [action] [entity], [result] occurs.")
            patterns.Add("The advantages of [entity] are [advantages], and the disadvantages are [disadvantages].")
            patterns.Add("To perform [action] with [entity], you can [method].")
            patterns.Add("The key features of [entity] include [features].")
            patterns.Add("The concept of [entity] can be explained using an analogy: [analogy].")
            patterns.Add("Some safety precautions when dealing with [entity] are [precautions].")
            patterns.Add("[Entity] can be compared to [similar entity] based on [comparison].")
            patterns.Add("The properties of [entity] include [properties].")
            patterns.Add("[Entity] interacts with [other entity] through [interaction].")
            patterns.Add("Studying [subject] is important because [reasons].")
            patterns.Add("The limitations of [entity] are [limitations].")
            patterns.Add("The practical uses of [concept] are [uses].")
            patterns.Add("[Entity] impacts the environment by [impact].")
            patterns.Add("The concept of [entity] can be explained through an experiment: [experiment].")
            patterns.Add("Future advancements in [subject] may include [advancements].")
            patterns.Add("[Entity] contributes to sustainable development by [contribution].")
            patterns.Add("Ethical considerations related to [entity] include [considerations].")
            patterns.Add("To solve [problem] using [entity], you can [solution].")
            patterns.Add("The challenges faced in studying [subject] include [challenges].")
            patterns.Add("[Entity] supports scientific research by [support].")
            patterns.Add("Some key discoveries or inventions related to [subject] are [discoveries].")
            patterns.Add("We can use [entity] to improve our daily lives by [improvement].")
            patterns.Add("The different branches or fields within [subject] are [branches].")
            patterns.Add("Common misconceptions about [entity] include [misconceptions].")
            patterns.Add("[Entity] impacts the economy by [impact].")
            patterns.Add("A step-by-step explanation of [process] is as follows: [explanation].")
            patterns.Add("The implications of [entity] on society include [implications].")
            patterns.Add("[Entity] contributes to technological advancements by [contribution].")
            patterns.Add("Current research areas in [subject] include [research areas].")
            patterns.Add("The concept of [entity] can be visualized using [visual aids].")
            patterns.Add("Career opportunities in [subject] include [careers].")

            ' Biology Answers
            patterns.Add("[Organism] reproduces through [reproductive process].")
            patterns.Add("Photosynthesis is the process by which [plants] convert sunlight into [energy].")
            patterns.Add("The different parts of a cell include [cell components].")
            patterns.Add("The digestive system works by [digestive process].")
            patterns.Add("DNA is a molecule that contains genetic information and determines [traits].")
            patterns.Add("Plants make food through a process called [photosynthesis].")
            patterns.Add("The respiratory system is responsible for [respiratory function].")
            patterns.Add("The stages of the life cycle are [life cycle stages].")
            patterns.Add("The circulatory system works by [circulatory process].")
            patterns.Add("Examples of ecosystems include [ecosystem examples].")
            patterns.Add("The classification of [animal/plant] is [classification].")
            patterns.Add("The nervous system functions by [nervous system process].")
            patterns.Add("There are different types of animals such as [animal types].")
            patterns.Add("Biodiversity is important for [reasons].")
            patterns.Add("Vaccines work by [vaccine mechanism].")
            patterns.Add("Evolution is the process of [evolution process].")
            patterns.Add("Plants and animals adapt to their environment through [adaptation process].")
            patterns.Add("Examples of food chains include [food chain examples].")
            patterns.Add("DNA plays a role in heredity by [heredity mechanism].")
            patterns.Add("Different animals communicate through [animal communication methods].")

            ' Computer Science Answers
            patterns.Add("Programming is the process of creating instructions for a computer to follow.")
            patterns.Add("There are different programming languages such as [programming languages].")
            patterns.Add("Computers store and process information using [computer mechanism].")
            patterns.Add("An algorithm is a step-by-step set of instructions for solving a problem.")
            patterns.Add("Coding involves using programming languages to write instructions for computers.")
            patterns.Add("Computers communicate with each other through [communication methods].")
            patterns.Add("There are different types of computer networks such as [network types].")
            patterns.Add("Cybersecurity is important to protect computer systems from [security threats].")
            patterns.Add("Computers generate and display graphics using [graphics technology].")
            patterns.Add("Artificial intelligence is used to create intelligent computer systems.")
            patterns.Add("The components of a computer system include [computer components].")
            patterns.Add("Data encryption ensures that data is secure and cannot be accessed by unauthorized parties.")
            patterns.Add("Software is a collection of programs that perform specific tasks on a computer.")
            patterns.Add("Computer games are developed using [game development techniques].")
            patterns.Add("The Internet is a global network of computers that are connected and can share information.")
            patterns.Add("Hardware refers to the physical components of a computer, while software refers to programs and data.")
            patterns.Add("Search engines like Google use algorithms to index and retrieve information from the web.")
            patterns.Add("Computer memory includes [memory types] used to store data and instructions.")
            patterns.Add("Cloud computing allows users to access and store data and applications on remote servers.")
            patterns.Add("Robots work through a combination of [robotics technology] and are used in various applications.")

            ' NLP Answers
            patterns.Add("Natural language processing is a field of study that focuses on how computers understand and generate human language.")
            patterns.Add("Chatbots understand and respond to text by analyzing the input and selecting appropriate responses.")
            patterns.Add("NLP has applications in everyday life such as [NLP applications].")
            patterns.Add("Voice recognition technology converts spoken language into written text.")
            patterns.Add("Sentiment analysis is used to analyze and understand emotions expressed in text.")
            patterns.Add("Machine translation involves translating text from one language to another using computer algorithms.")
            patterns.Add("Spam filters detect and block unwanted emails based on [spam detection techniques].")
            patterns.Add("Named entity recognition is the process of identifying and classifying named entities in text.")
            patterns.Add("Text summarization involves condensing a piece of text into a shorter version while retaining the main points.")
            patterns.Add("Text classification is the task of assigning predefined categories or labels to text based on its content.")
            patterns.Add("Topic modeling is a technique used to discover abstract topics in a collection of documents.")
            patterns.Add("Search engines like Google understand user queries by analyzing the keywords and context.")
            patterns.Add("Part-of-speech tagging assigns grammatical tags to words in a sentence to understand their roles.")
            patterns.Add("Text generation techniques include [text generation methods] used to create human-like text.")
            patterns.Add("Information extraction involves identifying and extracting structured information from unstructured text.")
            patterns.Add("Language models like GPT-3 use deep learning algorithms to generate coherent and contextually relevant text.")
            patterns.Add("Ethical considerations in NLP include [NLP ethical considerations] to ensure fair and unbiased use of the technology.")
            patterns.Add("NLP can be used for language translation by applying machine learning techniques to analyze and generate text.")
            patterns.Add("Current NLP systems have limitations in [NLP limitations].")

            patterns.Add("[Organism] [action] at [time].")
            patterns.Add("The [attribute] of [organism] is [value].")
            patterns.Add("[Organism] [action] by [method].")
            patterns.Add("[Organism]'s [relation] is [person].")
            patterns.Add("[Organism] [action] in [location].")
            patterns.Add("[Organism] [action] because [reason].")
            patterns.Add("There are [number] [organisms].")
            patterns.Add("[Organism] [action] [criteria].")
            patterns.Add("[Organism] [action] at [time].")
            patterns.Add("The [attribute] of [organism] is [value].")
            patterns.Add("[Organism] [action] by [method].")
            patterns.Add("[Organism]'s [relation] is [person].")
            patterns.Add("[Organism] [action] in [location].")
            patterns.Add("[Organism] [action] because [reason].")
            patterns.Add("There are [number] [organisms].")
            patterns.Add("[Organism] [action] [criteria].")

            patterns.Add("[Concept] [action] at [time].")
            patterns.Add("The [attribute] of [concept] is [value].")
            patterns.Add("[Concept] [action] by [method].")
            patterns.Add("[Concept] is used by [person].")
            patterns.Add("[Concept] [action] in [location].")
            patterns.Add("[Concept] [action] because [reason].")
            patterns.Add("There are [number] [concepts].")
            patterns.Add("[Concept] [action] [criteria].")

            patterns.Add("[Technology] [action] at [time].")
            patterns.Add("The [attribute] of [technology] is [value].")
            patterns.Add("[Technology] [action] by [method].")
            patterns.Add("[Person] uses [technology].")
            patterns.Add("[Technology] is used in [location].")
            patterns.Add("[Technology] is important because [reason].")
            patterns.Add("There are [number] [technologies].")
            patterns.Add("[Technology] [action] [criteria].")

            ' Biology Answers
            patterns.Add("[Organism] reproduces through [reproductive process].")
            patterns.Add("Photosynthesis is the process by which [plants] convert sunlight into [energy].")
            patterns.Add("The different parts of a cell include [cell components].")
            patterns.Add("The digestive system works by [digestive process].")
            patterns.Add("DNA is a molecule that contains genetic information and determines [traits].")
            patterns.Add("Plants make food through a process called [photosynthesis].")
            patterns.Add("The respiratory system is responsible for [respiratory function].")
            patterns.Add("The stages of the life cycle are [life cycle stages].")
            patterns.Add("The circulatory system works by [circulatory process].")
            patterns.Add("Examples of ecosystems include [ecosystem examples].")
            patterns.Add("The classification of [animal/plant] is [classification].")
            patterns.Add("The nervous system functions by [nervous system process].")
            patterns.Add("There are different types of animals such as [animal types].")
            patterns.Add("Biodiversity is important for [reasons].")
            patterns.Add("Vaccines work by [vaccine mechanism].")
            patterns.Add("Evolution is the process of [evolution process].")
            patterns.Add("Plants and animals adapt to their environment through [adaptation process].")
            patterns.Add("Examples of food chains include [food chain examples].")
            patterns.Add("DNA plays a role in heredity by [heredity mechanism].")
            patterns.Add("Different animals communicate through [animal communication methods].")

            ' Computer Science Answers
            patterns.Add("Programming is the process of creating instructions for a computer to follow.")
            patterns.Add("There are different programming languages such as [programming languages].")
            patterns.Add("Computers store and process information using [computer mechanism].")
            patterns.Add("An algorithm is a step-by-step set of instructions for solving a problem.")
            patterns.Add("Coding involves using programming languages to write instructions for computers.")
            patterns.Add("Computers communicate with each other through [communication methods].")
            patterns.Add("There are different types of computer networks such as [network types].")
            patterns.Add("Cybersecurity is important to protect computer systems from [security threats].")
            patterns.Add("Computers generate and display graphics using [graphics technology].")
            patterns.Add("Artificial intelligence is used to create intelligent computer systems.")
            patterns.Add("The components of a computer system include [computer components].")
            patterns.Add("Data encryption ensures that data is secure and cannot be accessed by unauthorized parties.")
            patterns.Add("Software is a collection of programs that perform specific tasks on a computer.")
            patterns.Add("Computer games are developed using [game development techniques].")
            patterns.Add("The Internet is a global network of computers that are connected and can share information.")
            patterns.Add("Hardware refers to the physical components of a computer, while software refers to programs and data.")
            patterns.Add("Search engines like Google use algorithms to index and retrieve information from the web.")
            patterns.Add("Computer memory includes [memory types] used to store data and instructions.")
            patterns.Add("Cloud computing allows users to access and store data and applications on remote servers.")
            patterns.Add("Robots work through a combination of [robotics technology] and are used in various applications.")

            ' NLP Answers
            patterns.Add("Natural language processing is a field of study that focuses on how computers understand and generate human language.")
            patterns.Add("Chatbots understand and respond to text by analyzing the input and selecting appropriate responses.")
            patterns.Add("NLP has applications in everyday life such as [NLP applications].")
            patterns.Add("Voice recognition technology converts spoken language into written text.")
            patterns.Add("Sentiment analysis is used to analyze and understand emotions expressed in text.")
            patterns.Add("Machine translation involves translating text from one language to another using computer algorithms.")
            patterns.Add("Spam filters detect and block unwanted emails based on [spam detection techniques].")
            patterns.Add("Named entity recognition is the process of identifying and classifying named entities in text.")
            patterns.Add("Text summarization involves condensing a piece of text into a shorter version while retaining the main points.")
            patterns.Add("Text classification is the task of assigning predefined categories or labels to text based on its content.")
            patterns.Add("Topic modeling is a technique used to discover abstract topics in a collection of documents.")
            patterns.Add("Search engines like Google understand user queries by analyzing the keywords and context.")
            patterns.Add("Part-of-speech tagging assigns grammatical tags to words in a sentence to understand their roles.")
            patterns.Add("Text generation techniques include [text generation methods] used to create human-like text.")
            patterns.Add("Information extraction involves identifying and extracting structured information from unstructured text.")
            patterns.Add("Language models like GPT-3 use deep learning algorithms to generate coherent and contextually relevant text.")
            patterns.Add("Ethical considerations in NLP include [NLP ethical considerations] to ensure fair and unbiased use of the technology.")
            patterns.Add("NLP can be used for language translation by applying machine learning techniques to analyze and generate text.")
            patterns.Add("Current NLP systems have limitations in [NLP limitations].")

            patterns.Add("[Entity] [action] at [time].")
            patterns.Add("The [attribute] of [entity] is [value].")
            patterns.Add("[Entity] [action] by [method].")
            patterns.Add("[Entity]'s [relation] is [person].")
            patterns.Add("[Entity] [action] in [location].")
            patterns.Add("[Entity] [action] because [reason].")
            patterns.Add("There are [number] [entity]s.")
            patterns.Add("[Entity] [action] [criteria].")

            Return patterns
        End Function
        ''' <summary>
        ''' When no lists are available, A mixed corpus of documents 
        ''' </summary>
        ''' <returns></returns>
        Public Shared Function GetDocumentCorpus() As List(Of String)
            ' Load paragraphs based on different topics
            Dim paragraphs As New List(Of String)

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

            Return paragraphs
        End Function
    End Class
    ''' <summary>
    ''' Used to Extract Random Batches of data 
    ''' Given a list of strings, 
    ''' The Tokens are  given a simple index, 
    ''' based on a Stored/Supplied vocabulary, or,
    ''' it will create a vocabulary based on the current document. 
    ''' This Vocabulary is a simple text List so its index is its ID_
    ''' Your personal vocabulary should maintain 
    ''' its ordering to Maintain decoding , 
    ''' This vocabulary file should also be used for encoding, 
    ''' Making it a universal Vocab list
    ''' </summary>
    Public Class DataExtraction
        ''' <summary>
        ''' The maximum sequence length for padding or truncating sequences.
        ''' </summary>
        Public Shared maxSequenceLength As Integer = 8

        ''' <summary>
        ''' The vocabulary used for encoding and decoding.
        ''' </summary>
        Public Shared TestVocab As New List(Of String)
        ''' <summary>
        ''' Initializes a new instance of the DataExtraction class with the specified maximum sequence length.
        ''' </summary>
        ''' <param name="MaxSeqLength">The maximum sequence length for padding or truncating sequences.</param>
        Public Sub New(ByRef MaxSeqLength As Integer)
            TestVocab = New List(Of String)
            maxSequenceLength = MaxSeqLength
        End Sub

        ''' <summary>
        ''' Initializes a new instance of the DataExtraction class.
        ''' </summary>
        Public Sub New()
            TestVocab = New List(Of String)
        End Sub

        ''' <summary>
        ''' Initializes a new instance of the DataExtraction class with the specified vocabulary.
        ''' </summary>
        ''' <param name="Vocab">The vocabulary used for encoding and decoding.</param>
        Public Sub New(ByRef Vocab As List(Of String))
            TestVocab = Vocab
        End Sub

        ''' <summary>
        ''' Initializes a new instance of the DataExtraction class with the specified vocabulary and maximum sequence length.
        ''' </summary>
        ''' <param name="Vocab">The vocabulary used for encoding and decoding.</param>
        ''' <param name="MaxSeqLength">The maximum sequence length for padding or truncating sequences.</param>
        Public Sub New(ByRef Vocab As List(Of String), ByRef MaxSeqLength As Integer)
            TestVocab = Vocab
            maxSequenceLength = MaxSeqLength
        End Sub
        Public Shared Function ComputeLoss(predictions As List(Of List(Of Double)), targets As List(Of List(Of Integer))) As Double
            Dim loss As Double = 0

            For i As Integer = 0 To predictions.Count - 1
                For j As Integer = 0 To predictions(i).Count - 1
                    loss += -Math.Log(predictions(i)(j)) * targets(i)(j)
                Next
            Next

            Return loss
        End Function

        ''' <summary>
        ''' Creates batches of data for training.
        ''' </summary>
        ''' <param name="trainingData">The training data as a list of string sequences.</param>
        ''' <param name="batchSize">The size of each batch.</param>
        Public Sub CreateData(ByRef trainingData As List(Of List(Of String)), ByRef batchSize As Integer)
            For batchStart As Integer = 0 To trainingData.Count - 1 Step batchSize
                Dim batchEnd As Integer = Math.Min(batchStart + batchSize - 1, trainingData.Count - 1)
                Dim batchInputs As List(Of List(Of Integer)) = GetBatchInputs(trainingData, batchStart, batchEnd)
                Dim batchTargets As List(Of List(Of Integer)) = GetBatchTargets(trainingData, batchStart, batchEnd)

                ' Perform further operations on the batches
            Next

            ' Compute loss
            ' Dim loss As Double = ComputeLoss(predictions, batchTargets)
        End Sub

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
                sequence.AddRange(Enumerable.Repeat(TestVocab.IndexOf("PAD"), length - sequence.Count))
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
            Dim shiftedSequence As New List(Of Integer) From {TestVocab.IndexOf("START")}

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
                If TestVocab.Contains(word) Then
                    indices.Add(TestVocab.IndexOf(word))
                Else
                    indices.Add(TestVocab.IndexOf("UNK")) ' Unknown word
                End If
            Next

            Return indices
        End Function
    End Class
    Public Class TrainingData
        Public Shared Function GenerateRandomTrainingData(ByRef Seeds As List(Of String)) As List(Of String)
            Dim trainingData As New List(Of String)()
            Dim random As New Random()

            ' List of entities
            Dim entities As New List(Of String)() From {
    "John", "Mary", "Susan", "The cat", "The book",
    "The president", "car", "book", "chair", "laptop",
    "Alice", "Bob", "The dog", "The car", "home", "office", "park", "store",
    "where is", "What is a", "where is "
}
            entities.AddRange(Seeds)
            ' List of pronouns
            Dim pronouns As New List(Of String)() From {
    "he", "she", "it", "they", "his", "her", "its", "their"
}

            ' Generate 100 random training examples
            For i = 1 To 100
                Dim sb As New StringBuilder()

                ' Generate a sentence with co-reference
                Dim numEntities = random.Next(2, 4) ' Randomly select 2 to 3 entities in a sentence
                Dim entityIndices = Enumerable.Range(0, entities.Count).OrderBy(Function(x) random.Next()).Take(numEntities)
                Dim pronounIndex = random.Next(numEntities)

                For j = 0 To numEntities - 1
                    Dim entity = entities(entityIndices(j))
                    sb.Append(entity)

                    If j = pronounIndex Then
                        sb.Append(" ").Append(pronouns(random.Next(pronouns.Count)))
                    End If

                    sb.Append(" ")
                Next

                trainingData.Add(sb.ToString().Trim())
            Next

            Return trainingData
        End Function
        Public Shared Function iGetTrainingSentences() As List(Of String)
            Dim documents As New List(Of String)()
            documents.AddRange(iGetTrainingQuestionSentences)
            ' Generate random documents related to NLP, machine learning, and data science
            documents.Add("If a sentence contains a noun and a verb, then it is grammatically correct.")
            documents.Add("In NLP, language models are often used to generate text based on input prompts.")
            documents.Add("Every time a new document is added to the dataset, the machine learning model is retrained.")
            documents.Add("If the temperature exceeds 30 degrees Celsius, the ice cream will melt.")
            documents.Add("Similar to image classification, text classification involves assigning labels to textual data.")
            documents.Add("Because of the high correlation between study time and test scores, it can be concluded that studying more leads to better performance.")
            documents.Add("In fact, the sun is a star.")
            documents.Add("Based on statistical analysis, there is a strong correlation between feature X and the target variable.")
            documents.Add("If the input data is preprocessed properly, the accuracy of the machine learning model improves.")
            documents.Add("The use of deep learning algorithms has led to significant advancements in natural language processing.")
            documents.Add("If a hypothesis is rejected based on the p-value, then the null hypothesis is accepted.")
            documents.Add("Analogous to image classification, text classification involves assigning labels to textual data.")
            documents.Add("By analyzing large volumes of data, data scientists can uncover valuable insights and patterns.")
            documents.Add("If all mammals are warm-blooded, and dogs are mammals, then dogs are warm-blooded.")
            documents.Add("Every time it rains, the streets get wet. It will rain today.")
            documents.Add("If it's raining, then the ground is wet. The ground is not wet.")
            documents.Add("If you study hard, you will get good grades. You studied hard.")
            documents.Add("Eating a healthy diet improves overall health. If you eat a healthy diet, your energy levels will increase.")
            documents.Add("An animal is a mammal if and only if it gives birth to live young. The platypus gives birth to live young.")
            documents.Add("All roses are flowers. The red object is a rose.")
            documents.Add("If it were not raining, the ground would not be wet. The ground is wet.")
            documents.Add("In a survey of 1,000 participants, 80% preferred tea over coffee. If a random person is chosen from the population, they are likely to prefer tea over coffee.")
            documents.Add("In the past, when the price of oil increased, the stock market declined. If the price of oil increases, the stock market will decline.")
            documents.Add("For a hypothesis to be accepted, it must be supported by evidence.")
            documents.Add("In general, cats are independent animals.")
            documents.Add("If the temperature exceeds 30 degrees Celsius, the ice cream will melt.")
            documents.Add("Similar to image classification, text classification involves assigning labels to textual data.")
            documents.Add("Because of the high correlation between study time and test scores, it can be concluded that studying more leads to better performance.")
            documents.Add("In fact, the sun is a star.")
            Return documents
        End Function
        Public Shared Function iGetTrainingQuestionSentences() As List(Of String)
            Dim trainingSentences As New List(Of String)

            ' Independent Question
            trainingSentences.Add("What is the capital of France?")
            trainingSentences.Add("Who won the World Cup in 2018?")
            trainingSentences.Add("Where can I find a good restaurant?")
            trainingSentences.Add("When is the next train arriving?")
            trainingSentences.Add("Why do birds fly?")
            trainingSentences.Add("How do I bake a cake?")

            ' Dependent Question
            trainingSentences.Add("If it rains, will the picnic be canceled?")
            trainingSentences.Add("Unless you study, you won't pass the exam.")
            trainingSentences.Add("Whether it's sunny or cloudy, we'll go to the beach.")
            trainingSentences.Add("In case of an emergency, call 911.")

            ' Comparative Question
            trainingSentences.Add("Which one is better, chocolate or vanilla?")
            trainingSentences.Add("What is the difference between a cat and a dog?")
            trainingSentences.Add("How does the price of gasoline compare to last year?")

            ' Cause-Effect Question
            trainingSentences.Add("Why does exercise improve cardiovascular health?")
            trainingSentences.Add("How does stress affect sleep quality?")
            trainingSentences.Add("What causes earthquakes?")

            ' Descriptive Question
            trainingSentences.Add("Describe the main features of a smartphone.")
            trainingSentences.Add("Explain the process of photosynthesis.")
            trainingSentences.Add("Tell me about the history of the Roman Empire.")

            ' Hypothetical Question
            trainingSentences.Add("What if humans could fly?")
            trainingSentences.Add("Imagine a world without internet.")
            trainingSentences.Add("Suppose you won the lottery, what would you do?")
            trainingSentences.Add("Assume you have unlimited resources, how would you solve world hunger?")

            Return trainingSentences
        End Function

    End Class
End Namespace
