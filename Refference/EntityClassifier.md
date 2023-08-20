# Entity Classifier

Entity Classifier is a .NET library for entity recognition, relationship discovery, and text transformation. It allows you to classify sentences, extract entities and their relationships, and replace entities with entity labels.



## Introduction

Entity Classifier is a powerful tool for natural language processing tasks that involve entity recognition and relationship extraction. It provides a flexible and rule-based approach to discover entities, detect relationships between them, and transform text by replacing entities with labels.

## Features

- Entity recognition: Detects entities from a list of predefined entity types.
- Relationship discovery: Finds relationships between entities based on user-defined patterns/rules.
- Text transformation: Replaces detected entities with their corresponding entity labels.
- Supports multiple entity types: Handles various types of entities and their respective entity lists.

## Usage

To use Entity Classifier in your .NET project, follow these steps:

1. Install the EntityClassifier NuGet package (See Installation section).
2. Create an instance of the `EntityClassifier` class by providing entity lists, entity types, and relationship patterns.
3. Call the `Classify` method to classify sentences and extract entities with their relationships.
4. Use the extracted information as needed in your application.

## Installation

To install the EntityClassifier NuGet package, run the following command in the Package Manager Console:


Install-Package EntityClassifier
## c#
 ```
using Recognition.Classifer;

// Define entity lists and relationship patterns
var entityList = new List<string> { "car", "driver", "manufacturer" };
var entityType = "Vehicle";
var relations = new List<string> { "drives", "is manufactured by" };

// Create an instance of the EntityClassifier
var classifier = new EntityClassifier(entityList, entityType, relations);

// Classify a sample sentence and extract entities
var sentence = "The driver drives a car that is manufactured by a company.";
var discoveredEntities = classifier.Classify(sentence);

// Use the extracted entities and relationships in your application
foreach (var entity in discoveredEntities)
{
    Console.WriteLine($"Entity: {entity.DiscoveredEntitys}, Type: {entity.EntityType}");
    foreach (var relationship in entity.Relationships)
    {
        Console.WriteLine($"Relationship: {relationship.SourceEntity} {relationship.RelationshipType} {relationship.TargetEntity}");
    }
}
```

## VB.NET
```
Imports Recognition.Classifer

Module Program
    Sub Main()
        ' Define entity lists and relationship patterns
        Dim entityList As New List(Of String) From {"car", "driver", "manufacturer"}
        Dim entityType As String = "Vehicle"
        Dim relations As New List(Of String) From {"drives", "is manufactured by"}

        ' Create an instance of the EntityClassifier
        Dim classifier As New EntityClassifier(entityList, entityType, relations)

        ' Classify a sample sentence and extract entities
        Dim sentence As String = "The driver drives a car that is manufactured by a company."
        Dim discoveredEntities As List(Of DiscoveredEntity) = classifier.Classify(sentence)

        ' Use the extracted entities and relationships in your application
        For Each entity As DiscoveredEntity In discoveredEntities
            Console.WriteLine($"Entity: {entity.DiscoveredEntitys}, Type: {entity.EntityType}")
            For Each relationship As DiscoveredEntity.EntityRelationship In entity.Relationships
                Console.WriteLine($"Relationship: {relationship.SourceEntity} {relationship.RelationshipType} {relationship.TargetEntity}")
            Next
        Next
    End Sub
End Module
```

For more examples and detailed usage, refer to the documentation.

## Contributing
Contributions are welcome! If you find any issues, have feature requests, or want to improve the library, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

