

# Bayesian Network Library for VB.NET

This is an open-source project that provides a library for creating and working with Bayesian networks in VB.NET. Bayesian networks are probabilistic graphical models that represent dependencies between variables using directed acyclic graphs. This library enables you to define nodes, states, conditional probability tables (CPTs), perform probabilistic inference, and visualize the network structure.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Training Data Format](#training-data-format)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Bayesian networks are powerful tools for probabilistic reasoning and decision-making under uncertainty. This library provides a user-friendly interface to create, manipulate, and analyze Bayesian networks. Whether you're interested in predicting outcomes based on probabilistic relationships or exploring conditional dependencies, this library can assist you in building and using Bayesian networks efficiently.

## Features

- Define nodes with associated states and parent nodes
- Create conditional probability tables (CPTs) to represent probabilistic relationships
- Perform probabilistic inference to calculate conditional probabilities
- Display network structure using a tree-like visualization
- Load training data from files to define CPTs
- Export network structure to text files for sharing and documentation

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/bayesian-network-vbnet.git
```

Open the solution in your preferred VB.NET development environment.

## Usage

1. Define nodes: Create instances of the `Node` class, specifying the node name and possible states.
2. Establish relationships: Add parent nodes to each node using the `AddEdge` method.
3. Set CPTs: Define conditional probability tables using the `DefineCPT` method.
4. Perform inference: Use the `InferenceEngine` class to calculate conditional probabilities and predictions.
5. Visualize: Display the network structure using the `DisplayAsTree` method.

Refer to the example code and documentation for detailed usage instructions.

## Example

```vb.net
' Create a BeliefNetwork instance
Dim network As New BeliefNetwork()

' Define nodes and relationships
Dim nodeA As New Node("A", {"True", "False"})
Dim nodeB As New Node("B", {"Yes", "No"})
network.AddNode(nodeA)
network.AddNode(nodeB)
network.AddEdge(nodeA, nodeB)

' Define CPTs
Dim cptValues As New Dictionary(Of List(Of String), Double)()
cptValues.Add(New List(Of String) From {"True"}, 0.7)
cptValues.Add(New List(Of String) From {"False"}, 0.3)
network.DefineCPT("A", cptValues)

' Perform inference and prediction
Dim evidence As Dictionary(Of Node, String) = network.CreateEvidence("A", "True")
Dim prediction As String = network.PredictWithEvidence("B", evidence)

Console.WriteLine("Prediction: " & prediction)
```

## Training Data Format

Training data should be provided in a text file, where each line represents a node and its associated conditional probabilities. The format is as follows:

```
NodeName
State Probability
State1 State2 ... StateN Probability
Parent1State Parent2State ... ParentNState State Probability
...
```

For example:

```
Weather
Sunny 0.4
Cloudy 0.3
Rainy 0.3

OutdoorActivity
Yes Sunny 0.8
Yes Cloudy 0.6
Yes Rainy 0.2
No Sunny 0.2
No Cloudy 0.4
No Rainy 0.8

```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


