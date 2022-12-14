# TravelingSalesman
# TSP problem solving with GNN and Attention Mechanism
 
 The Traveling Salesperson Problem (TSP) is one of the most popular NP-hard combinatorial problems in the theoretical computer science and operations research (OR) community. It asks the following question: “Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?”

TSP has many important applications, such as in logistics, planning, and scheduling. The problem has been studied for decades, and many traditional optimization algorithms have been proposed to solve it, such as dynamic programming and branch-and-bound. Although these optimization algorithms are capable of solving TSP with dozens of nodes, it is usually intractable to use these algorithms to solve optimally above thousands of nodes on modern computers due to their exponential execution times.

In this article, we’ll show how to train, deploy, and make inferences using deep learning to solve the Traveling Salesperson Problem.

# # Overview

Recently, deep learning-based algorithms, such as graph neural networks (GNNs) along with reinforcement learning (RL), have been proposed to solve TSP. The advantages of the deep reinforcement learning algorithms are:

Training a model from large synthetically (randomly) generated TSP instances.
Generalization to new problems with different number of nodes.
Quick inference time, relative to traditional optimization methods.

![song_traveling-salesperson_f1-learning_tsp_decoder](https://d2908q01vomqb2.cloudfront.net/ca3512f4dfa95a03169c5a670a4c91a19b3077b4/2022/05/23/song_traveling-salesperson_f3-routing-blog-demo-2-low.gif)

We are interested in applying open source TSP deep reinforcement learning algorithms in solving practical problems. In particular, we found the following capabilities important to successfully deploy RL-based solutions for supply chain operations:

-The ability to run model training distributed across multiple GPUs.

-The ability to host the model and provide routing in real time.

-The ability to interactively visualize the TSP solution to customers in order to receive feedback quickly.

-How to use Amazon SageMaker’s Distributed Data Parallel Library to train an open source deep learning-based TSP model across multiple GPUs.

-How to deploy the model to Amazon SageMaker inference endpoint. We also demonstrate how to perform batch inference.

-How to build a Streamlit interactive TSP demo using the SageMaker endpoint.


### Deep reinforcement learning TSP modeling

There are many options for designing the deep learning architecture for solving the Traveling Salesperson Problem. For this blog post, we will use a GNN to encode input nodes into dense feature vectors and use the Attention mechanism as a decoder to generate the ordered nodes in an autoregressive fashion.

The idea behind a GNN is that stops included in the route can be thought of as nodes on a graph. The edge representation can be thought of as a measure of distance between the stops or whether stops exist within the predefined neighborhood region.
The idea behind the Attention mechanism is to decode the routes in an autoregressive fashion by calculating the logit “attention” scores between nodes on the partial tour and the input nodes embedding.

![song_traveling-salesperson_f1-learning_tsp_decoder](https://user-images.githubusercontent.com/33089347/184277823-b46f0c4a-ac68-4d98-81cc-50b233b400b4.png)

Figure 1 shows the encoding and decoding processes based on Chaitanya Joshi’s repository. The dark gray rectangles during encoding represent various “fixed” embeddings projected from the original input—a set of nodes (Euclidean coordinates) and the nearest neighbor (NN) graph. The evolving state (the white box, bottom right), together with the fixed Node Embedding, is continuously projected onto the embedding known as Step Context (white rectangle) as the decoding process unfolds at a given step.

The State contains Node Embeddings of the first and last selected nodes in the partial tour. In this example, the first and fourth nodes’ Node Embeddings are included in the State. Fixed Context and Step Context are summed together to form the Query. Cross-product is performed between the Query and the Glimpse Key to form the Attention Score, which projects the Glimpse Values to the Logit Query.

Finally, another cross-product is performed between the Logit Key and the Logit Query to produce the final Logit Attention Score. This becomes the probability distribution from which the next unvisited node is drawn to join the partial tour.

The training approach utilizes the Reinforcement Learning Policy Gradient method called REINFORCE. The model training mechanics will look similar to the traditional Supervised Learning paradigm; however, there are a few key differences:

Rather than calculating the loss in relationship to ground truth labels, the training tries to minimize the total tour length.

-Introducing a baseline learner to achieve faster convergence of the parameters during training. The baseline does not need to be created before training but can be incrementally updated during the training process itself.
-Rather than making “predictions” for each route, the model is actually creating a policy (that is, a sequence of consecutive decisions) that recommends which node to connect next given the partially formed tour and the set of unvisited nodes.


