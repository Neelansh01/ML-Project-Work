# DOCUMENTATION

1. See the Dataset Analysis in the folder "DataSet Analysis". In the presentation we will get to see the subsample of the dataset, it's features and size, and it's PCA representation.

2. See the python notebooks in "Notebooks" folder to see the the steps for comparing Classification models for respective embedding models as depicted with the file name.

3. See the folder "Python_Files". This has:
    a. Classification Model Class: To import this class, we can use the syntax as used in the notebooks-
    
        Example:- from Python_Files.Classification_Models import ClassificationModel
                  _logistic_regression = ClassificationModel(model_name="LR", parameter_dict={'penalty':'l2'})
                  
                  The Inputs that can be given to this class are as mentioned below. We can pass any parameter dict using parameter_dict attribute as shown above.
                  1> Naive Bayes: model = ClassificationModel(model_name="NB", parameter_dict={}) // model name can be "NB" or "Naive Bayes".
                  2> Logistic Regression: model = ClassificationModel(model_name="LR", parameter_dict={'penalty':'l2'}) // model name can be "LR" or "Logistic Regression".
                  3> Multi Layer Perceptron: model = ClassificationModel(model_name="ANN", parameter_dict={}) // model name can be "ANN" or "Multi Layer Perceptron".
                  4> K Nearest Neighbour: model = ClassificationModel(model_name="KNN", parameter_dict={}) // model name can be "KNN" or "K Nearest Neighbour".
                  5> Support Vector Classifier: model = ClassificationModel(model_name="SVC", parameter_dict={}) // model name can be "SVC" or "Support Vector Machine".
                  6> Decision Tree = model = ClassificationModel(model_name="DT", parameter_dict={}) // model name can be "DT" or "Decision Tree".
                  7> Gradient Boosted Decision Tree: model = ClassificationModel(model_name="GBDT", parameter_dict={}) // model name can be "GBDT" or "Gradient Boosted Decision Tree".
                  
                  
    b. Embedding Model Class: To import this class, we can use the syntax as used in the notebooks-
        Example:- from Python_Files.Embedding_Models import PreTrained_EmbeddingModels
                  model_instance_bert = PreTrained_EmbeddingModels(model_url="sentence-transformers/bert-base-nli-mean-tokens", model_name="BERT")._get_model()
                  
                  The Inputs that can be given to this class are as mentioned below. We can pass any paramete dict using parameter_dict attribute as shown above.
                  1> BERT: model_instance_bert = PreTrained_EmbeddingModels(model_url="sentence-transformers/bert-base-nli-mean-tokens", model_name="BERT")._get_model()
                  2> SBERT: model_instance_sbert = PreTrained_EmbeddingModels(model_url="all-MiniLM-L6-v2", model_name="Sentence BERT")._get_model()
                  3> MPNET: model_instance_mpnet = PreTrained_EmbeddingModels(model_url="sentence-transformers/all-mpnet-base-v2", model_name="MPNET")._get_model()
                  
                  
