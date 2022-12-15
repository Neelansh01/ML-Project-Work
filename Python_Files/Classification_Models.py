class ClassificationModel:
    def __init__(self, model_name=None, parameter_dict=None):
        self.model_name = model_name
        self.parameter_dict = parameter_dict
        self.model = self.load_model()
        
    def load_model(self):            
        if self.model_name in ["Naive Bayes", "NB"]:
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(**self.parameter_dict)

        elif self.model_name in ["Support Vector Machine", "SVC"]:
            from sklearn.svm import SVC
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(), SVC(**self.parameter_dict))

        elif self.model_name in ["Logistic Regression", "LR"]:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**self.parameter_dict)

        elif self.model_name in ["Decision Tree", "DT"]:
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(**self.parameter_dict)

        elif self.model_name in ["K Nearest Neighbour", "KNN"]:
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**self.parameter_dict)

        elif self.model_name in ["Multi Layer Perceptron", "ANN"]:
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**self.parameter_dict)

        elif self.model_name in ["Gradient Boosted Decision Tree", "GBDT"]:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**self.parameter_dict)
        else:
            return None
        
    def fit(self, features, labels):
        self.model = self.model.fit(features, labels)
        
    def predict(self, test_data):
        return self.model.predict(test_data)
    
    def get_confusion_matrix(self, actual, prediction):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(actual,prediction)