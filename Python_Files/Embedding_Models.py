from sentence_transformers import SentenceTransformer
class PreTrained_EmbeddingModels:
    def __init__(self, model_url=None, model_name=None, model=None):
        self.model_url = model_url
        self.model_name = model_name
        self.model = model
        self._load(model_url)
        
    def _load(self, model_url):
        if self.model != None:
            print("Model Switching is disabled.")
            return
        self.model = SentenceTransformer(model_url)
        
    def _get_model(self):
        return self.model