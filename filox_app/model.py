import json
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, AutoConfig

class IndoBERTInference:
    def __init__(self, model_path='saved_model/indobert_hoax_model', tokenizer_path='saved_model/tokenizer_config.json'):
        self.model = None
        self.tokenizer = None
        self.max_length = None
        self.load_model(model_path, tokenizer_path)

    def load_model(self, model_path, tokenizer_path):
        try:
            with open(tokenizer_path, 'r') as f:
                config_tokenizer = json.load(f)

            self.max_length = config_tokenizer['max_length']
            model_name = config_tokenizer['model_name']

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            config_model = AutoConfig.from_pretrained(model_name)
            config_model.add_pooling_layer = False

            class IndoBERTClassifier(tf.keras.Model):
                def __init__(self, model_name_hf, max_len, model_config, **kwargs):
                    super(IndoBERTClassifier, self).__init__(**kwargs)
                    self.max_length = max_len
                    self.bert_model = TFAutoModel.from_pretrained(model_name_hf, config=model_config)
                    self.dropout1 = tf.keras.layers.Dropout(0.3)
                    self.dense1 = tf.keras.layers.Dense(256, activation='relu', name='dense_1')
                    self.dropout2 = tf.keras.layers.Dropout(0.2)
                    self.dense2 = tf.keras.layers.Dense(128, activation='relu', name='dense_2')
                    self.dropout3 = tf.keras.layers.Dropout(0.1)
                    self.classifier = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')

                def call(self, inputs, training=None):
                    input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']
                    bert_output = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        training=training
                    )
                    pooled_output = bert_output.last_hidden_state[:, 0, :]
                    x = self.dropout1(pooled_output, training=training)
                    x = self.dense1(x)
                    x = self.dropout2(x, training=training)
                    x = self.dense2(x)
                    x = self.dropout3(x, training=training)
                    return self.classifier(x)

            self.model = IndoBERTClassifier(
                model_name_hf=model_name,
                max_len=self.max_length,
                model_config=config_model,
                name='IndoBERT_Classifier'
            )

            dummy_input = {
                'input_ids': tf.zeros((1, self.max_length), dtype=tf.int32),
                'attention_mask': tf.zeros((1, self.max_length), dtype=tf.int32)
            }
            _ = self.model(dummy_input)

            if not model_path.endswith('.weights.h5'):
                model_path += '.weights.h5'

            self.model.load_weights(model_path)
            print("✅ IndoBERT model loaded successfully!")

        except Exception as e:
            print(f"❌ An unexpected error occurred while loading model: {e}")
            self.model = None
            self.tokenizer = None

    def predict(self, text):
        if self.model is None or self.tokenizer is None:
            return {'text': text, 'prediction': 'Error', 'confidence': 0.0, 'is_hoax': False}

        encoded = self.tokenizer(
            [text],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_attention_mask=True
        )
        predictions = self.model.predict({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }, verbose=0)

        confidence = float(predictions[0][0])
        is_hoax = confidence > 0.5
        return {
            'text': text,
            'prediction': 'HOAX' if is_hoax else 'NOT HOAX',
            'confidence': confidence,
            'is_hoax': is_hoax
        }