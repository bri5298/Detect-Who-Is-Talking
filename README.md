# Detect-Who-Is-Talking

### We have face, fingerprint and eye recognition but what about voice recognition?

## Project Structure:
- Modules (detect_who_is_talking):
  - data_preparation.py: Functions that perform data preparation tasks.
  - model_creation.py: Functions that create, train and evaluate the model.
  - gui_predict_from_file_selected.py: Creates a simple GUI where we can drag and drop a 
file from our computer and get shown a prediction of who is talking.
- Notebooks: 
  - Audio Classification - Conversation.ipynb: From start to finish the notebook takes files of people talking alone, 
trains the model with the data and uses it to identify when each person is talking during a conversation.
  - Audio Classification - Individual Clips.ipynb: From start to finish the notebook takes files of people talking 
alone, trains the model with the data and can make a prediction of which person is talking in an audio file.  
- Data: Data used for training and testing the model.
- Models: Where we save the models and labelencoders.