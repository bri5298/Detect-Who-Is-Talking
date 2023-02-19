import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tkinterdnd2 import *
from tkinter import *
from tkinter.ttk import *
import detect_who_is_talking.model_creation as mc


def get_path(event):
    pathLabel.configure(text=event.data)
    LABELENCODER = mc.get_labelencoder_from_file(r"/Users/brielle/Documents/Python/Detect-Who-Is-Talking/models/labelencoder.pkl")
    MODEL = mc.load_model_from_file(r"/Users/brielle/Documents/Python/Detect-Who-Is-Talking/models/audio_classification_model.hdf5")
    filepath = event.data
    prediction = mc.predict_single_audio_file(filepath, MODEL, LABELENCODER)
    pathLabel2.configure(text=f"Prediction is: {prediction}")
    return f"testing \n{event.data}"

root = TkinterDnD.Tk()
root.geometry("1000x500")
root.title("Get file path")
nameVar = StringVar()
entryWidget = Entry(root)
entryWidget.pack(side=TOP, padx=5, pady=5)
pathLabel = Label(root, text="Drag and drop file in the entry box")
pathLabel.pack(side=TOP, padx=0)
pathLabel2 = Label(root, text="")
pathLabel2.pack(side=TOP, padx=0)
entryWidget.drop_target_register(DND_ALL)
entryWidget.dnd_bind("<<Drop>>", get_path)
root.mainloop()
