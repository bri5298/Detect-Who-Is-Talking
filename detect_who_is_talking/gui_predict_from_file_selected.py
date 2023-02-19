import os
import tkinter as tk

import tkinterdnd2 as tkdnd

import detect_who_is_talking.model_creation as mc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # hide warnings

MODELS_FOLDER = r"/Users/brielle/Documents/Python/Detect-Who-Is-Talking/models"


def get_path(event):
    pathLabel.configure(text=event.data)
    LABELENCODER = mc.get_labelencoder_from_file(
        os.path.join(MODELS_FOLDER, "labelencoder.pkl")
    )
    MODEL = mc.load_model_from_file(
        os.path.join(MODELS_FOLDER, "audio_classification_model.hdf5")
    )
    filepath = event.data
    prediction = mc.predict_single_audio_file(filepath, MODEL, LABELENCODER)
    pathLabel2.configure(text=f"Prediction is: {prediction}")
    return f"testing \n{event.data}"


root = tkdnd.TkinterDnD.Tk()
root.geometry("1000x500")
root.title("Get file path")
nameVar = tk.StringVar()
entryWidget = tk.Entry(root)
entryWidget.pack(side=tk.TOP, padx=5, pady=5)
pathLabel = tk.Label(root, text="Drag and drop file in the entry box")
pathLabel.pack(side=tk.TOP, padx=0)
pathLabel2 = tk.Label(root, text="")
pathLabel2.pack(side=tk.TOP, padx=0)
entryWidget.drop_target_register(tkdnd.DND_ALL)
entryWidget.dnd_bind("<<Drop>>", get_path)
root.mainloop()
