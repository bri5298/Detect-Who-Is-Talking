import os
import tkinter as tk

import tkinterdnd2 as tkdnd

import detect_who_is_talking.model_creation as mc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide warnings

MODELS_FOLDER = r"/Users/brielle/Documents/Python/Detect-Who-Is-Talking/models"
LABELENCODER = mc.get_labelencoder_from_file(
    os.path.join(MODELS_FOLDER, "labelencoder.pkl")
)
MODEL = mc.load_model_from_file(
    os.path.join(MODELS_FOLDER, "audio_classification_model.hdf5")
)


def get_path(event):
    filepath = event.data
    if filepath.endswith(".wav"):
        pathLabel.configure(
            text="Drag and drop a '.wav' file in the entry box",
            font=("arial", 32),
            fg="black",
        )
        prediction = mc.predict_single_audio_file(filepath, MODEL, LABELENCODER)
        pathLabel2.configure(text=f"Prediction is: {prediction}")
        pathLabel3.configure(text=f"For file: {event.data}")
    else:
        pathLabel.configure(
            text=f"{event.data} is not a '.wav' file. \nPlease drop a '.wav' file",
            font=("arial", 14),
            fg="red",
        )


root = tkdnd.TkinterDnD.Tk()
root.geometry("650x500")
root.title("Get file path")
nameVar = tk.StringVar()
pathLabel = tk.Label(
    root, text="Drag and drop a '.wav' file in the entry box", font=("arial", 32)
)
pathLabel.pack(side=tk.TOP, padx=0, pady=50)
entryWidget = tk.Entry(root, font=("arial", 110))
entryWidget.pack(side=tk.TOP, padx=5, pady=10)
pathLabel2 = tk.Label(root, text="", font=("arial", 32), fg="blue")
pathLabel2.pack(side=tk.TOP, padx=0)
pathLabel3 = tk.Label(root, text="", font=("arial", 14))
pathLabel3.pack(side=tk.TOP, padx=0)
entryWidget.drop_target_register(tkdnd.DND_ALL)
entryWidget.dnd_bind("<<Drop>>", get_path)
root.mainloop()
