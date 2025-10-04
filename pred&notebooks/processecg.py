import numpy as np
import wfdb
from scipy.signal import resample
from keras.models import load_model
import matplotlib.pyplot as plt

def preprocess_ecg_with_mapping(record_path, target_fs=400, target_length=4096, plot_signal=False):
    # Load 12-lead ECG record using WFDB
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal
    fs = record.fs
    orig_leads = record.sig_name

    # Map original lead names to model expected names
    mapping = {
        'I': 'DI',
        'II': 'DII',
        'III': 'DIII',
        'AVR': 'AVR',
        'AVL': 'AVL',
        'AVF': 'AVF',
        'V1': 'V1',
        'V2': 'V2',
        'V3': 'V3',
        'V4': 'V4',
        'V5': 'V5',
        'V6': 'V6'
    }

    mapped_leads = [mapping.get(lead, None) for lead in orig_leads]

    # Expected model lead order
    model_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Reorder signals to match model expectation
    indices = []
    for lead in model_leads:
        if lead in mapped_leads:
            indices.append(mapped_leads.index(lead))
        else:
            raise ValueError(f"Lead {lead} is missing in the ECG record!")

    signals_selected = signals[:, indices]

    # Resample if necessary
    if fs != target_fs:
        num_samples = int(signals_selected.shape[0] * target_fs / fs)
        signals_selected = resample(signals_selected, num_samples, axis=0)

    # Crop or pad
    curr_len = signals_selected.shape[0]
    if curr_len > target_length:
        signals_selected = signals_selected[:target_length, :]
    elif curr_len < target_length:
        padding = np.zeros((target_length - curr_len, signals_selected.shape[1]))
        signals_selected = np.vstack([signals_selected, padding])

    # Normalize (zero mean, unit variance)
    signals_norm = (signals_selected - np.mean(signals_selected, axis=0)) / (np.std(signals_selected, axis=0) + 1e-8)

    # Add batch dimension
    input_tensor = np.expand_dims(signals_norm, axis=0)


    return input_tensor


if __name__ == "__main__":
    # --- Update paths ---
    record_path = r"data\ptb-xl-a-large-publicly-availabl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records100\00000\00674_lr"
    model_path = r"models\model.hdf5"

    # Load model
    model = load_model(model_path, compile=False)

    # Preprocess ECG record
    ecg_input = preprocess_ecg_with_mapping(record_path, plot_signal=True)

    # Predict probabilities
    prediction = model.predict(ecg_input)[0]

    # Define abnormality classes
    abnormalities = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

    # --- Decision logic ---
    threshold = 0.3  # adjust based on model behavior
    detected = [(ab, p) for ab, p in zip(abnormalities, prediction) if p >= threshold]

    print("\n--- ECG Classification Result ---")
    for ab, p in zip(abnormalities, prediction):
        print(f"{ab}: {p:.4f}")

    if len(detected) == 0:
        print("\n🟢 Result: NORMAL (no significant abnormality detected)")
    else:
        diseases = [ab for ab, _ in detected]
        print("\n🔴 Result: ABNORMAL")
        print(f"Detected conditions: {', '.join(diseases)}")
