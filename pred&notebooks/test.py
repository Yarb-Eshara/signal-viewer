import wfdb

record = wfdb.rdrecord(r"data\ptb-xl-a-large-publicly-availabl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records100\00000\00674_lr")
print("Sampling frequency:", record.fs)
print("Signal length:", record.sig_len)
print("Duration (s):", record.sig_len / record.fs)
