import os
import librosa
import math
import json
from pathlib import Path  # For cleaner path handling


def save_mfcc(
    dataset_path: str,
    json_path: str,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    num_segments=5,
):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }

    num_samples_per_segment = int((22050 * 30) / num_segments)
    expected_vector_length = math.ceil(num_samples_per_segment / hop_length)

    for i, (dir_path, dir_names, filenames) in enumerate(os.walk(dataset_path)):
        if dir_path is not dataset_path:
            semantic_label = Path(dir_path).name  # Extract label from path
            data["mapping"].append(semantic_label)

            print("\nProcessing: {}".format(semantic_label))

            for f in filenames:
                file_path = os.path.join(dir_path, f)

                # Check file extension before loading
                if not file_path.endswith(
                    (".wav", ".flac", ".ogg")
                ):  # Add supported extensions
                    print(f"Skipping unsupported format: {file_path}")
                    continue

                try:
                    signal, sr = librosa.load(file_path)

                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment

                        mfcc = librosa.feature.mfcc(
                            y=signal[start_sample:finish_sample],
                            sr=sr,
                            n_fft=n_fft,
                            n_mfcc=n_mfcc,
                            hop_length=hop_length,
                        )

                        mfcc = mfcc.T

                        if len(mfcc) == expected_vector_length:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i - 1)
                            print("{}, segment: {}".format(file_path, s))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(
        r"<PATH TO DATASET>",
        r"<PATH TO JSON FILE",
        num_segments=10,
    )
