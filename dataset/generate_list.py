import os


def get_snr_from_filename(filename):
    parts = filename.split('_')
    try:
        snr = int(parts[-2])  # 倒数第二个部分是SNR值
        return snr
    except ValueError:
        return None


def export_dataset_list(data_root):
    images_root = os.path.join(data_root, 'images')
    labels_root = os.path.join(data_root, 'lable')

    for split in ['train', 'test']:
        split_image_root = os.path.join(images_root, split)
        split_label_root = os.path.join(labels_root, split)

        if not os.path.exists(split_image_root):
            continue

        lst_files = {}

        for num in os.listdir(split_image_root):
            num_path = os.path.join(split_image_root, num)
            if not os.path.isdir(num_path):
                continue

            for signal_type in os.listdir(num_path):
                signal_path = os.path.join(num_path, signal_type)
                if not os.path.isdir(signal_path):
                    continue

                for img_file in os.listdir(signal_path):
                    if not img_file.endswith('.png'):
                        continue

                    snr = get_snr_from_filename(img_file)
                    if snr is None:
                        continue

                    img_path = os.path.join(split_image_root, num, signal_type, img_file)
                    label_path = os.path.join(split_label_root, num, signal_type, img_file)

                    lst_filename = f"snr_{snr}_num_{num}.lst"
                    if lst_filename not in lst_files:
                        lst_files[lst_filename] = []

                    lst_files[lst_filename].append(f"{img_path} {label_path}\n")

        output_dir = os.path.join(data_root, 'lst_files', split)
        os.makedirs(output_dir, exist_ok=True)

        for lst_filename, lines in lst_files.items():
            with open(os.path.join(output_dir, lst_filename), 'w') as f:
                f.writelines(lines)


if __name__ == "__main__":
    data_root = "../data"  # 根目录
    export_dataset_list(data_root)
