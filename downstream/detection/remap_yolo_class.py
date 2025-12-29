import os

def remap_classes_in_labels(input_folder, output_folder, class_mapping):
    """
    Read label files, remap class indices according to class_mapping,
    and save to a new folder.
    Annotations with classes not in the mapping are removed.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            with open(input_path, 'r') as infile:
                lines = infile.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue  # skip empty lines

                old_class = parts[0]
                # Check if class is in mapping
                if old_class in class_mapping:
                    new_class = class_mapping[old_class]
                    # Reconstruct line with new class index
                    new_line = ' '.join([str(new_class)] + parts[1:]) + '\n'
                    updated_lines.append(new_line)
                else:
                    # Skip annotations with classes not in mapping
                    continue

            with open(output_path, 'w') as outfile:
                outfile.writelines(updated_lines)

if __name__ == "__main__":
    input_folder = "/home/bhzhang/Documents/datasets/aigc_detection/second/synthetic_only/labels/train"    # Replace with your input folder path
    output_folder = "/home/bhzhang/Documents/datasets/aigc_detection/second_Gen1/synthetic_only/labels/train"
    class_mapping = {
        "0": "1",
        "2": "0",
        "3": "0",
        "4": "0",
        "7": "0",
        # "1": "2",
        # "5": "2",
        # "6": "2",
        # Add more mappings as needed
    }
    remap_classes_in_labels(input_folder, output_folder, class_mapping)