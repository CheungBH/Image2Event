import os
import tqdm


def process_yolo_annotations(input_dir, output_dir):
    """
    Process YOLO annotation files:
    - Read from input_dir
    - Clip bounding box corners within [0, 1]
    - Recalculate center, width, height
    - Save processed files to output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r') as infile:
                lines = infile.readlines()

            updated_lines = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip malformed lines

                class_id, x_center, y_center, width, height = parts

                # Convert to float
                x_center = float(x_center)
                y_center = float(y_center)
                width = float(width)
                height = float(height)

                # Convert center/size to corner coordinates
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2

                # Clip corners to [0, 1]
                xmin_clipped = max(0.0, xmin)
                ymin_clipped = max(0.0, ymin)
                xmax_clipped = min(1.0, xmax)
                ymax_clipped = min(1.0, ymax)

                # Recalculate center and size from clipped corners
                new_width = xmax_clipped - xmin_clipped
                new_height = ymax_clipped - ymin_clipped
                new_x_center = xmin_clipped + new_width / 2
                new_y_center = ymin_clipped + new_height / 2

                # Ensure values are within [0, 1]
                new_x_center = min(max(new_x_center, 0.0), 1.0)
                new_y_center = min(max(new_y_center, 0.0), 1.0)
                new_width = min(max(new_width, 0.0), 1.0)
                new_height = min(max(new_height, 0.0), 1.0)

                # Format the updated line
                updated_line = f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n"
                updated_lines.append(updated_line)

            # Write the updated annotations to the output directory
            with open(output_path, 'w') as outfile:
                outfile.writelines(updated_lines)


if __name__ == "__main__":
    input_directory = "/home/bhzhang/Documents/datasets/DSEC_detection_data/Gen1/baseline/full/labels_raw/val"  # Replace with your input folder path
    output_directory = "/home/bhzhang/Documents/datasets/DSEC_detection_data/Gen1/baseline/full/labels/val"  # Replace with your output folder path

    process_yolo_annotations(input_directory, output_directory)
