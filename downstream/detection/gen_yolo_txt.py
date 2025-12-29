import os
import argparse

def write_file_paths_to_txt(folder_path, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                f.write(file_path + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Write file paths to a text file.")
    parser.add_argument('--folder_path', type=str, default="/home/bhzhang/Documents/datasets/aigc_detection/second_Gen1/synthetic_only/images/train", help='Path to the folder containing files.')
    parser.add_argument('--output_file', type=str, default='', help='Output text file to save the paths.')
    args = parser.parse_args()

    folder_path = args.folder_path
    output_file = args.output_file
    if not output_file:
        output_file = "/".join(folder_path.split("/")[:-2]) + "/{}.txt".format(folder_path.split("/")[-1])
    write_file_paths_to_txt(folder_path, output_file)