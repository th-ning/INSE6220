import os
import glob
import pandas as pd


def merge_csv_files(input_folder, output_folder, dir_tag):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_folder}.")
        return

    first_csv = pd.read_csv(csv_files[0])
    first_csv_columns = first_csv.shape[1]

    print(f"第一个CSV文件：{csv_files[0]}，行数：{first_csv.shape[0]}，列数：{first_csv_columns}")

    combined_csv = first_csv

    for f in csv_files[1:]:
        csv = pd.read_csv(f)
        print(f"CSV文件：{f}，行数：{csv.shape[0]}，列数：{csv.shape[1]}")

        if csv.shape[1] != first_csv_columns:
            print(f"列数与第一个CSV文件不同，跳过：{f}")
            continue

        combined_csv = pd.concat([combined_csv, csv], ignore_index=True)

    folder_name = os.path.basename(input_folder)
    output_file = os.path.join(output_folder, f"{folder_name}_Merged_{dir_tag}.csv")
    combined_csv.to_csv(output_file, index=False)
    print(f"CSV files in {input_folder} have been merged and saved to {output_file}.")


def main(input_base_dir, output_base_dir, dir_tag):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for folder_name in os.listdir(input_base_dir):
        input_folder = os.path.join(input_base_dir, folder_name)
        output_folder = os.path.join(output_base_dir, folder_name)

        if os.path.isdir(input_folder):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            merge_csv_files(input_folder, output_folder, dir_tag)

if __name__ == "__main__":
    input_base_dir2 = r"C:\Users\123456\Documents\PCA project\OriginPeaks"
    output_base_dir2 = r"C:\Users\123456\Documents\PCA project\MergedPeaks"
    main(input_base_dir2, output_base_dir2, "Peaks")

    input_base_dir1 = r"C:\Users\123456\Documents\PCA project\OriginAll"
    output_base_dir1 = r"C:\Users\123456\Documents\PCA project\MergedAll"
    main(input_base_dir1, output_base_dir1, "All")


