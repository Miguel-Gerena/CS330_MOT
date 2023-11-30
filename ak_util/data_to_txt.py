import os

def write_subdir_names_to_file(main_directory):
    # Get the main directory name to use in the txt filename
    main_dir_name = os.path.basename(main_directory.strip('/').strip('\\'))
    output_filename = f"{main_dir_name}_subdirs.txt"
    output_filepath = os.path.join(main_directory, output_filename)
    
    # Open the output file
    with open(output_filepath, 'w') as file:
        # Loop through each item in the main directory
        for item in os.listdir(main_directory):
            # Construct the full path to the item
            item_path = os.path.join(main_directory, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                # Write the subdirectory name to the file
                file.write(item + '\n')
                
    print(f"Subdirectory names written to {output_filepath}")

# Example usage:
val_directory = 'C:/Users/akayl/Desktop/CS330_MOT/dataset/val'
train_directory = 'C:/Users/akayl/Desktop/CS330_MOT/dataset/train'
test_directory = 'C:/Users/akayl/Desktop/CS330_MOT/dataset/test'

write_subdir_names_to_file(val_directory)
write_subdir_names_to_file(train_directory)
write_subdir_names_to_file(test_directory)
