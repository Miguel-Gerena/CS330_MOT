import json

def check_string_in_files(target_file, basketball, football, volleyball):
    # Initialize a dictionary to hold the string and corresponding counts
    string_counts = {}

    # Dictionary to relate file paths to sport names
    file_to_sport = {basketball: 'Basketball', football: 'Football', volleyball: 'Volleyball'}

    # Open target file and loop through each line
    with open(target_file, 'r') as f_target:
        for line in f_target:
            # Strip any leading/trailing whitespace from the line
            string_to_check = line.strip()

            # Initialize counts for the current string
            if string_to_check not in string_counts:
                string_counts[string_to_check] = {'Basketball': 0, 'Football': 0, 'Volleyball': 0}

            # Check if the string is in any of the files
            for file_path, sport in file_to_sport.items():
                with open(file_path, 'r') as f:
                    if string_to_check in f.read():
                        string_counts[string_to_check][sport] += 1
                        break  # Assuming string can only be in one file
    return string_counts

# Function to save counts to a JSON file
def save_counts_to_json(string_counts, filename):
    with open(filename, 'w') as json_file:
        json.dump(string_counts, json_file, indent=4)

# File paths
basketball_path = './splits_txt/basketball.txt'
football_path = './splits_txt/football.txt'
volleyball_path = './splits_txt/volleyball.txt'
train_path = './splits_txt/train.txt'
val_path = './splits_txt/val.txt'
test_path = './splits_txt/test.txt'
combined_path = './splits_txt/combined_train_val_txt/combined_train_val.txt'

# # Perform the check and save results
# for path, name in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
#     counts = check_string_in_files(path, basketball_path, football_path, volleyball_path)
#     save_counts_to_json(counts, f'{name}_counts.json')


counts = check_string_in_files(combined_path, basketball_path, football_path, volleyball_path)
save_counts_to_json(counts, 'train_val_combined_counts.json')



