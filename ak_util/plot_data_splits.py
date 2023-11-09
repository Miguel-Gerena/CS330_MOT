import matplotlib.pyplot as plt


def check_string_in_files(target_file, basketball, football, volleyball):
    # Initialize counts for each sport
    counts = {'Basketball': 0, 'Football': 0, 'Volleyball': 0}

    # Dictionary to relate file paths to sport names
    file_to_sport = {basketball: 'Basketball', football: 'Football', volleyball: 'Volleyball'}

    # Open target file and loop through each line
    with open(target_file, 'r') as f_target:
        for line in f_target:
            # Strip any leading/trailing whitespace from the line
            string_to_check = line.strip()

            # Check if the string is in any of the files
            for file_path, sport in file_to_sport.items():
                with open(file_path, 'r') as f:
                    if string_to_check in f.read():
                        counts[sport] += 1
                        break  # Assuming string can only be in one file
    return counts


def plot_sports_counts(train_counts, val_counts, test_counts):
    
    sports = list(train_counts.keys())
    train_counts = [train_counts[sport] for sport in sports]
    val_counts = [val_counts[sport] for sport in sports]
    test_counts = [test_counts[sport] for sport in sports]

    bar_width = 0.25

    # Set the position of the bars on the x-axis
    r1 = range(len(train_counts))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Make the plot
    plt.bar(r1, train_counts, color='blue', width=bar_width, edgecolor='grey', label='Train')
    plt.bar(r2, val_counts, color='red', width=bar_width, edgecolor='grey', label='Validation')
    plt.bar(r3, test_counts, color='green', width=bar_width, edgecolor='grey', label='Test')

    # Add labels
    plt.xlabel('Sports', fontweight='bold', fontsize=15)
    plt.xticks([r + bar_width for r in range(len(train_counts))], sports)
    plt.ylabel('Number of Videos')

    # Create legend & Show graphic
    plt.legend()
    plt.title('Counts of Videos for Each Sport in the Train, Validation, and Test sets')
    plt.savefig('updated_data_distribution.png')




basketball_path = './splits_txt/basketball.txt'
football_path = './splits_txt/football.txt'
volleyball_path = './splits_txt/volleyball.txt'
train_path = './splits_txt/train.txt'
val_path = './splits_txt/val.txt'
test_path = './splits_txt/test.txt'

# Perform the check
train_counts = check_string_in_files(train_path, basketball_path, football_path, volleyball_path)
val_counts = check_string_in_files(val_path, basketball_path, football_path, volleyball_path)
test_counts = check_string_in_files(test_path, basketball_path, football_path, volleyball_path)

# Plot the counts
plot_sports_counts(train_counts, val_counts, test_counts)

