import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules,fpgrowth

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

def input_dataset():
    while True:
        print("Select an option:")
        print("1. Input dataset manually")
        print("2. Use a CSV file")
        option = input("Enter your choice (1 or 2): ")

        if option == '1':
            # Initialize an empty dataset
            dataset = []

            # Get the number of transactions from the user
            num_transactions = int(input("Enter the number of transactions: "))

            # Iterate through transactions
            for i in range(1, num_transactions + 1):
                transaction = []
                print(f"Transaction {i}:")

                while True:
                    print("Enter item name (or press Enter to finish this transaction):")
                    item = input()

                    if not item:
                        break  # Finish this transaction

                    transaction.append(item)

                # Append the transaction to the dataset
                dataset.append(transaction)

            break  # Exit the loop after input
        elif option == '2':
            csv_file_name = input("Enter the name of the CSV file (in the same directory): ")

            try:
                # Load the CSV file into a DataFrame, force all columns to be treated as string
                df = pd.read_csv(csv_file_name, header=None, dtype=str)

                # Convert the DataFrame into a list of transactions
                dataset = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()

                break  # Exit the loop after successfully reading the CSV
            except FileNotFoundError:
                print(f"File '{csv_file_name}' not found in the current directory. Please try again.")
        else:
            print("Invalid option. Please choose 1 or 2.")

    return dataset

def perform_apriori(dataset):
    # Encode the dataset
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Initialize the highest_k variable
    highest_k = 0

    # Calculate the support count for 1-itemsets
    support_count_1_itemsets = df.apply(lambda col: col.sum(), axis=0)

    # Print the support count for 1-Itemsets
    print("Support Count for 1-Itemsets:")
    print(support_count_1_itemsets)

    # Ask the user for a minimum support count
    min_support_count = int(input("Enter the minimum support count: "))

    # Prune 1-itemsets based on the user-defined minimum support count
    pruned_1_itemsets = support_count_1_itemsets[support_count_1_itemsets >= min_support_count]

    # Print 1-itemsets after pruning
    print("\n1-Itemsets after Pruning:")
    print(pruned_1_itemsets)

    # Update the highest_k variable
    highest_k = 1

    # Generate all possible 2-itemsets using pruned 1-itemsets
    # ...

    # Generate all possible 2-itemsets using pruned 1-itemsets
    all_2_itemsets = set()
    pruned_1_itemsets_list = pruned_1_itemsets.index.tolist()
    for i in range(len(pruned_1_itemsets_list)):
        for j in range(i + 1, len(pruned_1_itemsets_list)):
            item1 = pruned_1_itemsets_list[i]
            item2 = pruned_1_itemsets_list[j]
            itemset = tuple(sorted([item1, item2]))
            all_2_itemsets.add(itemset)

    # Calculate the support count for all 2-itemsets manually
    support_count_2_itemsets = {}
    for itemset in all_2_itemsets:
        support_count = df[list(itemset)].all(axis=1).sum()
        support_count_2_itemsets[itemset] = support_count

    # Print all 2-itemsets formed from joining 1-itemsets with support count
    print("\nAll 2-Itemsets Formed from Joining 1-Itemsets:")
    for itemset, support_count in support_count_2_itemsets.items():
        item1, item2 = itemset
        print(f"{item1}, {item2}    {support_count}")

    # Filter 2-itemsets to include only those that meet the minimum support count
    filtered_2_itemsets = {itemset: support for itemset, support in support_count_2_itemsets.items() if
                           support >= min_support_count}

    # Check if there are no 2-itemsets meeting the support count
    if not filtered_2_itemsets:
        print("\nNo 2-itemsets meet the minimum support count.")
        highest_k = 1  # Update the highest_k variable

    # If there are 2-itemsets meeting the support count, proceed to the next stage
    else:
        # Update the highest_k variable
        highest_k = 2

        # Print 2-itemsets that meet the minimum support count
        print("\n2-Itemsets that Meet Minimum Support Count:")
        for itemset, support_count in filtered_2_itemsets.items():
            item1, item2 = itemset
            print(f"{item1}, {item2}    {support_count}")

        # ...

        # Generate all possible 3-itemsets from the pruned 2-itemsets that meet support count
        possible_3_itemsets = set()
        for itemset1 in filtered_2_itemsets.keys():
            for itemset2 in filtered_2_itemsets.keys():
                if itemset1 != itemset2:
                    new_itemset = tuple(sorted(list(set(itemset1).union(set(itemset2)))))
                    if len(new_itemset) == 3:
                        possible_3_itemsets.add(new_itemset)

        # Calculate the support count for 3-itemsets manually
        support_count_3_itemsets = {}
        for itemset in possible_3_itemsets:
            support_count = df[list(itemset)].all(axis=1).sum()
            support_count_3_itemsets[itemset] = support_count

        # Prune 3-itemsets based on the user-defined minimum support count
        filtered_3_itemsets = {itemset: support for itemset, support in support_count_3_itemsets.items() if support >= min_support_count}

        # Check if there are no 3-itemsets meeting the support count
        if not filtered_3_itemsets:
            print("\nNo 3-itemsets meet the minimum support count.")
            highest_k = 2  # Update the highest_k variable

        # If there are 3-itemsets meeting the support count, proceed to the next stage
        else:
            # Update the highest_k variable
            highest_k = 3

            # Print 3-Itemsets that Meet Minimum Support Count:
            print("\n3-Itemsets that Meet Minimum Support Count:")
            for itemset, support_count in filtered_3_itemsets.items():
                print(f"{itemset[0]}, {itemset[1]}, {itemset[2]}: {support_count}")

            # Generate all possible 4-itemsets from the pruned 3-itemsets that meet support count
            possible_4_itemsets = set()
            for itemset1 in filtered_3_itemsets.keys():
                for itemset2 in filtered_3_itemsets.keys():
                    if itemset1 != itemset2:
                        new_itemset = tuple(sorted(list(set(itemset1).union(set(itemset2)))))
                        if len(new_itemset) == 4:
                            possible_4_itemsets.add(new_itemset)

            # Calculate the support count for 4-itemsets manually
            support_count_4_itemsets = {}
            for itemset in possible_4_itemsets:
                support_count = df[list(itemset)].all(axis=1).sum()
                support_count_4_itemsets[itemset] = support_count

            # Prune 4-itemsets based on the user-defined minimum support count
            filtered_4_itemsets = {itemset: support for itemset, support in support_count_4_itemsets.items() if support >= min_support_count}

            # Check if there are no 4-itemsets meeting the support count
            if not filtered_4_itemsets:
                print("\nNo 4-itemsets meet the minimum support count.")
                highest_k = 3  # Update the highest_k variable

            # If there are 4-itemsets meeting the support count, proceed to the next stage
            else:
                # Update the highest_k variable
                highest_k = 4

                # Print 4-Itemsets that Meet Minimum Support Count:
                print("\n4-Itemsets that Meet Minimum Support Count:")
                for itemset, support_count in filtered_4_itemsets.items():
                    print(f"{itemset[0]}, {itemset[1]}, {itemset[2]}, {itemset[3]}: {support_count}")

    # Print the highest_k value at the end
    print(f"\nThe highest k where itemsets meet the minimum support count is {highest_k}.")

    # Initialize an empty list to store the filtered itemsets
    filtered_itemsets = []

    # Check if there are itemsets with the highest_k level
    if highest_k > 0:
        # Use the appropriate filtered_x_itemsets variable based on the highest_k value
        filtered_x_itemsets = None
        if highest_k == 1:
            filtered_x_itemsets = pruned_1_itemsets
        elif highest_k == 2:
            filtered_x_itemsets = filtered_2_itemsets
        elif highest_k == 3:
            filtered_x_itemsets = filtered_3_itemsets
        elif highest_k == 4:
            filtered_x_itemsets = filtered_4_itemsets

        # Iterate through the filtered itemsets with the highest k-level
        for itemset in filtered_x_itemsets.keys():
            filtered_itemsets.append(list(itemset))

        # Filtered itemsets
        filtered_itemsets = [filtered_itemsets]

    # Print the filtered itemsets in the dataset format
    print("Filtered Itemsets in Dataset Format:")
    print(filtered_itemsets)

    return filtered_itemsets, highest_k
def perform_fpgrowth(dataset):
    # Encode the dataset
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Calculate the total number of transactions
    total_transactions = len(dataset)

    min_support_count = int(input("Enter the minimum support count: "))

    # Calculate the minimum support based on the user-defined minimum support count
    min_support = min_support_count / total_transactions

    # Use FP-Growth algorithm to find frequent itemsets
    frequent = fpgrowth(df, min_support=min_support, use_colnames=True)

    # Filter the results to show only 2-item and 3-item frequent itemsets
    filtered_frequent = frequent[frequent['itemsets'].apply(lambda x: len(x) == 2 or len(x) == 3)].copy()

    # Calculate the support count for each itemset
    filtered_frequent['support count'] = filtered_frequent['itemsets'].apply(
        lambda itemset: sum(df[list(itemset)].all(axis=1)))

    # Reorder the columns for display
    filtered_frequent = filtered_frequent[['itemsets', 'support count', 'support']]

    # Print the filtered frequent itemsets with support count
    print(filtered_frequent)

    # Find the highest k-level that exists
    highest_k = filtered_frequent['itemsets'].apply(len).max()

    # Filter the results to show itemsets with the same length as highest_k
    itemsets_with_highest_k = filtered_frequent[filtered_frequent['itemsets'].apply(len) == highest_k].copy()

    # Convert the itemsets from frozenset to list and return in the desired format
    itemsets_list = [list(itemset) for itemset in itemsets_with_highest_k['itemsets']]

    # Print the highest k-level
    print(f"The highest k-level is {highest_k}")

    # Print members of the highest k-level
    print(f"Members of the highest k-level ({highest_k}):")
    for itemset in itemsets_list:
        print(itemset)

    # Return the list of itemsets with the highest k-level and highest_k
    return [itemsets_list], highest_k

def find_association_rules(dataset, filtered_itemsets, highest_k):
    # Encode the dataset
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)


    k_level_rule = int(highest_k)


    if k_level_rule not in [2, 3]:
        print("Invalid choice. Please choose either '2' or '3'.")
        return

    # Generate frequent itemsets using Apriori with a minimum support threshold
    min_support = 0.2
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Filter the association rules based on the user's choice and k-level
    if k_level_rule == 2:
        # Use the 1 antecedent and 1 consequent rule
        filtered_association_rules_df = association_rules(frequent_itemsets, metric="confidence",
                                                         min_threshold=0.1)
        filtered_association_rules_df = filtered_association_rules_df[
            ((filtered_association_rules_df['antecedents'].apply(len) == 1) &
             (filtered_association_rules_df['consequents'].apply(len) == 1))]
    elif k_level_rule == 3:
        # Use the 2 antecedent and 1 consequent and vice versa rule
        filtered_association_rules_df = association_rules(frequent_itemsets, metric="confidence",
                                                         min_threshold=0.1)
        filtered_association_rules_df = filtered_association_rules_df[
            (((filtered_association_rules_df['antecedents'].apply(len) == 2) &
              (filtered_association_rules_df['consequents'].apply(len) == 1)) |
             ((filtered_association_rules_df['antecedents'].apply(len) == 1) &
              (filtered_association_rules_df['consequents'].apply(len) == 2)))]

    # Get the items present in the filtered itemsets
    filtered_items = set(item for itemset in filtered_itemsets[0] for item in itemset)

    # Filter the association rules to include only those with items present in the filtered itemsets
    filtered_association_rules_df = filtered_association_rules_df[
        filtered_association_rules_df['antecedents'].apply(lambda x: all(item in filtered_items for item in x)) &
        filtered_association_rules_df['consequents'].apply(lambda x: all(item in filtered_items for item in x))]

    # Print the association rules based on the user's choice and k-level
    print(f"Association Rules for k-level {highest_k} using Rule {k_level_rule}:")
    print(filtered_association_rules_df)

    # Ask the user for the minimum support and confidence thresholds
    min_support_threshold = float(input("Enter the minimum support threshold for filtering (between 0 and 1): "))
    min_confidence_threshold = float(input("Enter the minimum confidence threshold for filtering (between 0 and 1): "))

    # Filter the association rules based on the minimum support and confidence thresholds
    filtered_association_rules_df = filtered_association_rules_df[
        (filtered_association_rules_df['support'] >= min_support_threshold) &
        (filtered_association_rules_df['confidence'] >= min_confidence_threshold)]

    # Print the filtered association rules
    print(f"Filtered Association Rules for k-level {highest_k} using Rule {k_level_rule} "
          f"with Support >= {min_support_threshold} and Confidence >= {min_confidence_threshold}:")
    print(filtered_association_rules_df)
    print("\n")





if __name__ == "__main__":
    # Input dataset
    dataset = input_dataset()
    algorithm_choice = input("Choose the algorithm (1 for Apriori, 2 for FP-Growth): ")

    if algorithm_choice == '1':
        # Call Apriori function
        filtered_itemsets, highest_k = perform_apriori(dataset)
    elif algorithm_choice == '2':
        # Call FP-Growth function
        filtered_itemsets, highest_k = perform_fpgrowth(dataset)
    else:
        print("Invalid choice of algorithm. Please choose '1' for Apriori or '2' for FP-Growth.")

         # Call association rule function
    find_association_rules(dataset, filtered_itemsets, highest_k)
