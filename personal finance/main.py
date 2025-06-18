import pandas as pd
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from data import get_date, get_amount, get_category, get_description

class CSV:
    CSV_FILE = "finance_data.csv"
    COLUMNS = ["date", "amount", "category", "description"]
    FORMAT = "%d-%m-%Y"
    
    @classmethod
    def initialize_csv(cls):
        try:
            pd.read_csv(cls.CSV_FILE)
        except FileNotFoundError:
            df = pd.DataFrame(columns=cls.COLUMNS)
            df.to_csv(cls.CSV_FILE, index=False)


    @classmethod
    def add_entry(cls, date, amount, category, description):
        new_entry = {
            "date": date,
            "amount": amount,
            "category": category,
            "description": description,
        }
        with open(cls.CSV_FILE, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cls.COLUMNS)
            writer.writerow(new_entry)
        print("Entry added successfully")

    @classmethod
    def get_transactions(cls, start_date, end_date):
        df = pd.read_csv(cls.CSV_FILE)
        df["date"] = pd.to_datetime(df["date"], format = CSV.FORMAT)
        start_date = datetime.strptime(start_date, CSV.FORMAT)
        end_date = datetime.strptime(end_date, CSV.FORMAT)
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            print("No transactions found in the specified date range.")
        else:
            print(f"Transactions from", start_date.strftime(CSV.FORMAT), "to", end_date.strftime(CSV.FORMAT))
            print(filtered_df.to_string(index = False, formatters= {"date": lambda x: x.strftime(CSV.FORMAT)}))

            total_income = filtered_df[filtered_df["category"] == "Income"]["amount"].sum()
            total_expense = filtered_df[filtered_df["category"] == "Expense"]["amount"].sum()
            print("\nSummary:")
            print(f"Total Income: ${total_income:.2f}")
            print(f"Total Expense: ${total_expense:.2f}")
            print(f"Net Balance: ${total_income - total_expense:.2f}")

        return filtered_df

def add():
    CSV.initialize_csv()
    date = get_date("Enter the date (DD-MM-YYYY): " )
    amount = get_amount()
 
    category = get_category()
    description = get_description()
    CSV.add_entry(date, amount, category, description)

def plot_transactions(df):
    df.set_index("date", inplace=True)
    income_df = (
        df[df["category"] == "Income"]
        .resample("D")
        .sum()
        .reindex(df.index, fill_value=0)
    )
    expense_df = (
        df[df["category"] == "Expense"]
        .resample("D")
        .sum()
        .reindex(df.index, fill_value=0)
    )


    plt.figure(figsize=(12, 6))
    plt.plot(income_df.index, income_df["amount"], label="Income", color="green")
    plt.plot(expense_df.index, expense_df["amount"], label="Expense", color="red")
    plt.title("Daily Income and Expense")
    plt.xlabel("Date")
    plt.ylabel("Amount ($)")
    plt.legend()
    plt.grid(True)
    plt.show()    


def detect_anomalies():
    try:
        df = pd.read_csv(CSV.CSV_FILE)
        if df.empty:
            print("No transactions to analyze.")
            return

        df["date"] = pd.to_datetime(df["date"], format=CSV.FORMAT)

        anomalies = []

        grouped = df.groupby("category")
        threshold = 2

        for category, group in grouped:
            mean = group["amount"].mean()
            std = group["amount"].std()

            if pd.isna(std) or std == 0:
                continue

            for _, row in group.iterrows():
                if row["amount"] > mean + threshold * std:
                    anomalies.append(row)

        if anomalies:
            print("\n🔍 Spending Anomalies Detected:")
            for anomaly in anomalies:
                print(f"- [{anomaly['date'].strftime(CSV.FORMAT)}] {anomaly['category']}: ${anomaly['amount']} — {anomaly['description']}")
        else:
            print("✅ No anomalies found.")

    except FileNotFoundError:
        print("No data file found. Please add some transactions first.")

def export_to_excel(df, filename="exported_transactions.xlsx"):
    try:
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Transactions exported successfully to '{filename}'")
    except Exception as e:
        print(f"Failed to export to Excel: {e}")

def export_summary_report(df, filename="summary_report.csv"):
    if df.empty:
        print("No data to export.")
        return

    summary_data = {
        "Total Income": [df[df["category"] == "Income"]["amount"].sum()],
        "Total Expense": [df[df["category"] == "Expense"]["amount"].sum()],
    }
    summary_data["Net Balance"] = [summary_data["Total Income"][0] - summary_data["Total Expense"][0]]

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(filename, index=False)
    print(f"✅ Summary report exported to '{filename}'")


def search_edit_delete():
    try:
        df = pd.read_csv(CSV.CSV_FILE)
        if df.empty:
            print("No transactions found.")
            return
    except FileNotFoundError:
        print("Data file not found.")
        return

    keyword = input("Enter a keyword to search in descriptions: ").lower()
    filtered_df = df[df["description"].str.lower().str.contains(keyword)]

    if filtered_df.empty:
        print("No matching transactions found.")
        return

    filtered_df.reset_index(inplace=True)
    print("\nSearch Results:")
    print(filtered_df[["index", "date", "amount", "category", "description"]])

    try:
        selected_index = int(input("\nEnter the index of the transaction to edit/delete (or -1 to cancel): "))
        if selected_index == -1:
            return

        original_index = filtered_df.loc[filtered_df["index"] == selected_index, "index"].values[0]
        action = input("Type 'edit' to modify or 'delete' to remove the transaction: ").lower()

        if action == "edit":
            # Prompt for new values
            new_date = get_date("Enter new date (DD-MM-YYYY): ")
            new_amount = get_amount()
            new_category = get_category()
            new_description = get_description()

            # Update the row
            df.loc[original_index] = [new_date, new_amount, new_category, new_description]
            df.to_csv(CSV.CSV_FILE, index=False)
            print("Transaction updated successfully.")

        elif action == "delete":
            df.drop(index=original_index, inplace=True)
            df.to_csv(CSV.CSV_FILE, index=False)
            print("🗑 Transaction deleted successfully.")

        else:
            print("Invalid action. Please enter 'edit' or 'delete'.")
    except Exception as e:
        print(f"Error: {e}")




from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns

def cluster_spending_patterns():
    try:
        df = pd.read_csv(CSV.CSV_FILE)
        if df.empty:
            print("No transactions to analyze.")
            return
    except FileNotFoundError:
        print("Data file not found.")
        return

    df["date"] = pd.to_datetime(df["date"], format=CSV.FORMAT)


    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])

    features = df[["amount", "day_of_week", "is_weekend", "category_encoded"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(scaled_features)
    cluster_labels = {
    0: "Small Daily Expenses",
    1: "Large Infrequent Purchases",
    2: "Weekend Entertainment"
    }
    df["cluster_label"] = df["cluster"].map(cluster_labels)
    print("\n Cluster Summary:")
    print(df.groupby("cluster")[["amount"]].mean())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="amount", y="day_of_week", hue="cluster_label", palette="Set2")
    plt.title("Spending Pattern Clusters")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Day of Week (0 = Monday)")
    plt.legend(title="Spending Pattern")
    plt.grid(True)
    plt.show()


def suggest_budgets():
    try:
        df = pd.read_csv(CSV.CSV_FILE)
        if df.empty:
            print("No transactions available.")
            return
    except FileNotFoundError:
        print("Data file not found.")
        return

    df["date"] = pd.to_datetime(df["date"], format=CSV.FORMAT)
    df = df[df["category"] == "Expense"]  # Focus on spending

    # Use last 12 months of data
    recent_df = df[df["date"] > pd.Timestamp.now() - pd.DateOffset(months=12)]

    if recent_df.empty:
        print("Not enough recent expense data to suggest budgets.")
        return

    # Group by description or category — change as needed
    category_summary = (
        recent_df.groupby("description")["amount"]
        .mean()
        .reset_index()
        .rename(columns={"amount": "avg_monthly_spend"})
    )

    # Sort top 5 spending areas
    top_spending = category_summary.sort_values(by="avg_monthly_spend", ascending=False).head(5)

    print("\n Personalized Budget Suggestions (based on your last 12 months of expenses):")
    for _, row in top_spending.iterrows():
        suggested_budget = row["avg_monthly_spend"] * 0.9  # Suggest 10% reduction
        print(f"- {row['description']}: Avg Monthly Spend = ${row['avg_monthly_spend']:.2f} → Suggested Budget = ${suggested_budget:.2f}")

    # Optional: Export to CSV
    export = input("Do you want to export the suggested budgets to a file? (yes/no): ").strip().lower()
    if export == "yes":
        top_spending["suggested_budget"] = top_spending["avg_monthly_spend"] * 0.9
        top_spending.to_csv("budget_suggestions.csv", index=False)
        print("Budget suggestions exported to 'budget_suggestions.csv'.")


def main():
    while True:
        print("\n1. Add new transaction")   
        print("2. View transactions and summarize within a date range")
        print("3. Detect spending anomalies")
        print("4. Export summary report as CSV") 
        print("5. Search, Edit or Delete a transaction") 
        print("6. Cluster spending patterns")
        print("7. Get personalized budget suggestions")
        print("8. Exit")

        choice = input("Enter your choice: ")
        if choice == "1":
            add()
        elif choice == "2":
            start_date = get_date("Enter the start date (DD-MM-YYYY): ")
            end_date = get_date("Enter the end date (DD-MM-YYYY): ")
            
            start_dt = datetime.strptime(start_date, CSV.FORMAT)
            end_dt = datetime.strptime(end_date, CSV.FORMAT)

            if end_dt < start_dt:
                print("Error: End date cannot be earlier than start date.")
            else:
                df = CSV.get_transactions(start_date, end_date)
                plot_choice = input("Do you want to plot the transactions? (yes/no): ").lower()
                if plot_choice == "yes":
                    plot_transactions(df)
                elif plot_choice == "no":
                    print("Skipping plot.")
                else:
                    print("Error: Invalid input. Please answer 'yes' or 'no'.")
            if not df.empty:
                export_choice = input("Do you want to export these transactions to Excel? (yes/no): ").lower()
                if export_choice == "yes":
                    export_to_excel(df)
                elif export_choice == "no":
                    print("Skipping export.")
                else:
                    print("Error: Invalid input. Please answer 'yes' or 'no'.")
        elif choice == "3":
            detect_anomalies()
        elif choice == "4":
            try:
                df = pd.read_csv(CSV.CSV_FILE)
                if df.empty:
                    print("No transactions found. Please add some first.")
                else:
                    export_summary_report(df)
            except FileNotFoundError:
                print("Data file not found. Please add a transaction first.")
        elif choice == "5":
            search_edit_delete()

        elif choice == "6":
            cluster_spending_patterns()
        elif choice == "7":
            suggest_budgets()
        elif choice == "8":
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please try again.")
            return main()
        
        
if __name__ == "__main__":
    main() 