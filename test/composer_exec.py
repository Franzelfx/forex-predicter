import subprocess
from datetime import datetime, timedelta
import argparse

# Define a function to check if a date is a weekend (Saturday or Sunday)
def is_weekend(date):
    return date.weekday() >= 5  # 5 represents Saturday, 6 represents Sunday

def main():
    parser = argparse.ArgumentParser(description='Execute composer_test.py with date range excluding weekends')
    parser.add_argument('--end', type=str, help='End date in YYYY-MM-DD format')
    args = parser.parse_args()

    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        print("Please provide an end date using --end parameter in YYYY-MM-DD format.")
        return

    # Define the start date as today
    start_date = datetime.today()

    # Loop through dates and execute the command for weekdays
    current_date = start_date
    while current_date <= end_date:
        if not is_weekend(current_date):
            formatted_date = current_date.strftime('%Y-%m-%d')
            cmd = f'python composer_test.py --end {formatted_date}'
            subprocess.run(cmd, shell=True)
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
