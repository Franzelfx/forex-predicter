import yfinance as yahooFinance

GetFacebookInformation = yahooFinance.Ticker("META")

print(GetFacebookInformation.info)
# Write 2 months of 15 min financial data to csv file
GetFacebookInformation.history(period="2mo", interval="15m").to_csv("facebook.csv")