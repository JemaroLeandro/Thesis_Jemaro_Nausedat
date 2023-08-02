
Client_ID = 
Client_Secret = 
API_Key = 
API_Key_Secret = 
Bearer_Token = 
Access_Token =
Access_Token_Secret =

i = 9



Week_signifier = 'Week ' + str(i)

Week_signifier_previous = 'Week ' + str(i-1)

namer1 = 'Apple'

namer2 = 'Blizzard_Ent'


file_name1 = (Week_signifier + ' - Tweets addressed toward ' + namer1 +'.csv')

file_name2 = (Week_signifier + ' - Tweets addressed toward ' + namer2 +'.csv')

processed_file_name1 = Week_signifier + ' - Processed tweets addressed toward ' + namer1 + '.csv'

processed_file_name2 = Week_signifier + ' - Processed tweets addressed toward ' + namer2 + '.csv'

transformed_file_name1 = 'Weekly Sentiment Series for ' + namer1 + '.csv'

transformed_file_name2 = 'Weekly Sentiment Series for ' + namer2 + '.csv'

financial_file_1 = (namer1 + ' abnormal returns.csv')

financial_file_2 = (namer2 + ' abnormal returns.csv')

regression_ready_1 = (namer1 + ' Regression Dataset.csv')

regression_ready_2 = (namer2 + ' Regression Dataset.csv')