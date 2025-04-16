import nsepython as nse
import requests
import csv
import os
import yfinance as yf

nse_symbols = nse.nse_eq_symbols()

def get_website(identifier):
    try:
        # Try as ticker first
        ticker = yf.Ticker(identifier)
        if ticker.info.get('website'):
            return ticker.info['website']
    except:
        pass

    # If not found, try company name lookup
    try:
        symbol = get_symbol(identifier)
        if symbol:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('website', 'Website not found')
    except Exception as e:
        return f'Error: {str(e)}'
    
    return 'Not found'

def get_symbol(company_name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {'q': company_name, 'quotesCount': 1}
    response = requests.get(url, params=params)
    data = response.json()
    return data['quotes'][0]['symbol'] if data['quotes'] else None

os.makedirs('logos', exist_ok=True)

with open('company_websites.csv', mode='r', newline='') as csvfile:
    fieldnames = ['Ticker', 'Website']
    reader = csv.reader(csvfile)
    with open('new_website_list.csv', mode='w', newline='') as csvfile2:
        writer = csv.writer(csvfile2)
        writer.writerow(fieldnames)
        for i in reader:
            if i[1].startswith('https://'):
                website = i[1].replace('https://', '')
                writer.writerow([i[0], website])
                response = requests.get(f"https://img.logo.dev/{website}?token=pk_BxA5jV0UQSib3UcZVK_yyA&size=256")
                if response.status_code == 200:
                    with open(f"logos/{i[0]}.jpg", 'wb') as f:
                        f.write(response.content)
                    print(f"Logo saved for {i[0]}: {website}")
                else:
                    print(f"Failed to fetch logo for {website}")
                
            else:
                writer.writerow(i)

    '''
    for i in nse_symbols:
        website = get_website(i + ".NS")
        if website != 'Not found' and not website.startswith('Error'):
            website = website.replace('https://www.', '')
            writer.writerow({'Ticker': i, 'Website': website})

            response = requests.get(f"https://img.logo.dev/{website}?token=pk_BxA5jV0UQSib3UcZVK_yyA&size=256")
            if response.status_code == 200:
                with open(f"logos/{i}.jpg", 'wb') as f:
                    f.write(response.content)
                print(f"Logo saved for {i}: {website}")
            else:
                print(f"Failed to fetch logo for {i}")
        else:
            writer.writerow({'Ticker': i, 'Website': 'Not found'})
            print(f"Website not found for {i}")

    '''