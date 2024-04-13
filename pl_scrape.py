import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from io import StringIO

#place into csv that we can train on 

#scraping 2 years of premier league data
years = list(range(2024, 2022, -1))
all_matches = []
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

for year in years:
    print('year: ', year)
    data = requests.get(standings_url)
    print(data)

    #grab the season table
    soup = BeautifulSoup(data.text, 'lxml')
    standings_table = soup.select('table.stats_table')[0]

    #grab the links of links of the teams in the table
    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    print('links: ', links)

    #turn the links into full urls
    team_urls = [f"https://fbref.com{l}" for l in links]
    print('team_urls: ', team_urls)
    
    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"
    
    for team_url in team_urls:
        print('team: ', team_url)
        
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        print('team_name: ', team_name)
        data = requests.get(team_url)

        matches = pd.read_html(StringIO(data.text), match="Scores & Fixtures")
        if not matches:
            print("No 'Scores & Fixtures' table found for", team_name)
            continue
        matches = matches[0]
        
        soup = BeautifulSoup(data.text, 'lxml')
        links = [l.get("href") for l in soup.find_all('a')]

        links = [l for l in links if l and 'all_comps/shooting/' in l]
        if not links:
            print("No 'Shooting' link found for", team_name)
            continue
        
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(StringIO(data.text), match="Shooting")
        if not shooting:
            print("No 'Shooting' table found for", team_name)
            continue
        shooting = shooting[0]
        
        shooting.columns = shooting.columns.droplevel()
        
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            print("Merge failed for", team_name)
            continue
        
        team_data = team_data[team_data["Comp"] == "Premier League"]
        
        team_data["Season"] = year
        team_data["Team"] = team_name
        print('appending data: ', team_data)
        all_matches.append(team_data)
        time.sleep(3)
    

dataset = pd.concat(all_matches)
dataset.to_csv("pl_matches.csv")
