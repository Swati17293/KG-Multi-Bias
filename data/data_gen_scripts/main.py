import os
import sys
import argparse

path = os.getcwd()
sys.path.insert(1, path+'/scripts')

from create_dic import generate_uri_list
from load_data import create_datafiles
# from create_files import create_xml, create_json

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("api_key", type=str, help="<Required> set Event Registry api key")
    parser.add_argument("--category", help="<Optional> set one or more category", nargs='*', default=['news/Business', 'news/Politics', 'news/Technology','news/Environment', 'news/Health', 'news/Science','news/Sports', 'news/Arts_and_Entertainment'])
    parser.add_argument("--source", help="<Optional> set one or more source", nargs='*', default=['expressen.se','info.gp.se']) # 'hotnews.ro' ,'digi24.ro','agerpres.ro','delo.si','24ur.com','novinky.cz','expressen.se','dn.se','info.gp.se','hs.fi'
    parser.add_argument("--date_start", help="<Optional> set event start date", default='2021-01-01')
    parser.add_argument("--date_end", help="<Optional> set event end date", default='2021-12-31')
    # parser.add_argument("--lang", help="<Optional> set language") #ro sl cz sv fi
    
    args = parser.parse_args()

    print('\n\nGenerating folder structure...')

    if os.path.exists('data/') == False:
        os.mkdir('data/')
    if os.path.exists('data/intermediate/') == False:
        os.mkdir('data/intermediate/')
    if os.path.exists('data/raw/') == False:
        os.mkdir('data/raw/')

    print('\n\nGenerating uri dictionary...')

    generate_uri_list(args.api_key, args.category, args.source, args.date_start, args.date_end)

    print('\n\nGenerating raw data file...')
    create_datafiles(args.api_key, args.source)

    # print('\n\nGenerating XML file...')
    # create_xml(args.source)

    # print('\n\nGenerating JSON file...')
    # create_json(args.source)

    print('Finished...')
    print('\n\n')

if __name__ == "__main__":
    main()