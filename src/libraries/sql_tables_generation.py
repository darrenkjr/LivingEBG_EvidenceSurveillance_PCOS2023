import psycopg2
from pathlib import Path
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os 
from dotenv import load_dotenv
load_dotenv()


def create_basic_tables(cur): 
    print('Creating basic tables')
    sql_code_path = Path(__file__).parent / 'psql_tablesetup.sql'
    with open(sql_code_path, 'r') as file: 
        sql_code = file.read()
    cur.execute(sql_code)
    cur.commit()
    print('Basic tables created')

def create_database(db_name, user, pwd, host='localhost', port='5432'): 

    print('Connecting to default database')
    conn = psycopg2.connect(dbname='postgres', user=user, password=pwd, host=host, port=port)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # check if database exists 
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
    if cur.fetchone(): 
        print(f"Database {db_name} already exists")
    else: 
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"Database {db_name} created")
    #close connection 
    cur.close()
    conn.close()

db_name = os.getenv('db_name')
db_user = os.getenv('db_user')
db_pwd = os.getenv('db_pwd')
create_database(db_name = db_name, user = db_user, pwd = db_pwd)
conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_pwd)
cur = conn.cursor()
#check if tables exist 
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
tables = cur.fetchall()
if len(tables) == 0: 
    print('No tables found, creating basic tables')
    create_basic_tables(cur)
else: 
    print('Tables already exist')

for table in tables:
    print(table[0])

cur.close()
conn.close()


#check tables 



