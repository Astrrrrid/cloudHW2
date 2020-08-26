import mysql.connector
import os
import pandas as pd
#from reducer import current_word
#cmd = 'cat theStars.txt | python mapper.py | sort -k1,1| python reducer.py > result4.csv'
#os.system(cmd)

data = pd.read_csv('result.txt', sep="\t", header=None)
data.columns = ["word", "count"]


#print(data)

mydb = mysql.connector.connect(
  host="127.0.0.1",
  port="3306",
  user="root",
  password="39209093",
  database="mydatabase"
)



print(mydb)
mycursor = mydb.cursor()
mycursor = mydb.cursor(buffered=True)
dict1=dict()
dict1["j"]=1
#print(dict1)
#for x in data:
#    #dict1[x['word']]=int(x['count'])
#    print(x.word)
list=data.values.tolist()
print (list)

for x in list:
    sql = "INSERT INTO wcResult (word, count) VALUES (%s, %s)"
    val = (x[0],x[1])
    mycursor.execute(sql, val)
    mydb.commit()




print(mycursor.rowcount, "was inserted.")




#mycursor.execute("SHOW DATABASES")
#mycursor.execute("CREATE TABLE wcResult (word VARCHAR(255), count INT)")


